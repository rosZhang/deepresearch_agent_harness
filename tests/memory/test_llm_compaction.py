"""Behavioral checks for incremental summaries and lossless raw-window rollover."""

import asyncio
import json

import pytest
import pytest_asyncio
from langchain_core.messages import AIMessage

from deepresearch_agent.persistence.schemas import MessageCreate, RunCreate, SessionCreate
from deepresearch_agent.context import ContextBuilder, ContextCompactor
from deepresearch_agent.context.compactor import ContextCompactionError
from deepresearch_agent.context.tokens import count_tokens
from deepresearch_agent.harness import SourceMode, WorkflowMode
from deepresearch_agent.harness.errors import RunCancelled
from deepresearch_agent.harness.run_context import RunContext
from deepresearch_agent.memory import MemoryService
from deepresearch_agent.persistence import Database
from deepresearch_agent.persistence.repositories import (
    AuditRepository, MemoryRepository, MessageRepository, RunRepository, SessionRepository,
)
from deepresearch_agent.sessions import SessionSearchService


class SummaryLlm:
    model_name = "summary-test"

    def __init__(self):
        self.requests = []
        self.error = None

    async def ainvoke(self, messages):
        self.requests.append(messages)
        if self.error:
            raise self.error
        return AIMessage(content=f"保留目标与约束；最新决定优先。摘要版本 {len(self.requests)}。")

    @property
    def payloads(self):
        return [json.loads(request[-1].content) for request in self.requests]


@pytest_asyncio.fixture
async def setup(tmp_path):
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'compaction.db'}")
    await database.create_schema()
    sessions, messages = SessionRepository(database), MessageRepository(database)
    session = await sessions.create(SessionCreate(title="conversation"))
    yield database, sessions, messages, session.session_id
    await database.close()


async def append(messages, session_id, contents):
    result = []
    for index, content in enumerate(contents):
        message, _ = await messages.append(MessageCreate(
            session_id=session_id, role="user" if index % 2 == 0 else "assistant", content=content,
        ))
        result.append(message)
    return result


@pytest.mark.asyncio
async def test_below_threshold_keeps_every_unsummarized_message(setup):
    _, sessions, messages, session_id = setup
    originals = await append(messages, session_id, [f"short-{index}" for index in range(12)])
    llm = SummaryLlm()
    result = await ContextCompactor(
        sessions, messages, llm=llm, threshold_tokens=1000, protect_recent_messages=2,
    ).prepare(session_id)
    assert result.summary == {} and not llm.requests
    # A fixed last-N slice would silently lose ten messages here.
    assert [item.message_id for item in result.messages] == [item.message_id for item in originals]


@pytest.mark.asyncio
async def test_rollover_is_incremental_and_does_not_recompress_database_history(setup):
    _, sessions, messages, session_id = setup
    originals = await append(messages, session_id, [f"turn-{i} " + "background " * 30 for i in range(6)])
    llm = SummaryLlm()
    compactor = ContextCompactor(sessions, messages, llm=llm, threshold_tokens=200, protect_recent_messages=2)
    first = await compactor.prepare(session_id)
    assert first.status == "compacted"
    assert first.summary["covered_message_ids"] == [item.message_id for item in originals[:-2]]
    assert [item.message_id for item in first.messages] == [item.message_id for item in originals[-2:]]
    assert llm.payloads[0]["messages"][-1]["content"] == originals[-3].content
    assert await compactor.get_or_update(session_id) == first.summary
    assert len(llm.requests) == 1

    await append(messages, session_id, ["new short question", "new short answer"])
    await compactor.prepare(session_id)
    assert len(llm.requests) == 1  # Persisted raw history is still large, active input is small.

    newer = await append(messages, session_id, [f"new-{i} " + "new background " * 70 for i in range(4)])
    second = await compactor.prepare(session_id)
    assert second.status == "compacted" and len(llm.requests) == 2
    assert llm.payloads[1]["previous_summary"] == first.summary["content"]
    reprocessed = {item["message_id"] for item in llm.payloads[1]["messages"]}
    assert not reprocessed.intersection(first.summary["covered_message_ids"])
    assert [item.message_id for item in second.messages] == [item.message_id for item in newer[-2:]]
    assert (await sessions.get(session_id)).summary_json
    assert len(await messages.list_for_session(session_id, limit=500)) == 12


@pytest.mark.asyncio
async def test_batches_preserve_the_entire_long_message(setup):
    _, sessions, messages, session_id = setup
    long_text = "old report section " * 700 + "IMPORTANT FINAL CONSTRAINT"
    originals = await append(messages, session_id, [long_text, "old reply", "recent user", "recent reply"])
    llm = SummaryLlm()
    compactor = ContextCompactor(
        sessions, messages, llm=llm, threshold_tokens=100, protect_recent_messages=2,
        max_input_tokens=900, target_tokens=100,
    )
    result = await compactor.prepare(session_id)
    assert result.status == "compacted" and len(llm.requests) > 1
    pieces = [entry["content"] for payload in llm.payloads for entry in payload["messages"]
              if entry["message_id"] == originals[0].message_id]
    assert "".join(pieces) == long_text
    assert all(compactor._request_tokens(request) <= 900 for request in llm.requests)
    assert result.summary["covered_message_ids"] == [item.message_id for item in originals[:2]]


@pytest.mark.asyncio
async def test_failed_later_batch_does_not_advance_coverage(setup):
    _, sessions, messages, session_id = setup
    originals = await append(messages, session_id, ["long text " * 800, "old answer", "latest", "reply"])

    class FailingLlm(SummaryLlm):
        async def ainvoke(self, request):
            if self.requests:
                raise TimeoutError("temporary error")
            return await super().ainvoke(request)

    result = await ContextCompactor(
        sessions, messages, llm=FailingLlm(), threshold_tokens=100,
        protect_recent_messages=2, max_input_tokens=900,
    ).prepare(session_id)
    assert result.status == "failed" and result.summary == {}
    assert [item.message_id for item in result.messages] == [item.message_id for item in originals]
    assert (await sessions.get(session_id)).summary_json is None


@pytest.mark.asyncio
async def test_failure_retains_previous_summary_and_every_pending_message(setup):
    _, sessions, messages, session_id = setup
    await append(messages, session_id, ["old " * 80 for _ in range(6)])
    llm = SummaryLlm()
    compactor = ContextCompactor(sessions, messages, llm=llm, threshold_tokens=200, protect_recent_messages=2)
    first = await compactor.prepare(session_id)
    newer = await append(messages, session_id, ["new " * 200 for _ in range(4)])
    llm.error = ConnectionError("offline")
    failed = await compactor.prepare(session_id)
    assert failed.status == "failed" and failed.summary == first.summary
    assert [item.message_id for item in failed.messages] == [item.message_id for item in first.messages + newer]
    assert json.loads((await sessions.get(session_id)).summary_json) == first.summary
    with pytest.raises(ContextCompactionError, match="原文仍完整保留"):
        await compactor.prepare(session_id, max_tokens=400)
    assert json.loads((await sessions.get(session_id)).summary_json) == first.summary


@pytest.mark.asyncio
async def test_no_llm_never_silently_discards_history(setup):
    _, sessions, messages, session_id = setup
    originals = await append(messages, session_id, ["original " * 100 for _ in range(6)])
    compactor = ContextCompactor(sessions, messages, threshold_tokens=100, protect_recent_messages=2)
    result = await compactor.prepare(session_id)
    assert result.status == "disabled" and result.messages[-1].content == originals[-1].content
    assert len(result.messages) == len(originals)
    with pytest.raises(ContextCompactionError, match="未启用或未配置"):
        await compactor.prepare(session_id, max_tokens=200)


@pytest.mark.asyncio
async def test_cancellation_propagates_without_saving_summary(setup):
    _, sessions, messages, session_id = setup
    await append(messages, session_id, ["history " * 100 for _ in range(5)])
    llm = SummaryLlm()
    llm.error = RunCancelled("cancelled")
    with pytest.raises(RunCancelled):
        await ContextCompactor(sessions, messages, llm=llm, threshold_tokens=100, protect_recent_messages=2).prepare(session_id)
    assert (await sessions.get(session_id)).summary_json is None


@pytest.mark.asyncio
async def test_concurrent_preparations_share_committed_summary(setup):
    _, sessions, messages, session_id = setup
    await append(messages, session_id, ["history " * 40 for _ in range(8)])
    llm = SummaryLlm()
    compactor = ContextCompactor(sessions, messages, llm=llm, threshold_tokens=200, protect_recent_messages=2)
    first, second = await asyncio.gather(compactor.prepare(session_id), compactor.prepare(session_id))
    assert first.summary == second.summary and len(llm.requests) == 1


@pytest.mark.asyncio
async def test_history_past_500_messages_is_loaded_and_current_turn_excluded(setup):
    _, sessions, messages, session_id = setup
    originals = await append(messages, session_id, [f"message-{i}" for i in range(505)])
    llm = SummaryLlm()
    compactor = ContextCompactor(sessions, messages, llm=llm, threshold_tokens=100, protect_recent_messages=2)
    result = await compactor.prepare(session_id, before_message_id=originals[-1].message_id)
    represented = result.summary["covered_message_ids"] + [item.message_id for item in result.messages]
    assert represented == [item.message_id for item in originals[:-1]]
    assert originals[-1].message_id not in str(llm.payloads)


@pytest.mark.asyncio
async def test_context_contains_summary_plus_complete_raw_tail_without_memory_writes(setup):
    database, sessions, messages, session_id = setup
    originals = await append(messages, session_id, ["old discussion " * 80 for _ in range(6)])
    recent = await append(messages, session_id, ["recent question", "detail " * 350 + "TAIL MUST SURVIVE"])
    trigger, run, _ = await RunRepository(database).create_for_user_message(
        MessageCreate(session_id=session_id, role="user", content="current question", client_message_id="current"),
        RunCreate(session_id=session_id, trigger_message_id="atomic", source_mode=SourceMode.GRAPHRAG,
                  workflow_mode=WorkflowMode.DEEP_RESEARCH),
    )
    llm = SummaryLlm()
    service = MemoryService(MemoryRepository(database), AuditRepository(database))
    builder = ContextBuilder(
        messages=messages, sessions=sessions, memory_service=service,
        session_search=SessionSearchService(messages, sessions),
        compactor=ContextCompactor(sessions, messages, llm=llm, threshold_tokens=200, protect_recent_messages=2),
        max_tokens=1800, max_chars=12000,
    )
    context = await builder.build(RunContext(
        run_id=run.run_id, session_id=session_id, trigger_message_id=trigger.message_id,
        source_mode=SourceMode.GRAPHRAG, workflow_mode=WorkflowMode.DEEP_RESEARCH,
        status="context_building", original_query=trigger.content,
    ))
    summary = context.context_snapshot["session_summary"]
    assert summary["covered_message_ids"] == [item.message_id for item in originals]
    assert context.context_snapshot["recent_messages"][-1]["content"] == recent[-1].content
    assert recent[-1].content in context.model_input
    assert context.model_input.count(trigger.content) == 1
    assert "current question" not in str(llm.payloads)
    assert "covered_message_ids" not in context.model_input
    assert count_tokens(context.model_input) <= 1800
    assert context.context_snapshot["total_input_tokens"] == count_tokens(context.model_input)
    assert not (await service.build_snapshot())["items"]["user"]
    assert not (await service.build_snapshot())["items"]["project"]


@pytest.mark.asyncio
async def test_compression_client_is_independent_of_memory_extraction(setup, monkeypatch, tmp_path):
    from backend.app import bootstrap as module

    database, *_ = setup
    llm = SummaryLlm()
    created = []

    def get_model(**kwargs):
        created.append(kwargs)
        return llm

    monkeypatch.setattr(module, "get_llm_model", get_model)
    monkeypatch.setattr(module.settings, "OPENAI_API_KEY", "test-placeholder")
    monkeypatch.setattr(module.settings, "CONTEXT_COMPRESSION_MODEL", "compression-model")
    monkeypatch.setattr(module.settings, "CONTEXT_COMPRESSION_ENABLED", True)
    monkeypatch.setattr(module.settings, "MEMORY_BACKGROUND_REVIEW_ENABLED", False)
    monkeypatch.setattr(module.settings, "LEARNING_REVIEW_ENABLED", False)
    services = module.build_services(database_url=database.url, artifact_root=tmp_path / "artifacts", skills_root=tmp_path / "skills")
    with services.run_service._runtime_factory() as service:
        assert service.memory_extractor is None
        assert service.context_builder.compactor.llm is llm
    assert created == [{"model": "compression-model", "temperature": 0.0,
                        "max_tokens": module.settings.CONTEXT_COMPRESSION_MAX_OUTPUT_TOKENS}]
    monkeypatch.setattr(module.settings, "CONTEXT_COMPRESSION_ENABLED", False)
    disabled = module.build_services(database_url=database.url, artifact_root=tmp_path / "artifacts", skills_root=tmp_path / "skills")
    with disabled.run_service._runtime_factory() as runtime:
        assert runtime.context_builder.compactor.llm is None and len(created) == 1
    await services.database.close()
    await disabled.database.close()


@pytest.mark.asyncio
async def test_oversized_recent_report_is_summarized_in_full_instead_of_truncated(setup):
    _, sessions, messages, session_id = setup
    originals = await append(messages, session_id, ["recent user", "report " * 900 + "FINAL RECOMMENDATION"])
    llm = SummaryLlm()
    result = await ContextCompactor(
        sessions, messages, llm=llm, threshold_tokens=100, max_input_tokens=1000,
    ).prepare(session_id, max_tokens=300, max_chars=1600)
    assert result.status == "compacted" and result.messages == []
    assert result.summary["covered_message_ids"] == [item.message_id for item in originals]
    parts = [entry["content"] for payload in llm.payloads for entry in payload["messages"]
             if entry["message_id"] == originals[-1].message_id]
    assert "".join(parts) == originals[-1].content
    assert (await messages.get(originals[-1].message_id)).content == originals[-1].content


@pytest.mark.asyncio
async def test_existing_summary_can_shrink_when_current_task_leaves_less_room(setup):
    _, sessions, messages, session_id = setup
    originals = await append(messages, session_id, ["source message"])
    await sessions.update_summary(session_id, {
        "format": "llm-session-summary-v1", "content": "old summary " * 300,
        "covered_message_ids": [originals[0].message_id], "through_message_id": originals[0].message_id,
    })
    llm = SummaryLlm()
    result = await ContextCompactor(sessions, messages, llm=llm).prepare(session_id, max_tokens=200)
    assert result.status == "compacted" and result.messages == []
    assert llm.payloads[0]["messages"] == []
    assert llm.payloads[0]["previous_summary"].startswith("old summary")
    assert result.summary["covered_message_ids"] == [originals[0].message_id]


@pytest.mark.asyncio
async def test_legacy_keyword_summary_is_rebuilt_from_original_messages(setup):
    _, sessions, messages, session_id = setup
    originals = await append(messages, session_id, ["original " * 80 for _ in range(6)])
    await sessions.update_summary(session_id, {"goal": "legacy keyword summary", "covered_message_ids": ["wrong-id"]})
    llm = SummaryLlm()
    result = await ContextCompactor(
        sessions, messages, llm=llm, threshold_tokens=100, protect_recent_messages=2,
    ).prepare(session_id)
    assert llm.payloads[0]["previous_summary"] == ""
    assert result.summary["covered_message_ids"] == [item.message_id for item in originals[:-2]]
    assert "legacy keyword summary" not in str(llm.payloads)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["empty", "truncated", "oversized"])
async def test_invalid_summary_never_marks_raw_history_as_covered(setup, mode):
    _, sessions, messages, session_id = setup
    originals = await append(messages, session_id, ["history " * 70 for _ in range(5)])

    class InvalidLlm:
        async def ainvoke(self, request):
            if mode == "empty":
                return AIMessage(content="")
            if mode == "truncated":
                return AIMessage(content="partial summary", response_metadata={"finish_reason": "length"})
            return AIMessage(content="too long " * 300)

    result = await ContextCompactor(
        sessions, messages, llm=InvalidLlm(), threshold_tokens=100,
        protect_recent_messages=2, target_tokens=100,
    ).prepare(session_id)
    assert result.status == "failed" and result.summary == {}
    assert [item.message_id for item in result.messages] == [item.message_id for item in originals]
    assert (await sessions.get(session_id)).summary_json is None
