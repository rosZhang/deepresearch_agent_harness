"""Chat routing selects one operation and leaves research execution unchanged."""

import asyncio
import json
import uuid

import pytest
from langchain_core.messages import AIMessage

from backend.app.schemas import MessageSend
from backend.app.services.message_router import MessageRouter, RoutingDecision
from deepresearch_agent.harness.errors import AppError, ErrorCode
from tests.api.test_api_phase4 import ApiRouterLlm, client, new_session, send


def post_message(client, session_id, content, *, client_id=None):
    return client.post(f"/api/v1/sessions/{session_id}/messages", json={
        "client_message_id": client_id or str(uuid.uuid4()), "content": content,
        "source_mode": "graphrag", "workflow_mode": "deep_research",
    })


def test_duplicate_research_skips_llm_and_checks_conflicts(client):
    client.app.state.run_service.schedule = lambda _run_id: None
    session_id = new_session(client)
    first = send(client, session_id, client_id="retry-research")
    llm = client.app.state.chat_service.router._llm
    llm.error = RuntimeError("LLM must not run on a duplicate")
    retry = send(client, session_id, client_id="retry-research")
    assert retry.status_code == 202
    assert retry.json()["run_id"] == first.json()["run_id"]
    assert retry.json()["created"] is False
    assert len(llm.requests) == 1
    conflict = post_message(client, session_id, "different question", client_id="retry-research")
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"
    source_conflict = send(client, session_id, client_id="retry-research", source="web")
    assert source_conflict.status_code == 409
    assert len(llm.requests) == 1


def test_concurrent_retries_route_and_create_once(client):
    client.app.state.run_service.schedule = lambda _run_id: None
    session_id = new_session(client)
    request = MessageSend(client_message_id="concurrent-retry", content="研究数据库恢复", source_mode="graphrag")

    async def submit_twice():
        service = client.app.state.chat_service
        return await asyncio.gather(service.send(session_id, request), service.send(session_id, request))

    first, retry = client.portal.call(submit_twice)
    assert first.run_id == retry.run_id
    assert sorted([first.created, retry.created]) == [False, True]
    assert len(client.app.state.chat_service.router._llm.requests) == 1


@pytest.mark.parametrize("action,content,expected", [
    ("pause_run", "先放一放，过会再做", "paused"),
    ("cancel_run", "这个研究不用做了", "cancelled"),
])
def test_control_tool_and_duplicate_do_not_create_run(client, action, content, expected):
    client.app.state.run_service.schedule = lambda _run_id: None
    session_id = new_session(client)
    run_id = send(client, session_id).json()["run_id"]
    llm = client.app.state.chat_service.router._llm
    llm.action, llm.run_id = action, run_id
    response = post_message(client, session_id, content, client_id="control-retry")
    assert response.status_code == 202
    assert response.json()["run_id"] == run_id
    assert response.json()["created"] is False
    assert client.get(f"/api/v1/runs/{run_id}").json()["status"] == expected
    llm.error = RuntimeError("duplicate control must not route again")
    retry = post_message(client, session_id, content, client_id="control-retry")
    assert retry.status_code == 202
    assert retry.json()["message_id"] == response.json()["message_id"]
    detail = client.get(f"/api/v1/sessions/{session_id}").json()
    assert len(detail["runs"]) == 1
    assert len(detail["messages"]) == 2
    assert json.loads(detail["messages"][-1]["metadata_json"])["control"] == action.removesuffix("_run")
    assert len(llm.requests) == 2


def test_router_can_select_older_paused_run_and_receives_history(client):
    client.app.state.run_service.schedule = lambda _run_id: None
    session_id = new_session(client)
    old_run = post_message(client, session_id, "研究新能源汽车").json()["run_id"]
    assert client.post(f"/api/v1/runs/{old_run}/pause").status_code == 200
    newer_run = post_message(client, session_id, "研究电池回收").json()["run_id"]
    llm = client.app.state.chat_service.router._llm
    llm.action, llm.run_id = "resume_run", old_run
    scheduled = []
    client.app.state.run_service.schedule = scheduled.append
    response = post_message(client, session_id, "回到新能源汽车的研究，接着做", client_id="resume-retry")
    assert response.status_code == 202
    assert response.json()["run_id"] == old_run
    assert scheduled == [old_run]
    payload = json.loads(llm.requests[-1][-1].content)
    candidates = {run["run_id"]: run for run in payload["candidate_runs"]}
    assert candidates[old_run]["query"] == "研究新能源汽车"
    assert candidates[old_run]["status"] == "paused"
    assert candidates[newer_run]["status"] == "queued"
    assert payload["recent_messages"][-1]["content"] == "研究电池回收"
    assert len(llm.requests) == 3
    llm.error = RuntimeError("duplicate resume must not route again")
    retry = post_message(client, session_id, "回到新能源汽车的研究，接着做", client_id="resume-retry")
    assert retry.status_code == 202
    assert retry.json()["message_id"] == response.json()["message_id"]
    assert scheduled == [old_run]
    assert len(llm.requests) == 3


def test_router_cannot_control_other_session(client):
    client.app.state.run_service.schedule = lambda _run_id: None
    other_session = new_session(client)
    other_run = send(client, other_session).json()["run_id"]
    session_id = new_session(client)
    llm = client.app.state.chat_service.router._llm
    llm.action, llm.run_id = "cancel_run", other_run
    response = post_message(client, session_id, "取消另一个会话的任务")
    assert response.status_code == 409
    assert client.get(f"/api/v1/runs/{other_run}").json()["status"] == "queued"
    assert client.get(f"/api/v1/sessions/{session_id}").json()["runs"] == []

    # The service also checks ownership independently of the model output parser.
    class InvalidRouter:
        async def route(self, **kwargs):
            return RoutingDecision(action="cancel_run", run_id=other_run)

    client.app.state.chat_service.router = InvalidRouter()
    assert post_message(client, session_id, "取消").status_code == 404


def test_invalid_state_does_not_record_success_or_create_new_run(client):
    client.app.state.run_service.schedule = lambda _run_id: None
    session_id = new_session(client)
    run_id = send(client, session_id).json()["run_id"]
    assert client.post(f"/api/v1/runs/{run_id}/cancel").status_code == 200
    llm = client.app.state.chat_service.router._llm
    llm.action, llm.run_id = "resume_run", run_id
    response = post_message(client, session_id, "继续")
    assert response.status_code == 409
    detail = client.get(f"/api/v1/sessions/{session_id}").json()
    assert len(detail["messages"]) == len(detail["runs"]) == 1


@pytest.mark.parametrize("failure", ["exception", "no_tool", "multiple", "invalid_args", "unknown", "truncated", "malformed"])
def test_routing_failure_never_falls_back_to_new_research(client, failure):
    session_id = new_session(client)
    llm = client.app.state.chat_service.router._llm
    if failure == "exception":
        llm.error = ConnectionError("private gateway details")
    elif failure == "no_tool":
        llm.response = AIMessage(content="你想恢复哪一个研究任务？")
    elif failure == "truncated":
        llm.response = AIMessage(content="", tool_calls=[{"name": "start_research", "args": {}, "id": "1"}],
                                 response_metadata={"finish_reason": "length"})
    elif failure == "malformed":
        llm.response = AIMessage(content="", invalid_tool_calls=[{
            "name": "resume_run", "args": '{"run_id":', "id": "1", "error": "invalid JSON",
        }])
    else:
        calls = {
            "multiple": [{"name": "start_research", "args": {}, "id": "1"},
                         {"name": "start_research", "args": {}, "id": "2"}],
            "invalid_args": [{"name": "pause_run", "args": {}, "id": "1"}],
            "unknown": [{"name": "delete_everything", "args": {}, "id": "1"}],
        }
        llm.response = AIMessage(content="", tool_calls=calls[failure])
    response = post_message(client, session_id, "继续")
    assert response.status_code == (409 if failure == "no_tool" else 502)
    assert "private gateway" not in response.text
    if failure == "no_tool":
        assert response.json()["error"]["message"] == "你想恢复哪一个研究任务？"
    detail = client.get(f"/api/v1/sessions/{session_id}").json()
    assert detail["runs"] == detail["messages"] == []
    assert len(llm.requests) == 1


@pytest.mark.asyncio
async def test_router_initializes_separate_client_and_invokes_once(monkeypatch):
    from backend.app.services import message_router

    llm = ApiRouterLlm()
    options = []
    monkeypatch.setattr(message_router.settings, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(message_router.settings, "MESSAGE_ROUTER_MODEL", "router-model")
    monkeypatch.setattr(message_router.settings, "MESSAGE_ROUTER_MAX_OUTPUT_TOKENS", 512)
    monkeypatch.setattr(message_router, "get_llm_model", lambda **kwargs: options.append(kwargs) or llm)
    decision = await MessageRouter().route(
        content="研究市场", history=[], runs=[], summary=None,
        source_mode="web", workflow_mode="plan_execute_report",
    )
    assert decision.action == "start_research"
    assert options == [{"model": "router-model", "temperature": 0.0, "max_tokens": 512}]
    assert len(llm.requests) == 1
    assert {tool["function"]["name"] for tool in llm.tools} == {
        "start_research", "resume_run", "pause_run", "cancel_run",
    }


@pytest.mark.asyncio
async def test_missing_router_configuration_does_not_create_client(monkeypatch):
    from backend.app.services import message_router

    monkeypatch.setattr(message_router.settings, "OPENAI_API_KEY", "")
    with pytest.raises(AppError) as error:
        await MessageRouter().route(content="研究市场", history=[], runs=[], summary=None,
                                    source_mode="web", workflow_mode="deep_research")
    assert error.value.code is ErrorCode.CONFIGURATION_INVALID
