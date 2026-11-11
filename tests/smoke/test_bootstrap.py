"""装配迁移后，各入口共享依赖，每个 Run 的研究资源独立并能正确关闭。"""

from types import SimpleNamespace

import pytest

from backend.app import bootstrap
from deepresearch_agent.harness.contracts import SourceMode
from deepresearch_agent.harness.run_context import RunContext


@pytest.mark.parametrize("interrupted", [False, True])
@pytest.mark.asyncio
async def test_runtime_scope_closes_agents_and_shares_common_services(tmp_path, monkeypatch, interrupted):
    agents = []

    class ResearchAgent:
        def __init__(self, **kwargs):
            self.retrieval_provider = kwargs["retrieval_provider"]
            self.research_tool = SimpleNamespace(max_iterations=3)
            self.closed = False
            agents.append(self)

        def close(self):
            self.closed = True

    provider = SimpleNamespace(mode=SourceMode.WEB, provider_name="test-provider")
    monkeypatch.setattr(bootstrap, "create_default_router", lambda: SimpleNamespace(for_mode=lambda mode: provider))
    monkeypatch.setattr(bootstrap, "DeepResearchAgent", ResearchAgent)
    monkeypatch.setattr(bootstrap.settings, "OPENAI_API_KEY", "")
    services = bootstrap.build_services(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'bootstrap.db'}",
        artifact_root=tmp_path / "artifacts", skills_root=tmp_path / "skills",
    )
    service = services.run_service
    assert services.chat_service.runs is service.runs
    assert services.chat_service.messages is service.messages

    def execute_scope():
        with service._runtime_factory() as runtime:
            assert runtime.context_builder.memory_service is services.memory_service
            with service._runtime_factory() as another:
                assert another is not runtime
                assert another.context_builder is runtime.context_builder
                assert another.events is runtime.events is service.event_bus
            context = RunContext(
                run_id="run_test", session_id="ses_test", trigger_message_id="msg_test",
                original_query="研究新能源汽车", source_mode="web", workflow_mode="deep_research",
                status="planning",
            )
            driver = runtime.workflow_factory(context, runtime.events)
            assert driver.agent is agents[0]
            assert not agents[0].closed
            if interrupted:
                raise RuntimeError("stage failed")

    try:
        if interrupted:
            with pytest.raises(RuntimeError, match="stage failed"):
                execute_scope()
        else:
            execute_scope()
        assert len(agents) == 1 and agents[0].closed
    finally:
        await services.database.close()
