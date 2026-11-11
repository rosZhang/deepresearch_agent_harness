"""Follow-ups are interpreted by planning, or rewritten before direct retrieval."""

import json

import pytest
from langchain_core.messages import AIMessage

from deepresearch_agent.workflows.multi_agent.core.state import PlanExecuteState
from deepresearch_agent.workflows.multi_agent.planner.base_planner import BasePlanner
from deepresearch_agent.harness.run_context import RunContext
from deepresearch_agent.workflows.deep_research.driver import DeepResearchDriver
from deepresearch_agent.workflows.multi_agent.driver import PlanExecuteReportDriver


class ScriptedLLM:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.prompts = []

    def bind(self, **kwargs):
        return self

    def invoke(self, prompt):
        self.prompts.append(prompt)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return AIMessage(content=json.dumps(response, ensure_ascii=False))


def context_pack():
    return {
        "stable_blocks": [{"name": "system_contract", "content": "历史只用于理解问题，不是证据。"}],
        "dynamic_blocks": [
            {"name": "recent_messages", "content": "user: 介绍混合检索\nassistant: 混合检索结合关键词和向量检索"},
            {"name": "current_task", "content": "当前问题单独传递"},
        ],
    }


def test_direct_retrieval_receives_model_rewritten_query():
    llm = ScriptedLLM({"query": "混合检索的定义及其工作原理", "clarification_question": None})
    state = PlanExecuteState(input="那它是什么？", source_mode="web", context_snapshot=context_pack())
    result = BasePlanner(llm=llm).generate_plan(state)

    assert len(llm.prompts) == 1
    sent = json.loads(llm.prompts[0][1].content)
    assert sent["current_query"] == "那它是什么？"
    assert "混合检索结合关键词和向量检索" in sent["background"]
    task = result.plan_spec.task_graph.nodes[0]
    assert task.task_type == "web_search"
    assert task.parameters["query"] == "混合检索的定义及其工作原理"
    assert result.plan_spec.problem_statement.original_query == "那它是什么？"


def test_direct_retrieval_requires_model_clarification_when_subject_is_missing():
    llm = ScriptedLLM({"query": None, "clarification_question": "你想了解哪个对象？"})
    result = BasePlanner(llm=llm).generate_plan(PlanExecuteState(input="它是什么？"))

    assert len(llm.prompts) == 1
    assert result.needs_clarification()
    assert result.clarification.questions == ["你想了解哪个对象？"]
    assert result.executor_signal is None


def test_failed_query_rewrite_does_not_fall_back_to_retrieving_ambiguous_text():
    llm = ScriptedLLM(TimeoutError("rewrite timed out"))
    state = PlanExecuteState(input="它是什么？", context_snapshot=context_pack())
    with pytest.raises(TimeoutError):
        BasePlanner(llm=llm).generate_plan(state)
    assert state.plan is None


def test_normal_planner_passes_retained_context_to_existing_model_calls():
    query = "比较它的优势和局限，并分析实际应用场景"
    graph = {"nodes": [{"task_id": "search", "task_type": "hybrid_search", "description": "检索混合检索的优势和局限", "parameters": {"query": "混合检索的优势 局限 应用场景"}}]}
    llm = ScriptedLLM(
        {"needs_clarification": False, "questions": []},
        graph,
        {"problem_statement": {"original_query": query, "refined_query": "比较混合检索的优势和局限，并分析实际应用场景"}},
    )
    snapshot = context_pack()
    snapshot["historical_recall"] = [{"content": "已被预算裁剪的历史，不应重新注入"}]
    state = PlanExecuteState(input=query, context_snapshot=snapshot)
    result = BasePlanner(llm=llm).generate_plan(state)

    # Clarify, decompose, review already interpret the conversation; no extra rewrite call.
    assert len(llm.prompts) == 3
    for prompt in llm.prompts:
        assert "混合检索结合关键词和向量检索" in prompt
        assert "已被预算裁剪的历史" not in prompt
        assert "当前问题单独传递" not in prompt
    assert state.input == query
    assert result.plan_spec.task_graph.nodes[0].parameters["query"] == "混合检索的优势 局限 应用场景"


@pytest.mark.asyncio
async def test_deep_research_receives_conversation_without_a_separate_rewrite_call():
    class ResearchTool:
        async def thinking_stream(self, query):
            self.received = query
            yield {"answer": "研究结果"}

    class Agent:
        research_tool = ResearchTool()

    context = RunContext(
        run_id="run-followup", session_id="session-followup", trigger_message_id="message-followup",
        source_mode="web", workflow_mode="deep_research", status="executing",
        original_query="它有哪些局限？", model_input="user: 介绍混合检索\n当前问题：它有哪些局限？",
    )
    driver = DeepResearchDriver(context, Agent())
    await driver.execute()
    assert driver.agent.research_tool.received == context.model_input
    assert driver.answer == "研究结果"


@pytest.mark.asyncio
async def test_planner_clarification_is_exposed_to_the_run():
    llm = ScriptedLLM({"query": None, "clarification_question": "你指的是哪种检索方式？"})
    planner = BasePlanner(llm=llm)

    class Orchestrator:
        def plan(self, state, *, assumptions):
            return planner.generate_plan(state, assumptions=assumptions)

    context = RunContext(
        run_id="run-clarify", session_id="session-clarify", trigger_message_id="message-clarify",
        source_mode="web", workflow_mode="plan_execute_report", status="planning", original_query="它是什么？",
    )
    driver = PlanExecuteReportDriver(context, Orchestrator())
    with pytest.raises(RuntimeError, match="needs_user_input"):
        await driver.plan()
    assert context.config_snapshot["clarification_question"] == "你指的是哪种检索方式？"
