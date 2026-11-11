"""将研究工作流接入 Harness 的阶段执行协议。"""

from __future__ import annotations

import asyncio
from typing import Any

from deepresearch_agent.workflows.multi_agent.core.execution_record import ExecutionRecord
from deepresearch_agent.workflows.multi_agent.core.retrieval_result import RetrievalResult

from deepresearch_agent.harness.run_context import RunContext
from deepresearch_agent.harness.report_safety import add_inline_citations, citation_evidence_ids, has_internal_material, rank_evidence, sanitize_report
from deepresearch_agent.harness.report_safety import evidence_only_report


class DeepResearchDriver:
    """Keeps the existing DeepResearch loop as execute, with deterministic plan/report stages."""

    def __init__(self, context: RunContext, agent: Any, events: Any = None):
        self.context, self.agent = context, agent
        self.events = events
        self._iterations: dict[int, dict[str, Any]] = {}
        self.thinking: str | None = None
        research_tool = getattr(agent, "research_tool", None)
        if getattr(research_tool, "deep_research", None) is not None:
            research_tool = research_tool.deep_research
        max_iterations = context.config_snapshot.get("deep_research_max_iterations")
        if max_iterations is not None and research_tool is not None:
            research_tool.max_iterations = max(1, min(int(max_iterations), research_tool.max_iterations))
        payload = context.workflow_state
        self.answer: str | None = payload.get("answer")
        self.results = [RetrievalResult.from_dict(item) for item in payload.get("results", [])]
        self._report: str | None = payload.get("report")
        self._plan = payload.get("plan")
        self._repair_pending = bool(payload.get("repair_pending"))

    async def plan(self, failures: list[str] | None = None) -> None:
        if failures and set(failures) & {"min_evidence", "claim_support", "source_match"}:
            # A verification-driven replan must actually execute research again;
            # retaining the prior answer would turn replan into a no-op.
            self.answer = None
            self.results = []
        self._plan = {
            "plan_id": f"plan_{self.context.run_id}", "version": self.context.plan_version + 1,
            "status": "executing", "source_mode": self.context.source_mode.value,
            "tasks": [{"task_id": f"task_{self.context.run_id}_research", "task_type": "deep_research", "status": "pending", "source_mode": self.context.source_mode.value, "description": self.context.original_query, "parameters": {"query": self.context.resolved_query or self.context.original_query}}],
            "verification_failures": failures or [],
        }

    async def execute(self) -> None:
        if self.answer is None:
            tool = getattr(self.agent, "research_tool", None)
            if tool is not None and hasattr(tool, "thinking_stream"):
                # 直接流式消费 research_tool.thinking_stream（与 LangGraph 研究节点同源），
                # 逐轮迭代通过 progress_callback 实时发布 agent.progress / iteration.completed。
                async def _on_progress(msg: dict) -> None:
                    await self._on_tool_progress(msg)

                inner = getattr(tool, "deep_research", None)
                targets = [tool] + ([inner] if inner is not None else [])
                for target in targets:
                    target.progress_callback = _on_progress
                try:
                    async for chunk in tool.thinking_stream(
                        self.context.model_input or self.context.resolved_query or self.context.original_query
                    ):
                        if isinstance(chunk, dict):
                            if chunk.get("answer"):
                                self.answer = chunk["answer"]
                            self.thinking = chunk.get("thinking")
                finally:
                    for target in targets:
                        target.progress_callback = None
            else:
                # 无流式接口（测试桩等）：回退到闭路 ask 路径
                self.answer = await asyncio.to_thread(
                    self.agent.ask,
                    self.context.model_input or self.context.resolved_query or self.context.original_query,
                    self.context.session_id,
                    bypass_cache=self.context.source_mode.value == "web",
                )
        tool = getattr(self.agent, "research_tool", None)
        provider_results = list(getattr(tool, "provider_results", []) or [])
        if not provider_results and getattr(tool, "deep_research", None) is not None:
            provider_results = list(getattr(tool.deep_research, "provider_results", []) or [])
        if provider_results:
            self.results = provider_results

    async def _on_tool_progress(self, msg: dict) -> None:
        """聚合工具迭代进度并发布 agent.progress / iteration.completed 事件"""
        if self.events is None:
            return
        kind = msg.get("kind")
        idx = int(msg.get("iteration_index", 0))
        if kind == "iteration":
            self._iterations.setdefault(idx, {"queries": [], "total_results": 0, "info_snippets": []})
            await self.events.publish(self.context.run_id, "agent.progress", stage="executing",
                                      payload={"iteration_index": idx, "kind": "iteration"})
        elif kind == "search":
            acc = self._iterations.setdefault(idx, {"queries": [], "total_results": 0, "info_snippets": []})
            query = msg.get("query", "")
            result_count = int(msg.get("result_count") or 0)
            found_useful = bool(msg.get("found_useful"))
            acc["queries"].append({"query": query, "result_count": result_count, "found_useful": found_useful})
            acc["total_results"] += result_count
            preview = msg.get("useful_info_preview")
            if preview and len(acc["info_snippets"]) < 3:
                acc["info_snippets"].append(str(preview)[:200])
            await self.events.publish(self.context.run_id, "agent.progress", stage="executing",
                                      payload={"iteration_index": idx, "kind": "search", "query": query,
                                               "result_count": result_count, "found_useful": found_useful,
                                               "useful_info_preview": msg.get("useful_info_preview")})
        elif kind == "iteration_done":
            acc = self._iterations.get(idx, {"queries": [], "total_results": 0, "info_snippets": []})
            tool = getattr(self.agent, "research_tool", None)
            if getattr(tool, "deep_research", None) is not None:
                tool = tool.deep_research
            max_iterations = getattr(tool, "max_iterations", None)
            await self.events.publish(self.context.run_id, "iteration.completed", stage="executing",
                                      payload={"iteration_index": idx, "kind": "iteration",
                                               "queries": acc["queries"], "total_results": acc["total_results"],
                                               "info_snippets": acc["info_snippets"],
                                               "max_iterations": max_iterations,
                                               "iterations_total": len(self._iterations)})
        elif kind == "answer":
            await self.events.publish(self.context.run_id, "iteration.completed", stage="executing",
                                      payload={"iteration_index": idx, "kind": "answer",
                                               "answer_char_count": msg.get("answer_char_count"),
                                               "iterations_total": len(self._iterations)})

    async def report(self) -> str:
        if self._repair_pending and self._report is not None:
            self._repair_pending = False
            return self._report
        ranked = rank_evidence(self.results)
        body = add_inline_citations(sanitize_report(self.answer), citation_evidence_ids(ranked))
        limitation = "\n\n## 局限\n\n当前证据来源数量有限，结论应结合更多独立来源复核。" if len({item.metadata.source_id for item in ranked}) < 2 else ""
        self._report = f"# 深度研究报告\n\n{body}{limitation}".strip()
        return self._report

    async def repair_report(self, failures: list[str]) -> bool:
        if self._report is None:
            return False
        supported = {"citation_integrity", "required_section", "report_consistency", "source_diversity"}
        if not set(failures).issubset(supported):
            return False
        if set(failures) & {"citation_integrity", "report_consistency"}:
            self._report, selected = evidence_only_report(
                self.results,
                required_sections=list(self.context.config_snapshot.get("required_sections", [])),
            )
            if not selected:
                return False
        if "source_diversity" in failures and "局限" not in self._report:
            self._report += "\n\n## 局限\n\n当前证据来源数量有限。"
        self._repair_pending = True
        return True

    def snapshot(self) -> dict[str, Any]:
        return {"answer": self.answer, "report": self._report, "plan": self._plan, "results": [item.to_dict() for item in self.results], "repair_pending": self._repair_pending}

    def execution_records(self) -> list[ExecutionRecord]:
        task_id = f"task_{self.context.run_id}_research"
        tool = getattr(self.agent, "research_tool", None)
        if getattr(tool, "deep_research", None) is not None:
            tool = tool.deep_research
        calls = list(getattr(tool, "provider_calls", []) or [])
        if not calls and self.results:
            calls = [{"tool_call_id": f"call_{self.context.run_id}_research", "query": self.context.resolved_query or self.context.original_query, "results": self.results}]
        records = []
        tool_name = "tavily_search" if self.context.source_mode.value == "web" else "deep_research"
        for index, call in enumerate(calls):
            results = list(call.get("results", []))
            records.append(ExecutionRecord(
                record_id=f"record_{self.context.run_id}_{index}", task_id=task_id,
                session_id=self.context.session_id, worker_type="deep_research",
                inputs={"query": call.get("query")},
                tool_calls=[{
                    "tool_name": tool_name, "tool_call_id": call["tool_call_id"],
                    "source_mode": self.context.source_mode.value, "args": {"query": call.get("query")},
                    "result": {"result_ids": [item.result_id for item in results]}, "status": "success",
                }], evidence=results,
            ))
        return records

    def evidence_results(self) -> list[tuple[str | None, str | None, str, RetrievalResult]]:
        task_id = f"task_{self.context.run_id}_research"
        provider = getattr(self.agent, "retrieval_provider", None)
        provider_name = getattr(provider, "provider_name", "deep_research")
        tool = getattr(self.agent, "research_tool", None)
        if getattr(tool, "deep_research", None) is not None:
            tool = tool.deep_research
        calls = list(getattr(tool, "provider_calls", []) or [])
        if calls:
            return [(task_id, call["tool_call_id"], provider_name, item) for call in calls for item in call.get("results", [])]
        return [(task_id, f"call_{self.context.run_id}_research", provider_name, item) for item in self.results]

    def report_consistency(self) -> bool | None:
        return bool(self._report and self._report.strip()) and not has_internal_material(self._report)

    def evidence_card_coverage(self) -> dict[str, Any] | None:
        return None

    def report_metrics(self) -> dict[str, Any]:
        return {}

    def plan_record(self) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
        if not self._plan:
            return None
        return self._plan, list(self._plan["tasks"])

