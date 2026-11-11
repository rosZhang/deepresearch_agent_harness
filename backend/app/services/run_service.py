"""Run 的后台调度和控制；组件装配集中在 backend.app.bootstrap。"""

from __future__ import annotations

import asyncio
import json

from deepresearch_agent.config import settings
from deepresearch_agent.config.settings import AUTO_RESUME_RUNS
from deepresearch_agent.harness.contracts import SourceMode, WorkflowMode
from deepresearch_agent.harness.recovery import RecoveryManager
from deepresearch_agent.models.prefix_cache import clear_run_cancelled, mark_run_cancelled
from deepresearch_agent.persistence.repositories import ContractRepository
from deepresearch_agent.persistence.schemas import MessageCreate, RunCreate, SessionCreate


class RunService:
    def __init__(self, *, database, runs, messages, sessions, event_bus,
                 runtime_factory, skill_learning, skill_promotion, skill_registry):
        # 全部依赖由 bootstrap 传入；这里不创建 LLM、上下文或研究工作流。
        self.database = database
        self.runs, self.messages, self.sessions = runs, messages, sessions
        self.event_bus, self.events = event_bus, event_bus.repository
        self._runtime_factory = runtime_factory
        self.skill_learning, self.skill_promotion = skill_learning, skill_promotion
        self.skill_registry = skill_registry
        self._tasks: dict[str, asyncio.Task] = {}

    def schedule(self, run_id: str) -> asyncio.Task:
        existing = self._tasks.get(run_id)
        if existing is not None and not existing.done():
            return existing
        clear_run_cancelled(run_id)
        task = asyncio.create_task(self._execute(run_id), name=f"run:{run_id}")
        self._tasks[run_id] = task
        task.add_done_callback(lambda _task: self._tasks.pop(run_id, None))
        return task

    async def _execute(self, run_id: str):
        try:
            # 每个 Run 使用独立 Runtime；离开作用域时由工厂关闭研究 Agent。
            with self._runtime_factory() as runtime:
                return await runtime.execute_run(run_id)
        finally:
            run = await self.runs.get(run_id)
            config = json.loads(run.config_snapshot_json or "{}") if run else {}
            if not config.get("evaluation_run") and run and run.status not in {"paused", "pausing"}:
                await self.skill_learning.enqueue_for_run(run_id)
                snapshot = json.loads(run.model_snapshot_json or "{}") if run else {}
                selected = snapshot.get("skill") if isinstance(snapshot, dict) else None
                if isinstance(selected, dict) and selected.get("status") == "canary":
                    checks = await ContractRepository(self.database).list_for_run(run_id)
                    safety_violation = any(item.kind == "source_match" and item.passed != 1 for item in checks)
                    await self.skill_promotion.observe_run(
                        name=str(selected.get("name")), version=str(selected.get("version")),
                        succeeded=bool(run and run.status == "completed"), safety_violation=safety_violation,
                    )

    async def run_evaluation_case(self, *, query: str, source_mode: str, workflow_mode: str, forced_skill=None) -> dict[str, Any]:
        """Execute one isolated real Harness Run for paired Skill evaluation."""
        session = await self.sessions.create(SessionCreate(title="[Skill Eval]"))
        message, run, _ = await self.runs.create_for_user_message(
            MessageCreate(session_id=session.session_id, role="user", content=query, client_message_id=None, metadata={"evaluation": True}),
            RunCreate(
                session_id=session.session_id, trigger_message_id="assigned-atomically",
                source_mode=SourceMode(source_mode), workflow_mode=WorkflowMode(workflow_mode),
                config_snapshot={
                    "evaluation_run": True, "disable_skills": forced_skill is None,
                    "forced_skill": forced_skill, "min_evidence": 1,
                    "report_type": "brief", "deep_research_max_iterations": 1,
                    "schema_version": 1,
                },
                budget=settings.HARNESS_BUDGETS,
            ),
        )
        await self.runs.update_model_snapshot(run.run_id, {
            "llm_model": settings.OPENAI_LLM_MODEL,
            "learning_review_model": settings.LEARNING_REVIEW_MODEL,
            "evaluation_arm": "treatment" if forced_skill else "control",
        })
        await self._execute(run.run_id)
        finished = await self.runs.get(run.run_id)
        checks = await ContractRepository(self.database).list_for_run(run.run_id)
        usage = json.loads(finished.usage_json or "{}") if finished else {}
        check_map = {item.kind: bool(item.passed) for item in checks}
        limits = usage.get("limits", {}) if isinstance(usage, dict) else {}
        used = usage.get("usage", usage) if isinstance(usage, dict) else {}
        return {
            "run_id": run.run_id, "completed": bool(finished and finished.status == "completed"),
            "status": finished.status if finished else "missing",
            "citation_integrity": float(check_map.get("citation_integrity", False)),
            "claim_support": float(check_map.get("claim_support", False)),
            "tool_policy_violations": 0,
            "source_leakage": 0 if check_map.get("source_match", False) else 1,
            "safety_passed": bool(check_map.get("source_match", False)),
            "llm_tokens": int(used.get("llm_tokens", 0) or 0),
            "wall_time_seconds": float(used.get("elapsed_seconds", 0) or 0),
            "checks": check_map,
        }

    async def startup_recovery(self, *, auto_resume: bool = AUTO_RESUME_RUNS) -> list[str]:
        run_ids = await RecoveryManager(self.runs).scan(auto_resume=auto_resume)
        for run_id in run_ids:
            self.schedule(run_id)
        await self.skill_learning.recover()
        return run_ids

    async def cancel(self, run_id: str) -> bool:
        requested = await self.runs.request_cancel(run_id)
        if not requested:
            return False
        mark_run_cancelled(run_id)
        task = self._tasks.get(run_id)
        if task is not None and not task.done():
            task.cancel()
        else:
            run = await self.runs.get(run_id)
            if run is not None:
                await self.runs.update_status(
                    run_id, status="cancelled", current_stage="cancelled",
                    usage=json.loads(run.usage_json or "{}"),
                )
                await self.event_bus.publish(run_id, "run.cancelled", stage="cancelled", payload={"status": "cancelled"})
        return True

    async def pause(self, run_id: str) -> bool:
        requested = await self.runs.request_pause(run_id)
        if not requested:
            return False
        await self.event_bus.publish(
            run_id, "run.pause_requested", stage="pausing", payload={"status": "pausing"},
        )
        task = self._tasks.get(run_id)
        if task is None or task.done():
            if await self.runs.finalize_pause(run_id):
                run = await self.runs.get(run_id)
                await self.event_bus.publish(
                    run_id, "run.paused", stage="paused",
                    payload={"status": "paused", "resume_from_status": run.current_stage if run else "queued"},
                )
        return True

    async def resume(self, run_id: str, *, clarification: str | None = None) -> bool:
        before = await self.runs.get(run_id)
        pending_pause = bool(
            before and json.loads(before.config_snapshot_json or "{}").get("pause_requested")
        )
        cancelling_pending_pause = bool(
            pending_pause and before and before.status != "paused"
        )
        resumed = await self.runs.resume(run_id, clarification=clarification)
        if resumed:
            current = await self.runs.get(run_id)
            await self.event_bus.publish(
                run_id, "run.clarification_received" if clarification else "run.resume_requested",
                stage=current.current_stage if current else "queued",
                payload={
                    "has_clarification": bool(clarification),
                    "status": current.status if current else "queued",
                    "cancelled_pending_pause": cancelling_pending_pause,
                },
            )
            if cancelling_pending_pause:
                return True
            active = self._tasks.get(run_id)
            if active is not None and not active.done():
                loop = asyncio.get_running_loop()
                active.add_done_callback(lambda _task: loop.call_soon(self.schedule, run_id))
            else:
                self.schedule(run_id)
        return resumed

    async def shutdown(self) -> None:
        await self.skill_learning.shutdown()
        active = list(self._tasks.values())
        if not active:
            return
        for task in active:
            task.cancel()
        await asyncio.gather(*active, return_exceptions=True)
