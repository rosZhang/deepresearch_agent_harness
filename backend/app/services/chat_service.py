"""聊天消息入口：检查重复请求 → 加载路由上下文 → 单次 LLM 决策 → 执行业务操作。"""

import asyncio
import json
from weakref import WeakValueDictionary

from deepresearch_agent.persistence.schemas import MessageCreate, RunCreate
from backend.app.schemas import MessageSend, RunAccepted
from backend.app.services.message_router import MessageRouter, RoutingDecision
from deepresearch_agent.config.settings import HARNESS_BUDGETS, LEARNING_REVIEW_MODEL, OPENAI_LLM_MODEL
from deepresearch_agent.harness.errors import AppError, ErrorCode
from deepresearch_agent.persistence.repositories import MessageRepository, RunRepository, SessionRepository


class ChatService:
    """组织消息处理，研究执行、暂停和恢复的具体逻辑继续委托给 RunService。"""

    def __init__(self, sessions: SessionRepository, runs: RunRepository, messages: MessageRepository,
                 run_service, *, router: MessageRouter | None = None):
        self.sessions = sessions
        self.runs = runs
        self.messages = messages
        self.run_service = run_service
        self.router = router if router is not None else MessageRouter()
        # 当前是单进程部署：同一个客户端消息的并发重试共用一把锁，
        # 避免两个请求同时通过查询检查，重复调用 LLM 或执行控制操作。
        # 弱引用让没有请求再使用的锁可以回收，防止字典随历史消息持续增长。
        self._request_locks: WeakValueDictionary[tuple[str, str], asyncio.Lock] = WeakValueDictionary()

    async def send(self, session_id: str, request: MessageSend) -> RunAccepted:
        """按会话和客户端消息 ID 串行处理重复请求，不阻塞其他消息。"""
        key = (session_id, request.client_message_id)
        lock = self._request_locks.setdefault(key, asyncio.Lock())
        async with lock:
            return await self._send(session_id, request)

    async def _send(self, session_id: str, request: MessageSend) -> RunAccepted:
        session = await self.sessions.get(session_id)
        if session is None:
            raise AppError(ErrorCode.NOT_FOUND, "Session 不存在")
        if session.status != "active":
            raise AppError(ErrorCode.CONFLICT, "已归档 Session 不能发送消息")

        # 先查重复消息：已经处理过就返回原 Run，跳过后面的 LLM 路由和业务操作。
        existing = await self.messages.get_by_client_id(session_id, request.client_message_id)
        if existing is not None:
            metadata = json.loads(existing.metadata_json or "{}")
            # 同一个客户端 ID 不能被用于不同正文或不同运行参数。
            if existing.content != request.content or any(
                key in metadata and metadata[key] != value
                for key, value in self._request_metadata(request).items()
            ):
                raise AppError(ErrorCode.IDEMPOTENCY_CONFLICT, "client_message_id 已被不同的消息或运行参数使用")
            # 控制消息直接关联 message.run_id；普通研究消息则通过
            # Run.trigger_message_id 反查，兼容创建时没有填写 message.run_id 的记录。
            existing_run = (
                await self.runs.get(existing.run_id) if existing.run_id
                else await self.runs.get_by_trigger_message(existing.message_id)
            )
            if existing_run is not None:
                return RunAccepted(
                    message_id=existing.message_id, run_id=existing_run.run_id,
                    status=existing_run.status, events_url=f"/api/v1/runs/{existing_run.run_id}/events",
                    created=False,
                )

        # 让模型同时看到“用户在说什么”和“有哪些任务可操作”，从中选择目标 Run。
        # 这里只读取有限的最近消息和任务，不调用研究用的 ContextBuilder 或压缩 LLM。
        history = await self.messages.recent_for_session(session_id)
        runs = await self.runs.list_for_session(session_id, limit=20)
        # Run 本身不存问题正文，需读取触发消息，才能向模型说明每项任务的研究目标。
        triggers = await asyncio.gather(*(self.messages.get(run.trigger_message_id) for run in runs))
        summary = None
        if session.summary_json:
            try:
                summary = str(json.loads(session.summary_json).get("content") or "")[:4000]
            except (ValueError, AttributeError):
                pass
        # 下方截取只影响路由参考材料，数据库中的原始消息和研究输入保持完整。
        decision = await self.router.route(
            content=request.content,
            history=[{"role": msg.role, "content": msg.content[:2000]} for msg in history],
            runs=[{
                "run_id": run.run_id, "status": run.status, "current_stage": run.current_stage,
                "query": trigger.content[:1000] if trigger else "",
                "source_mode": run.source_mode, "workflow_mode": run.workflow_mode,
                "created_at": run.created_at,
                "pause_requested": bool(json.loads(run.config_snapshot_json or "{}").get("pause_requested")),
            } for run, trigger in zip(runs, triggers)],
            summary=summary, source_mode=request.source_mode.value, workflow_mode=request.workflow_mode.value,
        )
        if decision.action != "start_research":
            # 恢复、暂停、取消都操作已有 Run，不进入下面的新建研究流程。
            return await self._control(session_id, request, decision)

        return await self._start_research(session, request)

    @staticmethod
    def _request_metadata(request: MessageSend) -> dict:
        """保存原始请求参数，用于后续识别重试请求是否与首次提交一致。"""
        return {"source_mode": request.source_mode.value, "workflow_mode": request.workflow_mode.value,
                "report_type": request.report_type}

    async def _control(self, session_id: str, request: MessageSend, decision: RoutingDecision) -> RunAccepted:
        """执行已选中的控制操作，并将这条用户消息关联到原 Run。"""
        # 即使路由模型已经检查过候选范围，服务端仍独立校验任务归属。
        run = await self.runs.get(decision.run_id)
        if run is None or run.session_id != session_id:
            raise AppError(ErrorCode.NOT_FOUND, "当前会话中不存在该 Run")
        # 工具名映射到现有服务方法，不在入口重新实现暂停、恢复或取消机制。
        operations = {
            "resume_run": (self.run_service.resume, "resuming"),
            "pause_run": (self.run_service.pause, "pausing"),
            "cancel_run": (self.run_service.cancel, "cancelling"),
        }
        operation, status = operations[decision.action]
        # 模型判断的是意图，当前状态是否允许该动作由 RunService 决定。
        if not await operation(run.run_id):
            raise AppError(ErrorCode.CONFLICT, "Run 当前状态不允许执行该操作")
        # 操作被接受后再记录控制消息；执行被拒绝时，不留下“已成功处理”的消息记录。
        message, _ = await self.messages.append(MessageCreate(
            session_id=session_id, run_id=run.run_id, role="user", content=request.content,
            client_message_id=request.client_message_id,
            metadata={**self._request_metadata(request), "control": decision.action.removesuffix("_run")},
        ))
        # 这里返回控制请求已被接受，不等待研究完成，也不新建 Run。
        return RunAccepted(
            message_id=message.message_id, run_id=run.run_id, status=status,
            events_url=f"/api/v1/runs/{run.run_id}/events", created=False,
        )

    async def _start_research(self, session, request: MessageSend) -> RunAccepted:
        """沿用原来的研究创建流程：原子保存消息与 Run，再调度后台执行。"""
        session_id = session.session_id
        detailed_request = request.report_type == "long_document" or any(
            marker in request.content for marker in ("详细", "深入", "全面", "长篇")
        )
        message, run, created = await self.runs.create_for_user_message(
            MessageCreate(
                session_id=session_id, role="user", content=request.content,
                client_message_id=request.client_message_id,
                metadata=self._request_metadata(request),
            ),
            RunCreate(
                session_id=session_id, trigger_message_id="assigned-atomically",
                source_mode=request.source_mode, workflow_mode=request.workflow_mode,
                config_snapshot={
                    # 详细报告需要更多证据，不能凭单条弱相关材料就通过完成验证。
                    "min_evidence": 3 if detailed_request else 1,
                    "report_type": "long_document" if detailed_request else request.report_type,
                    "deep_research_max_iterations": 1 if request.source_mode.value == "graphrag" else 2,
                    "schema_version": 1,
                },
                budget=HARNESS_BUDGETS,
            ),
        )
        if created:
            await self.runs.update_model_snapshot(run.run_id, {
                "llm_model": OPENAI_LLM_MODEL,
                "learning_review_model": LEARNING_REVIEW_MODEL,
            })
            if session.title.strip() in {"新对话", "New conversation", "Untitled"}:
                title = " ".join(request.content.strip().split())[:36]
                if title:
                    await self.sessions.rename(session_id, title)
            # 只启动后台任务；检查点、预算和报告验证仍由原有 Harness 管理。
            self.run_service.schedule(run.run_id)
        return RunAccepted(
            message_id=message.message_id, run_id=run.run_id, status=run.status,
            events_url=f"/api/v1/runs/{run.run_id}/events", created=created,
        )
