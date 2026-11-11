"""应用装配入口：在这里创建依赖，Service 和 Harness 专注处理与执行任务。"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from backend.app.services import ChatService, EventStreamService, RunService
from backend.app.services.message_router import MessageRouter
from deepresearch_agent.config import settings
from deepresearch_agent.context import ArtifactEditContextBuilder, ContextBuilder, ContextCompactor
from deepresearch_agent.evolution import PromotionPolicy, SkillEvaluator, SkillLearningService, SkillLoader, SkillRegistry
from deepresearch_agent.harness.contracts import WorkflowMode
from deepresearch_agent.harness.event_bus import EventBus
from deepresearch_agent.harness.runtime import HarnessRuntime
from deepresearch_agent.memory import MemoryExtractor, MemoryService
from deepresearch_agent.models.get_models import get_llm_model
from deepresearch_agent.models.prefix_cache import tracker as prefix_tracker
from deepresearch_agent.persistence import ArtifactStore, Database
from deepresearch_agent.persistence.repositories import (
    ArtifactRepository, AuditRepository, CheckpointRepository, ContractRepository,
    EventRepository, EvidenceRepository, LearningReviewRepository, MemoryRepository,
    MessageRepository, PlanTaskToolRepository, RunRepository, SessionRepository, SkillRepository,
)
from deepresearch_agent.retrieval.base import TimeoutBoundProvider
from deepresearch_agent.retrieval.router import create_default_router
from deepresearch_agent.sessions import SessionSearchService
from deepresearch_agent.workflows.deep_research.agent import DeepResearchAgent
from deepresearch_agent.workflows.deep_research.driver import DeepResearchDriver
from deepresearch_agent.workflows.multi_agent.driver import PlanExecuteReportDriver
from deepresearch_agent.workflows.multi_agent.executor.worker_coordinator import WorkerCoordinator
from deepresearch_agent.workflows.multi_agent.factory import MultiAgentFactory


@dataclass
class ApplicationServices:
    """交给 FastAPI 使用的已装配组件；各入口复用同一组 Repository 和服务。"""

    database: Database
    run_service: RunService
    chat_service: ChatService
    event_stream: EventStreamService
    memory_service: MemoryService
    skill_services: dict[str, Any]


def build_services(*, database_url=None, workflow_factory=None, artifact_root=None,
                   skills_root=None, message_router=None) -> ApplicationServices:
    """装配公共依赖，具体研究 Agent 则在每个 Run 执行时创建。"""
    database = Database(database_url or settings.APP_DATABASE_URL)
    runs, messages, sessions = RunRepository(database), MessageRepository(database), SessionRepository(database)
    events, artifacts = EventRepository(database), ArtifactRepository(database)
    contracts, checkpoints = ContractRepository(database), CheckpointRepository(database)
    evidence, trajectory = EvidenceRepository(database), PlanTaskToolRepository(database)
    event_bus = EventBus(events)
    artifact_store = ArtifactStore(artifact_root or settings.ARTIFACT_ROOT)
    memory_service = MemoryService(
        MemoryRepository(database), AuditRepository(database),
        user_max_tokens=settings.MEMORY_USER_MAX_TOKENS, project_max_tokens=settings.MEMORY_PROJECT_MAX_TOKENS,
        user_max_chars=settings.MEMORY_USER_MAX_CHARS, project_max_chars=settings.MEMORY_PROJECT_MAX_CHARS,
    )
    skills = SkillRepository(database)
    registry = SkillRegistry(skills_root or settings.SKILLS_ROOT, skills)
    retrieval_router = None if workflow_factory is not None else create_default_router()

    # 压缩摘要、提取长期记忆、学习 Skill 使用独立客户端和开关。
    # 自定义工作流用于测试时不创建真实模型；这里只创建客户端，不执行模型请求。
    can_create_llm = workflow_factory is None and bool(settings.OPENAI_API_KEY)
    compression_llm = None
    if can_create_llm and settings.CONTEXT_COMPRESSION_ENABLED and settings.CONTEXT_COMPRESSION_MODEL:
        compression_llm = get_llm_model(
            model=settings.CONTEXT_COMPRESSION_MODEL, temperature=0.0,
            max_tokens=settings.CONTEXT_COMPRESSION_MAX_OUTPUT_TOKENS,
        )
    context_builder = ContextBuilder(
        messages=messages, sessions=sessions, memory_service=memory_service,
        session_search=SessionSearchService(messages, sessions),
        compactor=ContextCompactor(
            sessions, messages, llm=compression_llm,
            threshold_tokens=min(settings.CONTEXT_COMPRESSION_THRESHOLD_TOKENS,
                                 max(100, int(settings.CONTEXT_MAX_TOKENS * settings.CONTEXT_COMPRESSION_THRESHOLD_RATIO))),
            protect_recent_messages=settings.CONTEXT_PROTECT_RECENT_MESSAGES,
            target_tokens=settings.CONTEXT_COMPRESSION_TARGET_TOKENS,
            max_input_tokens=settings.CONTEXT_COMPRESSION_MAX_INPUT_TOKENS,
        ),
        skill_loader=SkillLoader(registry),
        artifact_context_builder=ArtifactEditContextBuilder(runs, artifacts, artifact_store),
        max_tokens=settings.CONTEXT_MAX_TOKENS, max_chars=settings.CONTEXT_MAX_CHARS,
        recent_turns=settings.CONTEXT_RECENT_TURNS, session_search_top_k=settings.SESSION_SEARCH_TOP_K,
        session_search_window=settings.SESSION_SEARCH_WINDOW,
    )
    memory_llm = None
    if can_create_llm and settings.MEMORY_BACKGROUND_REVIEW_ENABLED and settings.MEMORY_LLM_MODEL:
        memory_llm = get_llm_model(model=settings.MEMORY_LLM_MODEL, temperature=0.0,
                                   max_tokens=settings.MEMORY_LLM_MAX_OUTPUT_TOKENS)
    memory_extractor = None if memory_llm is None else MemoryExtractor(
        memory_service, runs, messages, contracts, llm=memory_llm,
        min_confidence=settings.MEMORY_LLM_MIN_CONFIDENCE, max_candidates=settings.MEMORY_LLM_MAX_CANDIDATES,
        max_message_chars=settings.MEMORY_LLM_MAX_MESSAGE_CHARS,
    )
    learning_llms = []
    if can_create_llm and settings.LEARNING_REVIEW_ENABLED and settings.LEARNING_REVIEW_MODEL:
        learning_llms = [get_llm_model(model=settings.LEARNING_REVIEW_MODEL, temperature=0.0,
                                      max_tokens=settings.LEARNING_REVIEW_MAX_OUTPUT_TOKENS) for _ in range(2)]
    learning = SkillLearningService(
        database, LearningReviewRepository(database), skills, registry,
        proposer_llm=learning_llms[0] if learning_llms else None,
        critic_llm=learning_llms[1] if learning_llms else None,
        events=event_bus, enabled=settings.LEARNING_REVIEW_ENABLED, max_retries=settings.LEARNING_REVIEW_MAX_RETRIES,
    )
    promotion = PromotionPolicy(skills, registry, AuditRepository(database), require_real_replay=True,
                                staged=True, canary_percent=settings.SKILL_CANARY_INITIAL_PERCENT)

    @contextmanager
    def runtime_factory():
        """每次执行创建 Runtime；任务结束、暂停或异常退出时关闭本次研究 Agent。"""
        agents = []

        def create_workflow(context, events=None):
            if workflow_factory is not None:
                return workflow_factory(context, events)
            provider = TimeoutBoundProvider(retrieval_router.for_mode(context.source_mode),
                                            timeout_seconds=context.budget_limits.tool_timeout_seconds)
            if context.workflow_mode is WorkflowMode.DEEP_RESEARCH:
                agent = DeepResearchAgent(use_deeper_tool=True, retrieval_provider=provider, run_id=context.run_id)
                agents.append(agent)
                return DeepResearchDriver(context, agent, events=events)
            worker = WorkerCoordinator(retrieval_provider=provider, max_parallel_workers=context.budget_limits.max_concurrency)
            bundle = MultiAgentFactory.create_default_bundle(retrieval_provider=provider, worker=worker)
            return PlanExecuteReportDriver(context, bundle.orchestrator, events=events)

        try:
            yield HarnessRuntime(
                run_repository=runs, message_repository=messages, event_repository=events,
                checkpoint_repository=checkpoints, evidence_repository=evidence, contract_repository=contracts,
                trajectory_repository=trajectory, workflow_factory=create_workflow, artifact_store=artifact_store,
                artifact_repository=artifacts, event_bus=event_bus, context_builder=context_builder,
                memory_extractor=memory_extractor, skill_distiller=None, prefix_tracker=prefix_tracker,
            )
        finally:
            for agent in agents:
                if hasattr(agent, "close"):
                    agent.close()

    run_service = RunService(
        database=database, runs=runs, messages=messages, sessions=sessions, event_bus=event_bus,
        runtime_factory=runtime_factory, skill_learning=learning, skill_promotion=promotion, skill_registry=registry,
    )
    return ApplicationServices(
        database=database, run_service=run_service,
        chat_service=ChatService(sessions, runs, messages, run_service,
                                 router=message_router if message_router is not None else MessageRouter()),
        event_stream=EventStreamService(events, runs, event_bus), memory_service=memory_service,
        skill_services={
            "repository": skills, "registry": registry,
            "evaluator": SkillEvaluator(skills, case_runner=run_service.run_evaluation_case if workflow_factory is None else None),
            "promotion": promotion if workflow_factory is None else PromotionPolicy(skills, registry, AuditRepository(database)),
            "learning": learning, "reviews": LearningReviewRepository(database), "events": event_bus,
            "test_mode": workflow_factory is not None,
        },
    )
