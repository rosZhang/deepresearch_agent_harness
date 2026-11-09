"""消息入口的单次 LLM 路由：只选择业务操作，不执行研究，也不运行工具循环。"""

from __future__ import annotations

import json
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from deepresearch_agent.config import settings
from deepresearch_agent.config.prompts.message_router_prompts import MESSAGE_ROUTER_SYSTEM_PROMPT
from deepresearch_agent.harness.errors import AppError, ErrorCode
from deepresearch_agent.models.get_models import get_llm_model


class RoutingDecision(BaseModel):
    """交给 ChatService 执行的路由结果；新研究没有 run_id，控制操作指定已有 Run。"""

    action: Literal["start_research", "resume_run", "pause_run", "cancel_run"]
    run_id: str | None = None


class _RunTarget(BaseModel):
    """校验模型生成的控制参数，拒绝 run_id 以外的额外字段。"""

    model_config = ConfigDict(extra="forbid")
    run_id: str = Field(min_length=1, max_length=64)


# 这里只定义给模型看的工具名称、用途和参数格式，并不会自动执行 Python 方法。
# 真正的执行入口在 ChatService._start_research() 和 ChatService._control()。
ROUTING_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "start_research",
            "description": "按当前用户消息创建研究任务，使用请求指定的来源和工作流。",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    *[
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": _RunTarget.model_json_schema(),
            },
        }
        for name, description in (
            ("resume_run", "恢复候选列表中暂停、正在暂停、中断或等待用户输入的原任务，不修改研究要求。"),
            ("pause_run", "请求在安全边界暂停候选列表中的执行中任务，保留后续恢复能力。"),
            ("cancel_run", "取消候选列表中的执行中或已暂停任务，结束该任务。"),
        )
    ],
]


class MessageRouter:
    """根据当前消息和已有任务信息做一次决策，任务执行仍由原有 Harness 负责。"""

    def __init__(self, llm=None):
        # 允许测试注入假模型；生产环境在第一次 route() 时才创建独立路由客户端。
        self._llm = llm
        # 缓存绑定了工具定义的客户端，不保存用户的对话历史。
        self._bound = None

    async def route(self, *, content: str, history: list[dict], runs: list[dict],
                    summary: str | None, source_mode: str, workflow_mode: str) -> RoutingDecision:
        """一次模型调用选择操作；history/runs/summary 由 ChatService 从当前会话加载。

        source_mode、workflow_mode 是请求中已选定的设置，路由不会改写它们。
        目标不明确或响应无效时返回业务错误，不会默认创建新的研究任务。
        """
        if self._llm is None and (not settings.OPENAI_API_KEY or not settings.MESSAGE_ROUTER_MODEL):
            raise AppError(ErrorCode.CONFIGURATION_INVALID, "消息路由需要配置 LLM API 密钥和模型")
        try:
            if self._bound is None:
                if self._llm is None:
                    self._llm = get_llm_model(
                        model=settings.MESSAGE_ROUTER_MODEL, temperature=0.0,
                        max_tokens=settings.MESSAGE_ROUTER_MAX_OUTPUT_TOKENS,
                    )
                # auto 允许模型在信息不足时不选工具，改为给出澄清提示。
                self._bound = self._llm.bind_tools(ROUTING_TOOLS, tool_choice="auto")
            # 系统提示固定路由职责；用户消息、历史和候选任务作为数据一起传入。
            # 本方法只调用一次 ainvoke，不把工具结果再交给模型继续推理。
            response = await self._bound.ainvoke([
                SystemMessage(content=MESSAGE_ROUTER_SYSTEM_PROMPT),
                HumanMessage(content=json.dumps({
                    "current_message": content, "recent_messages": history,
                    "candidate_runs": runs, "session_summary": summary,
                    "source_mode": source_mode, "workflow_mode": workflow_mode,
                }, ensure_ascii=False)),
            ])
        except Exception as exc:
            raise AppError(ErrorCode.MESSAGE_ROUTING_FAILED, "消息意图分析失败，请重试", retryable=True) from exc

        # LangChain 会把模型的工具选择解析为 tool_calls；此时工具尚未执行。
        calls = getattr(response, "tool_calls", None) or []
        finish_reason = (getattr(response, "response_metadata", None) or {}).get("finish_reason")
        # 仅接受一个完整、有效的操作，避免把截断响应或多个动作当成成功路由。
        if (getattr(response, "invalid_tool_calls", None) or len(calls) > 1
                or finish_reason in {"length", "content_filter"}):
            raise AppError(ErrorCode.MESSAGE_ROUTING_FAILED, "消息路由未返回唯一有效操作，请重试", retryable=True)
        if not calls:
            # 当前接口要求返回关联 Run；澄清文本沿用错误提示展示，不创建 Run。
            content = getattr(response, "content", None)
            question = content.strip()[:1000] if isinstance(content, str) else ""
            raise AppError(ErrorCode.CONFLICT, question or "请明确要研究的问题或要操作的任务")

        call = calls[0]
        name, args = call.get("name"), call.get("args")
        if name == "start_research" and args == {}:
            # 新任务沿用原始用户消息，不允许路由模型重写问题或切换研究工作流。
            return RoutingDecision(action=name)
        if name in {"resume_run", "pause_run", "cancel_run"}:
            try:
                target = _RunTarget.model_validate(args)
            except ValidationError as exc:
                raise AppError(ErrorCode.MESSAGE_ROUTING_FAILED, "消息路由返回的任务参数无效", retryable=True) from exc
            # 模型只能选择后端提供的候选任务，不能凭空编造或指定其他会话的 ID。
            # ChatService 执行时还会再次检查任务归属，RunService 检查可执行状态。
            if target.run_id not in {run["run_id"] for run in runs}:
                raise AppError(ErrorCode.CONFLICT, "未找到对应任务，请明确要操作的任务")
            return RoutingDecision(action=name, run_id=target.run_id)
        raise AppError(ErrorCode.MESSAGE_ROUTING_FAILED, "消息路由返回了不支持的操作", retryable=True)
