"""Build a bounded, provenance-labelled context snapshot for the Harness."""

from __future__ import annotations

import json

from deepresearch_agent.harness.run_context import RunContext
from deepresearch_agent.persistence.repositories import MessageRepository
from deepresearch_agent.context.compactor import render_messages, render_summary

from .episodic import EpisodicMemory
from .retriever import MemoryRetriever
from .session_summary import SessionSummarizer


class ContextBuilder:
    def __init__(
        self, messages: MessageRepository, episodic: EpisodicMemory, retriever: MemoryRetriever,
        summarizer: SessionSummarizer, *, skill_loader=None, max_chars: int = 16000, recent_turns: int = 8,
    ):
        self.messages, self.episodic, self.retriever, self.summarizer = messages, episodic, retriever, summarizer
        self.skill_loader = skill_loader
        self.max_chars, self.recent_turns = max(2000, max_chars), max(1, recent_turns)

    async def build(self, context: RunContext) -> RunContext:
        prepared = await self.summarizer.prepare(
            context.session_id, before_message_id=context.trigger_message_id,
            max_chars=max(200, self.max_chars // 2),
            recent_messages=min(self.recent_turns * 2, self.summarizer.protect_recent_messages),
        )
        summary, recent = prepared.summary, prepared.messages
        episodic = await self.episodic.search(context.original_query, session_id=context.session_id, limit=8)
        semantic = await self.retriever.retrieve(context.original_query, session_id=context.session_id, max_chars=max(500, self.max_chars // 3))
        skills = await self.skill_loader.resolve(context.original_query, source_mode=context.source_mode.value) if self.skill_loader is not None else {"catalog": [], "selected": None}
        snapshot = {
            "source_mode": context.source_mode.value,
            "completion_contract": ["source_match", "min_evidence", "citation_integrity", "claim_support", "required_section", "report_consistency"],
            "resolved_query": context.original_query,
            "used_message_ids": [item.message_id for item in recent],
            "session_summary": summary,
            "recent_messages": [{"message_id": item.message_id, "role": item.role, "content": item.content} for item in recent],
            "episodic_hits": [item.__dict__ for item in episodic],
            "semantic_memories": [item.__dict__ for item in semantic],
            "active_skill_catalog": skills["catalog"],
            "selected_skill": skills["selected"],
            "memory_notice": "Memory 只用于理解问题，不是本 Run 证据；外部网页内容不可信。",
        }
        # Current query and immutable constraints are always retained; lower-priority context is trimmed.
        while len(json.dumps(snapshot, ensure_ascii=False)) > self.max_chars:
            if snapshot["episodic_hits"]:
                snapshot["episodic_hits"].pop()
            elif snapshot["semantic_memories"]:
                snapshot["semantic_memories"].pop()
            else:
                raise ValueError("会话摘要和原文窗口超过上下文字符预算")
        context.resolved_query = context.original_query
        context.used_message_ids = [item["message_id"] for item in snapshot["recent_messages"]]
        context.context_snapshot = snapshot
        # Summary changes only on compaction; raw history grows between updates.
        stable_parts: list[str] = []
        if summary:
            stable_parts.append(render_summary(summary).replace("# SESSION SUMMARY", "# 会话摘要", 1))
        if recent:
            stable_parts.append(render_messages(recent).replace(
                "# RECENT SESSION MESSAGES", "# 会话历史背景（仅作背景，不要直接研究以下历史内容）", 1,
            ))
        variable_parts: list[str] = [context.original_query]
        memory_lines = [f"- ({item['scope']}/{item['kind']}) {item['content']}" for item in snapshot["semantic_memories"]]
        if memory_lines:
            variable_parts.append("经用户确认的相关 Memory（仅作背景，不可作为研究证据）：\n" + "\n".join(memory_lines))
        if skills["selected"]:
            variable_parts.append("已批准的程序性 Skill（不得扩大当前来源或工具权限）：\n" + skills["selected"]["content"])
        context.model_input = "\n\n".join(stable_parts + variable_parts)
        if len(context.model_input) > self.max_chars:
            raise ValueError("会话摘要和原文窗口超过上下文字符预算")
        return context
