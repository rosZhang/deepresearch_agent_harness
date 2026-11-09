"""Incremental LLM summaries plus an unabridged, rolling conversation window."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from deepresearch_agent.config.prompts.context_prompts import COMPACTION_SYSTEM_PROMPT
from deepresearch_agent.harness.errors import RunCancelled
from deepresearch_agent.persistence.repositories import MessageRepository, SessionRepository

from .tokens import count_tokens

_LOG = logging.getLogger(__name__)
_FORMAT = "llm-session-summary-v1"


class ContextCompactionError(ValueError):
    """History cannot fit without silently discarding unsummarized messages."""


@dataclass
class ConversationHistory:
    summary: dict[str, Any]
    messages: list[Any]
    status: str = "unchanged"


def render_summary(summary: dict[str, Any]) -> str:
    return "# SESSION SUMMARY\n" + summary["content"] if summary else ""


def render_messages(messages: list[Any]) -> str:
    if not messages:
        return ""
    return "# RECENT SESSION MESSAGES\n" + "\n".join(
        f"{item.role}: {item.content}" for item in messages
    )


class ContextCompactor:
    def __init__(
        self, sessions: SessionRepository, messages: MessageRepository, *,
        llm: Any | None = None, threshold_tokens: int = 8000,
        protect_recent_messages: int = 12, target_tokens: int = 2000,
        max_input_tokens: int = 16000,
    ):
        self.sessions, self.messages, self.llm = sessions, messages, llm
        self.threshold_tokens = max(100, threshold_tokens)
        self.protect_recent_messages = max(2, protect_recent_messages)
        self.target_tokens = max(64, target_tokens)
        self.max_input_tokens = max(512, max_input_tokens)
        self._locks: dict[str, asyncio.Lock] = {}

    async def _load_messages(self, session_id: str, before_message_id: str | None) -> list[Any]:
        # Repository pages are capped at 500; long sessions must not lose their tail.
        history: list[Any] = []
        offset = 0
        while True:
            page = await self.messages.list_for_session(session_id, limit=500, offset=offset)
            for item in page:
                if item.message_id == before_message_id:
                    return history
                if item.role in {"user", "assistant"}:
                    history.append(item)
            if len(page) < 500:
                return history
            offset += len(page)

    @staticmethod
    def _restore(raw: str | None, history: list[Any]) -> tuple[dict, list[Any]]:
        try:
            current = json.loads(raw or "{}")
        except (TypeError, ValueError):
            return {}, history
        # Rebuild old keyword summaries from durable original messages on demand.
        if (not isinstance(current, dict) or current.get("format") != _FORMAT
                or not isinstance(current.get("content"), str) or not current["content"]):
            return {}, history
        covered = current.get("covered_message_ids", [])
        if not covered or covered != [item.message_id for item in history[:len(covered)]]:
            return {}, history
        return current, history[len(covered):]

    @staticmethod
    def _text(summary: dict, messages: list[Any]) -> str:
        return "\n\n".join(filter(None, (render_summary(summary), render_messages(messages))))

    @classmethod
    def _fits(cls, summary: dict, messages: list[Any], max_tokens, max_chars) -> bool:
        text = cls._text(summary, messages)
        return ((max_tokens is None or count_tokens(text) <= max_tokens)
                and (max_chars is None or len(text) <= max_chars))

    @staticmethod
    def _request(previous: str, entries: list[dict], target_tokens: int, target_chars: int | None):
        payload = {
            "previous_summary": previous, "messages": entries,
            "target_tokens": target_tokens, "target_chars": target_chars,
        }
        return [SystemMessage(content=COMPACTION_SYSTEM_PROMPT),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False))]

    @staticmethod
    def _request_tokens(request) -> int:
        return sum(count_tokens(item.content) for item in request) + 32

    async def _invoke(self, request, target_tokens: int, target_chars: int | None) -> str:
        if self._request_tokens(request) > self.max_input_tokens:
            raise ContextCompactionError("摘要请求超过 CONTEXT_COMPRESSION_MAX_INPUT_TOKENS")
        response = await self.llm.ainvoke(request)
        content = getattr(response, "content", "")
        if isinstance(content, list):
            content = "".join(item if isinstance(item, str) else item.get("text", "")
                              for item in content if isinstance(item, (str, dict)))
        if not isinstance(content, str) or not content.strip():
            raise ContextCompactionError("压缩模型返回空摘要")
        if (getattr(response, "response_metadata", None) or {}).get("finish_reason") == "length":
            raise ContextCompactionError("压缩模型输出被截断，请提高压缩输出上限")
        text = content.strip()
        if count_tokens(text) > target_tokens or (target_chars is not None and len(text) > target_chars):
            # One bounded refinement, never cut a generated summary mid-sentence.
            retry = self._request(text, [], target_tokens, target_chars)
            if self._request_tokens(retry) > self.max_input_tokens:
                raise ContextCompactionError("模型返回的摘要过长")
            response = await self.llm.ainvoke(retry)
            text = getattr(response, "content", "")
            if not isinstance(text, str):
                raise ContextCompactionError("压缩模型返回无效摘要")
            text = text.strip()
            if (getattr(response, "response_metadata", None) or {}).get("finish_reason") == "length":
                raise ContextCompactionError("压缩模型输出被截断")
        if (not text or count_tokens(text) > target_tokens
                or (target_chars is not None and len(text) > target_chars)):
            raise ContextCompactionError("压缩摘要仍超过目标预算")
        return text

    async def _summarize(self, previous: str, messages: list[Any], target_tokens: int,
                         target_chars: int | None) -> str:
        pending = deque({"message_id": item.message_id, "role": item.role, "content": item.content}
                        for item in messages)
        while pending:
            batch: list[dict] = []
            while pending:
                entry = pending[0]
                request = self._request(previous, batch + [entry], target_tokens, target_chars)
                if self._request_tokens(request) <= self.max_input_tokens:
                    batch.append(pending.popleft())
                    continue
                if batch:
                    break
                # A single long report is split across model inputs, never discarded.
                content = entry["content"]
                low, high = 0, len(content)
                while low < high:
                    middle = (low + high + 1) // 2
                    part = {**entry, "content": content[:middle], "partial": True}
                    if self._request_tokens(self._request(previous, [part], target_tokens, target_chars)) <= self.max_input_tokens:
                        low = middle
                    else:
                        high = middle - 1
                if low == 0:
                    raise ContextCompactionError("摘要和提示词已占满压缩输入预算")
                batch.append({**entry, "content": content[:low], "partial": True})
                if low == len(content):
                    pending.popleft()
                else:
                    pending[0] = {**entry, "content": content[low:], "partial": True}
                break
            previous = await self._invoke(
                self._request(previous, batch, target_tokens, target_chars), target_tokens, target_chars,
            )
        return previous

    async def prepare(
        self, session_id: str, *, before_message_id: str | None = None,
        max_tokens: int | None = None, max_chars: int | None = None,
        recent_messages: int | None = None,
    ) -> ConversationHistory:
        async with self._locks.setdefault(session_id, asyncio.Lock()):
            session = await self.sessions.get(session_id)
            if session is None:
                return ConversationHistory({}, [])
            history = await self._load_messages(session_id, before_message_id)
            current, pending = self._restore(session.summary_json, history)
            text = self._text(current, pending)
            fits = self._fits(current, pending, max_tokens, max_chars)
            if count_tokens(text) < self.threshold_tokens and fits:
                return ConversationHistory(current, pending)
            if self.llm is None:
                if not fits:
                    raise ContextCompactionError("会话超出上下文预算，但压缩 LLM 未启用或未配置")
                return ConversationHistory(current, pending, "disabled")

            keep = min(len(pending), recent_messages or self.protect_recent_messages)
            target = self.target_tokens
            target_chars = None
            while keep >= 0:
                tail = pending[-keep:] if keep else []
                tail_text = self._text({}, tail)
                available_tokens = (max_tokens - count_tokens(tail_text) - 16
                                    if max_tokens is not None else self.target_tokens)
                available_chars = max_chars - len(tail_text) - 24 if max_chars is not None else None
                if available_tokens >= 64 and (available_chars is None or available_chars >= 128):
                    target = min(self.target_tokens, available_tokens)
                    target_chars = available_chars
                    break
                keep -= 1
            if keep < 0:
                if fits:
                    return ConversationHistory(current, pending)
                raise ContextCompactionError("上下文没有足够空间容纳摘要，请提高预算或缩小当前输入")
            # Preserve the requested raw suffix where possible. A single huge
            # report can itself enter the summary; its original remains durable.
            older = pending[:-keep] if keep else pending
            tail = pending[-keep:] if keep else []
            if not older and fits:
                return ConversationHistory(current, pending)
            try:
                if older:
                    content = await self._summarize(current.get("content", ""), older, target, target_chars)
                else:
                    content = await self._invoke(
                        self._request(current.get("content", ""), [], target, target_chars), target, target_chars,
                    )
                covered = list(current.get("covered_message_ids", [])) + [item.message_id for item in older]
                summary = {
                    "format": _FORMAT, "content": content,
                    "covered_message_ids": covered, "through_message_id": covered[-1],
                    "source_token_count": sum(count_tokens(item.content) for item in history[:len(covered)]),
                    "summary_tokens": count_tokens(content),
                    "model": getattr(self.llm, "model_name", ""),
                }
                if not self._fits(summary, tail, max_tokens, max_chars):
                    raise ContextCompactionError("压缩后的摘要和近期原文仍超过上下文预算")
            except RunCancelled:
                raise
            except Exception as exc:
                _LOG.warning("Session compaction failed (%s); retaining original messages", type(exc).__name__)
                if not fits:
                    raise ContextCompactionError("会话摘要生成失败，原文仍完整保留，但无法在当前上下文预算内继续") from exc
                return ConversationHistory(current, pending, "failed")
            # Commit coverage only after every batch and budget check has succeeded.
            await self.sessions.update_summary(session_id, summary)
            return ConversationHistory(summary, tail, "compacted")

    async def get_or_update(self, session_id: str) -> dict:
        """Compatibility entry point for callers that only need the summary."""
        return (await self.prepare(session_id)).summary


SessionSummarizer = ContextCompactor
