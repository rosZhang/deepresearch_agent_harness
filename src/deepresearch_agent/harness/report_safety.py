"""User-facing report sanitization and citation placement helpers."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

_THINK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.I | re.S)
_UNCLOSED_THINK = re.compile(r"<think\b[^>]*>.*$", re.I | re.S)
_FORBIDDEN = re.compile(
    r"SYSTEM\s+CONTRACT|CURRENT\s+TASK|SELECTED\s+SKILL|TOOL\s+ALLOWLIST|"
    r"created_from_runs|verification_failures|#\s*系统契约|#\s*当前任务",
    re.I,
)
_CITATION = re.compile(r"\[(ev_[A-Za-z0-9_-]+)\]")


def sanitize_report(text: str | None) -> str:
    """Remove hidden reasoning and internal harness material from model output."""
    value = str(text or "").replace("\x00", "")
    value = _THINK.sub("", value)
    value = _UNCLOSED_THINK.sub("", value)
    lines = value.splitlines()
    cleaned: list[str] = []
    skipping_internal = False
    for line in lines:
        if _FORBIDDEN.search(line):
            skipping_internal = True
            continue
        if skipping_internal:
            # Resume only at an explicit, user-facing report section.
            if re.match(r"^#{1,3}\s+(研究报告|深度研究报告|结论|摘要|回答|要点|分析)", line.strip(), re.I):
                skipping_internal = False
            else:
                continue
        cleaned.append(line.rstrip())
    value = "\n".join(cleaned).strip()
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value


def has_internal_material(text: str | None) -> bool:
    value = str(text or "")
    return bool(re.search(r"<think\b", value, re.I) or _FORBIDDEN.search(value))


def _authority_key(item: Any) -> tuple[int, float]:
    metadata = getattr(item, "metadata", None)
    domain = str(getattr(metadata, "domain", "") or "").lower()
    extra = getattr(metadata, "extra", {}) or {}
    authority = int(extra.get("authority_rank") or 0)
    if not authority:
        authority = int(
            domain.startswith(("docs.", "help.", "developer."))
            or domain.endswith((".gov", ".gov.cn", ".edu", ".edu.cn"))
        )
    return authority, float(getattr(item, "score", 0.0) or 0.0)


def rank_evidence(items: Sequence[Any]) -> list[Any]:
    """Prefer official/documentation domains, then provider relevance score."""
    return sorted(items, key=_authority_key, reverse=True)


def citation_evidence_ids(items: Sequence[Any], *, limit: int = 5) -> list[str]:
    """Use the strongest authority tier for citations instead of mixing weaker pages."""
    ranked = rank_evidence(items)
    if not ranked:
        return []
    strongest = _authority_key(ranked[0])[0]
    selected = [item for item in ranked if _authority_key(item)[0] == strongest] if strongest else ranked
    return [str(getattr(item, "result_id", "")) for item in selected[:limit] if getattr(item, "result_id", None)]


def add_inline_citations(text: str, evidence_ids: Sequence[str]) -> str:
    """Attach stable evidence IDs to substantive paragraphs or bullet claims."""
    ids = [item for item in evidence_ids if item.startswith("ev_")]
    if not ids:
        return text.strip()
    result: list[str] = []
    claim_index = 0
    excluded_section = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            excluded_section = bool(re.search(r"局限|方法|证据引用|参考来源|references?", stripped, re.I))
        is_claim = (
            len(stripped) >= 12
            and not stripped.startswith("#")
            and not stripped.startswith(("```", "|"))
            and not excluded_section
            and not re.search(r"局限|证据引用|参考来源", stripped, re.I)
        )
        if is_claim and not _CITATION.search(stripped):
            selected = ids[:2] if claim_index == 0 else [ids[claim_index % len(ids)]]
            line = f"{line.rstrip()} {' '.join(f'[{item}]' for item in selected)}"
            claim_index += 1
        result.append(line)
    return "\n".join(result).strip()


__all__ = ["add_inline_citations", "citation_evidence_ids", "has_internal_material", "rank_evidence", "sanitize_report"]


def evidence_only_report(
    results: list[Any],
    *,
    title: str = "修复后的研究报告",
    required_sections: list[str] | None = None,
) -> tuple[str, list[Any]]:
    """Build a conservative report that cannot introduce unsupported claims."""
    selected: list[Any] = []
    seen_ids: set[str] = set()
    for item in rank_evidence(results):
        if not item.result_id or item.result_id in seen_ids:
            continue
        seen_ids.add(item.result_id)
        selected.append(item)
        if len(selected) == 6:
            break

    lines: list[str] = []
    for index, item in enumerate(selected, 1):
        summary = re.sub(r"\[(?:ev_)?[A-Za-z0-9_-]+\]", "", str(item.evidence or ""))
        summary = re.sub(r"\s+", " ", summary).strip()
        if len(summary) > 520:
            summary = summary[:517].rstrip() + "…"
        if summary:
            lines.append(f"{index}. {summary} [{item.result_id}]")

    if not lines:
        return f"# {title}\n\n## 证据支持的要点\n\n当前没有可用于修复报告的有效证据。", []

    sections = [f"# {title}", "## 证据支持的要点", "\n".join(lines)]
    existing = {"证据支持的要点", "局限"}
    first_id = selected[0].result_id
    for name in required_sections or []:
        clean_name = str(name).strip()
        if clean_name and clean_name not in existing:
            sections.extend([f"## {clean_name}", f"本节仅保留上方证据能够直接支持的内容。 [{first_id}]"])
            existing.add(clean_name)
    sections.extend([
        "## 局限",
        "本报告由一致性门禁触发保守修复，仅保留当前 Run 证据能够直接支持的陈述；被判定为不一致或无法就近引用的原报告内容已移除。",
    ])
    return "\n\n".join(sections), selected
