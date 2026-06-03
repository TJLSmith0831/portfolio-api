"""
Convert SirenSpec workflow traces into the JSON shapes the frontend demos
already render.

Each shaper is total: missing keys fall back to sensible defaults so a
partial / failed workflow still produces a structured result the UI can
display.
"""

from __future__ import annotations

import re
from typing import Any

from app.models.sirenspec_models import (
    AgentOutput,
    CompressionGauntletResult,
    GradebookRow,
    GradingFactoryResult,
    GradingSummary,
    Integrity,
    PaperResult,
    RoundResult,
)
from app.services.sirenspec_runner import node_trace


def _count_words(text: str) -> int:
    return len([w for w in re.split(r"\s+", text.strip()) if w])


# =========================
# Compression Gauntlet
# =========================


def _node_text(node: dict[str, Any]) -> str:
    """
    Extract the model's text response from a node trace entry.

    SirenSpec uses `response_received` for agent nodes and `output`
    on some other trace shapes. Test fixtures use `output` for
    simplicity; we accept either.
    """
    text = node.get("response_received")
    if text is None:
        text = node.get("output")
    if text is None:
        return ""
    return text if isinstance(text, str) else str(text)


def shape_compression_gauntlet(
    trace: dict[str, Any], original_input: str
) -> CompressionGauntletResult:
    """
    Build a 4-round RoundResult list from a compression-gauntlet trace.
    Round N's input is round N-1's output; round 1's input is the user
    passage.
    """
    rounds: list[RoundResult] = []
    prev_output = original_input

    for round_id in ("round_1", "round_2", "round_3", "round_4"):
        node = node_trace(trace, round_id) or {}
        output_text = _node_text(node)

        rounds.append(
            RoundResult(
                input=prev_output,
                output=output_text,
                inputWords=_count_words(prev_output),
                outputWords=_count_words(output_text),
            )
        )
        prev_output = output_text

    summary = trace.get("summary") or {}
    return CompressionGauntletResult(
        rounds=rounds,
        totalTokens=int(summary.get("total_tokens") or 0),
        durationMs=float(summary.get("duration_ms") or 0.0),
    )


# =========================
# Grading Factory
# =========================

_GRADE_LINE = re.compile(r"^\s*Grade\s*:\s*([A-F])", re.IGNORECASE | re.MULTILINE)
_INTEGRITY_LINE = re.compile(
    r"^\s*Integrity\s*:\s*(LIKELY_HUMAN|UNCERTAIN|LIKELY_AI)",
    re.IGNORECASE | re.MULTILINE,
)
_FEEDBACK_LINE = re.compile(r"^\s*Feedback\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_INTEGRITY_TOKEN = re.compile(
    r"\b(LIKELY_HUMAN|UNCERTAIN|LIKELY_AI)\b", re.IGNORECASE
)


def _parse_synthesis(text: str) -> tuple[str, Integrity, str]:
    """
    Extract (grade letter, integrity tag, feedback) from a synthesis report.

    The synthesis prompt asks for a fixed three-line format but real models
    drift, so we fall back to scanning anywhere in the text.
    """
    grade_match = _GRADE_LINE.search(text)
    integrity_match = _INTEGRITY_LINE.search(text) or _INTEGRITY_TOKEN.search(text)
    feedback_match = _FEEDBACK_LINE.search(text)

    grade = grade_match.group(1).upper() if grade_match else "?"
    integrity_raw = integrity_match.group(1).upper() if integrity_match else "UNCERTAIN"
    # Coerce to one of the literals (defensive against accidental case drift).
    if integrity_raw not in ("LIKELY_HUMAN", "UNCERTAIN", "LIKELY_AI"):
        integrity_raw = "UNCERTAIN"

    if feedback_match:
        feedback = feedback_match.group(1).strip()
    else:
        # Strip the Grade/Integrity lines and use whatever's left as feedback.
        leftover = _GRADE_LINE.sub("", text)
        leftover = _INTEGRITY_LINE.sub("", leftover).strip()
        feedback = leftover.splitlines()[0].strip() if leftover else text.strip()[:240]

    return grade, integrity_raw, feedback  # type: ignore[return-value]


_MD_ROW = re.compile(r"^\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$")


def _parse_gradebook_markdown(markdown: str) -> list[GradebookRow]:
    """
    Parse the markdown table the gradebook_compiler agent produces.
    Tolerates the header / separator rows.
    """
    rows: list[GradebookRow] = []
    for line in markdown.splitlines():
        m = _MD_ROW.match(line)
        if not m:
            continue
        col1, col2, col3 = m.group(1), m.group(2), m.group(3)
        col1_clean = col1.strip().replace("#", "").strip()
        if not col1_clean.isdigit():
            continue
        idx = int(col1_clean) - 1
        rows.append(GradebookRow(index=idx, grade=col2.strip(), feedback=col3.strip()))
    return rows


def shape_grading_factory(
    trace: dict[str, Any], original_papers: list[str]
) -> GradingFactoryResult:
    """Build a GradingFactoryResult from a grading-factory trace."""
    factory = node_trace(trace, "grade_papers") or {}
    instances: list[dict[str, Any]] = factory.get("instances", []) or []

    papers: list[PaperResult] = []
    for inst in instances:
        idx = int(inst.get("index", len(papers)))
        item_text = inst.get("item")
        if not isinstance(item_text, str):
            item_text = (
                original_papers[idx]
                if 0 <= idx < len(original_papers)
                else str(item_text or "")
            )

        # SirenSpec nests the per-item swrm under `swrm`; tests may inline
        # agents/synthesis on the instance directly.
        swrm = inst.get("swrm") or inst

        agents_by_id: dict[str, str] = {}
        for a in swrm.get("agents", []) or []:
            aid = a.get("id")
            resp = a.get("response_received") or ""
            if aid:
                agents_by_id[aid] = resp

        synthesis = swrm.get("synthesis") or {}
        synth_text = (
            synthesis.get("response_received")
            or swrm.get("output")
            or inst.get("output")
            or ""
        )
        grade, integrity, feedback = _parse_synthesis(str(synth_text))

        ordered_agents = [
            AgentOutput(id="editor", output=agents_by_id.get("editor", "")),
            AgentOutput(id="ai_detector", output=agents_by_id.get("ai_detector", "")),
            AgentOutput(id="grader", output=agents_by_id.get("grader", "")),
        ]

        papers.append(
            PaperResult(
                index=idx,
                paper=item_text,
                agents=ordered_agents,
                grade=grade,
                integrity=integrity,  # type: ignore[arg-type]
                feedback=feedback,
            )
        )

    compile_node = node_trace(trace, "compile_gradebook") or {}
    gradebook_md = _node_text(compile_node)

    gradebook = _parse_gradebook_markdown(gradebook_md)

    # Fallback: synthesise the gradebook from per-paper results when the
    # compiler didn't return a parseable table.
    if not gradebook:
        gradebook = [
            GradebookRow(index=p.index, grade=p.grade, feedback=p.feedback)
            for p in papers
        ]

    summary = trace.get("summary") or {}
    return GradingFactoryResult(
        papers=sorted(papers, key=lambda p: p.index),
        gradebook=sorted(gradebook, key=lambda r: r.index),
        summary=GradingSummary(
            totalTokens=int(summary.get("total_tokens") or 0),
            durationMs=float(summary.get("duration_ms") or 0.0),
        ),
    )
