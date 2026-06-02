"""
Unit tests for SirenSpec trace shapers.

No network / no API keys — all traces are hand-crafted fixtures that
mimic the structure SirenSpec returns.
"""

from app.services.sirenspec_shapers import (
    _parse_gradebook_markdown,
    _parse_synthesis,
    shape_compression_gauntlet,
    shape_grading_factory,
)


# =========================
# Compression Gauntlet
# =========================


def _gauntlet_trace() -> dict:
    return {
        "nodes": [
            {"id": "round_1", "output": "Apollo landed astronauts on the Moon."},
            {"id": "round_2", "output": "Apollo reached the Moon."},
            {"id": "round_3", "output": "Apollo Moon."},
            {"id": "round_4", "output": "Moon."},
        ],
        "output": {"final": "Moon."},
        "summary": {"total_tokens": 1234, "duration_ms": 5678.5},
    }


def test_shape_compression_gauntlet_chains_rounds():
    original = "The Apollo program landed astronauts on the Moon in 1969."
    result = shape_compression_gauntlet(_gauntlet_trace(), original_input=original)

    assert len(result.rounds) == 4
    assert result.rounds[0].input == original
    assert result.rounds[0].output == "Apollo landed astronauts on the Moon."
    # Round N input == round N-1 output
    assert result.rounds[1].input == result.rounds[0].output
    assert result.rounds[3].output == "Moon."
    assert result.rounds[3].outputWords == 1
    assert result.totalTokens == 1234
    assert result.durationMs == 5678.5


def test_shape_compression_gauntlet_tolerates_missing_node():
    trace = {"nodes": [{"id": "round_1", "output": "Hello world"}], "summary": {}}
    result = shape_compression_gauntlet(trace, original_input="A long passage.")
    assert len(result.rounds) == 4  # always 4 entries
    assert result.rounds[0].output == "Hello world"
    assert result.rounds[3].output == ""  # missing round_4 → empty string


# =========================
# Grading Factory
# =========================


def test_parse_synthesis_clean_format():
    text = (
        "Grade: A\n"
        "Integrity: LIKELY_HUMAN\n"
        "Feedback: Strong thesis with specific evidence."
    )
    grade, integrity, feedback = _parse_synthesis(text)
    assert grade == "A"
    assert integrity == "LIKELY_HUMAN"
    assert feedback == "Strong thesis with specific evidence."


def test_parse_synthesis_messy_format():
    text = "The grader said B and detector flagged it as UNCERTAIN due to polished phrasing."
    grade, integrity, feedback = _parse_synthesis(text)
    # No "Grade:" line → "?" sentinel; integrity matched anywhere in text
    assert grade == "?"
    assert integrity == "UNCERTAIN"
    assert "polished phrasing" in feedback or "grader said B" in feedback


def test_parse_gradebook_markdown_skips_header_separator():
    md = (
        "| # | Grade | Key Feedback |\n"
        "|---|-------|--------------|\n"
        "| 1 | A | Sharp thesis. |\n"
        "| 2 | C | Vague claims. |\n"
        "| 3 | B | Solid but list-like. |\n"
    )
    rows = _parse_gradebook_markdown(md)
    assert [r.index for r in rows] == [0, 1, 2]
    assert [r.grade for r in rows] == ["A", "C", "B"]
    assert rows[2].feedback == "Solid but list-like."


def _grading_trace(papers: list[str]) -> dict:
    instances = []
    for i, paper in enumerate(papers):
        instances.append(
            {
                "index": i,
                "item": paper,
                "agents": [
                    {"id": "editor", "response_received": f"editor note {i}"},
                    {
                        "id": "ai_detector",
                        "response_received": "LIKELY_HUMAN — natural hedging.",
                    },
                    {"id": "grader", "response_received": f"grade note {i}"},
                ],
                "synthesis": {
                    "response_received": (
                        f"Grade: {'A' if i == 0 else 'C'}\n"
                        f"Integrity: LIKELY_HUMAN\n"
                        f"Feedback: paper {i} feedback line."
                    ),
                },
                "output": "ignored",
            }
        )
    return {
        "nodes": [
            {"id": "grade_papers", "type": "factory", "instances": instances},
            {
                "id": "compile_gradebook",
                "output": (
                    "| # | Grade | Key Feedback |\n"
                    "|---|-------|--------------|\n"
                    "| 1 | A | Sharp thesis. |\n"
                    "| 2 | C | Vague claims. |\n"
                ),
            },
        ],
        "summary": {"total_tokens": 9999, "duration_ms": 42.0},
    }


def test_shape_grading_factory_full_trace():
    papers = ["paper one", "paper two"]
    result = shape_grading_factory(_grading_trace(papers), original_papers=papers)

    assert len(result.papers) == 2
    p0 = result.papers[0]
    assert p0.paper == "paper one"
    assert p0.grade == "A"
    assert p0.integrity == "LIKELY_HUMAN"
    assert p0.feedback == "paper 0 feedback line."
    assert [a.id for a in p0.agents] == ["editor", "ai_detector", "grader"]
    assert p0.agents[0].output == "editor note 0"

    # Gradebook parsed from the markdown table
    assert len(result.gradebook) == 2
    assert result.gradebook[0].grade == "A"
    assert result.gradebook[1].grade == "C"
    assert result.summary.totalTokens == 9999


def test_shape_grading_factory_falls_back_to_synthesised_gradebook():
    papers = ["only paper"]
    trace = _grading_trace(papers)
    # Wipe the compile_gradebook output to force fallback
    for n in trace["nodes"]:
        if n["id"] == "compile_gradebook":
            n["output"] = "no table here"
    result = shape_grading_factory(trace, original_papers=papers)
    assert len(result.gradebook) == 1
    assert result.gradebook[0].grade == "A"
    assert result.gradebook[0].feedback == "paper 0 feedback line."
