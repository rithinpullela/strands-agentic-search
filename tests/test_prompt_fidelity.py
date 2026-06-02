"""Prove the prompts are a 1:1 copy of the OpenSearch source.

Two layers of verification:

1. The QPT prompt assets shipped under ``prompt_assets/`` are re-derived from the
   archived Java source (``docs/reference/QueryPlanningPromptTemplate.java``)
   via the repo's extractor and must match byte-for-byte. This guards against
   any drift between the Java source and the extracted text.

2. Structural invariants the assembled prompts must satisfy (all 13 examples,
   smart quotes, placeholders, exact ``DEFAULT_QUERY``).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from strands_agentic_search.tooling.query_planning import prompts

REPO = Path(__file__).resolve().parents[1]
JAVA_SRC = REPO / "docs" / "reference" / "QueryPlanningPromptTemplate.java"
EXTRACTOR = REPO / "scripts" / "extract_qpt_prompts.py"
ASSET_DIR = (
    REPO / "src" / "strands_agentic_search" / "tooling" / "query_planning"
    / "prompt_assets"
)


def _load_extractor():
    spec = importlib.util.spec_from_file_location("extract_qpt_prompts", EXTRACTOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not JAVA_SRC.exists(), reason="archived Java source not present")
def test_assets_match_java_source():
    extractor = _load_extractor()
    derived = extractor.assemble(JAVA_SRC.read_text(encoding="utf-8"))
    for name, expected in derived.items():
        shipped = (ASSET_DIR / name).read_text(encoding="utf-8")
        assert shipped == expected, f"{name} drifted from the Java source"


def test_default_query_exact():
    assert prompts.DEFAULT_QUERY == '{"size":10,"query":{"match_all":{}}}'


def test_placeholder_present_in_system_prompt():
    assert prompts.FALLBACK_QUERY_PROMPT_PLACEHOLDER == "{{FALLBACK_QUERY}}"
    assert (
        prompts.FALLBACK_QUERY_PROMPT_PLACEHOLDER
        in prompts.DEFAULT_QUERY_PLANNING_SYSTEM_PROMPT
    )


def test_all_thirteen_examples_present():
    sysp = prompts.DEFAULT_QUERY_PLANNING_SYSTEM_PROMPT
    for n in range(1, 14):
        assert f"Example {n} —" in sysp, f"missing Example {n}"


def test_special_characters_preserved():
    sysp = prompts.DEFAULT_QUERY_PLANNING_SYSTEM_PROMPT
    # smart quotes, bullet, and >= sign from the Java text blocks
    for ch in ("’", "“", "”", "•", "≥"):
        assert ch in sysp, f"missing special char {ch!r}"


def test_user_prompt_placeholders():
    up = prompts.DEFAULT_QUERY_PLANNING_USER_PROMPT
    for placeholder in (
        "${parameters.question}",
        "${parameters.index_mapping:-}",
        "${parameters.query_fields:-}",
        "${parameters.sample_document:-}",
        "${parameters.current_time:-}",
        "${parameters.embedding_model_id:- not provided}",
    ):
        assert placeholder in up


def test_section_headers_present():
    sysp = prompts.DEFAULT_QUERY_PLANNING_SYSTEM_PROMPT
    for header in (
        "==== PURPOSE ====",
        "==== RULES ====",
        "==== OUTPUT FORMAT ====",
        "==== EXAMPLES ====",
    ):
        assert header in sysp


def test_template_selection_prompt_present():
    tsp = prompts.DEFAULT_TEMPLATE_SELECTION_SYSTEM_PROMPT
    assert "OpenSearch Search Template selector" in tsp
    assert "==== GOAL ====" in tsp
    assert "==== SELECTION CRITERIA ====" in tsp


def test_default_search_template_is_mustache():
    dst = prompts.DEFAULT_SEARCH_TEMPLATE
    assert "{{lex_query}}" in dst
    assert "{{#sem_enabled}}" in dst
    assert "minimum_should_match" in dst
