"""Tests for the QueryPlanningTool control flow (MCP + LLM mocked)."""

from __future__ import annotations

import json

import pytest

from strands_agentic_search.config import QPTConfig
from strands_agentic_search.tooling.query_planning.prompts import DEFAULT_QUERY
from strands_agentic_search.tooling.query_planning.tool import (
    QueryContext,
    QueryPlanningTool,
)


def make_context(model_output, *, mapping='{"properties":{"price":{"type":"float"}}}',
                 sample='{"price":"9.99"}', resolve=None, capture=None):
    """Build a QueryContext with stubbed model + index helpers.

    ``capture`` (if a list) records each (system, user) prompt pair the model saw.
    ``model_output`` may be a value or a list (consumed per call).
    """
    outputs = list(model_output) if isinstance(model_output, list) else [model_output]

    def invoke_model(system_prompt, user_prompt):
        if capture is not None:
            capture.append((system_prompt, user_prompt))
        return outputs.pop(0) if len(outputs) > 1 else outputs[0]

    return QueryContext(
        invoke_model=invoke_model,
        fetch_index_mapping=lambda idx: mapping,
        fetch_sample_document=lambda idx: sample,
        resolve_template=resolve,
    )


def test_llm_generated_happy_path():
    ctx = make_context('{"query":{"term":{"price":50}}}')
    qpt = QueryPlanningTool(ctx, config=QPTConfig(generation_type="llmGenerated",
                                                  response_filter=None))
    out = qpt.plan(question="price 50", index_name="products")
    assert json.loads(out) == {"query": {"term": {"price": 50}}}


def test_missing_question_raises():
    ctx = make_context("{}")
    qpt = QueryPlanningTool(ctx)
    with pytest.raises(ValueError):
        qpt.plan(question="", index_name="products")


def test_missing_index_raises():
    ctx = make_context("{}")
    qpt = QueryPlanningTool(ctx)
    with pytest.raises(ValueError):
        qpt.plan(question="find shoes", index_name="")


def test_invalid_generation_type_raises():
    ctx = make_context("{}")
    with pytest.raises(ValueError):
        QueryPlanningTool(ctx, config=QPTConfig(generation_type="bogus"))


def test_blank_model_output_falls_back():
    ctx = make_context("   ")  # blank → fallback path
    qpt = QueryPlanningTool(ctx, config=QPTConfig(response_filter=None))
    out = qpt.plan(question="x", index_name="products")
    assert out == DEFAULT_QUERY


def test_literal_null_output_falls_back():
    ctx = make_context("null")
    qpt = QueryPlanningTool(ctx, config=QPTConfig(response_filter=None))
    out = qpt.plan(question="x", index_name="products")
    assert out == DEFAULT_QUERY


def test_custom_fallback_query_used():
    custom = '{"size":1,"query":{"match_none":{}}}'
    ctx = make_context("")  # empty → fallback
    qpt = QueryPlanningTool(
        ctx, config=QPTConfig(response_filter=None, fallback_query=custom)
    )
    out = qpt.plan(question="x", index_name="products")
    assert out == custom


def test_garbage_output_extracts_default_via_parser():
    ctx = make_context("I really cannot help with that")
    qpt = QueryPlanningTool(ctx, config=QPTConfig(response_filter=None))
    out = qpt.plan(question="x", index_name="products")
    assert out == DEFAULT_QUERY


def test_prompt_contains_mapping_and_sample_and_fallback():
    capture: list = []
    ctx = make_context('{"query":{"match_all":{}}}', capture=capture,
                        mapping='{"properties":{"color":{"type":"keyword"}}}',
                        sample='{"color":"red"}')
    qpt = QueryPlanningTool(ctx, config=QPTConfig(response_filter=None))
    qpt.plan(question="red things", index_name="products",
             query_fields=["color"], embedding_model_id="m-1")
    system, user = capture[0]
    # fallback placeholder was replaced with the escaped default query
    assert "{{FALLBACK_QUERY}}" not in system
    assert '{\\"size\\":10' in system  # escaped DEFAULT_QUERY injected
    # user prompt carries the mapping, sample doc, and question
    assert "red things" in user
    assert "color" in user
    assert "keyword" in user
    assert "m-1" in user


def test_user_templates_selects_and_resolves_template():
    # First model call returns a template id; second returns the DSL.
    capture: list = []
    resolved = {}

    def resolver(tid):
        resolved["id"] = tid
        return '{"query":{"match":{"name":"{{q}}"}}}'

    ctx = make_context(
        ["product-search-template", '{"query":{"match_all":{}}}'],
        capture=capture,
        resolve=resolver,
    )
    qpt = QueryPlanningTool(
        ctx,
        config=QPTConfig(generation_type="user_templates", response_filter=None),
    )
    out = qpt.plan(
        question="find shoes",
        index_name="products",
        search_templates=[
            {"template_id": "product-search-template",
             "template_description": "Searches products."}
        ],
    )
    assert resolved["id"] == "product-search-template"
    assert json.loads(out) == {"query": {"match_all": {}}}
    # The query-planning (second) call's system prompt embeds the resolved template.
    planning_system = capture[1][0]
    assert "{{q}}" in planning_system


def test_user_templates_null_selection_uses_default_template():
    ctx = make_context(["null", '{"query":{"match_all":{}}}'])
    qpt = QueryPlanningTool(
        ctx,
        config=QPTConfig(generation_type="user_templates", response_filter=None),
    )
    out = qpt.plan(
        question="find shoes",
        index_name="products",
        search_templates=[
            {"template_id": "t1", "template_description": "desc"}
        ],
    )
    assert json.loads(out) == {"query": {"match_all": {}}}
