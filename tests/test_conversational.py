"""Tests for the conversational agent's prompt rendering + output parsing."""

from __future__ import annotations

from strands_agentic_search.agents.conversational import prompts as conv_prompts
from strands_agentic_search.agents.conversational.agent import _parse_dsl_query


def test_system_prompt_is_verbatim_resource():
    sysp = conv_prompts.SYSTEM_PROMPT
    assert "==== PURPOSE ====" in sysp
    assert "query_planner_tool" in sysp
    assert '{"dsl_query": <OpenSearch DSL Object>}' in sysp
    assert "==== OPERATING LOOP (QPT-CENTRIC) ====" in sysp


def test_user_prompt_template_renders_all_fields():
    out = conv_prompts.render_user_prompt(
        question="find shoes", index_name="products", embedding_model_id="m-1"
    )
    assert "find shoes" in out
    assert "products" in out
    assert "m-1" in out


def test_user_prompt_template_optional_fields_blank():
    out = conv_prompts.render_user_prompt(question="find shoes")
    # index_name + model id absent → their :- defaults (empty) apply
    assert "find shoes" in out
    assert "index_name is: ," in out  # empty default leaves nothing between : and ,


def test_parse_dsl_query_with_wrapper():
    out = _parse_dsl_query('{"dsl_query":{"query":{"match_all":{}}}}')
    assert out == {"dsl_query": {"query": {"match_all": {}}}}


def test_parse_dsl_query_bare_dsl_gets_wrapped():
    out = _parse_dsl_query('{"query":{"term":{"x":1}}}')
    assert out == {"dsl_query": {"query": {"term": {"x": 1}}}}


def test_parse_dsl_query_with_prose_and_fence():
    text = 'Final answer:\n```json\n{"dsl_query":{"query":{"match_all":{}}}}\n```'
    out = _parse_dsl_query(text)
    assert out == {"dsl_query": {"query": {"match_all": {}}}}


def test_parse_dsl_query_garbage_uses_failure_mode():
    out = _parse_dsl_query("the model said nothing useful")
    assert out == {"dsl_query": {"query": {"match_all": {}}}}
