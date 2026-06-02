"""Tests for the ${parameters.*} substitutor."""

from __future__ import annotations

from strands_agentic_search.tooling.substitutor import substitute


def test_simple_substitution():
    assert substitute("Q: ${parameters.question}", {"question": "shoes"}) == "Q: shoes"


def test_default_used_when_missing():
    assert substitute("${parameters.x:-fallback}", {}) == "fallback"


def test_default_ignored_when_present():
    assert substitute("${parameters.x:-fallback}", {"x": "real"}) == "real"


def test_empty_default():
    assert substitute("[${parameters.x:-}]", {}) == "[]"


def test_default_with_leading_space():
    # mirrors ${parameters.embedding_model_id:- not provided}
    assert substitute("${parameters.m:- not provided}", {}) == " not provided"


def test_unknown_without_default_is_left_intact():
    assert substitute("${parameters.unknown}", {}) == "${parameters.unknown}"


def test_none_value_falls_back_to_default():
    assert substitute("${parameters.x:-d}", {"x": None}) == "d"


def test_multiple_placeholders():
    out = substitute(
        "${parameters.a}/${parameters.b:-B}/${parameters.a}",
        {"a": "1"},
    )
    assert out == "1/B/1"


def test_fallback_query_substitution_for_default_query():
    # The Java substitutes ${parameters.*} into the fallback query string.
    fallback = '{"size":10,"query":{"match_all":{}}}'
    assert substitute(fallback, {"question": "x"}) == fallback
