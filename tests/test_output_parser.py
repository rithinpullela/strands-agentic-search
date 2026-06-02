"""Tests for the QPT output parser (extract_json + response_filter + fallback)."""

from __future__ import annotations

import json

from strands_agentic_search.tooling.query_planning.output_parser import (
    apply_response_filter,
    extract_first_json_object,
    parse_query_output,
)
from strands_agentic_search.tooling.query_planning.prompts import DEFAULT_QUERY


def test_extract_plain_object():
    assert json.loads(extract_first_json_object('{"query":{"match_all":{}}}')) == {
        "query": {"match_all": {}}
    }


def test_extract_from_code_fence():
    text = '```json\n{"size":5,"query":{"match_all":{}}}\n```'
    assert json.loads(extract_first_json_object(text)) == {
        "size": 5,
        "query": {"match_all": {}},
    }


def test_extract_amid_prose():
    text = 'Here is your query: {"a":{"b":1}} hope it helps!'
    assert json.loads(extract_first_json_object(text)) == {"a": {"b": 1}}


def test_extract_handles_braces_in_strings():
    text = '{"q":"a } b { c"}'
    assert json.loads(extract_first_json_object(text)) == {"q": "a } b { c"}


def test_no_json_returns_default():
    assert extract_first_json_object("no json here") == DEFAULT_QUERY


def test_none_returns_default():
    assert extract_first_json_object(None) == DEFAULT_QUERY


def test_response_filter_bedrock_envelope():
    envelope = {
        "output": {"message": {"content": [{"text": '{"query":{"term":{"x":1}}}'}]}}
    }
    out = apply_response_filter(envelope, "$.output.message.content[0].text")
    assert out == '{"query":{"term":{"x":1}}}'


def test_response_filter_openai_envelope():
    envelope = {"choices": [{"message": {"content": '{"query":{"match_all":{}}}'}}]}
    out = apply_response_filter(envelope, "$.choices[0].message.content")
    assert out == '{"query":{"match_all":{}}}'


def test_response_filter_plain_string_passthrough():
    assert apply_response_filter("just text", None) == "just text"


def test_full_pipeline_with_filter_and_prose():
    envelope = {
        "output": {
            "message": {
                "content": [{"text": 'Sure: {"size":3,"query":{"match_all":{}}}'}]
            }
        }
    }
    out = parse_query_output(envelope, "$.output.message.content[0].text")
    assert json.loads(out) == {"size": 3, "query": {"match_all": {}}}


def test_full_pipeline_no_json_returns_default():
    envelope = {"output": {"message": {"content": [{"text": "I cannot help"}]}}}
    out = parse_query_output(envelope, "$.output.message.content[0].text")
    assert out == DEFAULT_QUERY
