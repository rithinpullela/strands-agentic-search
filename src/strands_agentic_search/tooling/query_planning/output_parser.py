"""Output parsing for the QueryPlanningTool.

Replicates two pieces of the Java pipeline:

1. ``response_filter`` — a JSONPath applied to the *raw* model envelope to pull
   out the generated text (e.g. Bedrock Converse
   ``$.output.message.content[0].text``). In ml-commons this is handled by
   ``MLModelTool``; Strands hands us the text directly, but we keep the hook so
   raw-envelope inputs are still supported.

2. The default output parser — ``QueryPlanningTool.Factory`` always prepends an
   ``extract_json`` processor with ``extract_type=object`` and
   ``default=DEFAULT_QUERY``. It extracts the first balanced JSON object from the
   model text (tolerating markdown/code fences/prose) and falls back to the
   match_all default when no object can be extracted.
"""

from __future__ import annotations

import json
from typing import Any

from jsonpath_ng import parse as jsonpath_parse

from .prompts import DEFAULT_QUERY


def apply_response_filter(raw: Any, response_filter: str | None) -> str | None:
    """Extract the model's text using a JSONPath ``response_filter``.

    ``raw`` may already be a plain string (Strands' common case) — then the
    filter is a no-op. If it is a dict/JSON envelope and a filter is provided,
    the first JSONPath match is returned as a string.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        # Could be a JSON envelope serialized as a string; try to filter it.
        if response_filter:
            try:
                parsed = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                return raw
            return _json_path_first(parsed, response_filter, fallback=raw)
        return raw
    if response_filter:
        return _json_path_first(raw, response_filter, fallback=json.dumps(raw))
    return str(raw)


def _json_path_first(obj: Any, expr: str, *, fallback: str) -> str:
    matches = jsonpath_parse(expr).find(obj)
    if not matches:
        return fallback
    value = matches[0].value
    return value if isinstance(value, str) else json.dumps(value)


def find_first_json_object(text: str) -> str | None:
    """Return the first balanced JSON object in ``text``, or ``None`` if absent.

    Scans for the first ``{`` that opens a parseable, brace-balanced object
    (ignoring braces inside strings). Lets callers choose their own default,
    since QPT and the conversational agent fall back differently.
    """
    if text is None:
        return None
    start = text.find("{")
    while start != -1:
        candidate = _balanced_slice(text, start)
        if candidate is not None:
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return json.dumps(obj)
            except (json.JSONDecodeError, ValueError):
                pass
        start = text.find("{", start + 1)
    return None


def extract_first_json_object(text: str) -> str:
    """Like :func:`find_first_json_object` but returns ``DEFAULT_QUERY`` if none.

    Mirrors the ``extract_json`` processor (``extract_type=object``,
    ``default=DEFAULT_QUERY``) that QueryPlanningTool.Factory always prepends.
    """
    found = find_first_json_object(text)
    return found if found is not None else DEFAULT_QUERY


def _balanced_slice(text: str, start: int) -> str | None:
    """Return the substring from ``start`` to the matching closing brace."""
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_query_output(raw: Any, response_filter: str | None) -> str:
    """Full output pipeline: response_filter → extract first JSON object.

    Returns a compact JSON DSL string (or ``DEFAULT_QUERY`` on failure).
    """
    text = apply_response_filter(raw, response_filter)
    if text is None:
        return DEFAULT_QUERY
    return extract_first_json_object(text)
