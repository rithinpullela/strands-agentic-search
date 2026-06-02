"""The conversational agent — a ReAct agent orchestrating QPT + MCP tools.

Equivalent to an OpenSearch ``type: "conversational"`` / ``os_chat`` agent: the
verbatim ``agentic-system-prompt.txt`` drives a ReAct loop whose central
mandated action is calling the Query Planner Tool. The agent additionally has
the OpenSearch MCP tools (ListIndexTool, IndexMappingTool, SearchIndexTool, …)
for context gathering and index discovery, so ``index_name`` is OPTIONAL here.

Output contract (from the system prompt): a strict JSON object
``{"dsl_query": <DSL>}``; failure mode yields ``{"query": {"match_all": {}}}``.
"""

from __future__ import annotations

import json
from typing import Any

from strands import Agent

from ...config import QPTConfig
from ...tooling import factory
from ...tooling.mcp_client import MCPClient
from ...tooling.query_planning.output_parser import find_first_json_object
from . import prompts

# A bounded loop keeps a misbehaving model from looping forever; generous enough
# for discovery (list → mapping → qpt → validate). Mirrors the OpenSearch
# conversational agent's max_iteration (commonly 15-20).
_DEFAULT_MAX_TURNS = 20

_FAILURE_DSL = {"query": {"match_all": {}}}


class ConversationalAgent:
    """ReAct agent that returns ``{"dsl_query": <DSL>}``."""

    def __init__(
        self,
        client: MCPClient,
        *,
        model: Any | None = None,
        qpt_config: QPTConfig | None = None,
        max_turns: int = _DEFAULT_MAX_TURNS,
    ):
        toolset = factory.build_toolset(
            client, model=model, qpt_config=qpt_config, include_mcp_tools=True
        )
        self._agent = Agent(
            model=model or factory.build_model(),
            system_prompt=prompts.SYSTEM_PROMPT,
            tools=toolset.all_tools,
            callback_handler=None,
        )
        self._max_turns = max_turns

    def run(
        self,
        question: str,
        *,
        index_name: str | None = None,
        embedding_model_id: str | None = None,
    ) -> dict[str, Any]:
        """Run the ReAct loop and return ``{"dsl_query": <DSL object>}``."""
        user_prompt = prompts.render_user_prompt(
            question=question,
            index_name=index_name,
            embedding_model_id=embedding_model_id,
        )
        result = self._agent(user_prompt, limits={"turns": self._max_turns})
        return _parse_dsl_query(str(result))


def _parse_dsl_query(text: str) -> dict[str, Any]:
    """Extract ``{"dsl_query": ...}`` from the model's final message.

    Per the OUTPUT CONTRACT the model returns exactly that JSON object. We
    extract the first JSON object defensively (tolerating any stray prose) and
    normalize to ``{"dsl_query": <DSL>}``, applying the prompt's failure mode if
    nothing usable is produced.
    """
    obj_str = find_first_json_object(text)
    if obj_str is None:
        return {"dsl_query": _FAILURE_DSL}
    try:
        obj = json.loads(obj_str)
    except (json.JSONDecodeError, ValueError):
        return {"dsl_query": _FAILURE_DSL}

    if isinstance(obj, dict) and "dsl_query" in obj:
        return {"dsl_query": obj["dsl_query"]}
    # Model returned the DSL directly without the wrapper key.
    if isinstance(obj, dict) and obj:
        return {"dsl_query": obj}
    return {"dsl_query": _FAILURE_DSL}
