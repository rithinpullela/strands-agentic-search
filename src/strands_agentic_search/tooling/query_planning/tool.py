"""QueryPlanningTool — a 1:1 Python replica of OpenSearch's ``QueryPlanningTool``.

This is the deterministic core shared by both agents. Given a natural-language
``question`` and an ``index_name`` it:

1. validates inputs (``question`` + ``index_name`` required),
2. strips agent-context parameters,
3. selects a search template (``user_templates`` mode) or uses the default
   search template (``llmGenerated`` mode),
4. fetches the index mapping + a sample document (via the MCP tooling layer),
5. builds the verbatim system/user prompts with ``${parameters.*}`` substitution,
6. calls the LLM once, and
7. extracts the first JSON object, falling back to ``match_all`` when the model
   returns nothing usable.

The control flow mirrors ``QueryPlanningTool.java`` /
``QueryPlanningPromptTemplate.java``; see ``docs/reference/`` for the originals.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from strands import tool

from ...config import DEFAULT_DATETIME_STRFTIME, QPTConfig, qpt as default_qpt_config
from ..substitutor import substitute
from . import prompts
from .output_parser import parse_query_output

# Agent-context parameter keys to ignore (QueryPlanningTool.AGENT_CONTEXT_EXCLUDED_PARAMS)
_AGENT_CONTEXT_EXCLUDED_PARAMS = frozenset(
    {"_chat_history", "_tools", "_interactions", "tool_configs"}
)

_LLM_GENERATED = "llmGenerated"
_USER_TEMPLATES = "user_templates"

# A callable that takes (system_prompt, user_prompt) and returns the raw model
# output (string text, or a JSON envelope to be filtered by response_filter).
ModelInvoker = Callable[[str, str], Any]

# A callable that resolves a stored search-template id to its source string.
# (Optional; only used in user_templates mode. Returns None if not found.)
TemplateResolver = Callable[[str], "str | None"]


@dataclass
class QueryContext:
    """Per-call context the tool needs to gather index info + call the model."""

    invoke_model: ModelInvoker
    fetch_index_mapping: Callable[[str], str]
    fetch_sample_document: Callable[[str], "str | None"]
    resolve_template: TemplateResolver | None = None


def _current_datetime() -> str:
    """AgentUtils.getCurrentDateTime(DEFAULT_DATETIME_FORMAT) in UTC."""
    return time.strftime(DEFAULT_DATETIME_STRFTIME, time.gmtime())


def _strip_agent_context(parameters: dict[str, Any]) -> dict[str, Any]:
    """Drop nulls and agent-specific metadata keys."""
    return {
        k: v
        for k, v in parameters.items()
        if v is not None and k not in _AGENT_CONTEXT_EXCLUDED_PARAMS
    }


class QueryPlanningTool:
    """Stateful, reusable query planner. Construct once, call ``plan`` per query."""

    def __init__(self, context: QueryContext, config: QPTConfig | None = None):
        self.context = context
        self.config = config or default_qpt_config
        gen = self.config.generation_type or _LLM_GENERATED
        if gen not in (_LLM_GENERATED, _USER_TEMPLATES):
            raise ValueError(
                f"Invalid generation type: {gen}. "
                "The current supported types are llmGenerated and user_templates."
            )
        self.generation_type = gen

    # --- public API -----------------------------------------------------------

    def plan(
        self,
        question: str,
        index_name: str,
        *,
        query_fields: list[str] | None = None,
        embedding_model_id: str | None = None,
        search_templates: list[dict[str, str]] | None = None,
    ) -> str:
        """Return an OpenSearch query DSL JSON string for ``question``."""
        parameters: dict[str, Any] = _strip_agent_context(
            {
                "question": question,
                "index_name": index_name,
                "query_fields": query_fields,
                "embedding_model_id": embedding_model_id,
            }
        )
        if not self._validate(parameters):
            raise ValueError(
                "Validation error: missing or empty required parameters — "
                "index_name, question."
            )

        if self.generation_type != _USER_TEMPLATES:
            parameters["template"] = prompts.DEFAULT_SEARCH_TEMPLATE
            return self._execute_query_planning(parameters)

        # user_templates: run template selection first.
        template_id = self._select_template(parameters, search_templates or [])
        parameters["template"] = prompts.DEFAULT_SEARCH_TEMPLATE
        if template_id and template_id.strip() and template_id != "null":
            resolver = self.context.resolve_template
            source = resolver(template_id) if resolver else None
            if source is not None:
                parameters["template"] = json.dumps(source)
        return self._execute_query_planning(parameters)

    # --- internals ------------------------------------------------------------

    @staticmethod
    def _validate(parameters: dict[str, Any]) -> bool:
        return bool(
            parameters
            and parameters.get("question")
            and parameters.get("index_name")
        )

    def _select_template(
        self, parameters: dict[str, Any], search_templates: list[dict[str, str]]
    ) -> str | None:
        sel_params = dict(parameters)
        sel_params["search_templates"] = json.dumps(search_templates)
        system = prompts.DEFAULT_TEMPLATE_SELECTION_SYSTEM_PROMPT
        user = substitute(prompts.DEFAULT_TEMPLATE_SELECTION_USER_PROMPT, sel_params)
        raw = self.context.invoke_model(system, user)
        # Template selection returns the bare id; apply response_filter for envelopes.
        from .output_parser import apply_response_filter

        text = apply_response_filter(raw, self.config.response_filter)
        return text.strip() if isinstance(text, str) else None

    def _execute_query_planning(self, parameters: dict[str, Any]) -> str:
        # 1. fallback query → escaped → injected into the system prompt placeholder
        effective_fallback = self.config.fallback_query or prompts.DEFAULT_QUERY
        parameters["fallback_query"] = effective_fallback
        escaped_fallback = json.dumps(effective_fallback)[1:-1]  # strip surrounding quotes

        system_prompt = prompts.DEFAULT_QUERY_PLANNING_SYSTEM_PROMPT.replace(
            prompts.FALLBACK_QUERY_PROMPT_PLACEHOLDER, escaped_fallback
        )
        user_prompt = prompts.DEFAULT_QUERY_PLANNING_USER_PROMPT

        # 2. gson-encode query_fields if present
        if "query_fields" in parameters:
            parameters["query_fields"] = json.dumps(parameters["query_fields"])

        # 3. current time (gson.toJson → quoted string)
        parameters["current_time"] = json.dumps(_current_datetime())

        # 4. fetch index mapping then sample doc (via MCP tooling layer)
        index_name = parameters["index_name"]
        parameters["index_mapping"] = json.dumps(
            self.context.fetch_index_mapping(index_name)
        )
        parameters["sample_document"] = json.dumps(
            self.context.fetch_sample_document(index_name)
        )

        # 5. substitute ${parameters.*} into the prompts and call the model
        rendered_system = substitute(system_prompt, parameters)
        rendered_user = substitute(user_prompt, parameters)
        raw = self.context.invoke_model(rendered_system, rendered_user)

        # 6. handle null/blank/"null" → fallback (with ${parameters.*} substitution)
        from .output_parser import apply_response_filter

        text = apply_response_filter(raw, self.config.response_filter)
        if text is None or not str(text).strip() or str(text).strip() == "null":
            return substitute(effective_fallback, parameters)

        # 7. extract first JSON object, else DEFAULT_QUERY
        return parse_query_output(raw, self.config.response_filter)


def make_query_planning_tool(planner: QueryPlanningTool):
    """Wrap a :class:`QueryPlanningTool` as a Strands ``@tool``.

    The tool's name, description, and input schema match the OpenSearch
    ``QueryPlanningTool`` (``query_planner_tool`` / "qpt"), so the conversational
    agent's verbatim system prompt addresses it correctly.
    """

    @tool(name="query_planner_tool", description=prompts.DEFAULT_DESCRIPTION)
    def query_planner_tool(
        question: str,
        index_name: str,
        embedding_model_id: str | None = None,
    ) -> str:
        """Generate OpenSearch Query DSL from a natural-language question.

        Args:
            question: Complete natural language query with all necessary context
                to generate OpenSearch DSL. Include the question, any specific
                requirements, filters, or constraints.
            index_name: The name of the index against which the query needs to be
                generated.
            embedding_model_id: The model id to perform neural search.
        """
        return planner.plan(
            question=question,
            index_name=index_name,
            embedding_model_id=embedding_model_id,
        )

    return query_planner_tool
