"""The flow agent — deterministic single-shot query planning.

Equivalent to an OpenSearch ``type: "flow"`` agent whose sole tool is the
QueryPlanningTool: it runs QPT directly and returns ONLY the generated DSL — no
ReAct loop, no memory, no reasoning trace. ``index_name`` is required (a flow
agent cannot auto-discover indexes).
"""

from __future__ import annotations

import json
from typing import Any

from ..config import QPTConfig
from ..tooling import factory
from ..tooling.mcp_client import MCPClient


class FlowAgent:
    """Runs the QueryPlanningTool deterministically and returns query DSL."""

    def __init__(
        self,
        client: MCPClient,
        *,
        model: Any | None = None,
        qpt_config: QPTConfig | None = None,
    ):
        # The flow agent needs only QPT (its index context is fetched via MCP);
        # it does not expose the MCP tools to a reasoning loop.
        self._planner = factory.build_query_planner(
            client, model=model, qpt_config=qpt_config
        )

    def run(
        self,
        question: str,
        index_name: str,
        *,
        query_fields: list[str] | None = None,
        embedding_model_id: str | None = None,
    ) -> dict[str, Any]:
        """Return ``{"dsl_query": <DSL object>}`` for ``question`` over ``index_name``."""
        dsl_string = self._planner.plan(
            question=question,
            index_name=index_name,
            query_fields=query_fields,
            embedding_model_id=embedding_model_id,
        )
        return {"dsl_query": json.loads(dsl_string)}
