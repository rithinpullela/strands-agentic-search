"""Wiring for the tooling layer.

Builds, from config, the pieces the agents need:

* a live :class:`MCPClient` (context-managed by the caller),
* the QueryPlanningTool wrapped as a Strands tool, backed by MCP-driven index
  context and the configured model, and
* the filtered list of OpenSearch MCP tools.

Agents call :func:`build_toolset` inside the MCP client's ``with`` block and get
back a flat tool list, never touching MCP details themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import MCPConfig, QPTConfig
from ..llm import build_model
from . import mcp_client as mcp_mod
from .query_planning import (
    QueryContext,
    QueryPlanningTool,
    make_model_invoker,
    make_query_planning_tool,
)


@dataclass
class Toolset:
    """The assembled tools an agent operates with."""

    query_planning_tool: Any  # Strands @tool: query_planner_tool
    mcp_tools: list[Any]  # OpenSearch tools discovered from the MCP server

    @property
    def all_tools(self) -> list[Any]:
        """QPT first, then the MCP tools — a flat, source-agnostic list."""
        return [self.query_planning_tool, *self.mcp_tools]


def build_query_planner(
    client: mcp_mod.MCPClient,
    *,
    model: Any | None = None,
    qpt_config: QPTConfig | None = None,
) -> QueryPlanningTool:
    """Construct a QueryPlanningTool whose index context comes from MCP."""
    invoke_model = make_model_invoker(model or build_model())
    context = QueryContext(
        invoke_model=invoke_model,
        fetch_index_mapping=lambda idx: mcp_mod.fetch_index_mapping(client, idx),
        fetch_sample_document=lambda idx: mcp_mod.fetch_sample_document(client, idx),
    )
    return QueryPlanningTool(context, config=qpt_config)


def build_toolset(
    client: mcp_mod.MCPClient,
    *,
    model: Any | None = None,
    qpt_config: QPTConfig | None = None,
    include_mcp_tools: bool = True,
) -> Toolset:
    """Assemble the QPT tool + MCP tool list. Call inside ``with client:``."""
    planner = build_query_planner(client, model=model, qpt_config=qpt_config)
    qpt_tool = make_query_planning_tool(planner)
    mcp_tools = mcp_mod.list_tools(client) if include_mcp_tools else []
    return Toolset(query_planning_tool=qpt_tool, mcp_tools=mcp_tools)


def build_mcp_client(config: MCPConfig | None = None) -> mcp_mod.MCPClient:
    """Re-export for agents: construct (but don't start) the MCP client."""
    return mcp_mod.build_mcp_client(config)
