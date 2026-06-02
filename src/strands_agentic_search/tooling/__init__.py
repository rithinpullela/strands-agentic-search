"""The tooling layer: the only module that speaks OpenSearch/MCP.

Agents import from here to obtain the QueryPlanningTool and the MCP-discovered
OpenSearch tools, without any knowledge of MCP transports or OpenSearch clients.
"""

from .mcp_client import (
    build_mcp_client,
    fetch_index_mapping,
    fetch_sample_document,
    list_tools,
)

__all__ = [
    "build_mcp_client",
    "list_tools",
    "fetch_index_mapping",
    "fetch_sample_document",
]
