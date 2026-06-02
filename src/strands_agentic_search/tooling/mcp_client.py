"""The MCP tooling boundary.

Owns the connection to ``opensearch-mcp-server-py`` (launched over stdio as a
child process) and is the *only* place that speaks MCP. Agents receive a flat
list of ready-to-use Strands tools and never know they came from MCP.

It also exposes two deterministic helpers — :func:`fetch_index_mapping` and
:func:`fetch_sample_document` — that the QueryPlanningTool uses to gather index
context. In the OpenSearch Java these go through a direct cluster ``Client``;
here every OpenSearch access flows through the MCP server (single source of
truth), so QPT calls ``IndexMappingTool`` / ``SearchIndexTool`` directly via the
MCP session rather than the agent's tool loop.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from mcp import StdioServerParameters, stdio_client
from strands.tools.mcp import MCPClient

from ..config import MCPConfig, mcp as default_mcp_config


def build_mcp_client(config: MCPConfig | None = None) -> MCPClient:
    """Construct a Strands ``MCPClient`` for the OpenSearch MCP server (stdio).

    The returned client is *not* started; use it as a context manager
    (``with client:``) or pass it directly into ``Agent(tools=[client])`` for
    Strands-managed lifecycle. Tool filtering restricts the surface to the
    configured allowlist.
    """
    c = config or default_mcp_config

    def _transport():
        return stdio_client(
            StdioServerParameters(
                command=c.command,
                args=list(c.args),
                env=c.server_env,
            )
        )

    tool_filters = {"allowed": list(c.allowed_tools)} if c.allowed_tools else None
    return MCPClient(_transport, tool_filters=tool_filters)


def list_tools(client: MCPClient) -> list[Any]:
    """Return all (filtered) tools advertised by the server.

    Must be called inside the client's ``with`` block.
    """
    return client.list_tools_sync()


# --- Deterministic helpers used by the QueryPlanningTool ----------------------

# Default OpenSearch index-mapping / search tool names exposed by the server.
INDEX_MAPPING_TOOL = "IndexMappingTool"
SEARCH_INDEX_TOOL = "SearchIndexTool"

_MAX_TRUNCATE_CHARS = 250  # QueryPlanningTool.MAX_TRUNCATE_CHARS
_TRUNC_PREFIX = "[truncated]"  # QueryPlanningTool.TRUNC_PREFIX


def _is_error(result: Any) -> bool:
    """True if the MCP tool call reported an error.

    ``MCPToolResult`` is a TypedDict carrying both a ``status`` literal
    (``success``/``error``) and an optional ``isError`` flag; honor either.
    """
    if not isinstance(result, dict):
        return False
    return result.get("isError") is True or result.get("status") == "error"


def _tool_text(result: Any) -> str:
    """Flatten an MCP ``call_tool_sync`` result into a text string."""
    content = result.get("content") if isinstance(result, dict) else None
    if not content:
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            if "text" in block:
                parts.append(block["text"])
            elif "json" in block:
                parts.append(json.dumps(block["json"]))
        else:
            text = getattr(block, "text", None)
            if text is not None:
                parts.append(text)
    return "\n".join(parts)


def fetch_index_mapping(client: MCPClient, index_name: str) -> str:
    """Fetch the index mapping via the MCP ``IndexMappingTool``.

    Returns the mapping serialized as a string (parity with the Java, which
    sets ``index_mapping = gson.toJson(mapping)``). Raises ``ValueError`` if the
    server reports the index is unavailable.
    """
    result = client.call_tool_sync(
        tool_use_id="qpt-index-mapping",
        name=INDEX_MAPPING_TOOL,
        arguments={"index": index_name},
    )
    if _is_error(result):
        raise ValueError(f"Index does not exist or is not available: {index_name}")
    text = _tool_text(result)
    if not text.strip():
        raise ValueError(f"Failed to extract index mapping for {index_name}")
    return text


def fetch_sample_document(client: MCPClient, index_name: str) -> str | None:
    """Fetch one sample document via the MCP ``SearchIndexTool``.

    Replicates the Java sample-doc behavior: size-1 match_all, then each field
    value longer than 250 codepoints is truncated with a ``[truncated]`` prefix.
    Returns a JSON string of the (truncated) source map, or ``None`` if empty.
    """
    query_dsl = {"size": 1, "query": {"match_all": {}}, "track_total_hits": False}
    result = client.call_tool_sync(
        tool_use_id="qpt-sample-doc",
        name=SEARCH_INDEX_TOOL,
        arguments={"index": index_name, "query_dsl": query_dsl},
    )
    if _is_error(result):
        return None
    source = _extract_first_source(_tool_text(result))
    if not source:
        return None
    truncated = {k: _truncate(str(v)) for k, v in source.items()}
    return json.dumps(truncated)


def _truncate(value: str) -> str:
    if len(value) > _MAX_TRUNCATE_CHARS:
        return _TRUNC_PREFIX + value[:_MAX_TRUNCATE_CHARS]
    return value


def _extract_first_source(text: str) -> dict[str, Any] | None:
    """Pull the first hit's ``_source`` out of a SearchIndexTool response."""
    if not text or not text.strip():
        return None
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    hits = _dig_hits(payload)
    if not hits:
        return None
    first = hits[0]
    source = first.get("_source") if isinstance(first, dict) else None
    return source if isinstance(source, dict) and source else None


def _dig_hits(payload: Any) -> Sequence[Any] | None:
    """Locate the hits array in common SearchIndexTool response shapes."""
    if isinstance(payload, dict):
        hits = payload.get("hits")
        if isinstance(hits, dict) and isinstance(hits.get("hits"), list):
            return hits["hits"]
        if isinstance(hits, list):
            return hits
    return None
