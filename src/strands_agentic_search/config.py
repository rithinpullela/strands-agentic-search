"""Central, environment-driven configuration.

Everything that varies between deployments lives here so the agents and the
tooling layer stay declarative. OpenSearch connection details deliberately do
NOT live here — they are configured on the opensearch-mcp-server-py process
(see ``MCP_SERVER_ENV`` below), keeping the MCP server the single source of
OpenSearch truth.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str) -> str:
    val = os.environ.get(key)
    return val if val is not None and val != "" else default


# --- Verbatim constants from the OpenSearch Java source -----------------------

# QueryPlanningPromptTemplate.DEFAULT_QUERY
DEFAULT_QUERY: str = '{"size":10,"query":{"match_all":{}}}'

# AgentUtils.DEFAULT_DATETIME_FORMAT (Java SimpleDateFormat) → strftime equivalent.
# Java: yyyy-MM-dd'T'HH:mm:ss'Z'  →  Python: %Y-%m-%dT%H:%M:%SZ
DEFAULT_DATETIME_STRFTIME: str = "%Y-%m-%dT%H:%M:%SZ"


@dataclass(frozen=True)
class LLMConfig:
    """Bedrock model configuration (Strands' default provider)."""

    provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "bedrock"))
    model_id: str = field(
        default_factory=lambda: _env(
            "BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0"
        )
    )
    region_name: str = field(default_factory=lambda: _env("AWS_REGION", "us-west-2"))
    # Deterministic query planning → temperature 0 by default.
    temperature: float = field(
        default_factory=lambda: float(_env("LLM_TEMPERATURE", "0"))
    )


@dataclass(frozen=True)
class QPTConfig:
    """QueryPlanningTool registration-time configuration.

    Mirrors the ``Factory.create`` params of QueryPlanningTool.java.
    """

    # "llmGenerated" (default) | "user_templates"
    generation_type: str = field(
        default_factory=lambda: _env("QPT_GENERATION_TYPE", "llmGenerated")
    )
    # JSONPath applied to the raw model envelope to extract the text.
    # Bedrock Converse (Claude): $.output.message.content[0].text
    # OpenAI:                     $.choices[0].message.content
    response_filter: str = field(
        default_factory=lambda: _env(
            "QPT_RESPONSE_FILTER", "$.output.message.content[0].text"
        )
    )
    # Optional override of the match_all fallback (JSON string). None → DEFAULT_QUERY.
    fallback_query: str | None = field(
        default_factory=lambda: os.environ.get("QPT_FALLBACK_QUERY") or None
    )


@dataclass(frozen=True)
class MCPConfig:
    """How to launch / reach the opensearch-mcp-server-py MCP server.

    Per design, transport is stdio: our process spawns the server as a child.
    OpenSearch URL + auth are passed to the *server* via these env vars.
    """

    # Default to launching the server with the same interpreter that installed
    # it (``python -m mcp_server_opensearch``), guaranteeing the version matches
    # what we pinned. Override with MCP_COMMAND/MCP_ARGS to use ``uvx`` instead.
    command: str = field(default_factory=lambda: _env("MCP_COMMAND", sys.executable))
    # space-separated args
    args: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            _env("MCP_ARGS", "-m mcp_server_opensearch").split()
        )
    )

    # Only expose the read tools the agents actually need. Empty → all tools.
    allowed_tools: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            t.strip()
            for t in _env(
                "MCP_ALLOWED_TOOLS",
                "ListIndexTool,IndexMappingTool,SearchIndexTool,GetIndexInfoTool",
            ).split(",")
            if t.strip()
        )
    )

    @property
    def server_env(self) -> dict[str, str]:
        """Environment passed to the spawned MCP server (OpenSearch connection)."""
        env: dict[str, str] = {}
        # Basic-auth placeholders defaulting to a local cluster.
        env["OPENSEARCH_URL"] = _env("OPENSEARCH_URL", "http://localhost:9200")
        username = os.environ.get("OPENSEARCH_USERNAME")
        password = os.environ.get("OPENSEARCH_PASSWORD")
        if username and password:
            env["OPENSEARCH_USERNAME"] = username
            env["OPENSEARCH_PASSWORD"] = password
        else:
            # No credentials supplied → tell the server to skip auth (local dev).
            env["OPENSEARCH_NO_AUTH"] = _env("OPENSEARCH_NO_AUTH", "true")
        # SSL verification toggle (local clusters often use self-signed certs).
        env["OPENSEARCH_SSL_VERIFY"] = _env("OPENSEARCH_SSL_VERIFY", "false")
        return env


@dataclass(frozen=True)
class ServerConfig:
    host: str = field(default_factory=lambda: _env("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(_env("PORT", "8080")))


llm = LLMConfig()
qpt = QPTConfig()
mcp = MCPConfig()
server = ServerConfig()
