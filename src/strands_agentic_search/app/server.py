"""FastAPI server exposing the flow and conversational agents on ``:8080``.

The opensearch-mcp-server-py MCP server is launched as a stdio child process and
its session is held open for the lifetime of the app (so both agents can call
OpenSearch tools per request). Agents are constructed once at startup.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .. import config
from ..agents.conversational import ConversationalAgent
from ..agents.flow_agent import FlowAgent
from ..llm import build_model
from ..tooling import factory

logger = logging.getLogger("strands_agentic_search")

# Holds the live MCP client + constructed agents between requests.
_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the MCP server (stdio) and build the agents once."""
    mcp_client = factory.build_mcp_client()
    logger.info("Starting OpenSearch MCP server over stdio: %s %s",
                config.mcp.command, " ".join(config.mcp.args))
    mcp_client.start()  # enter the stdio session; kept open for the app's life
    try:
        model = build_model()
        _state["mcp_client"] = mcp_client
        _state["flow_agent"] = FlowAgent(mcp_client, model=model)
        _state["conversational_agent"] = ConversationalAgent(mcp_client, model=model)
        logger.info("Agents ready.")
        yield
    finally:
        _state.clear()
        mcp_client.stop(None, None, None)
        logger.info("MCP server stopped.")


app = FastAPI(
    title="Strands Agentic Search",
    description="1:1 replication of OpenSearch agentic search (flow + conversational agents).",
    version="0.1.0",
    lifespan=lifespan,
)


# --- request/response models --------------------------------------------------


class FlowRequest(BaseModel):
    question: str = Field(..., description="Natural-language query.")
    index_name: str = Field(..., description="Target index (required for the flow agent).")
    query_fields: list[str] | None = Field(
        default=None, description="Optional field hints prioritized if present in the mapping."
    )
    embedding_model_id: str | None = Field(
        default=None, description="Optional model id for neural search."
    )


class ConversationalRequest(BaseModel):
    question: str = Field(..., description="Natural-language query.")
    index_name: str | None = Field(
        default=None,
        description="Optional target index; omitted → the agent discovers it via MCP tools.",
    )
    embedding_model_id: str | None = Field(
        default=None, description="Optional model id for neural search."
    )


class DslResponse(BaseModel):
    dsl_query: dict[str, Any]


# --- endpoints ----------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    ready = "flow_agent" in _state and "conversational_agent" in _state
    return {"status": "ok" if ready else "starting"}


@app.post("/flow_agent", response_model=DslResponse)
async def flow_agent(request: FlowRequest) -> DslResponse:
    """Deterministic single-shot QueryPlanningTool → DSL only."""
    agent: FlowAgent = _state["flow_agent"]
    try:
        result = agent.run(
            question=request.question,
            index_name=request.index_name,
            query_fields=request.query_fields,
            embedding_model_id=request.embedding_model_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surface upstream failures as 502
        logger.exception("flow_agent failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return DslResponse(**result)


@app.post("/conversational_agent", response_model=DslResponse)
async def conversational_agent(request: ConversationalRequest) -> DslResponse:
    """ReAct agent (QPT + MCP tools) → {"dsl_query": <DSL>}."""
    agent: ConversationalAgent = _state["conversational_agent"]
    try:
        result = agent.run(
            question=request.question,
            index_name=request.index_name,
            embedding_model_id=request.embedding_model_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("conversational_agent failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return DslResponse(**result)


def main() -> None:
    """Console-script entrypoint: ``strands-agentic-search``."""
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "strands_agentic_search.app.server:app",
        host=config.server.host,
        port=config.server.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
