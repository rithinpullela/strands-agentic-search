# Strands Agentic Search

A faithful, **1:1 replication** of [OpenSearch agentic search](https://docs.opensearch.org/latest/vector-search/ai-search/agentic-search/index/)
built on the [Strands Agents SDK](https://github.com/strands-agents/sdk-python), exposing **two agents behind two HTTP endpoints**:

| Endpoint | Agent | What it does |
|---|---|---|
| `POST /flow_agent` | **Flow agent** | Runs the **QueryPlanningTool (QPT)** deterministically and returns OpenSearch query DSL. No reasoning loop, no memory. `index_name` is **required**. |
| `POST /conversational_agent` | **Conversational agent** | A **ReAct** agent whose system prompt is the verbatim OpenSearch `agentic-system-prompt.txt`. It orchestrates QPT **plus** OpenSearch tools discovered from an MCP server. `index_name` is **optional** (auto-discovered). |

The two agents share one deterministic core — **QPT** — which is the load-bearing
piece of OpenSearch agentic search: a "mini agent" that turns a natural-language
question into a query DSL via a single, tightly-controlled LLM call.

## Why this is a faithful replica

The prompts and control flow are copied from the OpenSearch source, not paraphrased:

- **QPT prompts** are *extracted programmatically* from
  [`QueryPlanningPromptTemplate.java`](https://github.com/opensearch-project/ml-commons/blob/main/ml-algorithms/src/main/java/org/opensearch/ml/engine/tools/QueryPlanningTool.java)
  by [`scripts/extract_qpt_prompts.py`](scripts/extract_qpt_prompts.py) — evaluating the Java
  string/text-block concatenation so every smart quote, bullet, `≥`, and the
  `{{FALLBACK_QUERY}}` placeholder survive byte-for-byte. A test
  ([`tests/test_prompt_fidelity.py`](tests/test_prompt_fidelity.py)) re-derives them from the
  archived source and asserts equality.
- **QPT control flow** mirrors
  [`QueryPlanningTool.java`](https://github.com/opensearch-project/ml-commons/blob/main/ml-algorithms/src/main/java/org/opensearch/ml/engine/tools/QueryPlanningTool.java):
  validate → strip agent-context params → template selection (`user_templates`) or
  default search template (`llmGenerated`) → fetch index mapping + sample doc →
  `${parameters.*}` prompt substitution → one LLM call → extract first JSON object,
  else `match_all` fallback.
- **Conversational prompts** are the verbatim neural-search
  [`agentic-system-prompt.txt`](https://github.com/opensearch-project/neural-search/blob/main/src/main/resources/agentic-system-prompt.txt)
  and [`agentic-user-prompt.txt`](https://github.com/opensearch-project/neural-search/blob/main/src/main/resources/agentic-user-prompt.txt).

The original Java sources are archived under [`docs/reference/`](docs/reference/) so the 1:1 claim is auditable.

## Architecture

```
                         app/server.py  (FastAPI :8080)
                POST /flow_agent              POST /conversational_agent
                       │                                │
              agents/flow_agent.py        agents/conversational/agent.py
              run QPT once, return DSL     Strands ReAct Agent
                       │                    system = agentic-system-prompt.txt (verbatim)
                       │                    tools  = [qpt] + MCP tools
                       │                    output = {"dsl_query": <DSL>}
                       └───────────────┬────────────────┘
                                       ▼
        ┌────────────────────── tooling/  (the only module that speaks MCP) ──────┐
        │  query_planning/tool.py    QPT control-flow replica                     │
        │      prompts.py            verbatim QPT prompts (loaded from assets)     │
        │      output_parser.py      extract first JSON object / match_all         │
        │      model_invoker.py      single-shot LLM call via a tool-less Agent    │
        │  mcp_client.py             opensearch-mcp-server-py over stdio           │
        │  substitutor.py            ${parameters.x:-default} substitution         │
        │  factory.py                wires model + MCP + QPT into a Toolset        │
        └──────────────────────────────────────────────────────────────────────────┘
                                       ▼
                  opensearch-mcp-server-py (stdio child)  ─►  OpenSearch
                  tools: ListIndexTool, IndexMappingTool, SearchIndexTool, …
```

**The tooling layer is self-contained.** Agents receive a flat list of Strands
tools and never know which came from MCP. All OpenSearch access — including QPT's
own mapping/sample-doc fetch — flows through the MCP server, so it is the single
source of OpenSearch truth.

### Project layout

```
src/strands_agentic_search/
├── config.py                 # env-driven config (Bedrock, MCP launch, OpenSearch passthrough)
├── llm.py                    # model-provider factory (Bedrock default)
├── tooling/                  # ← the MCP/OpenSearch layer
│   ├── mcp_client.py         #   connect to opensearch-mcp-server-py (stdio); QPT index helpers
│   ├── substitutor.py        #   ${parameters.*} substitution
│   ├── factory.py            #   build model + MCP client + QPT → Toolset
│   └── query_planning/
│       ├── prompts.py        #   verbatim QPT prompts
│       ├── prompt_assets/    #   extracted prompt text (byte-for-byte from Java)
│       ├── output_parser.py  #   extract_json + response_filter
│       ├── model_invoker.py  #   single-shot model call
│       └── tool.py           #   QueryPlanningTool replica + @tool wrapper
├── agents/
│   ├── flow_agent.py
│   └── conversational/
│       ├── agent.py
│       └── prompts/          #   agentic-system-prompt.txt, agentic-user-prompt.txt (verbatim)
└── app/server.py             # FastAPI on :8080
```

## Prerequisites

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) (used to launch the MCP server via `uvx`)
- AWS credentials with Bedrock access (the default model is Claude Sonnet on Bedrock)
- An OpenSearch cluster reachable by the MCP server (defaults to `http://localhost:9200`)

## Setup

```bash
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

cp .env.example .env   # then edit as needed
```

Configuration is entirely environment-driven — see [`.env.example`](.env.example). Highlights:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `bedrock` | `bedrock` \| `anthropic` \| `openai` |
| `BEDROCK_MODEL_ID` | `us.anthropic.claude-sonnet-4-20250514-v1:0` | Bedrock model |
| `AWS_REGION` | `us-west-2` | Bedrock region |
| `QPT_GENERATION_TYPE` | `llmGenerated` | `llmGenerated` \| `user_templates` |
| `QPT_RESPONSE_FILTER` | `$.output.message.content[0].text` | JSONPath into the model envelope |
| `MCP_COMMAND` / `MCP_ARGS` | `uvx` / `opensearch-mcp-server-py` | How the MCP server is launched (stdio) |
| `MCP_ALLOWED_TOOLS` | `ListIndexTool,IndexMappingTool,SearchIndexTool,GetIndexInfoTool` | MCP tool allowlist |
| `OPENSEARCH_URL` | `http://localhost:9200` | Passed through to the MCP server |
| `OPENSEARCH_USERNAME` / `OPENSEARCH_PASSWORD` | — | Basic auth (omit → no-auth for local dev) |
| `PORT` | `8080` | HTTP port |

> **OpenSearch connection details live with the MCP server**, not in app code — set
> them via the `OPENSEARCH_*` variables and they are forwarded to the spawned server.

## Run

```bash
source .venv/bin/activate
strands-agentic-search            # serves on http://localhost:8080
# or: python -m strands_agentic_search.app.server
```

Check health:

```bash
curl localhost:8080/health
```

### Flow agent — deterministic DSL generation

```bash
curl -s localhost:8080/flow_agent -H 'content-type: application/json' -d '{
  "question": "Find shoes under 500 dollars",
  "index_name": "products"
}' | jq
```

```json
{ "dsl_query": { "query": { "bool": {
  "must":   [ { "match": { "category": "Shoes" } } ],
  "filter": [ { "range": { "price": { "lte": 500 } } } ]
} } } }
```

### Conversational agent — ReAct with auto-discovery

```bash
curl -s localhost:8080/conversational_agent -H 'content-type: application/json' -d '{
  "question": "Find shoes under 500 dollars. I am so excited for shoes yay!"
}' | jq
```

`index_name` is optional here — the agent uses `ListIndexTool` / `IndexMappingTool`
to discover and pick an index, composes a clean natural-language question for QPT,
and returns `{"dsl_query": <DSL>}`.

Interactive API docs are at `http://localhost:8080/docs`.

## Tests

```bash
source .venv/bin/activate
pytest -q
```

Covers prompt fidelity (re-derivation from the Java source), the `${parameters.*}`
substitutor, the output parser (`extract_json` + `response_filter` + fallback), the
full QPT control flow (MCP + LLM mocked), and the conversational agent's prompt
rendering + output normalization. These run without Bedrock or a live cluster; live
LLM/OpenSearch calls are exercised by running the server.

## Refreshing the QPT prompts

If the OpenSearch source changes, re-extract:

```bash
python scripts/extract_qpt_prompts.py \
  docs/reference/QueryPlanningPromptTemplate.java \
  src/strands_agentic_search/tooling/query_planning/prompt_assets
```

The fidelity test will confirm the shipped assets match the source.

## Design notes & deviations

- **QPT fetches its index context via MCP**, whereas the Java uses a direct cluster
  `Client`. This is the one deliberate deviation, made to keep the MCP server the
  single source of OpenSearch truth.
- **`generation_type: user_templates`** is implemented (template selection + stored-
  template resolution), but resolving a stored template id requires a resolver hook;
  by default only `llmGenerated` is wired end-to-end.
- **Out of scope (YAGNI):** OpenSearch search-pipeline processors
  (`agentic_query_translator`, `agentic_context`), cross-request persistent memory,
  multi-cluster MCP mode.

## License

Apache-2.0. Prompt text and the archived Java sources are © OpenSearch Contributors (Apache-2.0).
