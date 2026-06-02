# Strands Agentic Search — Design

A 1:1 replication of [OpenSearch agentic search](https://docs.opensearch.org/latest/vector-search/ai-search/agentic-search/index/)
built with the [Strands Agents SDK](https://github.com/strands-agents/sdk-python), exposing **two agents behind two HTTP endpoints**.

## Goals

1. **Flow agent** (`POST /flow_agent`) — deterministically run the **QueryPlanningTool (QPT)** and return OpenSearch query DSL. No reasoning loop, no memory.
2. **Conversational agent** (`POST /conversational_agent`) — a **ReAct agent** whose system prompt is the verbatim OpenSearch `agentic-system-prompt.txt`, with QPT plus tools discovered from the **opensearch-mcp-server-py** MCP server.
3. **1:1 fidelity** — QPT prompts/control-flow replicate `QueryPlanningTool.java` + `QueryPlanningPromptTemplate.java`; the conversational prompts are the verbatim neural-search `.txt` files.
4. **Tooling layer is its own module** — everything OpenSearch/MCP lives under `tooling/`. Agents never speak MCP directly.

## Reference sources (replicated verbatim)

| Source | Role |
|---|---|
| `ml-commons …/tools/QueryPlanningTool.java` | QPT control flow, params, validation, fallback |
| `ml-commons …/tools/QueryPlanningPromptTemplate.java` | All QPT prompts (system/user, template-selection, 13 examples, default search template) |
| `neural-search …/resources/agentic-system-prompt.txt` | Conversational agent system prompt |
| `neural-search …/resources/agentic-user-prompt.txt` | Conversational agent user-message template |
| `opensearch-mcp-server-py` | MCP server exposing `ListIndexTool`, `IndexMappingTool`, `SearchIndexTool`, … |

## Key design decisions (locked with user)

- **LLM provider:** Amazon Bedrock (Strands default, Claude Sonnet). Env-driven `model_id`, `region`. Matches the OpenSearch Bedrock-Converse setup, hence the default `response_filter` JSONPath `$.output.message.content[0].text`.
- **OpenSearch access goes entirely through the MCP server.** OpenSearch URL/auth (basic auth, `localhost:9200` placeholder) is configured on the MCP server, not in our code. QPT therefore fetches index mapping + sample doc via the MCP `IndexMappingTool` / `SearchIndexTool` rather than its own `opensearch-py` client. This is the one deliberate deviation from the Java (which uses a direct cluster `Client`) and it keeps the tooling boundary clean — single source of OpenSearch truth.
- **MCP transport:** **stdio**, spawned by our process (`uvx opensearch-mcp-server-py`). The MCP server is a child of the Strands app — nothing extra to run.
- **Flow agent output:** **DSL only** (1:1 with the OpenSearch flow agent + `agentic_query_translator`).
- **`index_name` is a per-execution variable** on both endpoints: **required** for `/flow_agent`, **optional** for `/conversational_agent` (the ReAct loop auto-discovers via `ListIndexTool`/`IndexMappingTool` when omitted).

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
        ┌────────────────────── tooling/ ──────────────────────┐
        │  query_planning/tool.py   QPT control-flow replica    │
        │      prompts.py           verbatim QPT prompts        │
        │      output_parser.py     extract first JSON / match_all
        │  mcp_client.py            opensearch-mcp-server-py (stdio)
        │                           → IndexMappingTool, SearchIndexTool,
        │                             ListIndexTool, SearchIndexTool, …  │
        └────────────────────────────────────────────────────────┘
                                       ▼
                       opensearch-mcp-server-py ──► OpenSearch (localhost:9200)
```

## Module layout

```
src/strands_agentic_search/
├── config.py                 # env: Bedrock model/region, MCP launch cmd, response_filter, QPT generation_type
├── llm.py                    # BedrockModel factory
├── tooling/
│   ├── mcp_client.py         # build MCPClient(stdio uvx …); list (filtered) tools
│   ├── substitutor.py        # ${parameters.x:-default} substitution (Apache StringSubstitutor analogue)
│   └── query_planning/
│       ├── prompts.py        # verbatim constants from QueryPlanningPromptTemplate.java
│       ├── output_parser.py  # extract_json(object) first, else DEFAULT_QUERY
│       └── tool.py           # QueryPlanningTool replica + @tool wrapper
├── agents/
│   ├── flow_agent.py
│   └── conversational/
│       ├── agent.py
│       └── prompts/agentic-system-prompt.txt, agentic-user-prompt.txt   # verbatim
└── app/server.py
```

## QPT control flow (replicated from the Java)

1. **Validate** `question` + `index_name` present → else `ValueError`.
2. **Strip agent-context params** (`_chat_history`, `_tools`, `_interactions`, `tool_configs`) and nulls.
3. **Generation type:**
   - `llmGenerated` (default): set `template = DEFAULT_SEARCH_TEMPLATE`, go to planning.
   - `user_templates`: one LLM call with template-selection prompts + `search_templates` → a template id; if a valid id, fetch the stored script and use its source as `template`; else default template.
4. **Plan:** compute `effective_fallback = fallback_query or DEFAULT_QUERY`; inject escaped fallback into `{{FALLBACK_QUERY}}` in the system prompt; gson-encode `query_fields`; set `current_time` (`yyyy-MM-dd'T'HH:mm:ss'Z'`); **fetch index mapping then a sample doc via MCP**; build the verbatim system/user prompts with `${parameters.*}` substitution; call the LLM once.
5. **Parse:** if response null/blank/"null" → substitute `${parameters.*}` into the fallback and return that; else run the output parser (extract first JSON object; on none → `DEFAULT_QUERY`). Apply `response_filter` JSONPath to the raw model envelope first.

Sample-doc values > 250 codepoints are truncated with a `[truncated]` prefix; with multiple mappings, the first is used (warn).

## Defaults / constants (verbatim)

- `DEFAULT_QUERY = {"size":10,"query":{"match_all":{}}}`
- `DEFAULT_DATETIME_FORMAT = yyyy-MM-dd'T'HH:mm:ss'Z'`
- `response_filter` default (Bedrock Converse): `$.output.message.content[0].text`
- Conversational failure mode: `dsl_query = {"query":{"match_all":{}}}`

## HTTP contract

```
POST /flow_agent
  { "question": "...", "index_name": "products",
    "query_fields": ["..."]?, "embedding_model_id": "..."? }
  → 200 { "dsl_query": { ... } }            # DSL only

POST /conversational_agent
  { "question": "...", "index_name": "products"?,   # optional → auto-discover
    "embedding_model_id": "..."? }
  → 200 { "dsl_query": { ... } }            # per OUTPUT CONTRACT
```

## Testing

- **Prompt fidelity:** assert assembled QPT system/user prompts byte-match the Java constants; assert the two `.txt` files are unmodified.
- **Substitutor:** `${parameters.x}`, `${parameters.x:-default}`, missing keys.
- **Output parser:** extract first JSON object amid markdown/prose; `match_all` fallback; `response_filter` JSONPath.
- **QPT control flow:** generation-type branching, fallback path, sample-doc truncation — with the MCP/LLM layers mocked.
- LLM-live and MCP-live calls are not asserted in CI (require Bedrock + a cluster); covered by a smoke check.

## Out of scope (YAGNI)

Search-pipeline processors (`agentic_query_translator`, `agentic_context`), persistent multi-turn memory across requests, multi-cluster MCP mode, non-Bedrock providers (env-swappable but untested).
