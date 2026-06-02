"""Verbatim QueryPlanningTool prompts.

The prompt text is a 1:1 copy of OpenSearch's
``QueryPlanningPromptTemplate.java``. To eliminate transcription error, the
assembled prompt strings are *extracted* from the Java source by
``scripts/extract_qpt_prompts.py`` and stored under ``prompt_assets/``; this
module simply loads them. Re-run that script against a newer ml-commons source
to refresh.
"""

from __future__ import annotations

from pathlib import Path

_ASSETS = Path(__file__).parent / "prompt_assets"


def _load(name: str) -> str:
    return (_ASSETS / name).read_text(encoding="utf-8")


# QueryPlanningPromptTemplate.DEFAULT_QUERY
DEFAULT_QUERY: str = '{"size":10,"query":{"match_all":{}}}'

# QueryPlanningPromptTemplate.FALLBACK_QUERY_PROMPT_PLACEHOLDER
FALLBACK_QUERY_PROMPT_PLACEHOLDER: str = "{{FALLBACK_QUERY}}"

# Assembled prompts (verbatim from the Java constants of the same name).
DEFAULT_QUERY_PLANNING_SYSTEM_PROMPT: str = _load("query_planning_system_prompt.txt")
DEFAULT_QUERY_PLANNING_USER_PROMPT: str = _load("query_planning_user_prompt.txt")
DEFAULT_TEMPLATE_SELECTION_SYSTEM_PROMPT: str = _load(
    "template_selection_system_prompt.txt"
)
DEFAULT_TEMPLATE_SELECTION_USER_PROMPT: str = _load(
    "template_selection_user_prompt.txt"
)
DEFAULT_SEARCH_TEMPLATE: str = _load("default_search_template.txt")

# The QueryPlanningTool's default tool description (DEFAULT_DESCRIPTION in Java).
DEFAULT_DESCRIPTION: str = (
    "Use this tool to generate OpenSearch Query DSL from natural language queries."
    "Provide a 'question' parameter containing the complete natural language query "
    "with all necessary context, requirements, filters, and constraints."
    "The question should be self-contained with all information needed to generate "
    "the OpenSearch DSL."
    "Provide 'index_name' to help generate more accurate queries based on the index "
    "structure."
    "Optionally provide embedding model ID to be used for neural search "
    "The tool will return a valid OpenSearch query that can be used to search your data."
)
