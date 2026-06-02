"""Loaders for the conversational agent's verbatim prompts.

The text files in ``prompts/`` are byte-for-byte copies of the neural-search
resources ``agentic-system-prompt.txt`` and ``agentic-user-prompt.txt``. The
user prompt is a ``${parameters.*}`` template filled per request.
"""

from __future__ import annotations

from pathlib import Path

from ...tooling.substitutor import substitute

_DIR = Path(__file__).parent / "prompts"

SYSTEM_PROMPT: str = (_DIR / "agentic-system-prompt.txt").read_text(encoding="utf-8")
USER_PROMPT_TEMPLATE: str = (_DIR / "agentic-user-prompt.txt").read_text(
    encoding="utf-8"
)


def render_user_prompt(
    question: str,
    index_name: str | None = None,
    embedding_model_id: str | None = None,
) -> str:
    """Fill the verbatim user-prompt template with request parameters."""
    return substitute(
        USER_PROMPT_TEMPLATE,
        {
            "question": question,
            "index_name": index_name,
            "embedding_model_id": embedding_model_id,
        },
    )
