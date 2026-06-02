"""Bridge QPT's single-shot model call onto a Strands model.

The OpenSearch QueryPlanningTool delegates to an inner ``MLModelTool`` that does
one raw LLM call with a system + user prompt and returns the text. Strands has
no bare "complete once" primitive, but a tool-less ``Agent`` is exactly that: it
invokes the model once and, with no tools to call, terminates on the first
``end_turn``. We build a fresh tool-less agent per call so the system prompt
(which QPT rewrites each time) is applied cleanly and no conversation state
leaks between queries.
"""

from __future__ import annotations

from typing import Any

from strands import Agent


def make_model_invoker(model: Any):
    """Return an ``invoke_model(system_prompt, user_prompt) -> str`` callable."""

    def invoke_model(system_prompt: str, user_prompt: str) -> str:
        agent = Agent(
            model=model,
            system_prompt=system_prompt,
            tools=[],
            callback_handler=None,
        )
        result = agent(user_prompt)
        return str(result)

    return invoke_model
