"""Model-provider factory.

Strands' default provider is Amazon Bedrock (Claude). We centralize model
construction so both agents and the QueryPlanningTool's inner model call share
one configuration. The provider is env-swappable; only Bedrock is exercised.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from .config import LLMConfig, llm as default_llm_config


@lru_cache(maxsize=None)
def _build(provider: str, model_id: str, region_name: str, temperature: float) -> Any:
    if provider == "bedrock":
        from strands.models import BedrockModel

        return BedrockModel(
            model_id=model_id,
            region_name=region_name,
            temperature=temperature,
        )
    if provider == "anthropic":
        from strands.models.anthropic import AnthropicModel

        return AnthropicModel(model_id=model_id, params={"temperature": temperature})
    if provider == "openai":
        from strands.models.openai import OpenAIModel

        return OpenAIModel(model_id=model_id, params={"temperature": temperature})
    raise ValueError(f"Unsupported LLM_PROVIDER: {provider!r}")


def build_model(config: LLMConfig | None = None) -> Any:
    """Return a Strands model instance for the given (or default) config."""
    c = config or default_llm_config
    return _build(c.provider, c.model_id, c.region_name, c.temperature)
