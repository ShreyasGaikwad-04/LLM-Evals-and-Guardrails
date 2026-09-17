"""Provider-neutral LLM client with normalized usage and timing data."""

import time
from dataclasses import dataclass

from app.core.config import settings
from app.utils.cost_calculator import calculate_cost


@dataclass(frozen=True)
class ModelConfig:
    name: str
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    system_prompt: str = "Answer accurately and concisely."


@dataclass(frozen=True)
class LLMResponse:
    text: str
    latency_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    estimated_cost: float | None


class LLMService:
    def generate(self, question: str, config: ModelConfig, context: str | None = None) -> LLMResponse:
        if config.provider != "openai":
            raise ValueError(f"Unsupported provider: {config.provider}")
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required to run an LLM evaluation.")

        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key, timeout=settings.openai_timeout_seconds)
        user_content = question if not context else f"Context:\n{context}\n\nQuestion:\n{question}"
        started = time.perf_counter()
        response = client.chat.completions.create(
            model=config.model,
            temperature=config.temperature,
            messages=[
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        latency = time.perf_counter() - started
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else None
        completion_tokens = usage.completion_tokens if usage else None
        total_tokens = usage.total_tokens if usage else None
        cost = calculate_cost(config.model, prompt_tokens or 0, completion_tokens or 0)
        return LLMResponse(response.choices[0].message.content or "", latency, prompt_tokens, completion_tokens, total_tokens, cost)
