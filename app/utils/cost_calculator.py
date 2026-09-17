"""Centralized, updateable model pricing and cost calculations."""

from decimal import Decimal

# USD per 1M tokens. Pricing changes over time; update this table as providers change prices.
MODEL_PRICING: dict[str, dict[str, Decimal]] = {
    "gpt-4o-mini": {"input": Decimal("0.15"), "output": Decimal("0.60")},
    "gpt-4o": {"input": Decimal("2.50"), "output": Decimal("10.00")},
}


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        return None
    cost = (
        Decimal(prompt_tokens) * pricing["input"]
        + Decimal(completion_tokens) * pricing["output"]
    ) / Decimal(1_000_000)
    return float(cost)
