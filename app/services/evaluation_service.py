"""Evaluation orchestration and simple explainable quality metrics."""

import re
from dataclasses import asdict, dataclass

from app.services.guardrail_service import validate_input
from app.services.llm_service import LLMService, ModelConfig
from app.utils.dataset_loader import EvaluationRecord


@dataclass
class EvaluationRow:
    question: str
    expected_answer: str
    category: str | None
    model_response: str = ""
    correctness_score: float | None = None
    relevance_score: float | None = None
    faithfulness_score: float | None = None
    judge_score: float | None = None
    judge_reasoning: str | None = None
    latency: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost: float | None = None
    guardrail_passed: bool = False
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


def lexical_score(expected: str, actual: str) -> float:
    expected_tokens = _tokens(expected)
    if not expected_tokens:
        return 0.0
    return round(len(expected_tokens & _tokens(actual)) / len(expected_tokens), 3)


def relevance_score(question: str, actual: str) -> float:
    return lexical_score(question, actual)


def evaluate_records(records: list[EvaluationRecord], config: ModelConfig) -> list[EvaluationRow]:
    service = LLMService()
    rows: list[EvaluationRow] = []
    for record in records:
        row = EvaluationRow(record.question, record.expected_answer, record.category)
        input_check = validate_input(record.question)
        if not input_check.passed:
            row.error = input_check.message
            rows.append(row)
            continue
        try:
            response = service.generate(record.question, config, record.context)
            row.model_response = response.text
            row.correctness_score = lexical_score(record.expected_answer, response.text)
            row.relevance_score = relevance_score(record.question, response.text)
            row.faithfulness_score = lexical_score(record.context, response.text) if record.context else None
            row.judge_score = row.correctness_score
            row.judge_reasoning = "Deterministic lexical baseline; replace with an LLM judge when configured."
            row.latency = response.latency_seconds
            row.prompt_tokens = response.prompt_tokens
            row.completion_tokens = response.completion_tokens
            row.total_tokens = response.total_tokens
            row.estimated_cost = response.estimated_cost
            row.guardrail_passed = True
        except Exception as exc:
            row.error = str(exc)
        rows.append(row)
    return rows
