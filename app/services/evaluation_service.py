"""Evaluation orchestration using provider metrics and explicit fallbacks."""

from dataclasses import asdict, dataclass

from app.services.guardrail_service import validate_input
from app.services.llm_service import LLMService, ModelConfig
from app.services.quality_service import QualityEvaluator
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
    metric_details: dict | None = None
    latency: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost: float | None = None
    guardrail_passed: bool = False
    guardrail_reason: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_records(records: list[EvaluationRecord], config: ModelConfig, use_llm_metrics: bool | None = None) -> list[EvaluationRow]:
    service = LLMService()
    evaluator = QualityEvaluator(use_llm_metrics)
    rows: list[EvaluationRow] = []
    for record in records:
        row = EvaluationRow(record.question, record.expected_answer, record.category)
        input_check = validate_input(record.question)
        row.guardrail_passed = input_check.passed
        row.guardrail_reason = input_check.message
        if not input_check.passed:
            row.error = input_check.message
            rows.append(row)
            continue
        try:
            response = service.generate(record.question, config, record.context)
            row.model_response = response.content
            metrics = evaluator.evaluate(record.question, record.expected_answer, response.content, record.context)
            row.correctness_score = metrics["correctness"].score
            row.relevance_score = metrics["relevance"].score
            row.faithfulness_score = metrics["faithfulness"].score
            row.judge_score = metrics["judge"].score
            row.judge_reasoning = metrics["judge"].reason
            row.metric_details = {name: {"metric": metric.metric_name, "reason": metric.reason} for name, metric in metrics.items()}
            row.latency = response.latency_seconds
            row.prompt_tokens = response.prompt_tokens
            row.completion_tokens = response.completion_tokens
            row.total_tokens = response.total_tokens
            row.estimated_cost = response.estimated_cost
        except Exception as exc:
            row.error = str(exc)
        rows.append(row)
    return rows
