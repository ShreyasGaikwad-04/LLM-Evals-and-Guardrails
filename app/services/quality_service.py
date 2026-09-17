"""Quality metrics with explicit DeepEval and deterministic test modes."""

from dataclasses import dataclass
import re

from app.core.config import settings
from app.services.judge_service import JudgeService


@dataclass(frozen=True)
class MetricResult:
    score: float | None
    reason: str | None
    metric_name: str


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


def deterministic_score(expected: str, actual: str) -> float:
    expected_tokens = _tokens(expected)
    return round(len(expected_tokens & _tokens(actual)) / len(expected_tokens), 3) if expected_tokens else 0.0


class QualityEvaluator:
    """Runs real model-based metrics when enabled and transparent fallbacks otherwise."""

    def __init__(self, use_llm: bool | None = None) -> None:
        self.use_llm = bool(settings.openai_api_key) if use_llm is None else use_llm

    def evaluate(self, question: str, expected_answer: str, response: str, context: str | None = None) -> dict[str, MetricResult]:
        if not self.use_llm:
            return self._fallback(question, expected_answer, response, context)
        try:
            from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
            from deepeval.test_case import LLMTestCase, LLMTestCaseParams

            test_case = LLMTestCase(
                input=question,
                actual_output=response,
                expected_output=expected_answer,
                context=[context] if context else None,
            )
            correctness = GEval(
                name="Correctness",
                criteria="The answer must be factually correct, complete, and answer the question using the expected answer as a reference.",
                evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
                model=settings.judge_model,
                async_mode=False,
            )
            relevance = AnswerRelevancyMetric(model=settings.judge_model, include_reason=True, async_mode=False)
            result = {
                "correctness": MetricResult(correctness.measure(test_case), correctness.reason, "deepeval_geval_correctness"),
                "relevance": MetricResult(relevance.measure(test_case), relevance.reason, "deepeval_answer_relevancy"),
            }
            if context:
                faithfulness = FaithfulnessMetric(model=settings.judge_model, include_reason=True, async_mode=False)
                result["faithfulness"] = MetricResult(faithfulness.measure(test_case), faithfulness.reason, "deepeval_faithfulness")
            else:
                result["faithfulness"] = MetricResult(None, "No context supplied; faithfulness was not evaluated.", "not_evaluated")
            judge = JudgeService().evaluate(question, expected_answer, response, context)
            result["judge"] = MetricResult(judge.score, judge.reasoning, "llm_judge")
            return result
        except Exception as exc:
            fallback = self._fallback(question, expected_answer, response, context)
            return {name: MetricResult(value.score, f"LLM metric unavailable; deterministic fallback used: {exc}", "deterministic_fallback") for name, value in fallback.items()}

    @staticmethod
    def _fallback(question: str, expected_answer: str, response: str, context: str | None) -> dict[str, MetricResult]:
        score = deterministic_score(expected_answer, response)
        relevance = deterministic_score(question, response)
        result = {
            "correctness": MetricResult(score, "Deterministic fallback; configure OPENAI_API_KEY for DeepEval evaluation.", "deterministic_overlap"),
            "relevance": MetricResult(relevance, "Deterministic fallback; configure OPENAI_API_KEY for DeepEval evaluation.", "deterministic_overlap"),
            "faithfulness": MetricResult(deterministic_score(context, response) if context else None, "Deterministic context overlap fallback." if context else "No context supplied; faithfulness was not evaluated.", "deterministic_context_overlap" if context else "not_evaluated"),
            "judge": MetricResult(score, "Deterministic fallback; configure OPENAI_API_KEY for LLM-as-a-judge.", "deterministic_overlap"),
        }
        return result
