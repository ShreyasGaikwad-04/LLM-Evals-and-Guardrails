"""Optional structured LLM-as-a-judge evaluator."""

from pydantic import BaseModel, Field

from app.core.config import settings
from app.prompts.evaluation_prompts import JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE


class JudgeEvaluation(BaseModel):
    score: float = Field(ge=0, le=1)
    reasoning: str


class JudgeService:
    """Evaluate one answer with a configured judge model when explicitly requested."""

    def evaluate(self, question: str, expected_answer: str, response: str, context: str | None = None) -> JudgeEvaluation:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required to use the LLM judge.")
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key, timeout=settings.openai_timeout_seconds)
        completion = client.beta.chat.completions.parse(
            model=settings.judge_model,
            temperature=0,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": JUDGE_USER_TEMPLATE.format(question=question, expected_answer=expected_answer, response=response, context=context or "")},
            ],
            response_format=JudgeEvaluation,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise ValueError("Judge returned no structured result.")
        return parsed
