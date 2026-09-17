"""Pydantic API contracts."""

from pydantic import BaseModel, Field, field_validator


class ModelConfigRequest(BaseModel):
    name: str = "Baseline"
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = Field(default=0.0, ge=0, le=2)
    system_prompt: str = "Answer accurately and concisely."

    @field_validator("name", "provider", "model")
    @classmethod
    def required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()


class EvaluateRequest(BaseModel):
    dataset_path: str = "data/sample_eval_dataset.csv"
    dataset_name: str = "sample_eval_dataset.csv"
    configuration: ModelConfigRequest = Field(default_factory=ModelConfigRequest)
    run_name: str = "Evaluation run"


class CompareRequest(BaseModel):
    dataset_path: str = "data/sample_eval_dataset.csv"
    dataset_name: str = "sample_eval_dataset.csv"
    configurations: list[ModelConfigRequest] = Field(min_length=2, max_length=2)
    run_name: str = "Comparison run"


class RegressionRequest(BaseModel):
    baseline_run_id: int
    new_run_id: int
    threshold: float = Field(default=0.05, ge=0, le=1)


class GuardrailRequest(BaseModel):
    configuration: ModelConfigRequest | None = None
