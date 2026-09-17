"""Load and validate evaluation datasets from CSV or JSON files."""

from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator


class EvaluationRecord(BaseModel):
    id: str
    question: str = Field(min_length=1)
    expected_answer: str = Field(min_length=1)
    context: str | None = None
    category: str | None = None

    @field_validator("question", "expected_answer")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()


def load_dataset(path: str | Path) -> list[EvaluationRecord]:
    """Load records and raise a useful error for unsupported or malformed data."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Dataset not found: {source}")
    if source.suffix.lower() == ".csv":
        frame = pd.read_csv(source)
    elif source.suffix.lower() == ".json":
        frame = pd.read_json(source)
    else:
        raise ValueError("Unsupported dataset format. Use .csv or .json.")

    required = {"id", "question", "expected_answer"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

    records: list[EvaluationRecord] = []
    errors: list[str] = []
    for index, raw_record in enumerate(frame.fillna("").to_dict(orient="records"), start=1):
        try:
            records.append(EvaluationRecord.model_validate(_normalize_record(raw_record)))
        except ValidationError as exc:
            errors.append(f"row {index}: {exc.errors()[0]['msg']}")
    if errors:
        raise ValueError("Invalid dataset records: " + "; ".join(errors))
    if not records:
        raise ValueError("Dataset contains no records.")
    return records


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(record.get("id", "")),
        "question": record.get("question", ""),
        "expected_answer": record.get("expected_answer", ""),
        "context": record.get("context") or None,
        "category": record.get("category") or None,
    }
