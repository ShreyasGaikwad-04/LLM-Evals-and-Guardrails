"""FastAPI routes for evaluation operations."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.schemas import CompareRequest, EvaluateRequest, GuardrailRequest, RegressionRequest
from app.db.database import get_db
from app.db.models import EvaluationResult, EvaluationRun
from app.services.evaluation_service import evaluate_records
from app.services.guardrail_service import run_guardrail_suite
from app.services.llm_service import ModelConfig
from app.services.regression_service import compare_metrics
from app.utils.dataset_loader import load_dataset

router = APIRouter()


def _run(request: EvaluateRequest, db: Session) -> EvaluationRun:
    try:
        records = load_dataset(request.dataset_path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    config = ModelConfig(**request.configuration.model_dump())
    rows = evaluate_records(records, config)
    run = EvaluationRun(name=request.run_name, model_configuration=request.configuration.model_dump(), dataset_name=request.dataset_name)
    run.results = [EvaluationResult(**row.to_dict()) for row in rows]
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


@router.post("/evaluate")
def evaluate(request: EvaluateRequest, db: Session = Depends(get_db)) -> dict:
    run = _run(request, db)
    return {"run_id": run.id, "status": run.status, "result_count": len(run.results)}


@router.post("/compare")
def compare(request: CompareRequest, db: Session = Depends(get_db)) -> dict:
    runs = []
    for configuration in request.configurations:
        run = _run(EvaluateRequest(dataset_path=request.dataset_path, dataset_name=request.dataset_name, configuration=configuration, run_name=f"{request.run_name} - {configuration.name}"), db)
        runs.append({"run_id": run.id, "configuration": configuration.name, "result_count": len(run.results)})
    return {"runs": runs}


@router.get("/runs")
def list_runs(db: Session = Depends(get_db)) -> list[dict]:
    return [{"id": run.id, "name": run.name, "created_at": run.created_at, "status": run.status} for run in db.scalars(select(EvaluationRun).order_by(EvaluationRun.created_at.desc())).all()]


@router.get("/runs/{run_id}")
def get_run(run_id: int, db: Session = Depends(get_db)) -> dict:
    run = db.get(EvaluationRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return {"id": run.id, "name": run.name, "configuration": run.model_configuration, "dataset_name": run.dataset_name, "results": [_result_dict(result) for result in run.results]}


@router.post("/regression")
def regression(request: RegressionRequest, db: Session = Depends(get_db)) -> list[dict]:
    baseline = db.get(EvaluationRun, request.baseline_run_id)
    new = db.get(EvaluationRun, request.new_run_id)
    if not baseline or not new:
        raise HTTPException(status_code=404, detail="Both evaluation runs are required")
    baseline_summary = _averages(baseline)
    new_summary = _averages(new)
    baseline_scores = _regression_values(baseline_summary)
    new_scores = _regression_values(new_summary)
    return [metric.__dict__ for metric in compare_metrics(baseline_scores, new_scores, request.threshold)]


@router.get("/metrics/summary/{run_id}")
def summary(run_id: int, db: Session = Depends(get_db)) -> dict:
    run = db.get(EvaluationRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return _averages(run)


@router.post("/guardrails/test")
def guardrails(request: GuardrailRequest | None = None) -> list[dict]:
    config = ModelConfig(**request.configuration.model_dump()) if request and request.configuration else None
    return [result.to_dict() for result in run_guardrail_suite(config)]


def _averages(run: EvaluationRun) -> dict[str, float | int | None]:
    def average(field: str) -> float | None:
        values = [getattr(row, field) for row in run.results if getattr(row, field) is not None]
        return round(sum(values) / len(values), 4) if values else None

    return {
        "total_items": len(run.results),
        "successful_items": sum(result.error is None for result in run.results),
        "failed_items": sum(result.error is not None for result in run.results),
        "avg_correctness": average("correctness_score"),
        "avg_relevance": average("relevance_score"),
        "avg_faithfulness": average("faithfulness_score"),
        "avg_judge_score": average("judge_score"),
        "avg_latency": average("latency"),
        "total_prompt_tokens": sum(result.prompt_tokens or 0 for result in run.results),
        "total_completion_tokens": sum(result.completion_tokens or 0 for result in run.results),
        "total_tokens": sum(result.total_tokens or 0 for result in run.results),
        "total_estimated_cost": round(sum(result.estimated_cost or 0 for result in run.results), 8),
        "guardrail_pass_rate": round(sum(result.guardrail_passed for result in run.results) / len(run.results), 4) if run.results else None,
    }


def _result_dict(result: EvaluationResult) -> dict:
    return {column.name: getattr(result, column.name) for column in EvaluationResult.__table__.columns}


def _regression_values(summary: dict[str, float | int | None]) -> dict[str, float]:
    return {
        key: float(summary[key])
        for key in ("avg_correctness", "avg_relevance", "avg_faithfulness", "avg_latency", "total_estimated_cost", "guardrail_pass_rate")
        if summary.get(key) is not None
    }
