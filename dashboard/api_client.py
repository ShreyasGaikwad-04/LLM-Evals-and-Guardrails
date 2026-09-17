"""Small HTTP client used by the Streamlit dashboard."""

import os
from typing import Any

import httpx

BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")


class BackendUnavailable(RuntimeError):
    pass


def _request(method: str, path: str, **kwargs: Any) -> Any:
    try:
        response = httpx.request(method, f"{BASE_URL}{path}", timeout=120, **kwargs)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        raise RuntimeError(f"Backend request failed ({exc.response.status_code}): {detail}") from exc
    except httpx.RequestError as exc:
        raise BackendUnavailable(f"Backend is unavailable. Start FastAPI with: uvicorn app.main:app --reload ({exc})") from exc


def health_check() -> dict:
    return _request("GET", "/health")


def run_evaluation(payload: dict) -> dict:
    return _request("POST", "/evaluate", json=payload)


def compare_models(payload: dict) -> dict:
    return _request("POST", "/compare", json=payload)


def get_runs() -> list[dict]:
    return _request("GET", "/runs")


def get_run(run_id: int) -> dict:
    return _request("GET", f"/runs/{run_id}")


def get_summary(run_id: int) -> dict:
    return _request("GET", f"/metrics/summary/{run_id}")


def run_regression(payload: dict) -> list[dict]:
    return _request("POST", "/regression", json=payload)


def run_guardrail_suite(payload: dict | None = None) -> list[dict]:
    return _request("POST", "/guardrails/test", json=payload or {})
