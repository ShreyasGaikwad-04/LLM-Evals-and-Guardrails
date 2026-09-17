"""Deterministic regression comparison logic."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RegressionMetric:
    metric: str
    baseline: float
    new: float
    change: float
    status: str


def compare_metrics(
    baseline: dict[str, float],
    new: dict[str, float],
    threshold: float = 0.05,
) -> list[RegressionMetric]:
    results: list[RegressionMetric] = []
    for metric in sorted(set(baseline) & set(new)):
        change = round(new[metric] - baseline[metric], 4)
        if change < -threshold:
            status = "FAIL"
        elif change < 0:
            status = "WARNING"
        else:
            status = "PASS"
        results.append(RegressionMetric(metric, baseline[metric], new[metric], change, status))
    return results
