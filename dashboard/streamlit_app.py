"""Streamlit dashboard backed by the FastAPI service."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.api_client import (
    BackendUnavailable,
    compare_models,
    get_run,
    get_runs,
    get_summary,
    health_check,
    run_evaluation,
    run_guardrail_suite,
    run_regression,
)

st.set_page_config(page_title="LLM Evaluation Lab", page_icon="◎", layout="wide")
st.title("LLM Evaluation & Guardrails Platform")
st.caption("Compare quality, safety signals, performance, cost, and regression risk.")


def show_backend_error(exc: Exception) -> None:
    st.error(str(exc))


def config_form(prefix: str, default_name: str, default_temperature: float) -> dict:
    return {
        "name": st.text_input("Configuration name", default_name, key=f"{prefix}_name"),
        "provider": st.text_input("Provider", "openai", key=f"{prefix}_provider"),
        "model": st.text_input("Model", "gpt-4o-mini", key=f"{prefix}_model"),
        "temperature": st.slider("Temperature", 0.0, 2.0, default_temperature, 0.1, key=f"{prefix}_temperature"),
        "system_prompt": st.text_area("System prompt", "Answer accurately and concisely.", key=f"{prefix}_system"),
    }


def display_summary(summary: dict) -> None:
    columns = st.columns(4)
    columns[0].metric("Successful", summary.get("successful_items", 0))
    columns[1].metric("Correctness", _format_score(summary.get("avg_correctness")))
    columns[2].metric("Relevance", _format_score(summary.get("avg_relevance")))
    columns[3].metric("Guardrail pass", _format_score(summary.get("guardrail_pass_rate")))
    st.caption(f"Latency avg: {_format_score(summary.get('avg_latency'))} s | Total tokens: {summary.get('total_tokens', 0)} | Cost: ${summary.get('total_estimated_cost', 0):.6f}")


def _format_score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


page = st.sidebar.radio("Workspace", ["Overview", "Run Evaluation", "Compare Models", "Evaluation Details", "Regression Testing", "Guardrail Tests"])

try:
    if page == "Overview":
        st.header("Evaluation lab")
        health = health_check()
        runs = get_runs()
        first, second, third = st.columns(3)
        first.metric("Stored runs", len(runs))
        second.metric("Backend", health.get("status", "unknown").title())
        third.metric("Latest run", runs[0]["name"] if runs else "None")
        if runs:
            summary = get_summary(runs[0]["id"])
            display_summary(summary)
        else:
            st.info("No evaluation runs yet. Start with the sample dataset.")

    elif page == "Run Evaluation":
        st.header("Run evaluation")
        uploaded = st.file_uploader("Upload CSV or JSON dataset", type=["csv", "json"])
        use_second = st.checkbox("Compare a second configuration")
        config_a = config_form("config_a", "Baseline", 0.0)
        config_b = config_form("config_b", "Experimental", 0.5) if use_second else None
        if st.button("Run evaluation", type="primary"):
            path = "data/sample_eval_dataset.csv"
            dataset_name = "sample_eval_dataset.csv"
            if uploaded:
                target = Path("data") / uploaded.name
                target.write_bytes(uploaded.getvalue())
                path, dataset_name = str(target), uploaded.name
            payload = {"dataset_path": path, "dataset_name": dataset_name, "configuration": config_a, "run_name": f"{config_a['name']} evaluation"}
            with st.spinner("Running evaluation through FastAPI..."):
                result = compare_models({"dataset_path": path, "dataset_name": dataset_name, "configurations": [config_a, config_b], "run_name": "Model comparison"}) if config_b else run_evaluation(payload)
            st.success(f"Completed run(s): {result.get('run_id') or ', '.join(str(item['run_id']) for item in result.get('runs', []))}")
            ids = [result["run_id"]] if "run_id" in result else [item["run_id"] for item in result["runs"]]
            for run_id in ids:
                st.subheader(f"Run {run_id}")
                display_summary(get_summary(run_id))

    elif page == "Compare Models":
        st.header("Compare stored runs")
        runs = get_runs()
        if len(runs) < 2:
            st.info("Run at least two evaluations to compare persisted results.")
        else:
            labels = {f"{run['name']} (#{run['id']})": run["id"] for run in runs}
            selected = st.multiselect("Runs", list(labels), default=list(labels)[:2], max_selections=2)
            if len(selected) == 2:
                frames = []
                for label in selected:
                    run_id = labels[label]
                    summary = get_summary(run_id)
                    frames.append({"Run": label, "Correctness": summary.get("avg_correctness"), "Relevance": summary.get("avg_relevance"), "Faithfulness": summary.get("avg_faithfulness"), "Judge": summary.get("avg_judge_score")})
                frame = pd.DataFrame(frames).melt("Run", var_name="Metric", value_name="Score").dropna()
                st.plotly_chart(px.bar(frame, x="Metric", y="Score", color="Run", barmode="group", range_y=[0, 1]), use_container_width=True)
                summaries = pd.DataFrame([get_summary(labels[label]) | {"Run": label} for label in selected])
                st.dataframe(summaries, use_container_width=True)
                for column, title in (("avg_latency", "Average latency (seconds)"), ("total_tokens", "Total tokens"), ("total_estimated_cost", "Total estimated cost")):
                    chart = summaries[["Run", column]].rename(columns={column: "Value"}).dropna()
                    st.plotly_chart(px.bar(chart, x="Run", y="Value", title=title), use_container_width=True)

    elif page == "Evaluation Details":
        st.header("Evaluation details")
        runs = get_runs()
        if runs:
            labels = {f"{run['name']} (#{run['id']})": run["id"] for run in runs}
            run_id = labels[st.selectbox("Run", list(labels))]
            details = get_run(run_id)
            rows = pd.DataFrame(details["results"])
            if not rows.empty:
                category = st.multiselect("Category", sorted(rows["category"].dropna().unique()))
                low_score = st.checkbox("Low quality only")
                failed = st.checkbox("Failed guardrail or error")
                if category:
                    rows = rows[rows["category"].isin(category)]
                if low_score:
                    rows = rows[(rows["correctness_score"].fillna(0) < 0.5) | (rows["relevance_score"].fillna(0) < 0.5)]
                if failed:
                    rows = rows[(~rows["guardrail_passed"]) | rows["error"].notna()]
                st.dataframe(rows, use_container_width=True)
        else:
            st.info("No stored runs.")

    elif page == "Regression Testing":
        st.header("Regression testing")
        runs = get_runs()
        if len(runs) >= 2:
            labels = {f"{run['name']} (#{run['id']})": run["id"] for run in runs}
            baseline_label = st.selectbox("Baseline", list(labels), key="baseline")
            new_label = st.selectbox("New run", list(labels), index=min(1, len(labels) - 1), key="new")
            threshold = st.slider("Quality failure threshold", 0.0, 1.0, 0.05, 0.01)
            if st.button("Compare regression"):
                results = run_regression({"baseline_run_id": labels[baseline_label], "new_run_id": labels[new_label], "threshold": threshold})
                st.dataframe(pd.DataFrame(results), use_container_width=True)
        else:
            st.info("Run at least two evaluations first.")

    else:
        st.header("Guardrail tests")
        if st.button("Run guardrail suite", type="primary"):
            results = run_guardrail_suite()
            st.dataframe(pd.DataFrame(results), use_container_width=True)

except (BackendUnavailable, RuntimeError, KeyError, ValueError) as exc:
    show_backend_error(exc)
