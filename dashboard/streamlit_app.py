"""Streamlit dashboard for the evaluation platform."""

from pathlib import Path
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from app.services.guardrail_service import run_guardrail_suite
from app.services.regression_service import compare_metrics
from app.utils.dataset_loader import load_dataset

st.set_page_config(page_title="LLM Evaluation Lab", page_icon="◎", layout="wide")
st.title("LLM Evaluation & Guardrails Platform")
st.caption("Compare response quality, safety signals, performance, and regression risk.")

page = st.sidebar.radio("Workspace", ["Overview", "Run Evaluation", "Compare Models", "Regression Testing", "Guardrail Tests"])

if page == "Overview":
    st.header("Evaluation lab")
    st.write("A local control room for repeatable LLM experiments. The sample workflow is ready to run once an OpenAI key is configured.")
    dataset = Path("data/sample_eval_dataset.csv")
    records = load_dataset(dataset)
    first, second, third = st.columns(3)
    first.metric("Sample prompts", len(records))
    second.metric("Categories", len({record.category for record in records}))
    third.metric("API status", "Configured" if os.getenv("OPENAI_API_KEY") else "Set .env")
    st.info("Use Run Evaluation to submit a model configuration. Deterministic guardrail and regression checks do not require an API key.")

elif page == "Run Evaluation":
    st.header("Run an evaluation")
    uploaded = st.file_uploader("Evaluation dataset", type=["csv", "json"])
    model = st.text_input("Model", "gpt-4o-mini")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.0, 0.1)
    system_prompt = st.text_area("System prompt", "Answer accurately and concisely.")
    if st.button("Start evaluation", type="primary"):
        path = uploaded.name if uploaded else "data/sample_eval_dataset.csv"
        if uploaded:
            Path("data").mkdir(exist_ok=True)
            Path(path).write_bytes(uploaded.getvalue())
        try:
            records = load_dataset(path)
            st.success(f"Validated {len(records)} records. Call POST /evaluate to run the provider-backed evaluation.")
        except (ValueError, FileNotFoundError) as exc:
            st.error(str(exc))

elif page == "Compare Models":
    st.header("Compare models")
    st.write("Stored run comparison is available through the FastAPI `/compare` endpoint.")
    sample = pd.DataFrame({"Model": ["Baseline", "Experimental"], "Correctness": [0.82, 0.76], "Relevance": [0.88, 0.84]})
    st.plotly_chart(px.bar(sample, x="Model", y=["Correctness", "Relevance"], barmode="group"), use_container_width=True)

elif page == "Regression Testing":
    st.header("Regression testing")
    baseline = st.number_input("Baseline correctness", 0.0, 1.0, 0.82, 0.01)
    new = st.number_input("New correctness", 0.0, 1.0, 0.76, 0.01)
    threshold = st.slider("Failure threshold", 0.0, 1.0, 0.05, 0.01)
    result = compare_metrics({"correctness": baseline}, {"correctness": new}, threshold)[0]
    getattr(st, {"PASS": "success", "WARNING": "warning", "FAIL": "error"}[result.status])(f"{result.status}: change {result.change:+.2f}")

else:
    st.header("Guardrail test suite")
    if st.button("Run guardrail suite", type="primary"):
        results = run_guardrail_suite()
        st.dataframe(pd.DataFrame([result.__dict__ for result in results]), use_container_width=True)
