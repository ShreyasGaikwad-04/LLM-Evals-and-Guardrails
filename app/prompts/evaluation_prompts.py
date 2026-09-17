"""Prompts used by the optional LLM-as-a-judge evaluator."""

JUDGE_SYSTEM_PROMPT = """You are a concise LLM evaluation judge. Score the candidate answer from 0.0 to 1.0.
Consider factual correctness, completeness, instruction following, and whether claims are supported by context.
Return only the requested structured result and a brief explanation."""

JUDGE_USER_TEMPLATE = """Question:
{question}

Expected answer:
{expected_answer}

Candidate answer:
{response}

Context (may be empty):
{context}
"""
