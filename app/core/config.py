"""Application settings loaded from environment variables."""

from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


class Settings(BaseSettings):
    app_name: str = "LLM Evaluation & Guardrails Platform"
    database_url: str = f"sqlite:///{ROOT_DIR / 'evaluation.db'}"
    openai_api_key: str | None = None
    openai_timeout_seconds: float = 45.0
    judge_model: str = "gpt-4o-mini"

    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")


settings = Settings()
