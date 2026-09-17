"""FastAPI application entry point."""

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import settings
from app.core.logging_config import configure_logging
from app.db.database import init_db

configure_logging()
init_db()
app = FastAPI(title=settings.app_name, version="0.1.0")
app.include_router(router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}
