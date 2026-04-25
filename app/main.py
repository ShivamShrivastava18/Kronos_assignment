"""FastAPI application factory and startup lifecycle."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.routers import agent, dataset, generate, health, validate

settings = get_settings()

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN001
    """Pre-warm the generator on startup to avoid a slow first request."""
    logger.info("Starting up %s %s", settings.app_name, settings.app_version)
    try:
        from app.data.loader import load_dataset  # ensure dataset downloaded

        df = load_dataset()
        logger.info("Dataset loaded: %d rows", len(df))
    except Exception as exc:  # pragma: no cover
        logger.warning("Dataset pre-load failed (will retry on first request): %s", exc)

    try:
        from app.services import warm_up

        warm_up()
        logger.info("Generators warmed up.")
    except Exception as exc:  # pragma: no cover
        logger.warning("Generator warm-up failed: %s", exc)

    yield
    logger.info("Shutting down.")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Conditional synthetic tabular data generator for the UCI Adult Income dataset."
        " Includes a Gaussian Copula engine, a histogram fallback, a realism "
        "validator (TSTR / KS / JS), and a Groq-powered LLM agent."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "internal server error", "error": str(exc)},
    )


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(health.router)
app.include_router(dataset.router)
app.include_router(generate.router)
app.include_router(validate.router)
app.include_router(agent.router)


@app.get("/", include_in_schema=False)
async def root() -> Any:
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
    }
