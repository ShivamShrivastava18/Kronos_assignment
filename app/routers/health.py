"""Health, liveness, and aggregate metrics endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app.config import get_settings
from app.generators import AVAILABLE_STRATEGIES
from app.schemas import HealthResponse, MetricsSummaryResponse
from app.storage import store

router = APIRouter(tags=["meta"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()

    dataset_loaded = False
    n_rows = 0
    try:
        from app.data.loader import load_dataset  # deferred to avoid import-time cost

        df = load_dataset()
        dataset_loaded = True
        n_rows = len(df)
    except Exception:  # pragma: no cover
        pass

    return HealthResponse(
        status="ok" if dataset_loaded else "degraded",
        version=settings.app_version,
        dataset_loaded=dataset_loaded,
        n_rows=n_rows,
        strategies_available=list(AVAILABLE_STRATEGIES.keys()),
        groq_configured=bool(settings.groq_api_key),
        model=settings.groq_model,
    )


@router.get("/metrics/summary", response_model=MetricsSummaryResponse)
def metrics_summary() -> MetricsSummaryResponse:
    snap = store.snapshot()
    return MetricsSummaryResponse(
        total_generations=snap["total_generations"],
        total_validations=snap["total_validations"],
        avg_fidelity_score=snap["avg_fidelity_score"],
        pass_rate=snap["pass_rate"],
        strategies_used=snap["strategies_used"],
    )
