"""Dataset & schema routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.data.loader import get_schema, sample_rows
from app.schemas import DatasetInfo, DatasetSampleResponse

router = APIRouter(prefix="/dataset", tags=["dataset"])


@router.get("/info", response_model=DatasetInfo)
def dataset_info() -> DatasetInfo:
    try:
        s = get_schema()
    except Exception as e:  # pragma: no cover
        raise HTTPException(
            status_code=503, detail=f"dataset not available: {e}"
        ) from e
    return DatasetInfo(
        columns=s.columns,
        dtypes=s.dtypes,
        numeric_columns=s.numeric_columns,
        categorical_columns=s.categorical_columns,
        value_ranges=s.value_ranges,
        categorical_values=s.categorical_values,
        class_distribution=s.class_distribution,
        n_rows=s.n_rows,
        target_column=s.target_column,
    )


@router.get("/sample", response_model=DatasetSampleResponse)
def dataset_sample(
    n: int = Query(default=5, ge=1, le=100),
    random_seed: int | None = Query(default=None, ge=0),
) -> DatasetSampleResponse:
    rows = sample_rows(n=n, random_seed=random_seed)
    return DatasetSampleResponse(n=len(rows), rows=rows)
