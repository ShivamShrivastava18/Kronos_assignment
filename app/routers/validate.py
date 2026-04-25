"""Validation endpoints."""

from __future__ import annotations

import logging

import pandas as pd
from fastapi import APIRouter, HTTPException

from app.schemas import (
    CompareRequest,
    CompareResponse,
    ColumnMetric,
    ValidateRequest,
    ValidateResponse,
)
from app.services import _validation_to_dict, validate_rows
from app.storage import store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/validate", tags=["validate"])


def _to_response(payload: dict) -> ValidateResponse:
    return ValidateResponse(
        generation_id=payload.get("generation_id"),
        tstr_accuracy=payload["tstr_accuracy"],
        baseline_accuracy=payload["baseline_accuracy"],
        fidelity_score=payload["fidelity_score"],
        verdict=payload["verdict"],
        tolerance=payload["tolerance"],
        ks_tests=[ColumnMetric(**m) for m in payload["ks_tests"]],
        js_divergences=[ColumnMetric(**m) for m in payload["js_divergences"]],
        n_synthetic=payload["n_synthetic"],
        n_real_test=payload["n_real_test"],
        notes=payload.get("notes", []),
    )


@router.post("", response_model=ValidateResponse)
def post_validate(req: ValidateRequest) -> ValidateResponse:
    if req.generation_id is None and req.rows is None:
        raise HTTPException(
            status_code=400,
            detail="Provide either `generation_id` or `rows`.",
        )
    if req.generation_id and req.rows:
        raise HTTPException(
            status_code=400,
            detail="Provide only one of `generation_id` or `rows`, not both.",
        )

    if req.generation_id:
        record = store.get_generation(req.generation_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail=f"generation_id '{req.generation_id}' not found",
            )
        rows_df = record.rows_df
        gid = req.generation_id
        constraints = record.constraints_applied
    else:
        try:
            rows_df = pd.DataFrame(req.rows)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid rows: {e}") from e
        if rows_df.empty:
            raise HTTPException(status_code=400, detail="rows is empty")
        gid = None
        constraints = None

    try:
        result, _record = validate_rows(rows_df, generation_id=gid, constraints=constraints)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    payload = _validation_to_dict(result, gid)
    return _to_response(payload)


@router.get("/{generation_id}", response_model=ValidateResponse)
def get_validation(generation_id: str) -> ValidateResponse:
    rec = store.get_validation(generation_id)
    if rec is None:
        raise HTTPException(
            status_code=404,
            detail=f"no cached validation for generation_id '{generation_id}'",
        )
    return _to_response(rec.result)


@router.post("/compare", response_model=CompareResponse)
def post_validate_compare(req: CompareRequest) -> CompareResponse:
    rec_a = store.get_validation(req.generation_id_a)
    rec_b = store.get_validation(req.generation_id_b)
    if rec_a is None or rec_b is None:
        # Try to run validation on the fly if only the generation exists.
        missing: list[str] = []
        if rec_a is None:
            gen_a = store.get_generation(req.generation_id_a)
            if gen_a is None:
                missing.append(req.generation_id_a)
            else:
                result, rec_a = validate_rows(
                    gen_a.rows_df, generation_id=req.generation_id_a,
                    constraints=gen_a.constraints_applied,
                )
        if rec_b is None:
            gen_b = store.get_generation(req.generation_id_b)
            if gen_b is None:
                missing.append(req.generation_id_b)
            else:
                result, rec_b = validate_rows(
                    gen_b.rows_df, generation_id=req.generation_id_b,
                    constraints=gen_b.constraints_applied,
                )
        if missing:
            raise HTTPException(
                status_code=404,
                detail=f"unknown generation_ids: {missing}",
            )

    assert rec_a is not None and rec_b is not None
    a = _to_response(rec_a.result)
    b = _to_response(rec_b.result)
    diff = {
        "tstr_accuracy": round(b.tstr_accuracy - a.tstr_accuracy, 6),
        "fidelity_score": round(b.fidelity_score - a.fidelity_score, 6),
    }
    if b.fidelity_score > a.fidelity_score + 1e-4:
        winner: str = "B"
    elif a.fidelity_score > b.fidelity_score + 1e-4:
        winner = "A"
    else:
        winner = "TIE"
    return CompareResponse(a=a, b=b, diff=diff, winner=winner)  # type: ignore[arg-type]
