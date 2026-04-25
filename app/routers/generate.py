"""Generation endpoints."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from app.data.loader import get_schema
from app.parsers.text_parser import parse_text_to_constraints
from app.schemas import (
    GenerateFromTextRequest,
    GenerateFromTextResponse,
    GenerateRequest,
    GenerateResponse,
)
from app.services import generate
from app.storage import store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["generate"])


@router.post("/generate", response_model=GenerateResponse)
def post_generate(req: GenerateRequest) -> GenerateResponse:
    try:
        out = generate(
            constraints=req.constraints,
            n_rows=req.n_rows,
            strategy=req.strategy,
            random_seed=req.random_seed,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(e)) from e

    return GenerateResponse(
        generation_id=out.generation_id,
        strategy=out.strategy,  # type: ignore[arg-type]
        n_rows_requested=out.n_rows_requested,
        n_rows_returned=len(out.rows),
        constraints_applied=out.constraints_applied,
        constraints_rejected=out.constraints_rejected,
        random_seed=out.random_seed,
        rows=out.rows,
    )


@router.post("/generate/from-text", response_model=GenerateFromTextResponse)
def post_generate_from_text(req: GenerateFromTextRequest) -> GenerateFromTextResponse:
    schema = get_schema()
    parsed = parse_text_to_constraints(req.text, schema)
    if not parsed.constraints and parsed.warnings:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "no constraints could be parsed from the text",
                "warnings": parsed.warnings,
                "hint": (
                    "Try phrasings like: 'age between 40 and 60, income >50K,"
                    " occupation Exec-managerial'."
                ),
            },
        )

    try:
        out = generate(
            constraints=parsed.constraints,
            n_rows=req.n_rows,
            strategy=req.strategy,
            random_seed=req.random_seed,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return GenerateFromTextResponse(
        generation_id=out.generation_id,
        strategy=out.strategy,  # type: ignore[arg-type]
        n_rows_requested=out.n_rows_requested,
        n_rows_returned=len(out.rows),
        constraints_applied=out.constraints_applied,
        constraints_rejected=out.constraints_rejected,
        random_seed=out.random_seed,
        rows=out.rows,
        parsed_from_text=req.text,
        parser_warnings=parsed.warnings,
    )


@router.get("/generate/{generation_id}")
def get_generation(generation_id: str) -> Dict[str, Any]:
    """Retrieve a previously generated dataset by its ID."""
    record = store.get_generation(generation_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"generation_id '{generation_id}' not found",
        )
    rows: List[Dict[str, Any]] = record.rows_df.where(
        record.rows_df.notna(), None
    ).to_dict(orient="records")
    return {
        "generation_id": record.generation_id,
        "strategy": record.strategy,
        "n_rows": record.n_rows,
        "constraints_applied": record.constraints_applied,
        "constraints_rejected": record.constraints_rejected,
        "random_seed": record.random_seed,
        "rows": rows,
    }
