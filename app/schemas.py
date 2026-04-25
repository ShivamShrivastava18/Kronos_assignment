"""Pydantic request/response models for the API."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Constraint primitives
# ---------------------------------------------------------------------------


class NumericConstraint(BaseModel):
    """Range constraint on a numeric column. All bounds are inclusive."""

    model_config = ConfigDict(extra="forbid")

    gte: Optional[float] = Field(default=None, description="Greater than or equal")
    lte: Optional[float] = Field(default=None, description="Less than or equal")
    gt: Optional[float] = Field(default=None, description="Strictly greater than")
    lt: Optional[float] = Field(default=None, description="Strictly less than")
    eq: Optional[float] = Field(default=None, description="Exact numeric value")

    @field_validator("lte", "lt", "gte", "gt", "eq")
    @classmethod
    def _finite(cls, v):  # noqa: ANN001
        if v is None:
            return v
        try:
            float(v)
        except (TypeError, ValueError) as e:
            raise ValueError("must be a finite number") from e
        return float(v)


# Accepted shapes for a per-column constraint:
#   - scalar string / number / bool for equality ("income": ">50K")
#   - list of values for membership ("occupation": ["Exec-managerial", ...])
#   - dict of {gte|lte|gt|lt|eq} for numeric ranges
ConstraintValue = Union[
    str,
    int,
    float,
    bool,
    List[Union[str, int, float, bool]],
    NumericConstraint,
    Dict[str, Any],
]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


Strategy = Literal["copula", "histogram"]


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    constraints: Dict[str, ConstraintValue] = Field(default_factory=dict)
    n_rows: int = Field(default=100, ge=1, le=100_000)
    strategy: Strategy = "copula"
    random_seed: Optional[int] = Field(default=None, ge=0)


class GenerateFromTextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=2000)
    n_rows: int = Field(default=100, ge=1, le=100_000)
    strategy: Strategy = "copula"
    random_seed: Optional[int] = Field(default=None, ge=0)


class GenerateResponse(BaseModel):
    generation_id: str
    strategy: Strategy
    n_rows_requested: int
    n_rows_returned: int
    constraints_applied: Dict[str, Any]
    constraints_rejected: Dict[str, str] = Field(default_factory=dict)
    random_seed: Optional[int] = None
    rows: List[Dict[str, Any]]


class GenerateFromTextResponse(GenerateResponse):
    parsed_from_text: str
    parser_warnings: List[str] = Field(default_factory=dict)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class ValidateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generation_id: Optional[str] = None
    rows: Optional[List[Dict[str, Any]]] = None

    @field_validator("rows")
    @classmethod
    def _not_empty(cls, v):  # noqa: ANN001
        if v is not None and len(v) == 0:
            raise ValueError("rows must be non-empty if provided")
        return v


class ColumnMetric(BaseModel):
    column: str
    metric: str  # 'ks' or 'js'
    statistic: float
    p_value: Optional[float] = None


class ValidateResponse(BaseModel):
    generation_id: Optional[str] = None
    tstr_accuracy: float
    baseline_accuracy: float
    fidelity_score: float
    verdict: Literal["PASS", "FAIL"]
    tolerance: float
    ks_tests: List[ColumnMetric]
    js_divergences: List[ColumnMetric]
    n_synthetic: int
    n_real_test: int
    notes: List[str] = Field(default_factory=list)


class CompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generation_id_a: str
    generation_id_b: str


class CompareResponse(BaseModel):
    a: ValidateResponse
    b: ValidateResponse
    diff: Dict[str, float]  # key metric deltas (b - a)
    winner: Literal["A", "B", "TIE"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class DatasetInfo(BaseModel):
    columns: List[str]
    dtypes: Dict[str, str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    value_ranges: Dict[str, Dict[str, float]]
    categorical_values: Dict[str, List[str]]
    class_distribution: Dict[str, float]
    n_rows: int
    target_column: str


class DatasetSampleResponse(BaseModel):
    n: int
    rows: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class AgentChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)


class AgentChatResponse(BaseModel):
    session_id: str
    reply: str
    generation_id: Optional[str] = None
    validation_summary: Optional[Dict[str, Any]] = None
    tool_calls_made: List[str] = Field(default_factory=list)


class AgentHistoryEntry(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class AgentHistoryResponse(BaseModel):
    session_id: str
    messages: List[AgentHistoryEntry]
    tool_calls_total: int


# ---------------------------------------------------------------------------
# Health & metrics
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    dataset_loaded: bool
    n_rows: int
    strategies_available: List[str]
    groq_configured: bool
    model: str


class MetricsSummaryResponse(BaseModel):
    total_generations: int
    total_validations: int
    avg_fidelity_score: Optional[float] = None
    pass_rate: Optional[float] = None
    strategies_used: Dict[str, int]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    detail: str
    field_errors: Optional[Dict[str, str]] = None
