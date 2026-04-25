"""In-memory stores for generations, validations, and agent sessions.

Thread-safe enough for a single-worker uvicorn process. For multi-worker
deployments we'd swap these out for Redis; `docker-compose up` only starts
one worker, so this is adequate.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import pandas as pd


@dataclass
class GenerationRecord:
    generation_id: str
    strategy: str
    n_rows: int
    random_seed: Optional[int]
    constraints_applied: Dict[str, Any]
    constraints_rejected: Dict[str, str]
    rows_df: pd.DataFrame
    created_at: float = field(default_factory=time.time)


@dataclass
class ValidationRecord:
    generation_id: Optional[str]
    result: Dict[str, Any]
    verdict: str
    fidelity_score: float
    created_at: float = field(default_factory=time.time)


class MemoryStore:
    def __init__(self, max_entries: int = 200) -> None:
        self._lock = threading.RLock()
        self._generations: Dict[str, GenerationRecord] = {}
        self._validations_by_gen: Dict[str, ValidationRecord] = {}
        self._all_validations: Deque[ValidationRecord] = deque(maxlen=max_entries)
        self._gen_order: Deque[str] = deque(maxlen=max_entries)
        # agent session -> list of chat messages (openai-style)
        self._sessions: Dict[str, List[Dict[str, Any]]] = {}
        # agent session -> tool calls list
        self._session_tool_calls: Dict[str, List[Dict[str, Any]]] = {}

    # -- generations ----------------------------------------------------
    def put_generation(self, rec: GenerationRecord) -> None:
        with self._lock:
            if len(self._gen_order) == self._gen_order.maxlen and self._gen_order:
                oldest = self._gen_order[0]
                self._generations.pop(oldest, None)
            self._generations[rec.generation_id] = rec
            self._gen_order.append(rec.generation_id)

    def get_generation(self, generation_id: str) -> Optional[GenerationRecord]:
        with self._lock:
            return self._generations.get(generation_id)

    # -- validations ----------------------------------------------------
    def put_validation(self, rec: ValidationRecord) -> None:
        with self._lock:
            if rec.generation_id:
                self._validations_by_gen[rec.generation_id] = rec
            self._all_validations.append(rec)

    def get_validation(self, generation_id: str) -> Optional[ValidationRecord]:
        with self._lock:
            return self._validations_by_gen.get(generation_id)

    # -- metrics -------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            gens = list(self._generations.values())
            strategies: Dict[str, int] = {}
            for g in gens:
                strategies[g.strategy] = strategies.get(g.strategy, 0) + 1
            vals = list(self._all_validations)
            avg_fid = (
                sum(v.fidelity_score for v in vals) / len(vals) if vals else None
            )
            pass_rate = (
                sum(1 for v in vals if v.verdict == "PASS") / len(vals)
                if vals
                else None
            )
            return {
                "total_generations": len(gens),
                "total_validations": len(vals),
                "avg_fidelity_score": avg_fid,
                "pass_rate": pass_rate,
                "strategies_used": strategies,
            }

    # -- agent sessions -------------------------------------------------
    def get_session(self, session_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def set_session(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        with self._lock:
            self._sessions[session_id] = list(messages)

    def append_tool_call(self, session_id: str, call: Dict[str, Any]) -> None:
        with self._lock:
            self._session_tool_calls.setdefault(session_id, []).append(call)

    def get_tool_calls(self, session_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._session_tool_calls.get(session_id, []))

    def clear_session(self, session_id: str) -> bool:
        with self._lock:
            existed = session_id in self._sessions or session_id in self._session_tool_calls
            self._sessions.pop(session_id, None)
            self._session_tool_calls.pop(session_id, None)
            return existed


store = MemoryStore()
