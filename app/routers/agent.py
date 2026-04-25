"""LLM agent endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from app.agent.groq_agent import get_agent
from app.schemas import (
    AgentChatRequest,
    AgentChatResponse,
    AgentHistoryEntry,
    AgentHistoryResponse,
)
from app.storage import store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/chat", response_model=AgentChatResponse)
def post_agent_chat(req: AgentChatRequest) -> AgentChatResponse:
    try:
        agent = get_agent()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    try:
        reply = agent.chat(req.session_id, req.message)
    except RuntimeError as e:
        # Typically: missing API key.
        raise HTTPException(
            status_code=503,
            detail=str(e),
        ) from e
    except Exception as e:  # noqa: BLE001
        logger.exception("agent chat failed")
        err_msg = str(e)
        if "429" in err_msg or "rate_limit" in err_msg.lower():
            raise HTTPException(
                status_code=429,
                detail="Groq API rate limit reached. Wait a few minutes and try again, "
                       "or upgrade your Groq plan at https://console.groq.com/settings/billing",
            ) from e
        raise HTTPException(status_code=502, detail=f"agent error: {e}") from e

    return AgentChatResponse(
        session_id=req.session_id,
        reply=reply.reply,
        generation_id=reply.generation_id,
        validation_summary=reply.validation_summary,
        tool_calls_made=reply.tool_calls_made,
    )


@router.get("/history/{session_id}", response_model=AgentHistoryResponse)
def get_agent_history(session_id: str) -> AgentHistoryResponse:
    messages = store.get_session(session_id)
    if not messages:
        raise HTTPException(status_code=404, detail="session not found")
    tool_calls = store.get_tool_calls(session_id)
    entries = [
        AgentHistoryEntry(
            role=m.get("role", "unknown"),
            content=m.get("content"),
            tool_calls=m.get("tool_calls"),
            tool_call_id=m.get("tool_call_id"),
            name=m.get("name"),
        )
        for m in messages
    ]
    return AgentHistoryResponse(
        session_id=session_id,
        messages=entries,
        tool_calls_total=len(tool_calls),
    )


@router.delete("/session/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_agent_session(session_id: str):
    existed = store.clear_session(session_id)
    if not existed:
        raise HTTPException(status_code=404, detail="session not found")
    return None
