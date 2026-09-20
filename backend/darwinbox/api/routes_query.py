"""Question answering over SSE (SPEC 9).

Trace events stream as they happen, and the final event carries the QueryResult.
The planner is a blocking generator -- it calls a local model and DuckDB -- so it is
pumped on a worker thread and the event loop stays free.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import anyio
from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from darwinbox.api.deps import get_session
from darwinbox.api.errors import error_body
from darwinbox.models import QueryResult, QuestionRequest, TraceEvent
from darwinbox.session import Session

log = logging.getLogger("darwinbox")

router = APIRouter(prefix="/api/sessions/{session_id}", tags=["queries"])

_DONE = object()


@router.post("/queries")
async def ask(
    body: QuestionRequest, session: Session = Depends(get_session)
) -> EventSourceResponse:
    """Stream TraceEvents, then one final `result` event carrying the QueryResult."""
    question = body.question.strip()
    if not question:
        raise ValueError("question must not be empty")

    return EventSourceResponse(_stream(session, question))


async def _stream(session: Session, question: str) -> AsyncIterator[dict]:
    iterator = session.ask(question)

    def pump() -> object:
        return next(iterator, _DONE)

    while True:
        try:
            item = await anyio.to_thread.run_sync(pump)
        except Exception as exc:  # noqa: BLE001 - the stream owns its own error shape
            # Status codes are already sent, so a mid-stream failure has to be
            # reported inside the stream rather than by the exception handler.
            log.exception("query stream failed")
            yield {
                "event": "error",
                "data": json.dumps(
                    error_body("QUERY_FAILED", _safe_message(exc))
                ),
            }
            return

        if item is _DONE:
            return
        if isinstance(item, TraceEvent):
            yield {"event": "trace", "data": item.model_dump_json()}
        elif isinstance(item, QueryResult):
            yield {"event": "result", "data": item.model_dump_json()}


def _safe_message(exc: Exception) -> str:
    """Never leak a traceback; known failures keep their own wording."""
    from darwinbox.llm.client import LLMError, unavailable_message

    if isinstance(exc, LLMError):
        return unavailable_message()
    return "Something went wrong while answering that question."
