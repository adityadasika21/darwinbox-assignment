"""One error shape, one place that decides status codes (SPEC 9).

Internals raise typed exceptions and never think about HTTP. This module is the
boundary where they become responses, and the only place allowed to decide a code.
A stack trace never reaches the client.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from darwinbox.execute.validator import ValidationError
from darwinbox.ingest.loader import UnreadableFileError
from darwinbox.llm.client import LLMError
from darwinbox.profile.registry import TableNotFoundError
from darwinbox.relate.graph import RelationshipNotFoundError
from darwinbox.samples import UnknownSampleError
from darwinbox.session import FileTooLargeError, SessionNotFoundError

log = logging.getLogger("darwinbox")

# exception type -> (status, code). Order matters only for subclasses.
_MAPPING: list[tuple[type[Exception], int, str]] = [
    (SessionNotFoundError, 404, "SESSION_NOT_FOUND"),
    (RelationshipNotFoundError, 404, "RELATIONSHIP_NOT_FOUND"),
    (TableNotFoundError, 404, "TABLE_NOT_FOUND"),
    (UnknownSampleError, 404, "SAMPLE_NOT_FOUND"),
    (FileTooLargeError, 413, "FILE_TOO_LARGE"),
    (UnreadableFileError, 422, "UNPARSEABLE_FILE"),
    (ValidationError, 400, "INVALID_SQL"),
    (LLMError, 502, "MODEL_UNAVAILABLE"),
    (ValueError, 400, "BAD_REQUEST"),
]

_MESSAGES: dict[str, str] = {
    "SESSION_NOT_FOUND": "No such session. It may have expired when the server restarted.",
    "RELATIONSHIP_NOT_FOUND": "No such relationship in this session.",
    "TABLE_NOT_FOUND": "No such table or column in this session.",
    "SAMPLE_NOT_FOUND": "No such sample dataset.",
    "FILE_TOO_LARGE": "That file is above the 100 MB limit.",
    "UNPARSEABLE_FILE": "That file could not be read as CSV or Excel.",
    "MODEL_UNAVAILABLE": "The model is not responding.",
}


def error_body(code: str, message: str, detail: dict | None = None) -> dict:
    return {"error": {"code": code, "message": message, "detail": detail or {}}}


def register(app: FastAPI) -> None:
    """Install the handlers that give every failure the SPEC 9 shape."""

    @app.exception_handler(RequestValidationError)
    async def _on_request_validation(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=400,
            content=error_body(
                "BAD_REQUEST", "The request body was not valid.", {"errors": exc.errors()[:5]}
            ),
        )

    @app.exception_handler(HTTPException)
    async def _on_http(request: Request, exc: HTTPException):
        code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED", 413: "FILE_TOO_LARGE"}.get(
            exc.status_code, "HTTP_ERROR"
        )
        return JSONResponse(
            status_code=exc.status_code, content=error_body(code, str(exc.detail))
        )

    # Each domain exception is registered by type. A bare `Exception` handler is not
    # enough on its own: Starlette routes that one through ServerErrorMiddleware, which
    # re-raises under TestClient, so mapped failures would never get their status code.
    for kind, status, code in _MAPPING:
        app.add_exception_handler(kind, _handler_for(status, code))

    @app.exception_handler(Exception)
    async def _on_unexpected(request: Request, exc: Exception):
        # Anything unmapped is a bug in our code, so log it fully and tell the client
        # nothing beyond the fact that it failed.
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content=error_body("INTERNAL_ERROR", "Something went wrong on the server."),
        )


def _handler_for(status: int, code: str):
    async def handle(request: Request, exc: Exception) -> JSONResponse:
        message = _MESSAGES.get(code) or str(exc)
        return JSONResponse(
            status_code=status, content=error_body(code, message, _detail(exc, code))
        )

    return handle


def _detail(exc: Exception, code: str) -> dict:
    """The offending name is useful to the client; the traceback is not."""
    text = str(exc).strip()
    return {"resource": text} if text and code.endswith("NOT_FOUND") else {}
