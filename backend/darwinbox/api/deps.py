"""Shared dependencies. Routes get a Session; they never touch the store directly."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Depends, Request

from darwinbox.session import Session, SessionStore


def get_store(request: Request) -> SessionStore:
    return request.app.state.store


def get_session(
    session_id: str, store: SessionStore = Depends(get_store)
) -> Iterator[Session]:
    """Resolve the path's session_id, raising SessionNotFoundError for a bad one.

    Checked out for the duration of the request and released afterwards, so the store
    never evicts a session while one of its own requests is still running -- doing so
    closed the DuckDB connection mid-ingest and produced a 500 from a valid upload.
    """
    with store.acquire(session_id) as session:
        yield session
