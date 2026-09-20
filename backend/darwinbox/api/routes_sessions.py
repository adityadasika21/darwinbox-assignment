"""Session lifecycle and file upload (SPEC 9).

Routes parse, delegate to the Session aggregate, and serialise. No business logic.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile, status

from darwinbox import samples as sample_sets
from darwinbox.api.deps import get_session, get_store
from darwinbox.models import SampleRequest, SessionCreated, UploadResponse
from darwinbox.session import Session, SessionStore

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("", status_code=status.HTTP_201_CREATED, response_model=SessionCreated)
def create_session(store: SessionStore = Depends(get_store)) -> SessionCreated:
    """Start a session. It owns a DuckDB connection until it is deleted."""
    return SessionCreated(session_id=store.create().id)


@router.post(
    "/{session_id}/files",
    status_code=status.HTTP_201_CREATED,
    response_model=UploadResponse,
)
async def upload_files(
    files: list[UploadFile] = File(...),
    session: Session = Depends(get_session),
) -> UploadResponse:
    """Ingest one or more CSV/Excel files and re-discover relationships."""
    payload = [(f.filename or "upload", await f.read()) for f in files]
    outcome = session.add_files(payload)
    return UploadResponse(tables=outcome.tables, warnings=outcome.warnings)


@router.post(
    "/{session_id}/samples",
    status_code=status.HTTP_201_CREATED,
    response_model=UploadResponse,
)
def load_sample(
    body: SampleRequest, session: Session = Depends(get_session)
) -> UploadResponse:
    """Load a bundled sample set through the ordinary ingestion path."""
    chosen = sample_sets.get(body.name)
    outcome = session.add_files(sample_sets.read(chosen))
    return UploadResponse(tables=outcome.tables, warnings=outcome.warnings)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(session_id: str, store: SessionStore = Depends(get_store)) -> None:
    store.delete(session_id)
