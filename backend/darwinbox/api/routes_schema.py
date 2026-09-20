"""Schema and relationship editing (SPEC 9)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from darwinbox.api.deps import get_session
from darwinbox.models import (
    Relationship,
    RelationshipCreate,
    RelationshipPatch,
    SchemaResponse,
)
from darwinbox.session import Session

router = APIRouter(prefix="/api/sessions/{session_id}", tags=["schema"])


@router.get("/schema", response_model=SchemaResponse)
def get_schema(session: Session = Depends(get_session)) -> SchemaResponse:
    """Tables, every discovered relationship, and the connected components."""
    return SchemaResponse(
        tables=session.tables,
        relationships=session.relationships,
        components=session.components(),
    )


@router.patch("/relationships/{relationship_id}", response_model=Relationship)
def update_relationship(
    relationship_id: str,
    patch: RelationshipPatch,
    session: Session = Depends(get_session),
) -> Relationship:
    """Confirm or reject a proposed edge. Rejected edges never reach the model."""
    return session.set_relationship_status(relationship_id, patch.status)


@router.post(
    "/relationships", status_code=status.HTTP_201_CREATED, response_model=Relationship
)
def create_relationship(
    body: RelationshipCreate, session: Session = Depends(get_session)
) -> Relationship:
    """Add an edge the statistics missed. A human said so, so it is confirmed."""
    return session.add_relationship(
        body.left_table, body.left_columns, body.right_table, body.right_columns
    )
