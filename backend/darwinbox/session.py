"""The Session aggregate root.

A Session owns its DuckDB connection, its registry and its relationship graph. All
business logic for upload / schema / query lives here or below; the API layer only
parses, delegates and serialises.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field

import duckdb

from darwinbox.ingest.blocks import detect_blocks
from darwinbox.ingest.loader import UnreadableFileError, load_file
from darwinbox.llm.planner import answer
from darwinbox.llm.router import TableRouter
from darwinbox.models import QueryResult, Relationship, TableProfile, TraceEvent
from darwinbox.profile.registry import Registry, TableNotFoundError
from darwinbox.relate.candidates import (
    ValueSets,
    adjudicate,
    build_relationship_id,
    containment,
    discover,
)
from darwinbox.relate.candidates import name_similarity as _name_similarity
from darwinbox.relate.graph import RelationshipGraph, RelationshipNotFoundError

# Ingestion costs roughly 25x the file size in memory -- a 5.9 MB CSV of 100k rows
# peaks around 300 MB once pandas and DuckDB have both held a copy. On a machine with
# real RAM 100 MB is a reasonable ceiling; on a 512 MB instance it is a way to get the
# process OOM-killed, which loses every other session in it rather than failing one
# upload. So the ceiling is configurable, and the hosted deployment sets it low enough
# that an oversized file is refused with a 413 instead of taking the service down.
MAX_FILE_MB = int(os.environ.get("DARWINBOX_MAX_FILE_MB", "100"))
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024

# How long a session survives without being touched, and how many may be live at once.
# Both exist to bound the memory of a long-running process; see SessionStore.
SESSION_TTL_SECONDS = int(os.environ.get("DARWINBOX_SESSION_TTL", "3600"))
MAX_SESSIONS = int(os.environ.get("DARWINBOX_MAX_SESSIONS", "50"))


class FileTooLargeError(Exception):
    """Raised when an upload exceeds the size ceiling (API maps to 413)."""


class SessionNotFoundError(Exception):
    """Raised for an unknown session id (API maps to 404)."""


@dataclass
class UploadOutcome:
    tables: list[TableProfile] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class Session:
    """One upload-and-ask session. Nothing survives its deletion."""

    def __init__(self, client, session_id: str | None = None) -> None:
        self.id = session_id or uuid.uuid4().hex[:12]
        self.touched = time.monotonic()
        self.in_use = 0  # requests currently served by this session; see SessionStore
        self.conn = duckdb.connect(":memory:")
        self.registry = Registry(self.conn)
        self.graph = RelationshipGraph(self.registry)
        self.client = client
        self._router: TableRouter | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #

    def add_files(self, files: list[tuple[str, bytes]]) -> UploadOutcome:
        """Ingest, profile and re-discover relationships across the whole session."""
        outcome = UploadOutcome()

        with self._lock:
            taken_ids = set(self.registry.table_ids)
            taken_aliases = {p.alias for p in self.registry.profiles.values()}

            for filename, data in files:
                if len(data) > MAX_FILE_BYTES:
                    raise FileTooLargeError(filename)
                try:
                    grids = load_file(filename, data)
                except UnreadableFileError as exc:
                    raise UnreadableFileError(f"{filename}: {exc}") from exc

                found = 0
                for loaded in grids:
                    blocks = detect_blocks(
                        filename, loaded.sheet, loaded.grid, taken_ids, taken_aliases
                    )
                    if not blocks and loaded.sheet:
                        outcome.warnings.append(
                            f"{filename} / {loaded.sheet}: no table found on this sheet."
                        )
                    for detected in blocks:
                        outcome.tables.append(self.registry.register(detected))
                        found += 1

                if found == 0:
                    outcome.warnings.append(f"{filename}: no table could be read from this file.")

            for merged_id, members in self.registry.merge_identical_tables():
                outcome.tables = [t for t in outcome.tables if t.table_id not in members]
                merged = self.registry.profile(merged_id)
                if merged not in outcome.tables:
                    outcome.tables.append(merged)
                message = (
                    f"{len(members)} tables with identical columns were combined into "
                    f"'{merged.alias}' ({merged.n_rows:,} rows)."
                )
                if message not in outcome.warnings:
                    outcome.warnings.append(message)

            self.registry.refresh_catalog()
            self._rediscover(outcome)
            self._router = None  # the schema changed; the BM25 index is stale

        return outcome

    def _rediscover(self, outcome: UploadOutcome) -> None:
        """Re-run discovery over every table, preserving the user's own decisions.

        Discovery is global rather than incremental: a file uploaded second can be the
        parent of one uploaded first, and rerunning is cheap next to re-reading files.
        """
        decisions = {
            edge.id: edge.status
            for edge in self.graph.all
            if edge.status in {"confirmed", "rejected"}
        }

        discovery = discover(self.registry)
        fresh = RelationshipGraph(self.registry)
        fresh.extend(discovery.relationships)

        try:
            fresh.extend(adjudicate(self.registry, discovery, self.client))
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, never swallowed
            outcome.warnings.append(
                f"Relationship adjudication was skipped ({exc}); "
                "statistical edges are unaffected."
            )

        for edge in fresh.all:
            if edge.id in decisions:
                edge.status = decisions[edge.id]  # type: ignore[assignment]

        for edge in self.graph.all:
            if edge.derivation == "user" and edge.id not in {e.id for e in fresh.all}:
                fresh.add(edge)  # hand-added edges survive re-discovery

        self.graph = fresh

    # ------------------------------------------------------------------ #
    # Schema
    # ------------------------------------------------------------------ #

    @property
    def tables(self) -> list[TableProfile]:
        """Uploaded tables. The schema catalog is queryable but is not a file."""
        return [self.registry.profile(t) for t in self.registry.data_table_ids]

    @property
    def relationships(self) -> list[Relationship]:
        return self.graph.all

    def components(self) -> list[list[str]]:
        return self.graph.components()

    def set_relationship_status(self, relationship_id: str, status: str) -> Relationship:
        return self.graph.set_status(relationship_id, status)

    def add_relationship(
        self,
        left_table: str,
        left_columns: list[str],
        right_table: str,
        right_columns: list[str],
    ) -> Relationship:
        """Add a user-defined edge. It is confirmed by definition: a human said so."""
        left = self.registry.resolve(left_table)
        right = self.registry.resolve(right_table)
        if left == right:
            raise ValueError("a relationship must connect two different tables")
        if len(left_columns) != len(right_columns) or not left_columns:
            raise ValueError("both sides must name the same number of columns")

        for table_id, columns in ((left, left_columns), (right, right_columns)):
            for column in columns:
                if not self.registry.has_column(table_id, column):
                    raise TableNotFoundError(f"{table_id}.{column}")

        values = ValueSets(self.registry)
        if len(left_columns) == 1:
            overlap = containment(
                values.get(left, left_columns[0]), values.get(right, right_columns[0])
            )
        else:
            overlap = containment(
                set(values.tuple_values(left, left_columns)),
                set(values.tuple_values(right, right_columns)),
            )

        left_profile = self.registry.profile(left)
        right_profile = self.registry.profile(right)
        name_sim = max(
            _name_similarity(
                (left_profile.column(lc).normalized_name_tokens if left_profile.column(lc) else []),
                (
                    right_profile.column(rc).normalized_name_tokens
                    if right_profile.column(rc)
                    else []
                ),
            )
            for lc, rc in zip(left_columns, right_columns, strict=True)
        )

        edge = Relationship(
            id=build_relationship_id(left, left_columns, right, right_columns),
            left_table=left,
            left_columns=left_columns,
            right_table=right,
            right_columns=right_columns,
            kind="composite" if len(left_columns) > 1 else "fk",
            containment=round(overlap, 4),
            name_similarity=round(name_sim, 4),
            score=1.0,
            parent_side=None,
            evidence=(
                f"Added by you. {overlap:.0%} of values on the left appear on the right."
            ),
            status="confirmed",
            derivation="user",
        )
        self.graph.add(edge)
        return edge

    # ------------------------------------------------------------------ #
    # Querying
    # ------------------------------------------------------------------ #

    @property
    def router(self) -> TableRouter:
        if self._router is None:
            self._router = TableRouter(self.registry)
        return self._router

    def ask(self, question: str) -> Iterator[TraceEvent | QueryResult]:
        """Stream trace events and one final QueryResult."""
        return answer(question, self.registry, self.graph, self.router, self.client)

    def close(self) -> None:
        self.conn.close()


class SessionStore:
    """In-process session registry. Sessions die with the process, by design.

    They also have to die before it, which is the part that is easy to miss. A session
    holds a DuckDB connection and every table ingested into it, and the only thing that
    freed one was an explicit DELETE -- which a browser does not send when its tab is
    closed. Every visitor therefore leaked a session for the lifetime of the process.
    On a laptop that is untidy; on the single hosted instance it is the whole service
    falling over after a few dozen visitors.

    So idle sessions are evicted, and there is a ceiling on how many can be live at
    once. Eviction runs when a session is created rather than on a timer, because a
    background thread would have to be owned, shut down and kept out of the tests for
    no benefit -- nothing accumulates unless sessions are being made.
    """

    def __init__(self, client_factory) -> None:
        self._sessions: dict[str, Session] = {}
        self._client_factory = client_factory
        self._lock = threading.Lock()

    def create(self) -> Session:
        session = Session(self._client_factory())
        with self._lock:
            self._evict_locked()
            self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session:
        try:
            session = self._sessions[session_id]
        except KeyError as exc:
            raise SessionNotFoundError(session_id) from exc
        session.touched = time.monotonic()
        return session

    @contextlib.contextmanager
    def acquire(self, session_id: str) -> Iterator[Session]:
        """Check a session out for the length of one request.

        Idleness is measured from when a request *starts*, so under enough load the
        ceiling picked a session whose own ingest was still running, closed its DuckDB
        connection underneath it and turned a valid upload into a 500. A session in use
        is not idle by any useful definition, so eviction skips it until it is done.
        """
        with self._lock:
            try:
                session = self._sessions[session_id]
            except KeyError as exc:
                raise SessionNotFoundError(session_id) from exc
            session.in_use += 1
        try:
            yield session
        finally:
            with self._lock:
                session.in_use -= 1
                session.touched = time.monotonic()

    def _evict_locked(self) -> None:
        """Drop idle sessions, then the oldest ones if still over the ceiling."""
        now = time.monotonic()
        stale = [
            sid
            for sid, s in self._sessions.items()
            if not s.in_use and now - s.touched > SESSION_TTL_SECONDS
        ]
        for sid in stale:
            self._close_quietly(self._sessions.pop(sid))

        # The ceiling is a backstop for traffic faster than the TTL, so the process is
        # bounded even then. Least recently used goes first, and a session serving a
        # request is never a candidate -- if every one of them is busy the ceiling
        # gives way rather than break a request that is already underway.
        while len(self._sessions) >= MAX_SESSIONS:
            idle = [sid for sid, s in self._sessions.items() if not s.in_use]
            if not idle:
                break
            oldest = min(idle, key=lambda sid: self._sessions[sid].touched)
            self._close_quietly(self._sessions.pop(oldest))

    @staticmethod
    def _close_quietly(session: Session) -> None:
        # A connection that will not close must not stop the others being reclaimed.
        with contextlib.suppress(Exception):
            session.close()

    def delete(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            raise SessionNotFoundError(session_id)
        session.close()

    def __len__(self) -> int:
        return len(self._sessions)


__all__ = [
    "FileTooLargeError",
    "RelationshipNotFoundError",
    "Session",
    "SessionNotFoundError",
    "SessionStore",
    "TableNotFoundError",
    "UnreadableFileError",
    "UploadOutcome",
]
