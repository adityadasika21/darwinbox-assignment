"""DuckDB execution with row caps and a wall-clock limit (SPEC 8.6)."""

from __future__ import annotations

import contextlib
import datetime as dt
import decimal
import math
import threading
from dataclasses import dataclass, field

import duckdb

from darwinbox.profile.registry import Registry

QUERY_TIMEOUT_SECONDS = 20
MAX_ANSWER_ROWS = 5000
MAX_EVIDENCE_ROWS = 100


class QueryTimeoutError(Exception):
    """Raised when a query exceeds the wall-clock budget."""


class QueryExecutionError(Exception):
    """Raised when DuckDB refuses a query that passed validation."""


@dataclass
class ExecutionResult:
    columns: list[str] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)
    truncated: bool = False


def execute(
    registry: Registry,
    sql: str,
    limit: int = MAX_ANSWER_ROWS,
    timeout: int = QUERY_TIMEOUT_SECONDS,
) -> ExecutionResult:
    """Run validated SQL and return JSON-ready rows."""
    holder: dict[str, object] = {}

    def run() -> None:
        try:
            cursor = registry.conn.cursor()
            cursor.execute(sql)
            holder["columns"] = [d[0] for d in cursor.description or []]
            holder["rows"] = cursor.fetchmany(limit + 1)
        except duckdb.Error as exc:
            holder["error"] = exc

    # DuckDB has no per-statement timeout, so the query runs on a worker thread and
    # the caller stops waiting. The thread is left to finish rather than killed:
    # interrupting mid-query risks corrupting the shared connection state.
    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout)

    if worker.is_alive():
        # Best-effort: if interrupt is unsupported the timeout still stands.
        with contextlib.suppress(Exception):
            registry.conn.interrupt()
        raise QueryTimeoutError(f"query exceeded {timeout}s")

    error = holder.get("error")
    if error is not None:
        raise QueryExecutionError(str(error).splitlines()[0])

    columns = list(holder.get("columns") or [])
    raw = list(holder.get("rows") or [])
    truncated = len(raw) > limit

    return ExecutionResult(
        columns=columns,
        rows=[
            {col: to_json_value(value) for col, value in zip(columns, row, strict=False)}
            for row in raw[:limit]
        ],
        truncated=truncated,
    )


def to_json_value(value: object) -> object:
    """Convert a DuckDB value to something json.dumps accepts."""
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.isoformat(sep=" ", timespec="seconds")
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, dt.timedelta):
        return str(value)
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return [to_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_json_value(v) for k, v in value.items()}
    return value
