"""DuckDB registration and the table profile store.

The registry owns the mapping table_id -> (DataFrame, TableProfile) and is the only
place that turns a detected block into a queryable DuckDB table.
"""

from __future__ import annotations

import re

import duckdb
import pandas as pd

from darwinbox.ingest.blocks import DetectedBlock
from darwinbox.models import SourceRef, TableProfile
from darwinbox.profile.columns import coerce_series, profile_column

_PLACEHOLDER = re.compile(r"col_\d+")
FORM_SIGNATURE = ("section", "item", "value")

CATALOG_TABLES = "schema_tables"
CATALOG_COLUMNS = "schema_columns"


class TableNotFoundError(Exception):
    """Raised when a table_id or alias is not registered (API maps to 404)."""


class Registry:
    """Holds every table discovered in one session, plus its DuckDB projection."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self.conn = conn
        self.frames: dict[str, pd.DataFrame] = {}
        self.profiles: dict[str, TableProfile] = {}
        self.derived: dict[str, str] = {}  # "table_id.derived_col" -> sql expression

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #

    def register(self, detected: DetectedBlock) -> TableProfile:
        """Coerce a detected block into a DataFrame, load it into DuckDB, profile it."""
        block = detected.block
        frame = _build_frame(detected)

        self.conn.register(f"_staging_{block.table_id}", frame)
        self.conn.execute(
            f'CREATE OR REPLACE TABLE "{block.table_id}" AS '
            f'SELECT * FROM "_staging_{block.table_id}"'
        )
        self.conn.unregister(f"_staging_{block.table_id}")

        n_rows = len(frame)
        raw_names = dict(zip(block.columns, block.raw_columns, strict=False))
        profile = TableProfile(
            table_id=block.table_id,
            alias=block.alias,
            source=block.source,
            n_rows=n_rows,
            columns=[
                profile_column(name, frame[name], n_rows, raw_names.get(name))
                for name in frame.columns
            ],
        )

        self.frames[block.table_id] = frame
        self.profiles[block.table_id] = profile
        return profile

    # ------------------------------------------------------------------ #
    # Lookup
    # ------------------------------------------------------------------ #

    @property
    def table_ids(self) -> list[str]:
        return list(self.profiles)

    def profile(self, table_id: str) -> TableProfile:
        try:
            return self.profiles[table_id]
        except KeyError as exc:
            raise TableNotFoundError(table_id) from exc

    def frame(self, table_id: str) -> pd.DataFrame:
        try:
            return self.frames[table_id]
        except KeyError as exc:
            raise TableNotFoundError(table_id) from exc

    def resolve(self, name: str) -> str:
        """Map an alias (what the LLM sees) or a table_id to a table_id."""
        if name in self.profiles:
            return name
        for table_id, profile in self.profiles.items():
            if profile.alias == name:
                return table_id
        raise TableNotFoundError(name)

    def alias_of(self, table_id: str) -> str:
        return self.profile(table_id).alias

    def has_column(self, table_id: str, column: str) -> bool:
        profile = self.profiles.get(table_id)
        if profile is None:
            return False
        if profile.column(column) is not None:
            return True
        return f"{table_id}.{column}" in self.derived

    # ------------------------------------------------------------------ #
    # Self-describing catalog
    # ------------------------------------------------------------------ #

    def refresh_catalog(self) -> None:
        """Expose the schema itself as two queryable tables.

        "What is in these files?" is a perfectly ordinary question, and answering it
        with "look at the sidebar" is a non-answer. Rather than special-casing the
        phrasing -- which only ever matches the wordings someone thought of -- the
        catalog becomes data. The planner then handles "what columns are there",
        "which table has the most rows" and "which columns hold dates" as plain SQL,
        with the same validation and the same visible trace as any other question.
        """
        rows = [
            (p.alias, p.source.filename, p.source.sheet, p.n_rows, len(p.columns))
            for p in self.profiles.values()
            if p.kind == "data"
        ]
        columns = [
            (p.alias, c.name, c.semantic_type, c.n_distinct, c.n_null,
             ", ".join(c.samples[:3]))
            for p in self.profiles.values()
            if p.kind == "data"
            for c in p.columns
        ]

        self._materialise_catalog(
            CATALOG_TABLES,
            ["table_name", "source_file", "sheet", "n_rows", "n_columns"],
            rows,
        )
        self._materialise_catalog(
            CATALOG_COLUMNS,
            ["table_name", "column_name", "data_type", "n_distinct", "n_nulls",
             "example_values"],
            columns,
        )

    def _materialise_catalog(
        self, alias: str, columns: list[str], rows: list[tuple]
    ) -> None:
        frame = pd.DataFrame(rows, columns=columns)
        self.conn.register(f"_staging_{alias}", frame)
        self.conn.execute(
            f'CREATE OR REPLACE TABLE "{alias}" AS SELECT * FROM "_staging_{alias}"'
        )
        self.conn.unregister(f"_staging_{alias}")

        self.frames[alias] = frame
        self.profiles[alias] = TableProfile(
            table_id=alias,
            alias=alias,
            source=SourceRef(
                filename="(schema catalog)", sheet=None, block_index=0,
                header_row=0, data_start_row=0,
                data_end_row=max(0, len(frame) - 1), col_start=0,
                col_end=len(columns) - 1,
            ),
            n_rows=len(frame),
            columns=[profile_column(c, frame[c], len(frame)) for c in frame.columns],
            kind="catalog",
        )

    @property
    def data_table_ids(self) -> list[str]:
        """Uploaded tables only -- what relationship discovery and the UI care about."""
        return [t for t, p in self.profiles.items() if p.kind == "data"]

    # ------------------------------------------------------------------ #
    # Sheet fusion (SPEC 11.1: split_sheets)
    # ------------------------------------------------------------------ #

    def merge_identical_tables(self) -> list[tuple[str, list[str]]]:
        """Fuse tables that share an identical column signature into one table.

        One logical table exported across three sheets is three blocks to the
        detector and should be one table to the user. Leaving them separate is not
        merely untidy: "average hours per department" then needs a three-way UNION
        that a 7B model will not reliably write, so a correct answer becomes
        unreachable for a reason that has nothing to do with the question.

        The merge is by exact normalized column signature, deliberately: anything
        looser risks fusing two genuinely different tables, and a wrong union is far
        worse than a missed one -- it silently doubles rows in every later answer.
        """
        groups: dict[tuple[str, ...], list[str]] = {}
        for table_id, profile in self.profiles.items():
            if profile.kind != "data":
                continue
            signature = tuple(c.name for c in profile.columns)
            # A signature carries no information when it comes from the shape of the
            # reader rather than from the file: col_1..col_n is the absence of a
            # header, and (section, item, value) is what every extracted form looks
            # like. Fusing on either merged unrelated documents -- a tax computation
            # with a statutory certificate -- into one nonsense table.
            if all(_PLACEHOLDER.fullmatch(name) for name in signature):
                continue
            if signature == FORM_SIGNATURE:
                continue
            groups.setdefault(signature, []).append(table_id)

        merged: list[tuple[str, list[str]]] = []
        for members in groups.values():
            if len(members) < 2:
                continue
            merged.append((self._fuse(members), list(members)))
        return merged

    def _fuse(self, members: list[str]) -> str:
        first = self.profiles[members[0]]
        frame = pd.concat([self.frames[m] for m in members], ignore_index=True)

        # Sheets of one workbook are best named after the workbook ("attendance_split");
        # tables fused across files (jan.csv, feb.csv) keep their shared alias stem.
        filenames = {self.profiles[m].source.filename for m in members}
        if len(filenames) == 1:
            alias = _slugify(next(iter(filenames)).rsplit(".", 1)[0])
        else:
            alias = _common_prefix([self.profiles[m].alias for m in members]) or first.alias
        table_id = f"{alias}__merged"
        sheets = [self.profiles[m].source.sheet for m in members]
        source = first.source.model_copy(
            update={
                "sheet": "+".join(s for s in sheets if s) or None,
                "data_end_row": first.source.data_start_row + len(frame) - 1,
            }
        )

        for member in members:
            self.conn.execute(f'DROP TABLE IF EXISTS "{member}"')
            self.frames.pop(member, None)
            self.profiles.pop(member, None)

        self.conn.register(f"_staging_{table_id}", frame)
        self.conn.execute(
            f'CREATE OR REPLACE TABLE "{table_id}" AS SELECT * FROM "_staging_{table_id}"'
        )
        self.conn.unregister(f"_staging_{table_id}")

        n_rows = len(frame)
        self.frames[table_id] = frame
        self.profiles[table_id] = TableProfile(
            table_id=table_id,
            alias=alias,
            source=source,
            n_rows=n_rows,
            columns=[profile_column(name, frame[name], n_rows) for name in frame.columns],
        )
        return table_id

    # ------------------------------------------------------------------ #
    # Derived columns (SPEC 7.4)
    # ------------------------------------------------------------------ #

    def add_derived_column(self, table_id: str, column: str, expression: str) -> None:
        """Materialise a computed column so generated SQL can reference it by name.

        Tradeoff: ALTER TABLE ... ADD COLUMN over a view, because a view would shadow
        the base table name and every later CREATE would have to re-derive the stack.
        """
        key = f"{table_id}.{column}"
        if key in self.derived:
            return
        self.conn.execute(f'ALTER TABLE "{table_id}" ADD COLUMN "{column}" TIMESTAMP')
        self.conn.execute(f'UPDATE "{table_id}" SET "{column}" = {expression}')
        self.derived[key] = expression


def _slugify(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", value.strip().lower())).strip("_")


def _common_prefix(aliases: list[str]) -> str:
    """Longest shared alias stem, e.g. week1/week2/week3 -> 'week'."""
    if not aliases:
        return ""
    shortest = min(aliases, key=len)
    for i, char in enumerate(shortest):
        if any(alias[i] != char for alias in aliases):
            return shortest[:i].rstrip("_0123456789")
    return shortest.rstrip("_0123456789")


def _build_frame(detected: DetectedBlock) -> pd.DataFrame:
    """Rows of strings -> a typed DataFrame with the block's normalized column names."""
    columns = detected.block.columns
    width = len(columns)
    padded = [(row + [None] * width)[:width] for row in detected.rows]

    frame = pd.DataFrame(padded, columns=columns, dtype="object")
    for name in columns:
        frame[name] = coerce_series(frame[name])
    return frame
