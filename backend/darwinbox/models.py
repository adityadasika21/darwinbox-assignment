"""All pydantic data contracts for Darwinbox Assignment.

This module is written in full before any logic elsewhere in the package, and it
imports nothing from the rest of the codebase. Every other module depends on
these shapes; none of them redefine one locally.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #

SemanticType = Literal["id", "numeric", "date", "category", "text", "boolean"]


class SourceRef(BaseModel):
    """Where a table block physically came from, in original-grid coordinates."""

    filename: str
    sheet: str | None = None
    block_index: int
    header_row: int  # 0-based index in the original grid; -1 => headers synthesised
    data_start_row: int
    data_end_row: int
    col_start: int
    col_end: int

    def human_location(self) -> str:
        """Render as 'Sheet1, rows 4-210' for display in the UI."""
        where = self.sheet or self.filename
        return f"{where}, rows {self.data_start_row + 1}-{self.data_end_row + 1}"


class TableBlock(BaseModel):
    """One rectangular table found inside one grid.

    The DataFrame itself is deliberately held outside the model, keyed by
    table_id in the registry, so that contracts stay cheap to serialise.
    """

    table_id: str  # "{file_slug}__{sheet_slug}__b{n}", snake_case, unique
    alias: str  # short human name shown to the LLM, e.g. "orders"
    source: SourceRef
    columns: list[str]  # normalized, unique, snake_case
    raw_columns: list[str]  # as found in the file
    n_rows: int


# --------------------------------------------------------------------------- #
# Profiling
# --------------------------------------------------------------------------- #


class ColumnProfile(BaseModel):
    name: str
    semantic_type: SemanticType
    dtype: str
    n_distinct: int
    n_null: int
    null_rate: float
    is_unique: bool
    min_value: str | None = None
    max_value: str | None = None
    samples: list[str] = Field(default_factory=list)  # max 5, each truncated to 40 chars
    regex_signature: str | None = None  # e.g. "^EMP\\d{4}$" if >=95% of values match
    normalized_name_tokens: list[str] = Field(default_factory=list)


TableKind = Literal["data", "catalog"]


class TableProfile(BaseModel):
    table_id: str
    alias: str
    source: SourceRef
    n_rows: int
    columns: list[ColumnProfile]
    # "catalog" marks the self-describing schema tables: queryable like any other
    # table, but not uploaded data, so they are kept out of relationship discovery
    # and out of the file listing in the UI.
    kind: TableKind = "data"

    def column(self, name: str) -> ColumnProfile | None:
        """Look up a column profile by normalized name."""
        for col in self.columns:
            if col.name == name:
                return col
        return None


# --------------------------------------------------------------------------- #
# Relationships
# --------------------------------------------------------------------------- #

RelationshipKind = Literal["fk", "composite", "derived", "unrelated"]
RelationshipStatus = Literal["proposed", "confirmed", "rejected"]


class Relationship(BaseModel):
    id: str
    left_table: str
    left_columns: list[str]
    right_table: str
    right_columns: list[str]
    kind: RelationshipKind
    containment: float
    name_similarity: float
    score: float
    parent_side: Literal["left", "right"] | None = None
    evidence: str  # one human sentence, shown in the UI
    status: RelationshipStatus = "proposed"
    derivation: str | None = None  # e.g. "date_trunc('month', orders.order_date)"
    projection: str | None = None  # normalization both sides must apply when joining


# --------------------------------------------------------------------------- #
# Planning and execution
# --------------------------------------------------------------------------- #

ChartIntent = Literal[
    "trend_over_time",
    "comparison",
    "distribution",
    "single_value",
    "breakdown",
    "none",
]


class QueryPlan(BaseModel):
    """The exact JSON shape the planner LLM must return."""

    reasoning: str = ""
    tables_used: list[str] = Field(default_factory=list)
    answer_sql: str = ""
    evidence_sql: str | None = None
    chart_intent: ChartIntent = "none"
    followups: list[str] = Field(default_factory=list)
    clarification_needed: str | None = None  # non-null => refuse and ask


class TraceEvent(BaseModel):
    stage: str  # route | plan | validate | repair | execute | chart
    message: str
    detail: dict | None = None
    ms: int = 0


class ChartSpec(BaseModel):
    type: Literal["line", "bar", "scatter", "big_number", "table"]
    x: str | None = None
    y: str | None = None
    series: str | None = None
    title: str = ""


class QueryResult(BaseModel):
    summary: str = ""  # one or two sentences stating the finding, in prose
    answer_rows: list[dict] = Field(default_factory=list)
    answer_columns: list[str] = Field(default_factory=list)
    evidence_rows: list[dict] = Field(default_factory=list)
    sql: str = ""
    chart: ChartSpec | None = None
    followups: list[str] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    clarification: str | None = None


# --------------------------------------------------------------------------- #
# API envelopes
# --------------------------------------------------------------------------- #


class SessionCreated(BaseModel):
    session_id: str


class UploadResponse(BaseModel):
    tables: list[TableProfile]
    warnings: list[str] = Field(default_factory=list)


class SchemaResponse(BaseModel):
    tables: list[TableProfile]
    relationships: list[Relationship]
    components: list[list[str]]


class RelationshipPatch(BaseModel):
    status: RelationshipStatus


class RelationshipCreate(BaseModel):
    left_table: str
    left_columns: list[str]
    right_table: str
    right_columns: list[str]


class SampleSet(BaseModel):
    name: str
    title: str
    description: str
    files: list[str]


class SampleRequest(BaseModel):
    name: str


class QuestionRequest(BaseModel):
    question: str


class ErrorBody(BaseModel):
    code: str
    message: str
    detail: dict = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorBody
