"""Prompt templates and the data-dictionary renderer (SPEC 8.3, 8.4).

The dictionary is the entire world the planner sees. It contains names, types,
ranges and at most a few sample values -- never data rows. On an 8 GB card the
budget is hard, so the renderer drops the lowest-ranked table and re-renders rather
than letting the prompt overflow the context window.
"""

from __future__ import annotations

from darwinbox.ingest.forms import FORM_COLUMNS
from darwinbox.models import Relationship, TableProfile
from darwinbox.profile.registry import Registry
from darwinbox.relate.graph import RelationshipGraph

DICTIONARY_TOKEN_BUDGET = 2200
SUMMARY_MAX_ROWS = 20
FORM_MAX_ITEMS = 40
FORM_ITEM_CHARS = 42
SAMPLE_CHARS = 30
MAX_SAMPLES = 2
MAX_CATEGORY_VALUES = 6

PLANNER_SYSTEM = """You translate questions into DuckDB SQL over the tables described below.

Rules:
- Use ONLY the tables and columns listed. Never invent a name.
- Join ONLY on the relationships listed. If the question needs two tables with no
  listed relationship between them, do not guess a join - set clarification_needed.
- If the question is ambiguous or cannot be answered from these tables, set
  clarification_needed to a single short question and leave answer_sql empty.
- If the question names a metric, concept or column that is not in the schema, set
  clarification_needed. Do NOT substitute a different column, and do NOT invent a
  formula to derive it. "commission" is not "salary"; asking is the correct answer.
- Aggregate each table at its own grain. To average a per-row attribute of a table,
  query that table directly; joining it to a larger one first double-counts rows.
- To group by month, quarter or year, use date_trunc('month', <date_column>). Never
  EXTRACT(MONTH ...): it discards the year, so January 2024 and January 2025 collapse
  into one row and the answer is silently wrong on any data spanning two years.
- answer_sql must be one SELECT statement. No CTE chains longer than 2. No DDL, no DML.
- evidence_sql returns the underlying un-aggregated rows behind the answer, LIMIT 100.
- Prefer explicit column aliases so results are readable.
- Questions about the data itself - what files or tables there are, what columns
  exist, their types or how many rows - are answered with SQL over the schema_tables
  and schema_columns tables, which describe every uploaded table.
- A table marked FORM holds one labelled figure per row. Query it by matching the
  label, e.g. SELECT item, value FROM <form> WHERE item ILIKE '%deduction%'. Its
  value column is text, so wrap it in TRY_CAST(value AS DOUBLE) to total or compare.

Reply with JSON only, matching this schema exactly:
{"reasoning": str, "tables_used": [str], "answer_sql": str, "evidence_sql": str,
 "chart_intent": "trend_over_time"|"comparison"|"distribution"|"single_value"|"breakdown"|"none",
 "followups": [str, str, str], "clarification_needed": str|null}"""

REPAIR_SYSTEM = """You fix a single DuckDB SQL statement that failed validation.

Use ONLY the tables, columns and relationships in the schema below. Change as little
as possible. If the query cannot be fixed with the columns available, set
clarification_needed and leave answer_sql empty.

If the error mentions GROUP BY, every selected column that is not inside an aggregate
must appear in the GROUP BY clause. Add the missing ones rather than removing columns.

Reply with JSON only, matching the same schema as before."""


# Nudges for the second attempt. Each addresses a shape the 7B was observed to give
# up on: monthly grouping, and reaching an attribute that lives in another table.
_REMINDERS = (
    "Reminders:\n"
    "- To group by month use date_trunc('month', <date_column>).\n"
    "- To filter or group by an attribute held in another table, JOIN to it using one "
    "of the listed relationships.\n"
    "- Every selected column that is not inside an aggregate must appear in GROUP BY.\n"
    "- Put answer_sql in the JSON. Only set clarification_needed if the question "
    "genuinely cannot be answered from these tables."
)


def estimate_tokens(text: str) -> int:
    """Rough token count. Deliberately cheap: a tokenizer dependency for a budget
    check would pull a model download into a module that must stay import-light."""
    return max(1, len(text) // 4)


# --------------------------------------------------------------------------- #
# Dictionary rendering
# --------------------------------------------------------------------------- #


def render_table(registry: Registry, table_id: str) -> str:
    profile = registry.profile(table_id)
    location = profile.source.filename
    if profile.source.sheet:
        location = f"{location} / {profile.source.sheet}"

    derived = sorted(
        key.split(".", 1)[1]
        for key in registry.derived
        if key.split(".", 1)[0] == table_id
    )

    if _is_form(profile):
        return _render_form(registry, profile, location)

    lines = [f"TABLE {profile.alias} — {profile.n_rows:,} rows  ({location})"]
    width = max([len(c.name) for c in profile.columns] + [len(d) for d in derived] + [10])
    for column in profile.columns:
        lines.append(f"  {column.name.ljust(width)}  {_render_column(column)}")

    # Derived granularity columns are real columns in DuckDB and are named in the
    # RELATIONSHIPS block, so they must be listed here too. Without this the model is
    # told to join on a column it cannot see in the schema, and invents its own
    # grouping instead.
    for name in derived:
        unit = name.rsplit("__", 1)[-1]
        lines.append(f"  {name.ljust(width)}  DATE      derived: start of {unit}")
    return "\n".join(lines)


def _is_form(profile) -> bool:
    return tuple(c.name for c in profile.columns) == FORM_COLUMNS


def _render_form(registry: Registry, profile, location: str) -> str:
    """A form table is queried by matching `item`, so the model must see the items.

    Listing two sample values the way an ordinary column is rendered tells it nothing:
    the labels *are* the schema here, and without them it cannot write a WHERE clause
    that matches anything.
    """
    items = registry.frame(profile.table_id)["item"].dropna().astype(str)
    shown = [i[:FORM_ITEM_CHARS] for i in items.head(FORM_MAX_ITEMS)]
    more = f"\n  ... and {len(items) - len(shown)} more items" if len(items) > len(shown) else ""

    return (
        f"TABLE {profile.alias} — {profile.n_rows:,} rows  ({location})\n"
        f"  This sheet is a FORM, not a table: one labelled figure per row.\n"
        f"  section  TEXT     the heading a row sits under\n"
        f"  item     TEXT     the label — match this, e.g. "
        f"WHERE item ILIKE '%standard deduction%'\n"
        f"  value    TEXT     the figure or text beside it; CAST to a number to compute\n"
        f"  items available: {', '.join(shown)}{more}"
    )


def _render_column(column) -> str:
    kind = column.semantic_type.upper()
    detail = _detail(column)
    return f"{kind.ljust(9)} {detail}".rstrip()


def _detail(column) -> str:
    samples = ", ".join(s[:SAMPLE_CHARS] for s in column.samples[:MAX_SAMPLES])

    if column.semantic_type == "id":
        cardinality = "unique" if column.is_unique else f"{column.n_distinct:,} distinct"
        return f"{cardinality.ljust(18)} e.g. {samples}" if samples else cardinality

    if column.semantic_type == "date":
        return f"{column.min_value} → {column.max_value}"

    if column.semantic_type == "numeric":
        return f"{column.min_value} → {column.max_value}"

    if column.semantic_type == "category":
        values = ", ".join(s[:SAMPLE_CHARS] for s in column.samples[:MAX_CATEGORY_VALUES])
        return f"{column.n_distinct} values: {values}"

    if column.semantic_type == "boolean":
        return "true / false"

    return f"e.g. {samples}" if samples else "text"


def render_relationships(
    registry: Registry, graph: RelationshipGraph, table_ids: list[str]
) -> str:
    """Only edges whose both ends are in the prompt, and never a rejected edge."""
    chosen = set(table_ids)
    edges = [
        e
        for e in graph.active
        if e.left_table in chosen and e.right_table in chosen
    ]
    if not edges:
        return ""
    lines = ["RELATIONSHIPS"] + [f"  {graph.describe(e)}" for e in edges]
    return "\n".join(lines)


def render_unrelated(registry: Registry, graph: RelationshipGraph, table_ids: list[str]) -> str:
    """Name the components explicitly so the model cannot assume a join exists."""
    chosen = set(table_ids)
    groups = [sorted(chosen & set(c)) for c in graph.components()]
    groups = [g for g in groups if g]
    if len(groups) < 2:
        return ""

    rendered = " | ".join(
        ", ".join(registry.alias_of(t) for t in group) for group in groups
    )
    return (
        "NOT RELATED\n"
        f"  These groups have no relationship between them: {rendered}\n"
        "  A question spanning two groups cannot be answered - ask for clarification."
    )


def render_dictionary(
    registry: Registry,
    graph: RelationshipGraph,
    table_ids: list[str],
    budget: int = DICTIONARY_TOKEN_BUDGET,
) -> tuple[str, list[str]]:
    """Render the dictionary, dropping lowest-ranked tables until it fits the budget.

    Returns the text and the tables that survived, so the trace can report a drop.
    """
    chosen = list(table_ids)
    while chosen:
        text = _assemble(registry, graph, chosen)
        if estimate_tokens(text) <= budget or len(chosen) == 1:
            return text, chosen
        chosen.pop()  # the router returns best-first, so the tail is least relevant
    return "", []


def _assemble(registry: Registry, graph: RelationshipGraph, table_ids: list[str]) -> str:
    blocks = [render_table(registry, t) for t in table_ids]
    relationships = render_relationships(registry, graph, table_ids)
    unrelated = render_unrelated(registry, graph, table_ids)
    parts = ["\n\n".join(blocks)]
    if relationships:
        parts.append(relationships)
    if unrelated:
        parts.append(unrelated)
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# User messages
# --------------------------------------------------------------------------- #


def planner_user(dictionary: str, question: str) -> str:
    return f"{dictionary}\n\nQUESTION: {question}"


def retry_user(dictionary: str, question: str) -> str:
    """Second attempt when the first returned no SQL at all.

    Asking the repair prompt to fix an empty string is incoherent -- there is nothing
    to correct, and the model tends to return empty again. This re-plans instead, and
    names the shapes it was most often missing.
    """
    return (
        f"{dictionary}\n\n"
        f"QUESTION: {question}\n\n"
        "Your previous reply contained no SQL. Write exactly one SELECT statement that "
        "answers this question using only the tables and columns above.\n"
        f"{_REMINDERS}"
    )


def repair_user(dictionary: str, question: str, sql: str, error: str) -> str:
    return (
        f"{dictionary}\n\n"
        f"QUESTION: {question}\n\n"
        f"This SQL failed validation:\n{sql}\n\n"
        f"The error was:\n{error}\n\n"
        f"{_REMINDERS}\n"
        "Return corrected JSON with answer_sql filled in."
    )


SUMMARY_SYSTEM = """You state the finding in a result set, in plain English.

Rules:
- One or two sentences. No preamble, no "based on the data", no restating the question.
- Lead with the actual figure or name that answers it. Quote numbers as they appear.
- If the rows are a list of things, say how many and name the notable ones.
- If the rows describe tables and columns, say what the data is ABOUT - what a person
  could ask of it - not just how many columns there are.
- Never invent a number that is not in the rows.
- The FACTS block is computed from the full result and is authoritative. Never
  contradict it, and never claim a different highest or lowest than it states.

Reply with JSON only: {"summary": str}"""


def summary_user(
    question: str,
    columns: list[str],
    rows: list[dict],
    total: int,
    facts: list[str] | None = None,
) -> str:
    """Prompt the summariser with the ANSWER, never with source tables.

    SPEC 14 forbids sending data rows to the model, which protects the planning stage:
    it must reason from profiles so it cannot be led by a handful of values, and the
    prompt budget cannot absorb real data. A summary is a different job -- the rows it
    sees are the aggregated result the user is already looking at on screen, never the
    uploaded tables, and they are capped hard.
    """
    shown = rows[:SUMMARY_MAX_ROWS]
    lines = [
        " | ".join(f"{k}={_short(v)}" for k, v in row.items()) for row in shown
    ]
    more = f"\n... and {total - len(shown):,} more rows" if total > len(shown) else ""
    computed = "\n\nFACTS (computed, authoritative):\n" + "\n".join(
        f"- {f}" for f in facts
    ) if facts else ""
    return (
        f"QUESTION: {question}\n\n"
        f"RESULT ({total:,} rows, columns: {', '.join(columns)}):\n"
        + "\n".join(lines)
        + more
        + computed
    )


def _short(value: object) -> str:
    text = str(value)
    return text if len(text) <= 40 else text[:37] + "..."


def summarize_relationship(registry: Registry, edge: Relationship) -> str:
    """One-line description used in traces and warnings."""
    left = registry.alias_of(edge.left_table)
    right = registry.alias_of(edge.right_table)
    lc = "+".join(edge.left_columns)
    rc = "+".join(edge.right_columns)
    return f"{left}.{lc} = {right}.{rc} ({edge.kind}, {edge.containment:.0%})"


def profile_headline(profile: TableProfile) -> str:
    return f"{profile.alias}: {profile.n_rows:,} rows, {len(profile.columns)} columns"
