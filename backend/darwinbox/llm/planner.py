"""Question -> QueryResult, with a trace of every decision (SPEC 8.4-8.7).

The planner orchestrates: route, render, plan, validate, repair once, execute,
chart. It emits a TraceEvent at every stage because the visible trace is the
product's answer to "why should I believe this number".

It knows nothing about HTTP. The API layer streams whatever this yields.
"""

from __future__ import annotations

import ast
import time
from collections.abc import Iterator

from darwinbox.execute.charts import select_chart
from darwinbox.execute.grounding import caveat, ungrounded_labels
from darwinbox.execute.runner import (
    MAX_ANSWER_ROWS,
    MAX_EVIDENCE_ROWS,
    QueryExecutionError,
    QueryTimeoutError,
    execute,
)
from darwinbox.execute.summarize import summarize
from darwinbox.execute.validator import ValidatedSQL, ValidationError, validate
from darwinbox.llm.client import LLMError, unavailable_message
from darwinbox.llm.prompts import (
    PLANNER_SYSTEM,
    REPAIR_SYSTEM,
    estimate_tokens,
    planner_user,
    render_dictionary,
    repair_user,
    retry_user,
)
from darwinbox.llm.router import TableRouter
from darwinbox.models import QueryPlan, QueryResult, TraceEvent
from darwinbox.profile.registry import Registry
from darwinbox.relate.graph import RelationshipGraph

MAX_PLAN_TOKENS = 700


class Clock:
    """Milliseconds since the last checkpoint, for TraceEvent.ms."""

    def __init__(self) -> None:
        self._last = time.perf_counter()

    def lap(self) -> int:
        now = time.perf_counter()
        elapsed = int((now - self._last) * 1000)
        self._last = now
        return elapsed


def answer(
    question: str,
    registry: Registry,
    graph: RelationshipGraph,
    router: TableRouter,
    client,
) -> Iterator[TraceEvent | QueryResult]:
    """Stream trace events, then exactly one QueryResult as the final item."""
    trace: list[TraceEvent] = []
    clock = Clock()

    def emit(stage: str, message: str, detail: dict | None = None) -> TraceEvent:
        event = TraceEvent(stage=stage, message=message, detail=detail, ms=clock.lap())
        trace.append(event)
        return event

    def refuse(message: str) -> QueryResult:
        return QueryResult(clarification=message, trace=list(trace))

    # data_table_ids, not table_ids: the schema catalog exists from the start, and
    # describing zero tables is not something to answer a question from.
    if not registry.data_table_ids:
        yield emit("route", "No files have been uploaded yet.")
        yield refuse("Upload a file first, then ask me about it.")
        return

    # ---------------------------------------------------------------- route
    routing = router.route(question, graph)
    if not routing.table_ids:
        yield emit("route", "No table matched the question.")
        yield refuse("I could not find a table related to that. What should I look at?")
        return

    yield emit(
        "route",
        routing.summary(registry),
        {
            "tables": [registry.alias_of(t) for t in routing.table_ids],
            "matched_terms": {
                registry.alias_of(t): terms for t, terms in routing.matched_terms.items()
            },
            "expanded": [registry.alias_of(t) for t in routing.expanded],
        },
    )

    dictionary, kept = render_dictionary(registry, graph, routing.table_ids)
    if len(kept) < len(routing.table_ids):
        dropped = [registry.alias_of(t) for t in routing.table_ids[len(kept) :]]
        yield emit(
            "route",
            f"Schema too large for the prompt budget; dropped {', '.join(dropped)}.",
            {"dropped": dropped, "tokens": estimate_tokens(dictionary)},
        )

    # ----------------------------------------------------------------- plan
    try:
        plan = _plan(client, dictionary, question)
    except LLMError as exc:
        yield emit("plan", f"The model could not be reached: {exc}")
        yield refuse(unavailable_message())
        return

    if plan.clarification_needed:
        yield emit("plan", "The model asked for clarification instead of guessing.")
        result = refuse(plan.clarification_needed)
        result.followups = plan.followups[:3]
        yield result
        return

    yield emit(
        "plan",
        plan.reasoning.strip() or "Planned a query.",
        {"tables_used": plan.tables_used, "sql": plan.answer_sql},
    )

    # ------------------------------------------------- validate and repair
    try:
        validated, repair_note = _validate_with_repair(
            plan, registry, client, dictionary, question
        )
    except ValidationError as exc:
        yield emit("validate", f"Validation failed twice: {exc.message}", {"code": exc.code})
        # An empty answer_sql means the model understood the question and had nothing
        # to compute -- almost always a question about the schema rather than the
        # data. Telling that user to "rephrase" sends them in circles; the answer they
        # want is already on screen.
        if exc.code == "EMPTY_SQL":
            yield refuse(
                "I can only answer questions that are computed from the data itself. "
                "To see what these files contain, open the Files & Schema pane on the "
                "left - it lists every table with its columns, types and sample values."
            )
        else:
            yield refuse(
                "I could not write a query I trust for that. Could you rephrase it, "
                "or name the columns you mean?"
            )
        return
    except LLMError as exc:
        yield emit("repair", f"The model could not be reached during repair: {exc}")
        yield refuse("The local model stopped responding while fixing the query.")
        return

    if repair_note:
        yield emit("repair", repair_note, {"sql": validated.sql})
    yield emit(
        "validate",
        f"SQL validated against {len(validated.tables)} table(s); read-only and parseable.",
        {"tables": [registry.alias_of(t) for t in validated.tables]},
    )

    # -------------------------------------------------------------- execute
    try:
        answer_result = execute(registry, validated.sql, limit=MAX_ANSWER_ROWS)
    except QueryTimeoutError as exc:
        yield emit("execute", f"Query timed out: {exc}")
        yield refuse("That query took too long. Could you narrow it down?")
        return
    except QueryExecutionError as exc:
        yield emit("execute", f"DuckDB rejected the query: {exc}")
        yield refuse("The query failed to run. Could you rephrase the question?")
        return

    yield emit(
        "execute",
        f"Returned {len(answer_result.rows):,} row(s).",
        {"truncated": answer_result.truncated},
    )

    evidence_rows: list[dict] = []
    if plan.evidence_sql:
        try:
            checked = validate(plan.evidence_sql, registry)
            evidence = execute(registry, checked.sql, limit=MAX_EVIDENCE_ROWS)
            evidence_rows = evidence.rows
            yield emit("execute", f"Fetched {len(evidence_rows)} row(s) of supporting evidence.")
        except (ValidationError, QueryExecutionError, QueryTimeoutError) as exc:
            # The answer is still sound without its evidence rows, so this degrades to
            # a visible trace note rather than failing the whole question.
            yield emit("execute", f"Could not fetch the rows behind the answer: {exc}")

    # ---------------------------------------------------------------- chart
    chart = select_chart(
        answer_result.columns, answer_result.rows, plan.chart_intent, title=question.strip()
    )
    yield emit("chart", f"Chose a {chart.type.replace('_', ' ')} from the result shape.")

    # ----------------------------------------------------------- grounding
    # A valid join, valid SQL and correct arithmetic can still produce a false answer
    # if the label is invented: "average temperature" over files holding only rainfall.
    # The substitution is reported rather than blocked, because a synonym looks the
    # same as an invention and refusing on it would reject "sum(amount) AS revenue".
    schema_columns = [
        c.name for t in validated.tables for c in registry.profile(t).columns
    ]
    relabelled = ungrounded_labels(
        validated.sql, schema_columns, [registry.alias_of(t) for t in validated.tables]
    )
    if relabelled:
        yield emit(
            "validate",
            "Renamed measures that no column is named for: "
            + ", ".join(f"{a} <- {c}" for a, c in sorted(relabelled.items())),
            {"labels": relabelled},
        )

    # -------------------------------------------------------------- summary
    summary, from_model = summarize(
        client, question, answer_result.columns, answer_result.rows, answer_result.truncated
    )
    note = caveat(relabelled)
    if note:
        summary = f"{summary} {note}"
    yield emit(
        "summary",
        "Summarised the result." if from_model else "Described the result from its shape.",
        {"from_model": from_model},
    )

    yield QueryResult(
        summary=summary,
        answer_rows=answer_result.rows,
        answer_columns=answer_result.columns,
        evidence_rows=evidence_rows,
        sql=validated.sql,
        chart=chart,
        followups=plan.followups[:3],
        trace=list(trace),
        clarification=None,
    )


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #


def _plan(client, dictionary: str, question: str) -> QueryPlan:
    reply = client.complete_json(
        PLANNER_SYSTEM, planner_user(dictionary, question), max_tokens=MAX_PLAN_TOKENS
    )
    return _coerce_plan(reply)


# A 7B model routinely writes JSON nulls as prose. Left alone, the literal string
# "None" is truthy and non-empty, so a plan carrying perfectly good SQL is reported as
# a refusal -- which is what "give me sales per region" hit: valid SQL in answer_sql,
# "None" in clarification_needed, and the whole answer thrown away.
_NOT_A_CLARIFICATION = {"none", "null", "nil", "n/a", "na", "false", "no", "-", "undefined"}


def _as_optional_text(value: object) -> str | None:
    """A model's string field, or None when it is really a spelled-out null."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    return None if not text or text.lower() in _NOT_A_CLARIFICATION else text


def _as_list(value: object) -> list[str]:
    """Coerce a list field the model may have serialised as a string."""
    if isinstance(value, list):
        return [str(v) for v in value if v]
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, list):
                return [str(v) for v in parsed if v]
        return [text] if text else []
    return []


def _coerce_plan(reply: object) -> QueryPlan:
    """Build a QueryPlan from whatever the model returned, without trusting its shape.

    This is the boundary where a small model's sloppiness has to be absorbed: string
    "None" for null, a Python-repr list where JSON was asked for, missing keys.
    """
    if not isinstance(reply, dict):
        raise LLMError(f"expected a JSON object, got {type(reply).__name__}")

    return QueryPlan(
        reasoning=str(reply.get("reasoning") or ""),
        tables_used=_as_list(reply.get("tables_used")),
        answer_sql=str(reply.get("answer_sql") or "").strip(),
        evidence_sql=_as_optional_text(reply.get("evidence_sql")),
        chart_intent=_chart_intent(reply.get("chart_intent")),
        followups=_as_list(reply.get("followups"))[:3],
        clarification_needed=_as_optional_text(reply.get("clarification_needed")),
    )


_INTENTS = {
    "trend_over_time", "comparison", "distribution", "single_value", "breakdown", "none"
}


def _chart_intent(value: object) -> str:
    text = str(value or "none").strip().lower()
    return text if text in _INTENTS else "none"


def _validate_with_repair(
    plan: QueryPlan, registry: Registry, client, dictionary: str, question: str
) -> tuple[ValidatedSQL, str | None]:
    """Validate; on failure send exactly one repair message, then validate again."""
    try:
        return validate(plan.answer_sql, registry), None
    except ValidationError as first:
        # No SQL at all is a different failure from wrong SQL, and needs a different
        # second attempt: there is nothing to repair, so re-plan with a nudge. Sending
        # an empty string to the repair prompt just produces another empty reply.
        if first.code == "EMPTY_SQL":
            reply = client.complete_json(
                PLANNER_SYSTEM, retry_user(dictionary, question), max_tokens=MAX_PLAN_TOKENS
            )
        else:
            reply = client.complete_json(
                REPAIR_SYSTEM,
                repair_user(dictionary, question, plan.answer_sql, first.message),
                max_tokens=MAX_PLAN_TOKENS,
            )
        repaired = _coerce_plan(reply)

        if repaired.clarification_needed:
            raise ValidationError(first.code, first.message) from first
        if not repaired.answer_sql:
            # The retry gave up. Surface the ORIGINAL failure: saying "returned no SQL"
            # when the first attempt produced a query, just an invalid one, sends the
            # user (and me) looking in entirely the wrong place.
            raise ValidationError(first.code, first.message) from first

        validated = validate(repaired.answer_sql, registry)
        plan.answer_sql = repaired.answer_sql
        if repaired.evidence_sql:
            plan.evidence_sql = repaired.evidence_sql
        return validated, _repair_message(first, plan.answer_sql)


def _repair_message(error: ValidationError, fixed_sql: str) -> str:
    """Human-readable repair line; the most persuasive line in the demo (SPEC 8.5)."""
    if error.code == "COLUMN_NOT_FOUND":
        name = error.message.split('"')[1] if '"' in error.message else "a column"
        resolved = _find_resolution(name, fixed_sql)
        if resolved:
            return (
                f'Validation failed: unknown column "{name}" '
                f'→ retried and resolved to "{resolved}"'
            )
    if error.code == "TABLE_NOT_FOUND":
        name = error.message.split('"')[1] if '"' in error.message else "a table"
        return f'Validation failed: unknown table "{name}" → retried against the real schema'
    return f"Validation failed: {error.message} → retried and the corrected query passed"


def _find_resolution(missing: str, fixed_sql: str) -> str | None:
    """Guess which identifier in the repaired SQL replaced the missing one."""
    import re

    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", fixed_sql))
    parts = set(missing.lower().split("_"))
    best, best_overlap = None, 0
    for token in sorted(tokens):
        if token.lower() == missing.lower():
            continue
        overlap = len(parts & set(token.lower().split("_")))
        if overlap > best_overlap:
            best, best_overlap = token, overlap
    return best
