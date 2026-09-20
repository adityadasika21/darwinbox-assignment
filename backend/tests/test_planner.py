"""Phase 4 acceptance: routing, prompt rendering and the plan/repair/refuse loop.

Runs entirely on FakeLLMClient, so the whole suite works without a GPU.
"""

from __future__ import annotations

import pytest
from conftest import alias_for, load_session
from darwinbox.llm.client import FakeLLMClient
from darwinbox.llm.planner import _coerce_plan, answer
from darwinbox.llm.prompts import (
    DICTIONARY_TOKEN_BUDGET,
    estimate_tokens,
    render_dictionary,
)
from darwinbox.models import QueryResult, TraceEvent


def run(question, registry, graph, router, client) -> tuple[list[TraceEvent], QueryResult]:
    events = list(answer(question, registry, graph, router, client))
    result = events[-1]
    assert isinstance(result, QueryResult), "the last item must be the QueryResult"
    return [e for e in events[:-1] if isinstance(e, TraceEvent)], result


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #


def test_router_picks_the_table_the_question_names(registry, graph, router):
    result = router.route("how many employees are in each department?", graph)
    chosen = [registry.alias_of(t) for t in result.table_ids]
    assert "headcount" in chosen


def test_router_matches_category_values_not_just_column_names(registry, graph, router):
    # "North" is a value inside customers.region, never a column name.
    result = router.route("which customers are in the North region?", graph)
    assert alias_for(registry, "customers") in [registry.alias_of(t) for t in result.table_ids]


def test_router_expands_along_confirmed_relationships(registry, graph, router):
    for edge in graph.all:
        edge.status = "confirmed"

    result = router.route("total amount by customer region", graph)
    chosen = {registry.alias_of(t) for t in result.table_ids}
    assert {"orders", "customers"} <= chosen


def test_router_reports_the_terms_that_matched(registry, graph, router):
    result = router.route("total salary by department", graph)
    terms = [t for terms in result.matched_terms.values() for t in terms]
    assert "salary" in terms or "department" in terms


# --------------------------------------------------------------------------- #
# Data dictionary
# --------------------------------------------------------------------------- #


def test_dictionary_contains_schema_but_never_data_rows(registry, graph):
    text, kept = render_dictionary(registry, graph, registry.table_ids)

    assert "TABLE orders" in text
    assert "ID" in text and "NUMERIC" in text and "DATE" in text
    assert len(kept) == len(registry.table_ids)
    # Sample values are allowed; a full row of one is not.
    assert "ORD-1" in text
    assert "ORD-1,C-88,2024-01-05,120.50" not in text


def test_dictionary_names_disconnected_groups_explicitly(registry, graph):
    text, _ = render_dictionary(registry, graph, registry.table_ids)
    assert "NOT RELATED" in text
    assert "headcount" in text


def test_dictionary_is_dropped_table_by_table_to_fit_the_budget(registry, graph):
    text, kept = render_dictionary(registry, graph, registry.table_ids, budget=40)
    assert len(kept) == 1
    assert estimate_tokens(text) <= DICTIONARY_TOKEN_BUDGET


def test_rejected_relationships_never_reach_the_prompt(registry, graph):
    for edge in graph.all:
        edge.status = "rejected"
    text, _ = render_dictionary(registry, graph, registry.table_ids)
    assert "RELATIONSHIPS" not in text


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def plan_for(sql: str, **extra) -> dict:
    plan = {
        "reasoning": "Sum order amounts by customer region.",
        "tables_used": ["orders", "customers"],
        "answer_sql": sql,
        "evidence_sql": None,
        "chart_intent": "comparison",
        "followups": ["By month?", "Top customer?", "Average order size?"],
        "clarification_needed": None,
    }
    plan.update(extra)
    return plan


def test_a_cross_file_question_is_answered_by_a_real_join(registry, graph, router):
    orders, customers = alias_for(registry, "orders"), alias_for(registry, "customers")
    sql = (
        f"SELECT c.region, round(sum(o.amount), 2) AS total "
        f"FROM {orders} o JOIN {customers} c ON o.cust_ref = c.code "
        f"GROUP BY c.region ORDER BY c.region"
    )
    client = FakeLLMClient(default=plan_for(sql))

    trace, result = run("total revenue by region", registry, graph, router, client)

    assert result.clarification is None
    assert {r["region"] for r in result.answer_rows} == {"East", "North", "South"}
    assert result.chart.type == "bar"
    assert result.followups == ["By month?", "Top customer?", "Average order size?"]
    assert [e.stage for e in trace] == [
        "route", "plan", "validate", "execute", "chart", "summary",
    ]
    assert result.summary, "every answer leads with a sentence"


def test_evidence_rows_accompany_the_answer(registry, graph, router):
    orders, customers = alias_for(registry, "orders"), alias_for(registry, "customers")
    client = FakeLLMClient(
        default=plan_for(
            f"SELECT c.region, sum(o.amount) AS total FROM {orders} o "
            f"JOIN {customers} c ON o.cust_ref = c.code GROUP BY c.region",
            evidence_sql=(
                f"SELECT o.order_id, c.region, o.amount FROM {orders} o "
                f"JOIN {customers} c ON o.cust_ref = c.code LIMIT 100"
            ),
        )
    )
    _, result = run("total revenue by region", registry, graph, router, client)

    assert len(result.evidence_rows) == 6
    assert set(result.evidence_rows[0]) == {"order_id", "region", "amount"}


# --------------------------------------------------------------------------- #
# Refusal: the system asks rather than inventing
# --------------------------------------------------------------------------- #


def test_model_clarification_is_returned_as_a_question(registry, graph, router):
    client = FakeLLMClient(
        default=plan_for("", clarification_needed="Which file holds the salaries you mean?")
    )
    _, result = run("compare salary to revenue", registry, graph, router, client)

    assert result.clarification == "Which file holds the salaries you mean?"
    assert result.answer_rows == []
    assert result.sql == ""


def test_invented_column_is_refused_rather_than_executed(registry, graph, router):
    orders = alias_for(registry, "orders")
    # The model invents a column twice; the system must refuse, not return bad SQL.
    client = FakeLLMClient(default=plan_for(f"SELECT made_up_column FROM {orders}"))

    trace, result = run("what is the made up column", registry, graph, router, client)

    assert result.clarification is not None
    assert result.answer_rows == []
    assert any(e.stage == "validate" and "twice" in e.message for e in trace)


def test_a_write_query_from_the_model_never_reaches_duckdb(registry, graph, router):
    orders = alias_for(registry, "orders")
    client = FakeLLMClient(default=plan_for(f"DROP TABLE {orders}"))

    _, result = run("delete everything", registry, graph, router, client)

    assert result.clarification is not None
    assert registry.conn.execute(f'SELECT count(*) FROM "{registry.table_ids[0]}"').fetchone()[0]


# --------------------------------------------------------------------------- #
# Repair: one retry, with a human-readable trace line
# --------------------------------------------------------------------------- #


class RepairingClient:
    """Returns a broken plan first, then a correct one -- exactly one repair.

    Summary calls are counted separately: they are a different job, and folding them
    into `calls` would hide whether the planner really retried only once.
    """

    def __init__(self, broken: dict, fixed: dict) -> None:
        self.replies = [broken, fixed]
        self.calls = 0
        self.summary_calls = 0

    def complete_json(self, system: str, user: str, max_tokens: int = 700) -> dict:
        if system.startswith("You state the finding"):
            self.summary_calls += 1
            return {"summary": "A summary."}
        self.calls += 1
        return self.replies[min(self.calls - 1, len(self.replies) - 1)]


def test_one_repair_round_fixes_a_wrong_column_name(registry, graph, router):
    orders = alias_for(registry, "orders")
    client = RepairingClient(
        broken=plan_for(f"SELECT cust_name, amount FROM {orders}"),
        fixed=plan_for(f"SELECT cust_ref, amount FROM {orders}"),
    )

    trace, result = run("show customer refs and amounts", registry, graph, router, client)

    assert client.calls == 2, "expected exactly one repair call"
    assert result.clarification is None
    assert len(result.answer_rows) == 6

    (repair,) = [e for e in trace if e.stage == "repair"]
    assert 'unknown column "cust_name"' in repair.message
    assert "cust_ref" in repair.message


def test_two_failures_refuse_instead_of_returning_broken_sql(registry, graph, router):
    orders = alias_for(registry, "orders")
    client = RepairingClient(
        broken=plan_for(f"SELECT nope FROM {orders}"),
        fixed=plan_for(f"SELECT still_nope FROM {orders}"),
    )

    _, result = run("show me nope", registry, graph, router, client)

    assert client.calls == 2, "the model must not be retried more than once"
    assert result.clarification is not None
    assert result.sql == ""


# --------------------------------------------------------------------------- #
# Degradation
# --------------------------------------------------------------------------- #


def test_broken_evidence_sql_degrades_but_keeps_the_answer(registry, graph, router):
    orders = alias_for(registry, "orders")
    client = FakeLLMClient(
        default=plan_for(
            f"SELECT count(*) AS n FROM {orders}",
            evidence_sql="SELECT * FROM a_table_that_does_not_exist",
        )
    )

    trace, result = run("how many orders", registry, graph, router, client)

    assert result.answer_rows == [{"n": 6}]
    assert result.evidence_rows == []
    assert any("Could not fetch the rows behind" in e.message for e in trace)


def test_question_with_no_tables_uploaded_asks_for_a_file(graph, router):
    empty_registry, empty_graph, empty_router = load_session({})
    _, result = run("anything", empty_registry, empty_graph, empty_router, FakeLLMClient())
    assert "Upload a file" in (result.clarification or "")


# --------------------------------------------------------------------------- #
# Coercing a small model's sloppy JSON
#
# Every case here reached a user as a refused answer. A 7B writes JSON nulls as
# prose and lists as Python reprs, and the planner is the boundary that has to
# absorb that -- an unparsed "None" is truthy, so a plan carrying perfectly good
# SQL was reported as a refusal.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "spelled_null", ["None", "none", "null", "NULL", "nil", "N/A", "na", "false", "-", ""]
)
def test_spelled_out_nulls_are_not_treated_as_a_clarification(spelled_null):
    plan = _coerce_plan(
        {"answer_sql": "SELECT 1", "clarification_needed": spelled_null}
    )
    assert plan.clarification_needed is None
    assert plan.answer_sql == "SELECT 1"


def test_a_real_clarification_still_survives():
    plan = _coerce_plan({"answer_sql": "", "clarification_needed": "Which file has salaries?"})
    assert plan.clarification_needed == "Which file has salaries?"


def test_list_fields_serialised_as_python_reprs_are_parsed():
    plan = _coerce_plan(
        {
            "answer_sql": "SELECT 1",
            "tables_used": "['orders', 'customers']",
            "followups": "['a', 'b', 'c', 'd']",
        }
    )
    assert plan.tables_used == ["orders", "customers"]
    assert plan.followups == ["a", "b", "c"]  # still capped at three


def test_a_bare_string_table_becomes_a_single_entry():
    assert _coerce_plan({"answer_sql": "SELECT 1", "tables_used": "orders"}).tables_used == [
        "orders"
    ]


def test_evidence_sql_of_none_is_dropped():
    assert _coerce_plan({"answer_sql": "SELECT 1", "evidence_sql": "None"}).evidence_sql is None


def test_the_sales_per_region_reply_that_was_wrongly_refused(registry, graph, router):
    """The exact shape the model returned for "give me sales per region?"."""
    orders, customers = alias_for(registry, "orders"), alias_for(registry, "customers")
    client = FakeLLMClient(
        default={
            "reasoning": "Join orders to customers and sum by region.",
            "tables_used": f"['{orders}', '{customers}']",
            "answer_sql": (
                f"SELECT c.region, SUM(o.amount) AS total_sales FROM {orders} o "
                f"JOIN {customers} c ON o.cust_ref = c.code GROUP BY c.region"
            ),
            "evidence_sql": "None",
            "chart_intent": "breakdown",
            "followups": "['Which region is highest?']",
            "clarification_needed": "None",
        }
    )

    _, result = run("give me sales per region?", registry, graph, router, client)

    assert result.clarification is None, "valid SQL was thrown away as a refusal"
    assert len(result.answer_rows) == 3


def test_empty_sql_refusal_points_at_the_schema_pane(registry, graph, router):
    # A question about what the files contain has no SQL to write. Telling that user
    # to rephrase sends them in circles; the answer is already on screen.
    client = FakeLLMClient(default={"reasoning": "The question asks about contents."})

    _, result = run("what is present in the files?", registry, graph, router, client)

    assert result.clarification is not None
    assert "Files & Schema" in result.clarification


# --------------------------------------------------------------------------- #
# The schema is data too
#
# "What is present in these files?" is an ordinary question. Answering it with
# "look at the sidebar" is a non-answer, and matching phrasings with keywords only
# ever covers the wordings someone thought of. Instead the catalog is registered as
# two real DuckDB tables, so such questions become plain SQL with the same
# validation and the same visible trace as any other.
# --------------------------------------------------------------------------- #


def test_catalog_tables_describe_every_uploaded_table(registry):
    rows = registry.conn.execute(
        "SELECT table_name, source_file, n_rows, n_columns FROM schema_tables ORDER BY 1"
    ).fetchall()
    by_name = {r[0]: r for r in rows}

    assert {"orders", "customers", "headcount"} <= set(by_name)
    assert by_name["orders"][2] == 6  # n_rows matches the real table
    assert by_name["orders"][3] == 4  # n_columns


def test_catalog_columns_carry_types_and_samples(registry):
    rows = registry.conn.execute(
        "SELECT column_name, data_type FROM schema_columns WHERE table_name = 'orders'"
    ).fetchall()
    kinds = dict(rows)

    assert kinds["order_date"] == "date"
    assert kinds["amount"] == "numeric"
    assert kinds["cust_ref"] == "id"


def test_catalog_is_queryable_end_to_end(registry, graph, router):
    client = FakeLLMClient(
        default=plan_for("SELECT table_name, n_rows FROM schema_tables ORDER BY n_rows DESC")
    )
    _, result = run("what is present in these files?", registry, graph, router, client)

    assert result.clarification is None, "a schema question was refused"
    assert len(result.answer_rows) == 3
    assert result.answer_rows[0]["table_name"] == "orders"


def test_schema_questions_route_to_the_catalog(registry, graph, router):
    for question in [
        "what is present in the files?",
        "what columns are there?",
        "which table has the most rows?",
        "describe the data",
    ]:
        chosen = [registry.alias_of(t) for t in router.route(question, graph).table_ids]
        assert any(a.startswith("schema_") for a in chosen), f"{question!r} -> {chosen}"


def test_data_questions_still_route_to_real_tables(registry, graph, router):
    # The catalog must not crowd out the data it describes.
    for question in ["total revenue by region", "average salary by department"]:
        chosen = [registry.alias_of(t) for t in router.route(question, graph).table_ids]
        assert any(not a.startswith("schema_") for a in chosen), f"{question!r} -> {chosen}"


def test_catalog_is_not_shown_as_an_uploaded_file(registry):
    # It is queryable, but it is not a file the user gave us.
    assert all(p.kind == "data" for p in registry.profiles.values() if "schema_" not in p.alias)
    assert "schema_tables" not in registry.data_table_ids
    assert "schema_tables" in registry.table_ids


def test_the_planner_is_told_not_to_extract_the_month():
    """EXTRACT(MONTH) discards the year, which is silently wrong across two years.

    The eval caught this as a formatting difference -- month numbers 1..12 where dates
    were expected -- but the totals matched only because that fixture covers a single
    year. On real data January 2024 and January 2025 would collapse into one row.
    """
    from darwinbox.llm.prompts import PLANNER_SYSTEM

    assert "date_trunc" in PLANNER_SYSTEM
    assert "EXTRACT(MONTH" in PLANNER_SYSTEM


def test_month_truncation_keeps_years_apart(registry):
    # The behaviour the rule protects, asserted against DuckDB rather than the prompt.
    from darwinbox.execute.runner import execute
    from darwinbox.execute.validator import validate

    orders = alias_for(registry, "orders")
    truncated = validate(
        f"SELECT date_trunc('month', order_date) AS m, count(*) AS n "
        f"FROM {orders} GROUP BY m ORDER BY m",
        registry,
    )
    extracted = validate(
        f"SELECT EXTRACT(MONTH FROM order_date) AS m, count(*) AS n "
        f"FROM {orders} GROUP BY m ORDER BY m",
        registry,
    )
    # Same here (one year of data), but only date_trunc carries the year at all.
    rows = execute(registry, truncated.sql).rows
    assert all("-" in str(r["m"]) for r in rows), rows
    assert all(isinstance(r["m"], (int, float)) for r in execute(registry, extracted.sql).rows)
