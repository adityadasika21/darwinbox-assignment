"""Phase 4 acceptance: validation, execution and chart selection (SPEC 8.5-8.7)."""

from __future__ import annotations

import pytest
from conftest import alias_for
from darwinbox.execute.charts import select_chart
from darwinbox.execute.runner import QueryExecutionError, execute
from darwinbox.execute.validator import ValidationError, validate

# --------------------------------------------------------------------------- #
# Validator: the guards
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE orders",
        "DELETE FROM orders",
        "UPDATE orders SET amount = 0",
        "INSERT INTO orders VALUES (1)",
        "CREATE TABLE x AS SELECT 1",
        "ATTACH 'evil.db'",
        "COPY orders TO 'out.csv'",
        "PRAGMA database_list",
        "INSTALL httpfs",
    ],
)
def test_write_statements_are_rejected(sql, registry):
    with pytest.raises(ValidationError) as exc:
        validate(sql, registry)
    assert exc.value.code in {"WRITE_STATEMENT", "NOT_A_SELECT", "PARSE_ERROR"}


def test_multiple_statements_are_rejected(registry):
    orders = alias_for(registry, "orders")
    with pytest.raises(ValidationError) as exc:
        validate(f"SELECT 1 FROM {orders}; SELECT 2 FROM {orders}", registry)
    assert exc.value.code == "MULTIPLE_STATEMENTS"


def test_unknown_table_is_rejected_with_the_available_names(registry):
    with pytest.raises(ValidationError) as exc:
        validate("SELECT * FROM invoices", registry)
    assert exc.value.code == "TABLE_NOT_FOUND"
    assert alias_for(registry, "orders") in exc.value.message


def test_unknown_column_is_rejected_and_suggests_a_real_one(registry):
    orders = alias_for(registry, "orders")
    with pytest.raises(ValidationError) as exc:
        validate(f"SELECT cust_name FROM {orders}", registry)
    assert exc.value.code == "COLUMN_NOT_FOUND"
    assert "cust_name" in exc.value.message


def test_empty_sql_is_rejected(registry):
    with pytest.raises(ValidationError) as exc:
        validate("   ", registry)
    assert exc.value.code == "EMPTY_SQL"


# --------------------------------------------------------------------------- #
# Validator: alias rewriting
# --------------------------------------------------------------------------- #


def test_aliases_are_rewritten_to_physical_table_ids(registry):
    orders = alias_for(registry, "orders")
    validated = validate(f"SELECT count(*) AS n FROM {orders}", registry)

    (table_id,) = validated.tables
    assert table_id == "orders__csv__b0"
    assert f'"{table_id}"' in validated.sql
    assert f"FROM {orders} " not in validated.sql + " "  # the bare alias is gone


def test_a_real_join_across_two_files_validates(registry):
    orders = alias_for(registry, "orders")
    customers = alias_for(registry, "customers")
    validated = validate(
        f"SELECT c.region, sum(o.amount) AS total "
        f"FROM {orders} o JOIN {customers} c ON o.cust_ref = c.code "
        f"GROUP BY c.region",
        registry,
    )
    assert len(validated.tables) == 2


def test_cte_names_are_not_treated_as_missing_tables(registry):
    orders = alias_for(registry, "orders")
    validated = validate(
        f"WITH monthly AS (SELECT amount FROM {orders}) SELECT sum(amount) FROM monthly",
        registry,
    )
    assert len(validated.tables) == 1


def test_select_alias_may_be_used_in_order_by(registry):
    orders = alias_for(registry, "orders")
    validate(
        f"SELECT cust_ref, sum(amount) AS total FROM {orders} "
        f"GROUP BY cust_ref ORDER BY total DESC",
        registry,
    )


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def test_execution_returns_json_ready_rows(registry):
    orders = alias_for(registry, "orders")
    validated = validate(
        f"SELECT order_id, order_date, amount FROM {orders} ORDER BY order_id", registry
    )
    result = execute(registry, validated.sql)

    assert result.columns == ["order_id", "order_date", "amount"]
    assert len(result.rows) == 6
    # Dates must be ISO strings, not datetime objects, or JSON serialisation fails.
    assert isinstance(result.rows[0]["order_date"], str)
    assert result.rows[0]["order_date"].startswith("2024-01-05")


def test_row_cap_marks_truncation(registry):
    orders = alias_for(registry, "orders")
    validated = validate(f"SELECT * FROM {orders}", registry)
    result = execute(registry, validated.sql, limit=2)

    assert len(result.rows) == 2
    assert result.truncated is True


def test_execution_error_is_typed(registry):
    with pytest.raises(QueryExecutionError):
        execute(registry, "SELECT * FROM nonexistent_table_xyz")


def test_cross_file_join_produces_the_right_numbers(registry):
    orders = alias_for(registry, "orders")
    customers = alias_for(registry, "customers")
    validated = validate(
        f"SELECT c.region, round(sum(o.amount), 2) AS total "
        f"FROM {orders} o JOIN {customers} c ON o.cust_ref = c.code "
        f"GROUP BY c.region ORDER BY c.region",
        registry,
    )
    rows = execute(registry, validated.sql).rows

    totals = {r["region"]: r["total"] for r in rows}
    assert totals == {"East": 310.00, "North": 266.50, "South": 650.00}


# --------------------------------------------------------------------------- #
# Charts: shape decides, intent only breaks ties
# --------------------------------------------------------------------------- #


def test_single_numeric_cell_is_a_big_number():
    chart = select_chart(["total"], [{"total": 48210.5}], "single_value")
    assert chart.type == "big_number"
    assert chart.y == "total"


def test_date_plus_numeric_is_a_line():
    rows = [{"month": "2024-01-01", "revenue": 10}, {"month": "2024-02-01", "revenue": 20}]
    chart = select_chart(["month", "revenue"], rows, "trend_over_time")
    assert chart.type == "line"
    assert chart.x == "month"


def test_category_plus_numeric_is_a_bar():
    rows = [{"region": "North", "total": 10}, {"region": "South", "total": 20}]
    chart = select_chart(["region", "total"], rows, "comparison")
    assert chart.type == "bar"
    assert chart.x == "region"


def test_high_cardinality_category_falls_back_to_a_table():
    rows = [{"id": f"K-{i}", "v": i} for i in range(40)]
    chart = select_chart(["id", "v"], rows, "comparison")
    assert chart.type == "table"


def test_two_numerics_over_many_rows_is_a_scatter():
    rows = [{"x": i, "y": i * 2} for i in range(30)]
    chart = select_chart(["x", "y"], rows, "distribution")
    assert chart.type == "scatter"


def test_intent_none_forces_a_table():
    rows = [{"month": "2024-01-01", "revenue": 10}, {"month": "2024-02-01", "revenue": 20}]
    assert select_chart(["month", "revenue"], rows, "none").type == "table"


def test_wrong_intent_cannot_override_the_result_shape():
    # The model says "trend", but two categories and a number are a bar chart.
    rows = [{"region": "North", "total": 10}, {"region": "South", "total": 20}]
    assert select_chart(["region", "total"], rows, "trend_over_time").type == "bar"


def test_columns_qualified_by_table_name_survive_the_rewrite(registry):
    """Rewriting a table must not orphan columns qualified by its original name.

    "orders.cust_ref = customers.code" is how a model naturally writes a join. Those
    qualifiers are column nodes, not table nodes, so rewriting only the table left
    every one of them dangling and DuckDB rejected perfectly valid SQL with
    'Referenced table "orders" not found'.
    """
    orders, customers = alias_for(registry, "orders"), alias_for(registry, "customers")
    validated = validate(
        f"SELECT sum({orders}.amount) AS total FROM {orders} "
        f"JOIN {customers} ON {orders}.cust_ref = {customers}.code "
        f"WHERE {customers}.region = 'North'",
        registry,
    )
    rows = execute(registry, validated.sql).rows
    assert rows == [{"total": 266.5}]  # matches the cross-file join test above


def test_an_explicit_alias_is_left_alone(registry):
    orders = alias_for(registry, "orders")
    validated = validate(f"SELECT o.amount FROM {orders} o WHERE o.amount > 100", registry)
    assert " AS o" in validated.sql or " o" in validated.sql
    assert len(execute(registry, validated.sql).rows) == 4


# --------------------------------------------------------------------------- #
# Result summaries
#
# A grid of numbers is data, not an answer. Every response leads with a sentence,
# and the fallback below cannot be wrong because it only restates what came back.
# --------------------------------------------------------------------------- #

from darwinbox.execute.summarize import describe, summarize  # noqa: E402


def test_single_value_reads_as_a_statement():
    assert describe(["total_revenue"], [{"total_revenue": 390157.01}]) == (
        "Total revenue: 390,157.01."
    )


def test_grouped_numbers_name_the_extremes():
    rows = [
        {"region": "North", "sales": 79772.7},
        {"region": "South", "sales": 109376.52},
        {"region": "East", "sales": 98726.54},
    ]
    text = describe(["region", "sales"], rows)
    assert "South" in text and "109,376.52" in text  # highest
    assert "North" in text and "79,772.7" in text  # lowest


def test_a_list_names_a_few_and_counts_the_rest():
    rows = [{"name": f"Person {i}"} for i in range(10)]
    text = describe(["name"], rows)
    assert "10 rows" in text and "Person 0" in text and "7 more" in text


def test_truncation_is_marked():
    rows = [{"a": "x", "n": i} for i in range(5000)]
    assert "5,000+" in describe(["a", "n"], rows, truncated=True)


def test_no_rows_says_so():
    assert describe(["a"], []) == "That query returned no rows."


def test_model_summary_is_used_when_available():
    from darwinbox.llm.client import FakeLLMClient

    client = FakeLLMClient(default={"summary": "Sales are highest in the South."})
    text, from_model = summarize(client, "sales per region", ["region", "sales"],
                                 [{"region": "South", "sales": 1}])
    assert text == "Sales are highest in the South."
    assert from_model is True


def test_a_broken_model_falls_back_instead_of_failing_the_answer():
    class Broken:
        def complete_json(self, system, user, max_tokens=700):
            raise RuntimeError("model is down")

    text, from_model = summarize(Broken(), "q", ["total"], [{"total": 42}])
    assert from_model is False
    assert text == "Total: 42."


def test_an_empty_model_reply_falls_back():
    from darwinbox.llm.client import FakeLLMClient

    text, from_model = summarize(FakeLLMClient(default={"summary": "   "}), "q",
                                 ["total"], [{"total": 42}])
    assert from_model is False
    assert "42" in text


def test_summary_sentence_is_punctuated():
    from darwinbox.llm.client import FakeLLMClient

    text, _ = summarize(FakeLLMClient(default={"summary": "no full stop here"}), "q",
                        ["a"], [{"a": 1}])
    assert text.endswith(".")


def test_superlatives_are_computed_not_left_to_the_model():
    """The model must never be the thing that decides which row is highest.

    Asked to rank four regions itself, a 7B wrote "North had the highest sales with
    $79,772.70" when North was the lowest of the four -- a confidently wrong sentence
    sitting directly above a chart showing the opposite. Ranking is arithmetic.
    """
    from darwinbox.execute.summarize import computed_facts

    rows = [
        {"region": "North", "total_sales": 79772.70},
        {"region": "South", "total_sales": 109376.52},
        {"region": "East", "total_sales": 98726.54},
        {"region": "West", "total_sales": 102281.25},
    ]
    facts = " ".join(computed_facts(["region", "total_sales"], rows))

    assert "highest is South" in facts
    assert "lowest is North" in facts
    assert "390,157.01" in facts  # total, so the model never has to add up


def test_facts_reach_the_model_prompt():
    from darwinbox.execute.summarize import summarize
    from darwinbox.llm.client import FakeLLMClient

    client = FakeLLMClient(default={"summary": "ok"})
    summarize(client, "sales per region", ["region", "sales"],
              [{"region": "A", "sales": 1}, {"region": "B", "sales": 9}])

    _, user = client.calls[0]
    assert "FACTS (computed, authoritative)" in user
    assert "highest is B" in user


# --------------------------------------------------------------------------- #
# Grounding: the label has to mean something
#
# Asked for "the average temperature for each customer region" against files holding
# no temperature, the planner wrote AVG(rainfall_mm) AS avg_temperature and the answer
# read "The average temperature ... North (244.5)". The join was real, the SQL valid,
# the arithmetic right. The lie was in the column name.
# --------------------------------------------------------------------------- #

from darwinbox.execute.grounding import caveat, ungrounded_labels  # noqa: E402

WEATHER = ["station_code", "city", "rainfall_mm"]
CUSTOMERS = ["customer_id", "region", "city"]


def test_an_invented_measure_is_reported():
    labels = ungrounded_labels(
        "SELECT c.region, AVG(w.rainfall_mm) AS avg_temperature "
        "FROM customers c JOIN weather w ON c.city = w.city GROUP BY c.region",
        [*WEATHER, *CUSTOMERS],
        ["weather", "customers"],
    )
    assert labels == {"avg_temperature": "rainfall_mm"}
    assert "rainfall_mm" in caveat(labels)


def test_a_faithful_alias_is_not_reported():
    labels = ungrounded_labels(
        "SELECT region, sum(rainfall_mm) AS total_rainfall_mm FROM weather GROUP BY region",
        [*WEATHER, *CUSTOMERS],
        ["weather", "customers"],
    )
    assert labels == {}


def test_an_aggregation_word_is_not_an_invention():
    """"total", "average" and friends describe the maths, not a thing in the data."""
    labels = ungrounded_labels(
        "SELECT city, avg(rainfall_mm) AS average_rainfall_mm FROM weather GROUP BY city",
        WEATHER,
        ["weather"],
    )
    assert labels == {}


def test_a_synonym_is_reported_rather_than_refused():
    """The eval's own ground truth writes sum(amount) AS total_revenue.

    A synonym and an invention are the same shape, so this reports instead of blocking
    -- refusing here would reject a perfectly good answer.
    """
    labels = ungrounded_labels(
        "SELECT sum(amount) AS total_revenue FROM orders", ["amount", "order_id"], ["orders"]
    )
    assert labels == {"total_revenue": "amount"}


def test_count_star_invents_nothing():
    assert ungrounded_labels("SELECT count(*) AS n FROM weather", WEATHER, ["weather"]) == {}


def test_unparseable_sql_never_raises():
    assert ungrounded_labels("this is not sql at all", WEATHER, ["weather"]) == {}


def test_no_caveat_when_everything_is_grounded():
    assert caveat({}) == ""


def test_several_relabelled_columns_are_all_named():
    text = caveat({"avg_temperature": "rainfall_mm", "total_profit": "amount"})
    assert "rainfall_mm" in text and "amount" in text and " and " in text
