"""Regenerate questions.yaml and holdout.yaml with ground truth from the clean seed.

Ground truth is computed by running ``truth_sql`` against eval/fixtures/clean, then
frozen into ``expect_rows``. The questions themselves are asked against the
*corrupted* fixtures, so a passing answer means the messy-data layer worked.

    python eval/corrupt.py --out eval/fixtures && python eval/make_questions.py
"""

from __future__ import annotations

import pathlib

import duckdb
import yaml

CLEAN = pathlib.Path("eval/fixtures/clean")
HERE = pathlib.Path("eval")

# (id, fixture set, question, kind, SQL over the clean source)
QUESTIONS = [
    ("q01", "sales", "What is the total order amount across all orders?",
     "single-table aggregate",
     "SELECT round(sum(amount),2) AS total_revenue FROM orders"),
    ("q02", "sales", "How many employees work in the Sales department?",
     "filter + aggregate",
     "SELECT count(*) AS n FROM employees WHERE department='Sales'"),
    ("q03", "sales", "What is the total order amount for each customer region?",
     "cross-file join",
     "SELECT c.region, round(sum(o.amount),2) AS revenue FROM orders o "
     "JOIN customers c ON o.customer_id=c.customer_id GROUP BY 1 ORDER BY 1"),
    ("q04", "sales", "What is the average employee salary?", "single-table aggregate",
     "SELECT round(avg(salary),2) AS avg_salary FROM employees"),
    ("q05", "sales", "What is the average salary by department?", "group-by aggregate",
     "SELECT department, round(avg(salary),2) AS avg_salary FROM employees GROUP BY 1 ORDER BY 1"),
    ("q06", "sales", "Which 3 customers have the highest total order amount?", "top-N + join",
     "SELECT c.customer_name, round(sum(o.amount),2) AS revenue FROM orders o "
     "JOIN customers c ON o.customer_id=c.customer_id GROUP BY 1 ORDER BY 2 DESC LIMIT 3"),
    ("q07", "sales", "Show total order amount by month", "trend over time",
     "SELECT strftime(date_trunc('month', CAST(order_date AS DATE)), '%Y-%m-%d') AS month, "
     "round(sum(amount),2) AS revenue FROM orders GROUP BY 1 ORDER BY 1"),
    ("q08", "sales", "What is the total order amount handled by each employee department?",
     "three-file join",
     "SELECT e.department, round(sum(o.amount),2) AS revenue FROM orders o "
     "JOIN employees e ON o.employee_id=e.employee_id GROUP BY 1 ORDER BY 1"),
    ("q09", "sales", "How many customers have never placed an order?", "anti-join",
     "SELECT count(*) AS n FROM customers c WHERE NOT EXISTS "
     "(SELECT 1 FROM orders o WHERE o.customer_id=c.customer_id)"),
    ("q11", "composite", "Compare total revenue against total target for each region",
     "composite key",
     "SELECT s.region, round(sum(s.revenue),2) AS revenue, "
     "round(sum(t.target_revenue),2) AS target "
     "FROM sales_by_region s JOIN targets t ON s.month=t.month AND s.region=t.region "
     "GROUP BY 1 ORDER BY 1"),
    ("q12", "attendance", "What are the average hours worked per department?", "cross-file join",
     "SELECT e.department, round(avg(a.hours),2) AS avg_hours FROM attendance a "
     "JOIN employees e ON a.employee_id=e.employee_id GROUP BY 1 ORDER BY 1"),
    ("q13", "hr", "What is the total gross pay and total deductions?", "single-table aggregate",
     "SELECT round(sum(gross_pay),2) AS gross, round(sum(deductions),2) AS deductions "
     "FROM payroll"),
    ("q15", "hr", "What is the total gross pay by department?", "noisy-key join",
     "SELECT e.department, round(sum(p.gross_pay),2) AS gross FROM payroll p "
     "JOIN employees e ON p.employee_id=e.employee_id GROUP BY 1 ORDER BY 1"),
    ("q16", "sales", "How many customers are in each region?", "group-by aggregate",
     "SELECT region, count(*) AS n FROM customers GROUP BY 1 ORDER BY 1"),
    ("q17", "sales", "What is the largest single order amount?", "single-value",
     "SELECT round(max(amount),2) AS biggest FROM orders"),
    ("q18", "sales", "Which region has the highest average order amount?",
     "comparison of groups",
     "SELECT c.region, round(avg(o.amount),2) AS avg_order FROM orders o "
     "JOIN customers c ON o.customer_id=c.customer_id GROUP BY 1 ORDER BY 2 DESC LIMIT 1"),
    ("q19", "messy", "How many products are in each category?", "messy-file aggregate",
     "SELECT category, count(*) AS n FROM products GROUP BY 1 ORDER BY 1"),
    # The question asks to compare two figures, so ground truth must carry both.
    ("q20", "granularity", "Compare the monthly order total against the monthly budget",
     "derived granularity",
     "WITH m AS (SELECT date_trunc('month', CAST(order_date AS DATE)) AS month, "
     "round(sum(amount),2) AS revenue FROM orders GROUP BY 1) "
     "SELECT strftime(m.month, '%Y-%m-%d') AS month, m.revenue, b.budget "
     "FROM m JOIN read_csv_auto('eval/fixtures/monthly_budget.csv') b "
     "ON m.month = CAST(b.budget_month AS DATE) ORDER BY 1"),
]

HOLDOUT = [
    ("h01", "sales", "What was the total order amount in the first quarter of 2024?",
     "filter + aggregate",
     "SELECT round(sum(amount),2) AS q1_revenue FROM orders "
     "WHERE CAST(order_date AS DATE) < DATE '2024-04-01'"),
    ("h02", "sales", "How many employees are in each region?", "group-by aggregate",
     "SELECT region, count(DISTINCT employee_id) AS n FROM employees GROUP BY 1 ORDER BY 1"),
    ("h03", "sales", "Which single product has sold the most units?", "top-N + join",
     "SELECT p.product_name, round(sum(oi.quantity),0) AS units FROM order_items oi "
     "JOIN products p ON oi.product_id=p.product_id GROUP BY 1 ORDER BY 2 DESC LIMIT 1"),
    ("h04", "composite", "What is the average monthly target for the North region?",
     "filter + aggregate",
     "SELECT round(avg(target_revenue),2) AS avg_target FROM targets WHERE region='North'"),
    ("h05", "hr", "What is the average net pay by department?", "noisy-key join",
     "SELECT e.department, round(avg(p.gross_pay - p.deductions),2) AS net FROM payroll p "
     "JOIN employees e ON p.employee_id=e.employee_id GROUP BY 1 ORDER BY 1"),
]

# The two required refusals. Neither has an answer to find: one spans unrelated
# files, the other names a column that does not exist anywhere.
REFUSALS = [
    (13, {"id": "q10", "fixture": "unrelated", "kind": "refusal (unrelated files)",
          "q": "Compare the average rainfall to the total headcount by department",
          "expect": "clarify"}),
    (99, {"id": "q14", "fixture": "sales", "kind": "refusal (no such column)",
          "q": "What is the total commission paid to each salesperson?",
          "expect": "clarify"}),
]


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    for path in sorted(CLEAN.glob("*.csv")):
        con.execute(f"CREATE TABLE {path.stem} AS SELECT * FROM read_csv_auto('{path}')")
    return con


def build(con, specs: list[tuple]) -> list[dict]:
    out = []
    for qid, fixture, question, kind, sql in specs:
        rows = [
            [float(v) if isinstance(v, (int, float)) else str(v) for v in row]
            for row in con.execute(sql).fetchall()
        ]
        out.append({
            "id": qid, "fixture": fixture, "q": question, "kind": kind,
            "truth_sql": sql, "expect_rows": rows, "tolerance": 0.01,
        })
    return out


def main() -> int:
    con = connect()

    questions = build(con, QUESTIONS)
    for index, refusal in REFUSALS:
        questions.insert(min(index, len(questions)), refusal)

    (HERE / "questions.yaml").write_text(
        "# Ground truth is computed from eval/fixtures/clean by truth_sql, then frozen\n"
        "# into expect_rows. Questions are asked against the CORRUPTED fixtures, so a\n"
        "# passing answer means the messy-data layer did its job.\n"
        "# Regenerate with: python eval/make_questions.py\n\n"
        + yaml.safe_dump(questions, sort_keys=False, width=100, allow_unicode=True)
    )
    (HERE / "holdout.yaml").write_text(
        "# HOLDOUT - deliberately not looked at while tuning. Same format as questions.yaml.\n\n"
        + yaml.safe_dump(build(con, HOLDOUT), sort_keys=False, width=100, allow_unicode=True)
    )

    print(f"wrote {len(questions)} questions and {len(HOLDOUT)} holdout questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
