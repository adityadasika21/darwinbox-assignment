"""Generate a QA set over data nobody here chose.

The tuning and holdout questions are mine: I wrote the questions *and* the fixtures
they run against, so 100% on them says the system handles the cases I thought of. That
is worth measuring and it is not external validity.

This builds questions over the 42 real open-data CSVs fetched by `fetch_gov.py --wide`
-- restaurant inspections, traffic crashes, library circulation, salaries -- whose
schemas, column names, null conventions and value distributions were chosen by the
agencies that published them. Ground truth is computed by executing SQL against the
ingested tables, so no expected answer is written by hand either.

What remains mine is the *shape* of the questions: the templates below. That is a real
limit and worth stating plainly. What the templates cannot do is know in advance which
column is a category, which is a measure, or what the answer is -- all of that comes
from the profiler and from the data.

    python eval/fetch_gov.py --wide          # once, needs network
    python eval/make_external_questions.py   # writes eval/external.yaml
    python eval/run_eval.py --stage qa --external --repeat 3
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

from darwinbox.llm.client import FakeLLMClient  # noqa: E402
from darwinbox.session import SessionStore  # noqa: E402

GOV = pathlib.Path("eval/fixtures/gov")
OUT = pathlib.Path("eval/external.yaml")
# Enough tables for variety, few enough that three repeats stay under ten minutes.
MAX_TABLES = 8
MIN_ROWS = 40
# A category has to have few enough values to group by and more than one to compare.
MAX_CATEGORY_DISTINCT = 25
MIN_CATEGORY_DISTINCT = 2


def usable_columns(profile) -> tuple[list, list]:
    """Split a table's columns into groupable categories and summable measures."""
    categories, measures = [], []
    for col in profile.columns:
        if col.null_rate > 0.2:
            continue
        if col.semantic_type == "numeric" and not col.is_unique:
            measures.append(col)
        elif col.semantic_type in {"category", "text"} and (
            MIN_CATEGORY_DISTINCT <= col.n_distinct <= MAX_CATEGORY_DISTINCT
        ):
            categories.append(col)
    return categories, measures


def questions_for(alias: str, table_id: str, profile) -> list[dict]:
    """Mechanically derive questions this table can actually answer."""
    categories, measures = usable_columns(profile)
    out: list[dict] = []
    quoted = f'"{table_id}"'

    out.append({
        "q": f"How many rows are in {alias}?",
        "kind": "row count",
        "sql": f"SELECT count(*) AS n FROM {quoted}",
    })

    if categories:
        cat = categories[0]
        out.append({
            "q": f"How many distinct values of {cat.name} are there in {alias}?",
            "kind": "distinct count",
            "sql": f'SELECT count(DISTINCT "{cat.name}") AS n FROM {quoted}',
        })
        out.append({
            "q": f"How many rows are there in {alias} for each {cat.name}?",
            "kind": "group by",
            "sql": f'SELECT "{cat.name}" AS k, count(*) AS n FROM {quoted} '
                   f'WHERE "{cat.name}" IS NOT NULL GROUP BY 1 ORDER BY 1',
        })

    if measures:
        m = measures[0]
        out.append({
            "q": f"What is the average {m.name} in {alias}?",
            "kind": "aggregate",
            "sql": f'SELECT round(avg("{m.name}"), 2) AS v FROM {quoted}',
        })

    if categories and measures:
        cat, m = categories[0], measures[0]
        out.append({
            "q": f"What is the total {m.name} by {cat.name} in {alias}?",
            "kind": "group by + aggregate",
            "sql": f'SELECT "{cat.name}" AS k, round(sum("{m.name}"), 2) AS v FROM {quoted} '
                   f'WHERE "{cat.name}" IS NOT NULL GROUP BY 1 ORDER BY 1',
        })
        out.append({
            "q": f"Which {cat.name} has the highest total {m.name} in {alias}?",
            "kind": "superlative",
            # The measure is part of the expected answer: naming the winner without
            # the figure answers less of the question, and the comparison is exact on
            # shape, so the ground truth has to be the better answer rather than the
            # minimal one.
            "sql": f'SELECT "{cat.name}" AS k, round(sum("{m.name}"), 2) AS v FROM {quoted} '
                   f'WHERE "{cat.name}" IS NOT NULL GROUP BY 1 '
                   f'ORDER BY 2 DESC NULLS LAST LIMIT 1',
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=OUT)
    parser.add_argument("--max-tables", type=int, default=MAX_TABLES)
    args = parser.parse_args()

    files = sorted(GOV.glob("*.csv"))
    if not files:
        print(f"no CSVs in {GOV} -- run: python eval/fetch_gov.py --wide")
        return 1

    session = SessionStore(lambda: FakeLLMClient()).create()
    session.add_files([(f.name, f.read_bytes()) for f in files])

    # Biggest tables first: more rows means the aggregates are less likely to be
    # degenerate, and a question whose answer is 0 tests nothing.
    profiles = sorted(
        (p for p in session.tables if p.n_rows >= MIN_ROWS),
        key=lambda p: p.n_rows,
        reverse=True,
    )[: args.max_tables]

    written: list[dict] = []
    for profile in profiles:
        for spec in questions_for(profile.alias, profile.table_id, profile):
            try:
                rows = session.conn.execute(spec["sql"]).fetchall()
            except Exception as exc:  # noqa: BLE001 - a bad template must name itself
                print(f"  skip  {spec['q'][:60]}: {exc}")
                continue
            if not rows or rows[0][0] is None:
                continue
            written.append({
                "id": f"x{len(written) + 1:02d}",
                "files": [f"gov/{f.name}" for f in files if f.stem == profile.alias.split("__")[0]]
                         or [f"gov/{profile.alias}.csv"],
                "q": spec["q"],
                "kind": spec["kind"],
                "truth_sql": spec["sql"],
                "expect_rows": [[_plain(v) for v in row] for row in rows],
                "tolerance": 0.01,
            })

    args.out.write_text(
        "# Generated by eval/make_external_questions.py over real open-data CSVs.\n"
        "# Questions are templated; the data, the columns chosen and every expected\n"
        "# answer come from the files themselves. Do not edit by hand.\n"
        + yaml.safe_dump(written, sort_keys=False, allow_unicode=True)
    )
    print(f"{len(written)} questions over {len(profiles)} tables -> {args.out}")
    return 0


def _plain(value):
    """YAML-safe scalars: Decimal and date objects round-trip badly."""
    if isinstance(value, (int, float, str)) or value is None:
        return value
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
