"""Ask questions from a terminal, without the API or the web app.

    python -m darwinbox.cli eval/fixtures/*.csv -q "total revenue by region"
    python -m darwinbox.cli data/*.xlsx --schema

Checkpoint 5 of the build order is verified through this: the whole pipeline runs
end to end with no HTTP layer in the way.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from darwinbox.llm.client import client_from_env
from darwinbox.models import QueryResult, TraceEvent
from darwinbox.session import Session

BOLD, DIM, GREEN, YELLOW, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[0m"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="darwinbox", description=__doc__)
    parser.add_argument("files", nargs="+", type=pathlib.Path)
    parser.add_argument("-q", "--question", action="append", default=[])
    parser.add_argument("--schema", action="store_true", help="print the discovered schema")
    parser.add_argument("--sql", action="store_true", help="print the generated SQL")
    args = parser.parse_args(argv)

    session = Session(client_from_env())
    payload = [(p.name, p.read_bytes()) for p in args.files]
    outcome = session.add_files(payload)

    print(f"{BOLD}Loaded {len(outcome.tables)} table(s) from {len(args.files)} file(s){RESET}")
    for warning in outcome.warnings:
        print(f"  {YELLOW}warning: {warning}{RESET}")

    if args.schema or not args.question:
        _print_schema(session)

    for question in args.question:
        _ask(session, question, show_sql=args.sql)

    return 0


def _print_schema(session: Session) -> None:
    print(f"\n{BOLD}Tables{RESET}")
    for profile in session.tables:
        where = profile.source.human_location()
        print(f"  {profile.alias:<20} {profile.n_rows:>7,} rows   {where}")
        for column in profile.columns:
            samples = ", ".join(column.samples[:3])
            print(f"      {column.name:<24} {column.semantic_type:<9} {DIM}{samples}{RESET}")

    print(f"\n{BOLD}Relationships{RESET}")
    if not session.relationships:
        print(f"  {DIM}none discovered{RESET}")
    for edge in session.relationships:
        mark = {"confirmed": GREEN + "confirmed", "rejected": DIM + "rejected"}.get(
            edge.status, YELLOW + "proposed"
        )
        left = session.registry.alias_of(edge.left_table)
        right = session.registry.alias_of(edge.right_table)
        predicate = " AND ".join(
            f"{left}.{lc} = {right}.{rc}"
            for lc, rc in zip(edge.left_columns, edge.right_columns, strict=True)
        )
        print(f"  [{mark}{RESET}] {predicate}  {DIM}({edge.kind}){RESET}")
        print(f"        {DIM}{edge.evidence}{RESET}")

    components = session.components()
    if len(components) > 1:
        print(f"\n{BOLD}Disconnected groups{RESET} (questions cannot span these)")
        for group in components:
            print("  - " + ", ".join(session.registry.alias_of(t) for t in group))


def _ask(session: Session, question: str, show_sql: bool) -> None:
    print(f"\n{BOLD}? {question}{RESET}")
    for item in session.ask(question):
        if isinstance(item, TraceEvent):
            print(f"  {DIM}[{item.stage:<8} {item.ms:>5}ms] {item.message}{RESET}")
        elif isinstance(item, QueryResult):
            _print_result(item, show_sql)


def _print_result(result: QueryResult, show_sql: bool) -> None:
    if result.clarification:
        print(f"\n  {YELLOW}I need to ask: {result.clarification}{RESET}")
        return

    if show_sql and result.sql:
        print(f"\n{DIM}{result.sql}{RESET}")

    rows, columns = result.answer_rows, result.answer_columns
    if not rows:
        print("  (no rows)")
        return

    widths = {
        c: max(len(str(c)), *(len(str(r.get(c, ""))) for r in rows[:20])) for c in columns
    }
    print("\n  " + "  ".join(str(c).ljust(widths[c]) for c in columns))
    print("  " + "  ".join("-" * widths[c] for c in columns))
    for row in rows[:20]:
        print("  " + "  ".join(str(row.get(c, "")).ljust(widths[c]) for c in columns))
    if len(rows) > 20:
        print(f"  {DIM}... {len(rows) - 20:,} more row(s){RESET}")

    if result.chart:
        print(f"\n  {DIM}chart: {result.chart.type}{RESET}")
    if result.followups:
        print(f"  {DIM}next: {' | '.join(result.followups)}{RESET}")


if __name__ == "__main__":
    sys.exit(main())
