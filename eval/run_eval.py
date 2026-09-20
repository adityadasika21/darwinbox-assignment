"""Three independently runnable eval stages (SPEC 11.3).

    python eval/run_eval.py --stage ingest         # block counts, headers, dtypes. No LLM.
    python eval/run_eval.py --stage relationships  # precision/recall vs the known FK set.
    python eval/run_eval.py --stage qa --repeat 3  # answers vs ground truth, with spread.
    python eval/run_eval.py --stage qa --holdout   # the questions not used for tuning.

The QA stage repeats because a 7B model is not perfectly deterministic even at
temperature 0, so a single number would overstate what the system does.
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

from darwinbox.ingest.blocks import detect_blocks  # noqa: E402
from darwinbox.ingest.loader import load_file  # noqa: E402
from darwinbox.llm.client import client_from_env  # noqa: E402
from darwinbox.models import QueryResult  # noqa: E402
from darwinbox.session import Session  # noqa: E402

FIXTURES = pathlib.Path("eval/fixtures")
EVAL = pathlib.Path("eval")

# Which files make up each named session in questions.yaml.
FIXTURE_SETS: dict[str, list[str]] = {
    "sales": ["orders_renamed.csv", "customers.csv", "employees.csv",
              "clean/products.csv", "clean/order_items.csv"],
    "composite": ["sales_by_region.csv", "targets.csv"],
    "granularity": ["orders_daily.csv", "monthly_budget.csv"],
    "hr": ["payroll_noisy.csv", "employees.csv"],
    "attendance": ["attendance_split.xlsx", "employees.csv"],
    "messy": ["titled_customers.xlsx", "stacked_tables.xlsx", "sales_with_footer.csv"],
    "unrelated": ["weather.csv", "headcount.csv"],
}

# SPEC 11.3 failure taxonomy.
TAXONOMY = [
    "bad_sql", "wrong_table_routed", "wrong_join", "refused_wrongly",
    "answered_wrongly", "timeout",
]

BOLD, DIM, GREEN, RED, YELLOW, RESET = (
    "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[33m", "\033[0m"
)


def ok(passed: bool) -> str:
    return f"{GREEN}pass{RESET}" if passed else f"{RED}FAIL{RESET}"


# --------------------------------------------------------------------------- #
# Stage: ingest
# --------------------------------------------------------------------------- #

# (file, expected block count, expected header rows, note)
INGEST_CASES: list[tuple[str, int, list[int], str]] = [
    ("stacked_tables.xlsx", 2, [0, 15], "two tables stacked in one sheet"),
    ("titled_customers.xlsx", 1, [3], "title + subtitle above the header"),
    ("sales_with_footer.csv", 1, [0], "Total: and Source: rows dropped"),
    ("attendance_split.xlsx", 4, [0, 0, 0, 0], "one table over 3 sheets + 1 unrelated"),
    ("orders_renamed.csv", 1, [0], "plain csv"),
]


def stage_ingest() -> bool:
    print(f"{BOLD}Stage: ingest{RESET}  (no LLM)")
    everything_passed = True

    for filename, expect_blocks, expect_headers, note in INGEST_CASES:
        path = FIXTURES / filename
        if not path.exists():
            print(f"  {RED}missing fixture {filename}{RESET} — run `make fixtures`")
            everything_passed = False
            continue

        blocks = []
        ids: set[str] = set()
        aliases: set[str] = set()
        for loaded in load_file(filename, path.read_bytes()):
            blocks.extend(detect_blocks(filename, loaded.sheet, loaded.grid, ids, aliases))

        headers = [b.block.source.header_row for b in blocks]
        passed = len(blocks) == expect_blocks and headers == expect_headers
        everything_passed &= passed

        print(f"  [{ok(passed)}] {filename:<26} {len(blocks)} block(s), headers {headers}")
        print(f"          {DIM}{note}; expected {expect_blocks} block(s), "
              f"headers {expect_headers}{RESET}")

    # Every discovered table must carry usable types, or the SQL layer has nothing to work with.
    session = build_session(FIXTURE_SETS["sales"], client=None)
    typed = [
        c.semantic_type
        for profile in session.tables
        for c in profile.columns
    ]
    untyped = sum(1 for t in typed if t == "text")
    print(f"  [{ok(untyped < len(typed) * 0.4)}] typing: {len(typed) - untyped}/{len(typed)} "
          f"columns got a specific semantic type")

    return everything_passed


# --------------------------------------------------------------------------- #
# Stage: relationships
# --------------------------------------------------------------------------- #


def normalise_edge(left: str, right: str) -> frozenset[str]:
    """Edges are undirected, so compare them as unordered pairs."""
    return frozenset({left, right})


@dataclass
class RelScore:
    expected: set = field(default_factory=set)
    found: set = field(default_factory=set)

    @property
    def true_positives(self) -> set:
        return self.expected & self.found

    @property
    def precision(self) -> float:
        return len(self.true_positives) / len(self.found) if self.found else 1.0

    @property
    def recall(self) -> float:
        return len(self.true_positives) / len(self.expected) if self.expected else 1.0


# The FK set to recover, expressed over the CORRUPTED files (renamed keys included).
RELATIONSHIP_CASES: list[tuple[str, list[str], list[tuple[str, str]]]] = [
    ("rename_key", ["orders_renamed.csv", "customers.csv", "employees.csv"],
     [("orders_renamed.cust_ref", "customers.customer_id"),
      ("orders_renamed.employeecode", "employees.employee_id")]),
    ("noisy_keys", ["payroll_noisy.csv", "employees.csv"],
     [("payroll_noisy.employee_id", "employees.employee_id")]),
    ("drop_fk", ["sales_by_region.csv", "targets.csv"],
     [("sales_by_region.month+region", "targets.month+region")]),
    ("change_granularity", ["orders_daily.csv", "monthly_budget.csv"],
     [("orders_daily.order_date__month", "monthly_budget.budget_month__month")]),
    # The three sheets are fused into one table before discovery runs, so the correct
    # answer here is one edge to the merged table, not three to the fragments.
    ("split_sheets", ["attendance_split.xlsx", "employees.csv"],
     [("attendance_split.employee_id", "employees.employee_id")]),
    ("unrelated_pair", ["weather.csv", "headcount.csv"], []),
]


def stage_relationships() -> bool:
    print(f"{BOLD}Stage: relationships{RESET}  (statistics only, no LLM)")
    total = RelScore()

    for name, files, expected_pairs in RELATIONSHIP_CASES:
        session = build_session(files, client=None)
        expected = {normalise_edge(a, b) for a, b in expected_pairs}

        found = set()
        for edge in session.graph.active:
            left = session.registry.alias_of(edge.left_table)
            right = session.registry.alias_of(edge.right_table)
            found.add(
                normalise_edge(
                    f"{left}.{'+'.join(edge.left_columns)}",
                    f"{right}.{'+'.join(edge.right_columns)}",
                )
            )

        case = RelScore(expected=expected, found=found)
        total.expected |= expected
        total.found |= found

        missed = expected - found
        spurious = found - expected
        print(f"  [{ok(not missed and not spurious)}] {name:<20} "
              f"recall {case.recall:.2f}  precision {case.precision:.2f}")
        for edge in sorted(missed, key=sorted):
            print(f"          {RED}missed{RESET}   {' = '.join(sorted(edge))}")
        for edge in sorted(spurious, key=sorted):
            print(f"          {YELLOW}extra{RESET}    {' = '.join(sorted(edge))}")

    print(f"\n  {BOLD}overall  recall {total.recall:.2f}  "
          f"precision {total.precision:.2f}{RESET}  "
          f"(target: recall >= 0.80 at precision >= 0.90)")
    return total.recall >= 0.80 and total.precision >= 0.90


# --------------------------------------------------------------------------- #
# Stage: qa
# --------------------------------------------------------------------------- #


def build_session(files: list[str], client) -> Session:
    session = Session(client)
    payload = []
    for name in files:
        path = FIXTURES / name
        if path.exists():
            payload.append((pathlib.Path(name).name, path.read_bytes()))
    session.add_files(payload)
    return session


def normalise_value(value: object) -> float | str:
    """Make a cell comparable: numbers as floats, dates without a midnight time."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text.endswith(" 00:00:00"):
        text = text[: -len(" 00:00:00")]
    try:
        return float(text)
    except ValueError:
        return text.lower()


def sort_key(value: float | str) -> tuple[int, float, str]:
    return (0, value, "") if isinstance(value, float) else (1, 0.0, value)


def compare(expected: list[list], rows: list[dict], tolerance: float) -> bool:
    """Order-insensitive comparison with a numeric tolerance (SPEC 11.3).

    Both row order and column order are ignored. Column order has to be, because the
    model chooses its own SELECT order and aliases: asked to compare revenue against
    target it may return either first, and that is not a wrong answer. The cost is
    that a row whose values are right but swapped between two columns would pass;
    that is the accepted trade for not failing correct answers on cosmetics.
    """
    if len(expected) != len(rows):
        return False

    remaining = [sorted((normalise_value(v) for v in row.values()), key=sort_key) for row in rows]

    for want in expected:
        target = sorted((normalise_value(v) for v in want), key=sort_key)
        match = next(
            (i for i, got in enumerate(remaining) if row_matches(target, got, tolerance)), None
        )
        if match is None:
            return False
        remaining.pop(match)
    return True


def row_matches(want: list, got: list, tolerance: float) -> bool:
    if len(want) != len(got):
        return False
    for a, b in zip(want, got, strict=True):
        if isinstance(a, float) and isinstance(b, float):
            if abs(a - b) > max(tolerance, abs(a) * tolerance):
                return False
        elif str(a) != str(b):
            return False
    return True


def classify(question: dict, result: QueryResult | None, error: str | None) -> str:
    """Bucket a failure into the SPEC 11.3 taxonomy."""
    if error == "timeout":
        return "timeout"
    if result is None:
        return "bad_sql"
    if question.get("expect") == "clarify":
        # Answering a question that should have been refused is the system inventing
        # something, which belongs in answered_wrongly, not refused_wrongly.
        return "answered_wrongly" if result.clarification is None else ""
    if result.clarification is not None:
        return "refused_wrongly"
    if not result.sql:
        return "bad_sql"

    stages = {e.stage for e in result.trace}
    if "execute" not in stages:
        return "bad_sql"

    # A join question answered from one table means routing or the graph let us down.
    expected_multi = question.get("kind", "") in {
        "cross-file join", "three-file join", "composite key",
        "derived granularity", "noisy-key join", "top-N + join",
    }
    if expected_multi and " join " not in result.sql.lower():
        return "wrong_join"
    return "answered_wrongly"


def ask_one(session: Session, question: str) -> tuple[QueryResult | None, str | None]:
    try:
        for item in session.ask(question):
            if isinstance(item, QueryResult):
                return item, None
    except Exception as exc:  # noqa: BLE001 - the harness must survive any single failure
        return None, str(exc)
    return None, "no result"


def stage_qa(repeat: int, holdout: bool, verbose: bool, external: bool = False) -> bool:
    if external:
        path, label = EVAL / "external.yaml", "external"
    elif holdout:
        path, label = EVAL / "holdout.yaml", "holdout"
    else:
        path, label = EVAL / "questions.yaml", "questions"
    if not path.exists():
        print(f"  {RED}{path} is missing{RESET}"
              + ("  — run: python eval/fetch_gov.py --wide && "
                 "python eval/make_external_questions.py" if external else ""))
        return False
    questions = yaml.safe_load(path.read_text())

    print(f"{BOLD}Stage: qa{RESET}  ({label}, {len(questions)} questions x {repeat} run(s))")

    accuracies: list[float] = []
    failures: Counter[str] = Counter()
    per_question: dict[str, list[bool]] = {q["id"]: [] for q in questions}
    latencies: list[float] = []

    for run in range(repeat):
        passed = 0
        # A fresh session per run, so nothing carries over between repeats.
        sessions: dict[str, Session] = {}

        for question in questions:
            # The external set names its files directly: they are downloaded rather
            # than checked in, so there is no fixed fixture group to refer to.
            fixture = question.get("fixture") or ",".join(question["files"])
            if fixture not in sessions:
                files = question.get("files") or FIXTURE_SETS.get(question["fixture"], [])
                sessions[fixture] = build_session(files, client_from_env())

            started = time.perf_counter()
            result, error = ask_one(sessions[fixture], question["q"])
            latencies.append(time.perf_counter() - started)

            if question.get("expect") == "clarify":
                good = result is not None and result.clarification is not None
            else:
                good = (
                    result is not None
                    and result.clarification is None
                    and compare(question["expect_rows"], result.answer_rows,
                                question.get("tolerance", 0.01))
                )

            per_question[question["id"]].append(good)
            passed += good
            if not good:
                bucket = classify(question, result, error)
                if bucket:
                    failures[bucket] += 1

            if verbose or not good:
                mark = ok(good)
                print(f"  [{mark}] run{run + 1} {question['id']} {question['kind']:<26} "
                      f"{question['q'][:52]}")
                if not good and result is not None:
                    if result.clarification:
                        print(f"          {DIM}asked: {result.clarification[:90]}{RESET}")
                    elif "expect_rows" in question:
                        print(f"          {DIM}got  {result.answer_rows[:2]}{RESET}")
                        print(f"          {DIM}want {question['expect_rows'][:2]}{RESET}")
                    else:
                        print(f"          {DIM}should have asked, but answered: "
                              f"{result.answer_rows[:2]}{RESET}")
                    if result.sql:
                        print(f"          {DIM}sql  {' '.join(result.sql.split())[:150]}{RESET}")

        accuracies.append(passed / len(questions))
        print(f"  run {run + 1}: {passed}/{len(questions)} = {accuracies[-1]:.0%}")

    print()
    mean = statistics.mean(accuracies)
    spread = (max(accuracies) - min(accuracies)) if len(accuracies) > 1 else 0.0
    print(f"  {BOLD}accuracy  mean {mean:.0%}  spread {spread:.0%}  "
          f"runs {[f'{a:.0%}' for a in accuracies]}{RESET}")
    print(f"  median latency {statistics.median(latencies):.1f}s")

    if repeat > 1:
        flaky = [qid for qid, runs in per_question.items() if 0 < sum(runs) < len(runs)]
        if flaky:
            print(f"  {YELLOW}non-deterministic: {', '.join(sorted(flaky))}{RESET}")

    print(f"\n  {BOLD}failure taxonomy{RESET}")
    if not failures:
        print(f"    {DIM}none{RESET}")
    for bucket in TAXONOMY:
        if failures[bucket]:
            print(f"    {bucket:<20} {failures[bucket]}")

    return mean >= 0.7


# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["ingest", "relationships", "qa"], required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument(
        "--external",
        action="store_true",
        help="run eval/external.yaml: real open data, questions and answers derived "
             "from the files rather than written by hand",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if not FIXTURES.exists():
        print(f"{RED}no fixtures — run `make fixtures` first{RESET}")
        return 2

    if args.stage == "ingest":
        passed = stage_ingest()
    elif args.stage == "relationships":
        passed = stage_relationships()
    else:
        passed = stage_qa(args.repeat, args.holdout, args.verbose, args.external)

    print()
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
