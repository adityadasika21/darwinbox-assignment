"""Score ingestion quality per genre, with no LLM involved.

Three objective checks per file: the number of tables found, whether the columns a
human would name actually appear, and what fraction of column names came out as
col_N placeholders. A genre passes only on all three.
"""
import glob
import pathlib
import re
import sys
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'backend'))
warnings.filterwarnings('ignore')
from darwinbox.llm.client import FakeLLMClient  # noqa: E402
from darwinbox.session import Session  # noqa: E402

D = 'eval/fixtures/genres/'
PLACEHOLDER = re.compile(r"col_\d+")

# file -> (expected tables, columns that MUST appear somewhere)
EXPECT = {
 # 2 tables: the schedule, plus its TOTAL row folded into a details table.
 "arch_room_schedule.xlsx": (2, ["room_no", "room_name", "area_sqm"]),
 "boq.xlsx":                (2, ["item", "description", "unit", "qty", "rate", "amount"]),
 "pnl.xlsx":                (1, ["apr", "may", "jun"]),
 "budget_tracker.csv":      (1, ["category", "item", "budget", "actual", "variance"]),
 "shopping_list.csv":       (1, ["item", "qty", "where", "done"]),
 "inventory.xlsx":          (2, ["sku", "product", "category", "on_hand"]),
 "sales_pivot.csv":         (1, ["region", "jan", "feb", "total"]),
 "todo.csv":                (1, ["done", "task", "owner", "due", "priority"]),
 "payroll_register.xlsx":   (1, ["emp_id", "name", "net_pay"]),
 "directory.csv":           (1, ["name", "department", "email", "phone", "location"]),
 "lookup_codes.csv":        (1, ["code", "description"]),
 "categories.csv":          (1, ["category", "count"]),
 "timesheet.xlsx":          (1, ["employee", "mon", "tue", "total"]),
 "gradebook.csv":           (1, ["roll", "student", "maths", "average"]),
 "expense_report.xlsx":     (2, ["date", "description", "category", "amount"]),
 "mixed_workbook.xlsx":     (3, ["order_id", "customer", "amount", "region"]),
 # round two
 "bank_statement.csv":      (1, ["txn_date", "description", "debit", "credit", "balance"]),
 "invoice.xlsx":            (2, ["item", "hsn", "qty", "rate", "amount"]),
 "attendance_register.xlsx":(1, ["emp_id", "name", "present", "absent"]),
 "survey.csv":              (1, ["respondent", "comments"]),
 "price_list.xlsx":         (1, ["sku", "product", "tier_1_qty", "tier_1_price"]),
 "asset_register.csv":      (1, ["asset_code", "description", "cost", "wdv"]),
 "crosstab.csv":            (1, ["product", "north", "south", "total"]),
 "with_errors.csv":         (1, ["item", "qty", "unit_price", "total"]),
 "minutes.xlsx":            (2, ["action", "owner", "due", "status"]),
 "multi_currency.csv":      (1, ["date", "vendor", "currency", "amount"]),
 "project_plan.csv":        (1, ["task", "owner", "w1", "w6"]),
 "bom.csv":                 (1, ["level", "part_no", "description", "qty", "uom"]),
}

rows = []
for path in sorted(glob.glob(D + "*")):
    name = pathlib.Path(path).name
    want_tables, want_cols = EXPECT.get(name, (1, []))
    try:
        s = Session(FakeLLMClient())
        s.add_files([(name, pathlib.Path(path).read_bytes())])
        tables = s.tables
        all_cols = {c.name for t in tables for c in t.columns}
        names = [c.name for t in tables for c in t.columns]
        junk = sum(1 for n in names if PLACEHOLDER.fullmatch(n)) / max(len(names), 1)
        missing = [c for c in want_cols if c not in all_cols]
        ok = len(tables) == want_tables and not missing and junk < 0.2
        rows.append((name, ok, len(tables), want_tables, junk, missing,
                     [(t.alias, t.n_rows, [c.name for c in t.columns][:5]) for t in tables]))
    except Exception as e:
        rows.append((name, False, 0, want_tables, 1.0, [f"ERROR {e}"], []))

good = sum(1 for r in rows if r[1])
for name, ok, got, want, junk, missing, detail in rows:
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<26} tables {got}/{want}  junk-cols {junk:.0%}")
    if not ok:
        if missing:
            print(f"         missing: {missing}")
        for alias, n, cols in detail:
            print(f"         {alias:<22} {n:>3}r {cols}")
print(f"\n{good}/{len(rows)} genres ingest cleanly")
