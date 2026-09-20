"""Clean relational data -> messy fixtures (SPEC 11.1).

Ground truth stays computable against the clean source, so every corrupted variant
has a known right answer and a known FK set. Each transform is independently
toggleable and seeded, so a fixture set is reproducible.

    python eval/corrupt.py --out eval/fixtures
    python eval/corrupt.py --out eval/fixtures --only rename_key,noisy_keys

The seed data is generated rather than downloaded. Northwind and Chinook are the
canonical choices, but a fixture corpus that needs the network is a fixture corpus
that fails in a demo; these tables are Northwind-shaped and exactly as relational.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import shutil
from dataclasses import dataclass, field

from openpyxl import Workbook

SEED = 42

REGIONS = ["North", "South", "East", "West"]
DEPARTMENTS = ["Sales", "Engineering", "Operations", "Finance", "People"]
CATEGORIES = ["Beverages", "Produce", "Hardware", "Stationery"]
CITIES = ["Pune", "Delhi", "Mumbai", "Chennai", "Bengaluru", "Hyderabad", "Kolkata"]
FIRST = ["Asha", "Ravi", "Lin", "Diego", "Mira", "Tom", "Sara", "Ken", "Ana", "Yusuf",
         "Priya", "Noah", "Ivy", "Omar", "Rhea", "Sam", "Nina", "Leo", "Zara", "Vik"]
LAST = ["Menon", "Kumar", "Wei", "Alvarez", "Shah", "Byrne", "Costa", "Ito", "Silva",
        "Khan", "Rao", "Frost", "Chen", "Haddad", "Nair", "Blake", "Roy", "Park"]


@dataclass
class Table:
    """A clean in-memory table: header plus rows of Python values."""

    name: str
    columns: list[str]
    rows: list[list[object]] = field(default_factory=list)

    def column(self, name: str) -> list[object]:
        i = self.columns.index(name)
        return [r[i] for r in self.rows]

    def copy(self) -> Table:
        return Table(self.name, list(self.columns), [list(r) for r in self.rows])

    def rename(self, old: str, new: str) -> None:
        self.columns[self.columns.index(old)] = new

    def drop(self, name: str) -> None:
        i = self.columns.index(name)
        self.columns.pop(i)
        for row in self.rows:
            row.pop(i)

    def as_grid(self) -> list[list[object]]:
        return [list(self.columns)] + [list(r) for r in self.rows]


# --------------------------------------------------------------------------- #
# Seed data
# --------------------------------------------------------------------------- #


def build_seed(rng: random.Random) -> dict[str, Table]:
    """A small Northwind-shaped sales schema plus an HR-shaped one."""
    customers = Table("customers", ["customer_id", "customer_name", "region", "city"])
    for i in range(1, 25):
        customers.rows.append(
            [f"C-{100 + i}", f"{rng.choice(LAST)} {rng.choice(['Ltd', 'Corp', 'Co'])}",
             REGIONS[i % len(REGIONS)], rng.choice(CITIES)]
        )

    products = Table("products", ["product_id", "product_name", "category", "unit_price"])
    for i in range(1, 13):
        products.rows.append(
            [f"P-{200 + i}", f"Item {i:02d}", CATEGORIES[i % len(CATEGORIES)],
             round(rng.uniform(5, 250), 2)]
        )

    employees = Table("employees", ["employee_id", "employee_name", "department", "salary",
                                    "joined_on", "region"])
    for i in range(1, 31):
        employees.rows.append(
            [f"EMP-{1000 + i}", f"{rng.choice(FIRST)} {rng.choice(LAST)}",
             DEPARTMENTS[i % len(DEPARTMENTS)], rng.randrange(45000, 160000, 1000),
             f"20{rng.randrange(18, 24)}-{rng.randrange(1, 13):02d}-{rng.randrange(1, 29):02d}",
             REGIONS[i % len(REGIONS)]]
        )

    orders = Table("orders", ["order_id", "customer_id", "employee_id", "order_date", "amount"])
    # Four customers are deliberately left with no orders, so the "which X have no Y"
    # anti-join question in questions.yaml has a real, non-empty answer to find.
    ordering_customers = customers.column("customer_id")[:20]
    for i in range(1, 181):
        month = (i % 12) + 1
        orders.rows.append(
            [f"ORD-{3000 + i}", rng.choice(ordering_customers),
             rng.choice(employees.column("employee_id")),
             f"2024-{month:02d}-{rng.randrange(1, 28):02d}",
             round(rng.uniform(50, 4000), 2)]
        )

    order_items = Table("order_items", ["order_id", "product_id", "quantity"])
    for order_id in orders.column("order_id"):
        for _ in range(rng.randrange(1, 4)):
            order_items.rows.append(
                [order_id, rng.choice(products.column("product_id")), rng.randrange(1, 12)]
            )

    # HR-shaped: Darwinbox is an HCM company, so the demo set should look like one.
    attendance = Table("attendance", ["employee_id", "work_date", "hours", "status"])
    for employee_id in employees.column("employee_id"):
        for day in range(1, 21):
            attendance.rows.append(
                [employee_id, f"2024-03-{day:02d}", round(rng.uniform(6, 9.5), 1),
                 rng.choice(["Present", "Present", "Present", "Remote", "Leave"])]
            )

    payroll = Table("payroll", ["employee_id", "pay_month", "gross_pay", "deductions"])
    for employee_id in employees.column("employee_id"):
        for month in range(1, 7):
            gross = rng.randrange(4000, 13000, 100)
            payroll.rows.append(
                [employee_id, f"2024-{month:02d}-01", gross, round(gross * 0.18, 2)]
            )

    headcount = Table("headcount", ["month", "department", "headcount"])
    for month in range(1, 7):
        for department in DEPARTMENTS:
            headcount.rows.append([f"2024-{month:02d}-01", department, rng.randrange(8, 40)])

    targets = Table("targets", ["month", "region", "target_revenue"])
    for month in range(1, 13):
        for region in REGIONS:
            targets.rows.append([f"2024-{month:02d}", region, rng.randrange(20000, 90000, 500)])

    sales_by_region = Table("sales_by_region", ["month", "region", "revenue"])
    for month in range(1, 13):
        for region in REGIONS:
            sales_by_region.rows.append(
                [f"2024-{month:02d}", region, round(rng.uniform(15000, 95000), 2)]
            )

    weather = Table("weather", ["station_code", "city", "rainfall_mm"])
    for i, city in enumerate(CITIES):
        weather.rows.append([f"W-{50 + i}", city, rng.randrange(0, 400)])

    return {
        t.name: t
        for t in [customers, products, employees, orders, order_items, attendance,
                  payroll, headcount, targets, sales_by_region, weather]
    }


# The FK set the relationship stage is scored against, in (table.col, table.col) form.
GROUND_TRUTH_FKS: list[tuple[str, str]] = [
    ("orders.customer_id", "customers.customer_id"),
    ("orders.employee_id", "employees.employee_id"),
    ("order_items.order_id", "orders.order_id"),
    ("order_items.product_id", "products.product_id"),
    ("attendance.employee_id", "employees.employee_id"),
    ("payroll.employee_id", "employees.employee_id"),
    ("sales_by_region.month+region", "targets.month+region"),
]


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #


def write_csv(path: pathlib.Path, grid: list[list[object]]) -> None:
    import csv

    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(grid)


def write_sheets(path: pathlib.Path, sheets: dict[str, list[list[object]]]) -> None:
    workbook = Workbook()
    workbook.remove(workbook.active)
    for name, grid in sheets.items():
        worksheet = workbook.create_sheet(title=name[:31])
        for row in grid:
            worksheet.append(row)
    workbook.save(path)


# --------------------------------------------------------------------------- #
# Transforms (SPEC 11.1)
# --------------------------------------------------------------------------- #


def t_stack_tables(seed: dict[str, Table], out: pathlib.Path, rng: random.Random) -> list[str]:
    """Two unrelated tables in one sheet, separated by blank rows."""
    grid = seed["products"].as_grid() + [[], []] + seed["weather"].as_grid()
    write_sheets(out / "stacked_tables.xlsx", {"Sheet1": grid})
    return ["stacked_tables.xlsx"]


def t_add_title_rows(seed: dict[str, Table], out: pathlib.Path, rng: random.Random) -> list[str]:
    """Title, subtitle and a blank line above the real header."""
    grid = [
        ["Customer Master — FY24"],
        ["Generated 2024-10-01 by the reporting team"],
        [],
        *seed["customers"].as_grid(),
    ]
    write_sheets(out / "titled_customers.xlsx", {"Customers": grid})
    return ["titled_customers.xlsx"]


def t_add_footer(seed: dict[str, Table], out: pathlib.Path, rng: random.Random) -> list[str]:
    """A 'Total:' row and a 'Source:' footnote under the data."""
    table = seed["sales_by_region"]
    total = round(sum(float(v) for v in table.column("revenue")), 2)
    grid = table.as_grid() + [["Total", None, total], ["Source: internal ledger export"]]
    write_csv(out / "sales_with_footer.csv", grid)
    return ["sales_with_footer.csv"]


def t_rename_key(seed: dict[str, Table], out: pathlib.Path, rng: random.Random) -> list[str]:
    """customer_id -> cust_ref on one side only; the join must survive it."""
    orders = seed["orders"].copy()
    orders.rename("customer_id", "cust_ref")
    orders.rename("employee_id", "EmployeeCode")
    write_csv(out / "orders_renamed.csv", orders.as_grid())
    write_csv(out / "customers.csv", seed["customers"].as_grid())
    write_csv(out / "employees.csv", seed["employees"].as_grid())
    return ["orders_renamed.csv", "customers.csv", "employees.csv"]


def t_split_sheets(seed: dict[str, Table], out: pathlib.Path, rng: random.Random) -> list[str]:
    """One table across three sheets, plus an unrelated sheet in the same workbook."""
    table = seed["attendance"]
    third = max(1, len(table.rows) // 3)
    chunks = [table.rows[:third], table.rows[third : third * 2], table.rows[third * 2 :]]
    sheets = {
        f"Week{i + 1}": [list(table.columns)] + [list(r) for r in chunk]
        for i, chunk in enumerate(chunks)
    }
    sheets["Holidays"] = [["holiday_name", "holiday_date"],
                          ["Holi", "2024-03-25"], ["Diwali", "2024-11-01"]]
    write_sheets(out / "attendance_split.xlsx", sheets)
    return ["attendance_split.xlsx"]


def t_drop_fk(seed: dict[str, Table], out: pathlib.Path, rng: random.Random) -> list[str]:
    """No ID anywhere: only (month, region) links these two files."""
    write_csv(out / "sales_by_region.csv", seed["sales_by_region"].as_grid())
    write_csv(out / "targets.csv", seed["targets"].as_grid())
    return ["sales_by_region.csv", "targets.csv"]


def t_change_granularity(
    seed: dict[str, Table], out: pathlib.Path, rng: random.Random
) -> list[str]:
    """Daily on one side, monthly on the other."""
    write_csv(out / "orders_daily.csv", seed["orders"].as_grid())
    monthly = Table("monthly_budget", ["budget_month", "budget"])
    for month in range(1, 13):
        monthly.rows.append([f"2024-{month:02d}-01", rng.randrange(50000, 250000, 1000)])
    write_csv(out / "monthly_budget.csv", monthly.as_grid())
    return ["orders_daily.csv", "monthly_budget.csv"]


def t_noisy_keys(seed: dict[str, Table], out: pathlib.Path, rng: random.Random) -> list[str]:
    """Case, whitespace and punctuation noise in the key column."""
    payroll = seed["payroll"].copy()
    index = payroll.columns.index("employee_id")
    for row in payroll.rows:
        value = str(row[index])
        style = rng.randrange(4)
        if style == 0:
            value = f" {value} "
        elif style == 1:
            value = value.lower()
        elif style == 2:
            value = value.replace("-", "_")
        row[index] = value
    write_csv(out / "payroll_noisy.csv", payroll.as_grid())
    write_csv(out / "employees.csv", seed["employees"].as_grid())
    return ["payroll_noisy.csv", "employees.csv"]


def t_unrelated_pair(seed: dict[str, Table], out: pathlib.Path, rng: random.Random) -> list[str]:
    """Two datasets with no relationship, in the same session."""
    write_csv(out / "weather.csv", seed["weather"].as_grid())
    write_csv(out / "headcount.csv", seed["headcount"].as_grid())
    return ["weather.csv", "headcount.csv"]


TRANSFORMS = {
    "stack_tables": t_stack_tables,
    "add_title_rows": t_add_title_rows,
    "add_footer": t_add_footer,
    "rename_key": t_rename_key,
    "split_sheets": t_split_sheets,
    "drop_fk": t_drop_fk,
    "change_granularity": t_change_granularity,
    "noisy_keys": t_noisy_keys,
    "unrelated_pair": t_unrelated_pair,
}


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def generate(out: pathlib.Path, only: list[str] | None = None) -> dict:
    rng = random.Random(SEED)
    seed = build_seed(rng)

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    clean = out / "clean"
    clean.mkdir()
    for table in seed.values():
        write_csv(clean / f"{table.name}.csv", table.as_grid())

    produced: dict[str, list[str]] = {}
    for name, transform in TRANSFORMS.items():
        if only and name not in only:
            continue
        produced[name] = transform(seed, out, random.Random(SEED))

    manifest = {
        "seed": SEED,
        "transforms": produced,
        "ground_truth_fks": [list(pair) for pair in GROUND_TRUTH_FKS],
        "tables": {name: {"columns": t.columns, "n_rows": len(t.rows)} for name, t in seed.items()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("eval/fixtures"))
    parser.add_argument("--only", type=str, default="", help="comma-separated transform names")
    args = parser.parse_args()

    only = [s.strip() for s in args.only.split(",") if s.strip()] or None
    manifest = generate(args.out, only)

    print(f"wrote fixtures to {args.out}")
    for name, files in manifest["transforms"].items():
        print(f"  {name:<20} {', '.join(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
