"""Phase 2 acceptance: coercion, semantic typing, signatures, DuckDB registration."""

from __future__ import annotations

import pathlib

import duckdb
import pandas as pd
import pytest
from darwinbox.ingest.blocks import detect_blocks
from darwinbox.ingest.loader import load_file
from darwinbox.profile.columns import (
    coerce_series,
    normalize_name_tokens,
    regex_signature,
    semantic_type,
)
from darwinbox.profile.registry import Registry, TableNotFoundError

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def series(*values: str) -> pd.Series:
    return pd.Series(list(values), dtype="object")


# --------------------------------------------------------------------------- #
# Coercion
# --------------------------------------------------------------------------- #


def test_currency_and_thousands_separators_become_numeric():
    out = coerce_series(series("$1,200.50", "$98.00", "$1,000,000"))
    assert pd.api.types.is_numeric_dtype(out)
    assert out.tolist() == [1200.5, 98.0, 1000000.0]


def test_percent_and_parenthesised_negatives():
    out = coerce_series(series("12%", "(50)", "7.5%"))
    assert out.tolist() == [0.12, -50.0, 0.075]


def test_dates_parse_when_most_values_are_dates():
    out = coerce_series(series("2024-01-05", "2024-02-11", "2024-03-02"))
    assert pd.api.types.is_datetime64_any_dtype(out)


def test_mostly_unparseable_dates_stay_text():
    out = coerce_series(series("2024-01-05", "not a date", "also not", "nope"))
    assert not pd.api.types.is_datetime64_any_dtype(out)


def test_booleans():
    out = coerce_series(series("yes", "no", "YES", "n"))
    assert pd.api.types.is_bool_dtype(out)
    assert out.tolist() == [True, False, True, False]


def test_single_valued_boolean_like_column_is_not_boolean():
    # All "1" would otherwise coerce to booleans and lose its numeric meaning.
    out = coerce_series(series("1", "1", "1"))
    assert pd.api.types.is_numeric_dtype(out)


# --------------------------------------------------------------------------- #
# Semantic typing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "values", "expected"),
    [
        ("cust_ref", ["C-1", "C-2", "C-1"], "id"),
        ("order_id", ["A", "A", "A"], "id"),  # name rule wins over distinctness
        ("sku", ["S1", "S2", "S3", "S4"], "id"),  # distinctness rule
        ("region", ["North", "South", "North", "South"], "category"),
        ("amount", ["10.5", "20.25", "30.0", "44.1"], "numeric"),
        ("hired_on", ["2024-01-05", "2024-02-11", "2024-03-02"], "date"),
        ("active", ["yes", "no", "yes"], "boolean"),
    ],
)
def test_semantic_types(name, values, expected):
    coerced = coerce_series(pd.Series(values, dtype="object"))
    assert semantic_type(name, coerced, len(values)) == expected


# --------------------------------------------------------------------------- #
# Signatures and name tokens
# --------------------------------------------------------------------------- #


def test_regex_signature_detects_prefixed_ids():
    assert regex_signature(series("EMP-1", "EMP-2", "EMP-33", "EMP-4")) == r"^[A-Z]{1,4}[-_]?\d+$"


def test_regex_signature_detects_fixed_width_digits():
    assert regex_signature(series("0011", "0242", "9930", "1234")) == r"^\d{4}$"


def test_regex_signature_none_when_values_are_mixed():
    assert regex_signature(series("EMP-1", "hello", "2024-01-01", "x")) is None


@pytest.mark.parametrize(
    ("name", "tokens"),
    [
        ("cust_ref", ["customer", "identifier"]),
        ("employeeID", ["employee", "identifier"]),
        ("dept_no", ["department", "number"]),
        ("order_qty", ["order", "quantity"]),
    ],
)
def test_name_tokens_expand_abbreviations(name, tokens):
    assert normalize_name_tokens(name) == tokens


# --------------------------------------------------------------------------- #
# Registry / DuckDB
# --------------------------------------------------------------------------- #


@pytest.fixture
def registry():
    return Registry(duckdb.connect(":memory:"))


def load_into(registry: Registry, name: str):
    data = (FIXTURES / name).read_bytes()
    ids: set[str] = set()
    aliases: set[str] = set()
    profiles = []
    for loaded in load_file(name, data):
        for detected in detect_blocks(name, loaded.sheet, loaded.grid, ids, aliases):
            profiles.append(registry.register(detected))
    return profiles


def test_registered_table_is_queryable_with_correct_types(registry):
    (profile,) = load_into(registry, "clean.csv")

    assert profile.n_rows == 5
    kinds = {c.name: c.semantic_type for c in profile.columns}
    assert kinds == {"order_id": "id", "order_date": "date", "amount": "numeric"}

    total = registry.conn.execute(f'SELECT sum(amount) FROM "{profile.table_id}"').fetchone()[0]
    assert total == pytest.approx(1894.5)


def test_both_tables_from_one_sheet_are_registered_separately(registry):
    profiles = load_into(registry, "two_tables_one_sheet.xlsx")
    assert len(profiles) == 2
    assert {p.n_rows for p in profiles} == {4, 3}

    for profile in profiles:
        count = registry.conn.execute(
            f'SELECT count(*) FROM "{profile.table_id}"'
        ).fetchone()[0]
        assert count == profile.n_rows


def test_resolve_by_alias_and_unknown_raises(registry):
    (profile,) = load_into(registry, "clean.csv")
    assert registry.resolve(profile.alias) == profile.table_id
    assert registry.resolve(profile.table_id) == profile.table_id
    with pytest.raises(TableNotFoundError):
        registry.resolve("nope")


def test_derived_column_is_queryable(registry):
    (profile,) = load_into(registry, "clean.csv")
    registry.add_derived_column(
        profile.table_id, "order_date__month", "date_trunc('month', order_date)"
    )

    assert registry.has_column(profile.table_id, "order_date__month")
    months = registry.conn.execute(
        f'SELECT DISTINCT order_date__month FROM "{profile.table_id}" ORDER BY 1'
    ).fetchall()
    assert len(months) == 3


# --------------------------------------------------------------------------- #
# Locale conventions
#
# Both of these produced silently wrong values rather than an error.
# --------------------------------------------------------------------------- #


def test_european_decimals_are_not_off_by_a_thousand():
    # "1.234,56" used to have its comma stripped as a thousands separator, leaving
    # "1.23456" -- a confident, wrong number three orders of magnitude out.
    out = coerce_series(series("1.234,56", "2.000,00", "987,65", "12,30"))
    assert out.tolist() == [1234.56, 2000.0, 987.65, 12.3]


def test_us_decimals_are_unaffected():
    out = coerce_series(series("1,234.56", "2,000.00", "987.65", "12.30"))
    assert out.tolist() == [1234.56, 2000.0, 987.65, 12.3]


def test_decimal_convention_is_decided_per_column_not_per_value():
    from darwinbox.profile.columns import uses_european_decimals

    assert uses_european_decimals(series("1.234,56", "987,65")) is True
    assert uses_european_decimals(series("1,234.56", "987.65")) is False


def test_day_first_dates_parse_consistently_across_the_column():
    # 07/03/2024 used to parse as 3 July while 13/01/2024 in the SAME column parsed
    # as 13 January, because format="mixed" infers per value.
    out = coerce_series(series("13/01/2024", "25/02/2024", "07/03/2024", "19/04/2024"))
    assert [str(v)[:10] for v in out] == [
        "2024-01-13", "2024-02-25", "2024-03-07", "2024-04-19",
    ]


def test_month_first_dates_parse_consistently_across_the_column():
    out = coerce_series(series("01/13/2024", "02/25/2024", "03/07/2024", "04/19/2024"))
    assert [str(v)[:10] for v in out] == [
        "2024-01-13", "2024-02-25", "2024-03-07", "2024-04-19",
    ]


def test_iso_dates_are_unaffected():
    out = coerce_series(series("2024-01-05", "2024-02-11", "2024-03-02"))
    assert [str(v)[:10] for v in out] == ["2024-01-05", "2024-02-11", "2024-03-02"]


def test_leading_zeros_are_never_coerced_to_numbers():
    # "01234" is a zip code, an account number or a product code -- never 1234.
    # Coercing destroys the value AND breaks every join that depends on its form.
    out = coerce_series(series("01234", "00567", "09876", "04321"))
    assert out.tolist() == ["01234", "00567", "09876", "04321"]


@pytest.mark.parametrize("token", ["NA", "N/A", "#N/A", "null", "NULL", "-", "none", "nan"])
def test_spreadsheet_null_spellings_become_null(token):
    out = coerce_series(series(token, "10", "20", "30"))
    assert out.isna().iloc[0], f"{token!r} was kept as a real value"
    assert out.dropna().tolist() == [10, 20, 30]


def test_null_tokens_do_not_count_as_distinct_values():
    # If "NA" survives, it inflates cardinality and pollutes containment downstream.
    out = coerce_series(series("NA", "N/A", "null", "5", "6"))
    assert out.nunique() == 2


def test_integers_beyond_int64_do_not_crash():
    out = coerce_series(series("123456789012345678901234567890", "2", "3"))
    assert len(out) == 3


def test_scientific_notation_parses():
    assert coerce_series(series("1.5e3", "2.0E-2", "3e10")).tolist() == [1500.0, 0.02, 3e10]
