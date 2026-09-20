"""Type coercion, semantic typing and column signatures.

Everything here operates on a pandas Series and knows nothing about DuckDB or the
session. ``normalize_name_tokens`` lives here rather than in ``relate`` because the
profile stores the tokens; relationship scoring consumes them.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
from pandas.tseries.api import guess_datetime_format

from darwinbox.models import ColumnProfile, SemanticType

MAX_SAMPLES = 5
SAMPLE_CHARS = 40
DATE_PARSE_MIN = 0.90
DATE_SAMPLE_ROWS = 50
NUMERIC_SAMPLE_ROWS = 1000
SIGNATURE_MIN = 0.95
ID_DISTINCT_MIN = 0.90
CATEGORY_MAX_DISTINCT = 50
CATEGORY_RATIO_MAX = 0.05

_ID_SUFFIXES = ("_id", "_code", "_ref", "_no", "_num", "_key")
_ID_EXACT = {"id", "code", "ref", "key", "no"}

_CURRENCY = re.compile(r"[$€£₹¥,\s]")
# "1.234,56" -- dot thousands, comma decimal. The comma must NOT be stripped as a
# thousands separator there, or the value silently becomes 1.23456.
_EU_DECIMAL = re.compile(r"^[^\d-]*-?\d{1,3}(\.\d{3})*,\d{1,2}[^\d]*$")
_US_DECIMAL = re.compile(r"^[^\d-]*-?\d{1,3}(,\d{3})*\.\d{1,2}[^\d]*$")
_PERCENT = re.compile(r"%$")
_PAREN_NEG = re.compile(r"^\((.*)\)$")

# Spellings a spreadsheet uses for "missing". Without these, a column holding "NA"
# stays text and never gets a numeric type, and "NA" counts as a real distinct value
# in every cardinality and containment calculation downstream.
_NULL_TOKENS = {
    "na", "n/a", "#n/a", "null", "none", "nil", "nan", "-", "--", "?",
    "#null!", "#value!", "#div/0!", "(blank)", "unknown", "not available",
}
_LEADING_ZERO = re.compile(r"^0\d+$")

_BOOL_TRUE = {"true", "yes", "y", "t", "1"}
_BOOL_FALSE = {"false", "no", "n", "f", "0"}

# Ordered: the first pattern matched by >=95% of values wins.
_SIGNATURES: list[str] = [
    r"^[A-Z]{1,4}[-_]?\d+$",
    r"^\d{4}-\d{2}-\d{2}$",
    r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
    r"^\d{10}$",
]

# Abbreviation expansion shared by profiling and relationship name similarity.
_ABBREV = {
    "emp": "employee",
    "empl": "employee",
    "cust": "customer",
    "custm": "customer",
    "dept": "department",
    "dep": "department",
    "qty": "quantity",
    "amt": "amount",
    "num": "number",
    "no": "number",
    "nbr": "number",
    "dt": "date",
    "id": "identifier",
    "code": "identifier",
    "ref": "identifier",
    "key": "identifier",
    "org": "organisation",
    "mgr": "manager",
    "addr": "address",
    "desc": "description",
    "prod": "product",
    "cat": "category",
    "txn": "transaction",
    "trans": "transaction",
    "acct": "account",
    "yr": "year",
    "mth": "month",
    "mon": "month",
}


# --------------------------------------------------------------------------- #
# Name tokens
# --------------------------------------------------------------------------- #


# Words whose trailing "s" is part of the stem, not a plural.
_NOT_PLURAL_ENDINGS = ("ss", "us", "is", "as", "os")


def singularize(token: str) -> str:
    """Fold a plural to its singular so "files" and "file" are one term.

    Applied to queries and to documents alike, so the goal is consistency rather
    than linguistic correctness -- without it "what is in the files" never matches a
    column called source_file, and "customers" never matches "customer".
    """
    if len(token) < 4 or token.endswith(_NOT_PLURAL_ENDINGS):
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith(("ches", "shes", "xes", "zes")):
        return token[:-2]
    return token[:-1] if token.endswith("s") else token


def normalize_name_tokens(name: str) -> list[str]:
    """Split camelCase/snake_case into tokens, expand abbreviations, fold plurals."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    parts = [p for p in re.split(r"[^0-9a-zA-Z]+", spaced) if p]
    tokens: list[str] = []
    for part in parts:
        low = part.lower()
        tokens.append(_ABBREV.get(low) or singularize(low))
    return tokens


# --------------------------------------------------------------------------- #
# Type coercion
# --------------------------------------------------------------------------- #


def coerce_series(raw: pd.Series) -> pd.Series:
    """Coerce a string series to numeric, datetime, boolean or string, in that order.

    Order matters: dates parse as numbers under some locales and booleans parse as
    numbers everywhere, so the most specific successful coercion must be tried first
    and each is all-or-nothing for the column.
    """
    text = raw.astype("string").str.strip().replace({"": pd.NA})
    text = text.mask(text.str.lower().isin(_NULL_TOKENS), pd.NA)
    non_null = text.dropna()
    if non_null.empty:
        return text

    numeric = _try_numeric(text, non_null)
    if numeric is not None:
        return numeric

    dates = _try_datetime(text, non_null)
    if dates is not None:
        return dates

    booleans = _try_boolean(text, non_null)
    if booleans is not None:
        return booleans

    return text


def uses_european_decimals(values: pd.Series) -> bool:
    """True when this column writes 1.234,56 rather than 1,234.56.

    Decided per column, never per value: a mixed verdict would parse some rows one
    way and some the other, which is worse than getting the whole column wrong.
    """
    sample = values.astype(str).str.strip()
    eu = sample.str.match(_EU_DECIMAL).sum()
    us = sample.str.match(_US_DECIMAL).sum()
    return bool(eu > us and eu > 0)


def _try_numeric(text: pd.Series, non_null: pd.Series) -> pd.Series | None:
    # "01234" is a zip code, an account number or a product code -- never the integer
    # 1234. Coercing it silently destroys the leading zero AND breaks every join that
    # depends on the key's exact form.
    if non_null.astype(str).str.match(_LEADING_ZERO).any():
        return None

    # Same argument, different mechanism: a number too big for int64 is an identifier,
    # not a quantity. The cast below does not raise on one -- pandas wraps it, so a
    # 30-digit reference number silently becomes -9223372036854775808 -- and Float64
    # would round it, which breaks the joins just as thoroughly.
    if _overflows_int64(non_null):
        return None

    # Fast path: a column of plain numerals needs no cleaning, and pd.to_numeric is
    # vectorised C. The cleaning path below is a Python call per cell, which on a
    # million-row file is six million calls and two thirds of the whole ingest time.
    direct = pd.to_numeric(non_null, errors="coerce")
    if direct.notna().all():
        out = pd.to_numeric(text, errors="coerce")
        return _as_integer_or_float(out, direct)

    # Reject on a sample before paying for the whole column. Coercion is
    # all-or-nothing, so one uncleanable value in the sample settles it -- and a text
    # column like "ORD-1" would otherwise cost a million Python calls to rule out.
    sample = non_null.head(NUMERIC_SAMPLE_ROWS)
    european = uses_european_decimals(sample)
    probe = pd.to_numeric(sample.map(lambda v: _clean_number(v, european)), errors="coerce")
    if probe.isna().any():
        return None

    cleaned = non_null.map(lambda v: _clean_number(v, european))
    parsed = pd.to_numeric(cleaned, errors="coerce")
    if parsed.isna().any():
        return None
    out = pd.to_numeric(text.map(lambda v: _clean_number(v, european)), errors="coerce")
    return _as_integer_or_float(out, parsed)


_PLAIN_INTEGER = re.compile(r"^[+-]?\d+$")
_INT64_MAX = 2**63 - 1
_INT64_DIGITS = 19  # len(str(2**63 - 1)); shorter than this cannot overflow


def _overflows_int64(non_null: pd.Series) -> bool:
    """True when a column of plain integers holds a value int64 cannot represent."""
    text = non_null.astype(str).str.strip()
    digits = text.str.lstrip("+-")
    long_enough = digits.str.len() >= _INT64_DIGITS
    if not long_enough.any():
        return False
    # Only now is it worth parsing: a plain integer that long may still fit.
    candidates = text[long_enough & text.str.match(_PLAIN_INTEGER)]
    return any(abs(int(value)) > _INT64_MAX for value in candidates)


def _as_integer_or_float(out: pd.Series, parsed: pd.Series) -> pd.Series:
    if _all_integral(parsed):
        try:
            return out.astype("Int64")
        except (TypeError, ValueError, OverflowError):
            # Integers beyond int64 arrive as floats that cannot be cast back.
            return out.astype("Float64")
    return out.astype("Float64")


def _clean_number(value: object, european: bool = False) -> object:
    if value is pd.NA or value is None:
        return value
    text = str(value).strip()
    if european:
        # Drop dot thousands separators first, then promote the comma to the point.
        text = text.replace(".", "").replace(",", ".")
    negative = bool(_PAREN_NEG.match(text))
    if negative:
        text = _PAREN_NEG.sub(r"\1", text)
    percent = bool(_PERCENT.search(text))
    text = _PERCENT.sub("", text)
    text = _CURRENCY.sub("", text)
    if not text or not re.fullmatch(r"-?\d*\.?\d+([eE][-+]?\d+)?", text):
        return None
    number = float(text)
    if percent:
        number /= 100.0
    return -number if negative else number


def _all_integral(values: pd.Series) -> bool:
    finite = values.dropna()
    return bool(finite.empty or np.all(np.mod(finite.astype(float), 1) == 0))


_SLASHED = re.compile(r"^\s*(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})")


def infer_dayfirst(values: pd.Series) -> bool:
    """Decide D/M/Y vs M/D/Y once for the whole column.

    format="mixed" infers per value, which silently produces an internally
    inconsistent column: in one real test 13/01/2024 parsed as 13 January while
    07/03/2024 in the same column parsed as 3 July. A column is written in one
    convention, so the convention is decided once, from whichever position is forced
    above 12 by some row.
    """
    first_over_12 = second_over_12 = 0
    for raw in values.astype(str):
        match = _SLASHED.match(raw)
        if not match:
            continue
        first, second = int(match.group(1)), int(match.group(2))
        first_over_12 += first > 12
        second_over_12 += second > 12
    return first_over_12 > second_over_12


def infer_date_format(values: pd.Series, dayfirst: bool) -> str | None:
    """The single strftime format that most values in this column are written in."""
    guesses: Counter[str] = Counter()
    for raw in values.head(DATE_SAMPLE_ROWS).astype(str):
        try:
            guess = guess_datetime_format(raw.strip(), dayfirst=dayfirst)
        except (ValueError, TypeError):
            guess = None
        if guess:
            guesses[guess] += 1
    return guesses.most_common(1)[0][0] if guesses else None


def _try_datetime(text: pd.Series, non_null: pd.Series) -> pd.Series | None:
    """Parse a column as dates under ONE format, or not at all.

    Letting pandas infer per value silently produces an internally inconsistent
    column: in one real test 13/01/2024 parsed as 13 January while 07/03/2024 in the
    same column parsed as 3 July. Here a value that does not match the column's format
    becomes NaT and counts against the 90% threshold, so a column we cannot read
    consistently stays text -- visibly unparsed rather than quietly wrong.
    """
    dayfirst = infer_dayfirst(non_null)
    fmt = infer_date_format(non_null, dayfirst)
    if fmt is None:
        return None

    parsed = pd.to_datetime(non_null, errors="coerce", format=fmt)
    if parsed.notna().mean() < DATE_PARSE_MIN:
        return None
    return pd.to_datetime(text, errors="coerce", format=fmt)


def _try_boolean(text: pd.Series, non_null: pd.Series) -> pd.Series | None:
    lowered = non_null.str.lower()
    if not lowered.isin(_BOOL_TRUE | _BOOL_FALSE).all():
        return None
    if lowered.nunique() < 2:
        return None
    return text.str.lower().map(
        lambda v: pd.NA if v is pd.NA or v is None else v in _BOOL_TRUE
    ).astype("boolean")


# --------------------------------------------------------------------------- #
# Semantic typing
# --------------------------------------------------------------------------- #


def semantic_type(name: str, series: pd.Series, n_rows: int) -> SemanticType:
    """Classify a coerced column (SPEC 6)."""
    dtype = series.dtype
    n_distinct = int(series.nunique(dropna=True))
    ratio = n_distinct / n_rows if n_rows else 0.0

    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "date"

    lowered = name.lower()
    if lowered in _ID_EXACT or lowered.endswith(_ID_SUFFIXES):
        return "id"

    is_int_like = pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_string_dtype(dtype)
    if is_int_like and ratio >= ID_DISTINCT_MIN and n_distinct > 1:
        return "id"

    if pd.api.types.is_numeric_dtype(dtype):
        return "numeric"

    if n_distinct <= CATEGORY_MAX_DISTINCT or ratio <= CATEGORY_RATIO_MAX:
        return "category"

    return "text"


def regex_signature(series: pd.Series) -> str | None:
    """Return a pattern matched by >=95% of non-null values, or None.

    A shared signature between two columns is a strong join signal that survives
    renaming, which is why it is stored on the profile rather than recomputed.
    """
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    if len(values) < 3:
        return None

    for pattern in _SIGNATURES:
        if values.str.match(pattern).mean() >= SIGNATURE_MIN:
            return pattern

    digits = values[values.str.fullmatch(r"\d+")]
    if len(digits) / len(values) >= SIGNATURE_MIN:
        widths = digits.str.len().value_counts()
        width = int(widths.index[0])
        if widths.iloc[0] / len(digits) >= SIGNATURE_MIN:
            return rf"^\d{{{width}}}$"

    return None


# --------------------------------------------------------------------------- #
# Profile assembly
# --------------------------------------------------------------------------- #


def profile_column(
    name: str, series: pd.Series, n_rows: int, raw_name: str | None = None
) -> ColumnProfile:
    """Build the full ColumnProfile for one already-coerced column.

    ``raw_name`` is the header as it appeared in the file. Tokens come from it when
    given, because normalization lowercases and so destroys the camelCase boundary:
    "EmployeeCode" normalises to "employeecode", which shares no token with
    "employee_id" and scores 0 name similarity against the key it actually matches.
    """
    n_null = int(series.isna().sum())
    n_distinct = int(series.nunique(dropna=True))
    kind = semantic_type(name, series, n_rows)
    non_null = series.dropna()

    return ColumnProfile(
        name=name,
        semantic_type=kind,
        dtype=str(series.dtype),
        n_distinct=n_distinct,
        n_null=n_null,
        null_rate=round(n_null / n_rows, 4) if n_rows else 0.0,
        is_unique=bool(n_rows and n_distinct == n_rows - n_null and n_null == 0),
        min_value=_bound(non_null, "min"),
        max_value=_bound(non_null, "max"),
        samples=_samples(non_null),
        regex_signature=regex_signature(series) if kind in {"id", "text", "category"} else None,
        normalized_name_tokens=normalize_name_tokens(raw_name or name),
    )


def _bound(non_null: pd.Series, which: str) -> str | None:
    if non_null.empty:
        return None
    try:
        value = non_null.min() if which == "min" else non_null.max()
    except TypeError:
        return None
    return _render(value)


def _samples(non_null: pd.Series) -> list[str]:
    seen = non_null.drop_duplicates().head(MAX_SAMPLES)
    return [_render(v)[:SAMPLE_CHARS] for v in seen]


def _render(value: object) -> str:
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, float) and float(value).is_integer():
        return str(int(value))
    return str(value)
