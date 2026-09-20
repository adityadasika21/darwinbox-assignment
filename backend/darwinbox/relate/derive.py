"""Normalized projections and derived granularity keys (SPEC 7.2, 7.4).

Two separate rescues for keys that do not match on raw values:

* a **projection** changes how a value is *compared* (case, whitespace, punctuation)
  without changing the column,
* a **granularity derivation** creates a new column (``date_trunc``) so that a daily
  table can join a monthly one.

Both carry a SQL expression so the generated query can reproduce them exactly.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

_NON_DIGIT = re.compile(r"[^0-9]")
_NON_ALNUM = re.compile(r"[^a-z0-9]")


@dataclass(frozen=True)
class Projection:
    """A value normalization applied identically in Python and in DuckDB."""

    name: str
    apply: Callable[[str], str]
    sql: Callable[[str], str]

    def express(self, qualified_column: str) -> str | None:
        """SQL for this projection over a qualified column, or None when it is a no-op."""
        return None if self.name == "raw" else self.sql(qualified_column)


RAW = Projection(
    name="raw",
    apply=lambda v: v,
    sql=lambda c: c,
)

LOWER_TRIM = Projection(
    name="lower_trim",
    apply=lambda v: v.strip().lower(),
    sql=lambda c: f"lower(trim({c}))",
)

DIGITS_ONLY = Projection(
    name="digits_only",
    apply=lambda v: _NON_DIGIT.sub("", v),
    sql=lambda c: f"regexp_replace({c}, '[^0-9]', '', 'g')",
)

ALNUM_ONLY = Projection(
    name="alnum_only",
    apply=lambda v: _NON_ALNUM.sub("", v.strip().lower()),
    sql=lambda c: f"regexp_replace(lower(trim({c})), '[^a-z0-9]', '', 'g')",
)

# Ordered cheapest-first; the first projection that clears threshold wins, so a key
# rescued by lowercasing is never reported as needing punctuation stripping.
PROJECTIONS: list[Projection] = [RAW, LOWER_TRIM, DIGITS_ONLY, ALNUM_ONLY]
RETRY_PROJECTIONS: list[Projection] = [LOWER_TRIM, DIGITS_ONLY, ALNUM_ONLY]


def render_series(values: pd.Series) -> pd.Series:
    """Render a column to comparable strings, preserving case and whitespace.

    Case and punctuation are preserved deliberately: deciding whether they matter is
    the projection's job, and a renderer that normalised here would make every column
    look clean and hide the very noise the projections exist to find.
    """
    text = values.dropna()
    if text.empty:
        return text.astype(str)

    if pd.api.types.is_datetime64_any_dtype(text):
        return text.dt.strftime("%Y-%m-%d")
    if pd.api.types.is_float_dtype(text):
        # 1.0 and "1" must compare equal, or a float FK never matches its int parent.
        return text.map(lambda v: str(int(v)) if float(v).is_integer() else str(v))
    return text.astype(str)


def project_values(values: pd.Series, projection: Projection) -> set[str]:
    """Distinct non-empty values of a column under one projection."""
    out = {projection.apply(v) for v in render_series(values)}
    out.discard("")
    return out


# --------------------------------------------------------------------------- #
# Granularity (SPEC 7.4)
# --------------------------------------------------------------------------- #

GRAINS: list[str] = ["month", "week"]


@dataclass(frozen=True)
class Grain:
    """A date truncation that becomes a real column so SQL can join on it."""

    unit: str
    source_column: str

    @property
    def column(self) -> str:
        return f"{self.source_column}__{self.unit}"

    def sql(self, qualified_column: str) -> str:
        return f"date_trunc('{self.unit}', {qualified_column})"

    def apply(self, values: pd.Series) -> pd.Series:
        if not pd.api.types.is_datetime64_any_dtype(values):
            return values
        freq = {"month": "M", "week": "W"}[self.unit]
        return values.dt.to_period(freq).dt.start_time
