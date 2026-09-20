"""Deterministic chart selection (SPEC 8.7).

The chart comes from the shape of the result, not from the model. ``chart_intent``
is a tiebreaker only. A model that picks its own chart type gets it wrong in ways
that are invisible until a demo, whereas result shape is checkable.
"""

from __future__ import annotations

import datetime as dt
import re

from darwinbox.models import ChartSpec

MAX_BAR_CATEGORIES = 25
MIN_SCATTER_ROWS = 15

_DATE_TEXT = re.compile(r"^\d{4}-\d{2}(-\d{2})?")


def select_chart(
    columns: list[str],
    rows: list[dict],
    chart_intent: str = "none",
    title: str = "",
) -> ChartSpec:
    """Choose a chart from the returned shape; intent only breaks ties."""
    if chart_intent == "none" or not rows or not columns:
        return ChartSpec(type="table", title=title)

    numeric = [c for c in columns if _is_numeric(rows, c)]
    temporal = [c for c in columns if _is_temporal(rows, c)]
    categorical = [
        c for c in columns if c not in numeric and c not in temporal and _is_categorical(rows, c)
    ]

    if len(rows) == 1 and len(columns) == 1 and numeric:
        return ChartSpec(type="big_number", y=numeric[0], title=title)

    if temporal and numeric:
        x = temporal[0]
        series = next((c for c in categorical if c != x), None)
        return ChartSpec(type="line", x=x, y=numeric[0], series=series, title=title)

    if categorical and numeric:
        x = categorical[0]
        if _distinct(rows, x) <= MAX_BAR_CATEGORIES:
            series = next((c for c in categorical if c != x), None)
            return ChartSpec(type="bar", x=x, y=numeric[0], series=series, title=title)

    if len(numeric) >= 2 and len(rows) > MIN_SCATTER_ROWS:
        return ChartSpec(type="scatter", x=numeric[0], y=numeric[1], title=title)

    # A single-value intent that produced one numeric cell still reads best as a KPI.
    if chart_intent == "single_value" and len(rows) == 1 and numeric:
        return ChartSpec(type="big_number", y=numeric[0], title=title)

    return ChartSpec(type="table", title=title)


def _values(rows: list[dict], column: str) -> list[object]:
    return [r.get(column) for r in rows if r.get(column) is not None]


def _is_numeric(rows: list[dict], column: str) -> bool:
    values = _values(rows, column)
    return bool(values) and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in values
    )


def _is_temporal(rows: list[dict], column: str) -> bool:
    values = _values(rows, column)
    if not values:
        return False
    # Dates arrive as ISO strings because the runner serialises them for JSON.
    return all(
        isinstance(v, (dt.date, dt.datetime)) or (isinstance(v, str) and _DATE_TEXT.match(v))
        for v in values
    )


def _is_categorical(rows: list[dict], column: str) -> bool:
    return bool(_values(rows, column))


def _distinct(rows: list[dict], column: str) -> int:
    return len({str(v) for v in _values(rows, column)})
