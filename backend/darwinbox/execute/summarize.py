"""Turn a result set into a sentence.

A grid of numbers is data, not an answer. Every response leads with prose that states
the finding, so the table and chart below it are supporting detail rather than
something the reader has to interpret unaided.

The model writes it when it is available; when it is not, or when it misbehaves, a
deterministic description takes over. There is always a sentence, and the fallback can
never be wrong because it only restates the shape of what was returned.
"""

from __future__ import annotations

from darwinbox.llm.prompts import SUMMARY_SYSTEM, summary_user

MAX_SUMMARY_TOKENS = 160
MAX_LIST_NAMES = 3


def summarize(
    client, question: str, columns: list[str], rows: list[dict], truncated: bool = False
) -> tuple[str, bool]:
    """Return (summary, came_from_model). Never raises."""
    if not rows:
        return "That query returned no rows.", False

    try:
        reply = client.complete_json(
            SUMMARY_SYSTEM,
            summary_user(question, columns, rows, len(rows), computed_facts(columns, rows)),
            max_tokens=MAX_SUMMARY_TOKENS,
        )
        text = (reply or {}).get("summary")
        if isinstance(text, str) and text.strip():
            return _tidy(text), True
    except Exception:  # noqa: BLE001 - a summary is never worth failing an answer over
        pass

    return describe(columns, rows, truncated), False


def computed_facts(columns: list[str], rows: list[dict]) -> list[str]:
    """Superlatives and totals, computed here rather than left to the model.

    Asked to rank four regions itself, a 7B wrote "North had the highest sales with
    $79,772.70" when North was the lowest of the four -- a confidently wrong sentence
    sitting directly above a chart that showed the opposite. Ranking is arithmetic, so
    it is done in Python and handed over as fact; the model only phrases it.
    """
    facts = [f"row count: {len(rows):,}"]
    labels = [c for c in columns if not _all_numeric(rows, c)]
    key = labels[0] if labels else None

    for column in columns:
        if not _all_numeric(rows, column) or len(rows) < 2:
            continue
        ranked = sorted(rows, key=lambda r: _as_float(r.get(column)), reverse=True)
        total = sum(_as_float(r.get(column)) for r in rows)
        if key:
            facts.append(
                f"{column}: highest is {_render(ranked[0].get(key))} at "
                f"{_render(ranked[0].get(column))}; lowest is "
                f"{_render(ranked[-1].get(key))} at {_render(ranked[-1].get(column))}; "
                f"total {_render(total)}"
            )
        else:
            facts.append(
                f"{column}: max {_render(ranked[0].get(column))}, "
                f"min {_render(ranked[-1].get(column))}, total {_render(total)}"
            )
    return facts


def _tidy(text: str) -> str:
    cleaned = " ".join(text.split())
    return cleaned if cleaned.endswith((".", "!", "?")) else cleaned + "."


def describe(columns: list[str], rows: list[dict], truncated: bool = False) -> str:
    """A factual description built from the result alone, so it cannot be wrong."""
    if not rows:
        return "That query returned no rows."

    count = f"{len(rows):,}{'+' if truncated else ''}"

    if len(rows) == 1 and len(columns) == 1:
        column = columns[0]
        return f"{_label(column)}: {_render(rows[0].get(column))}."

    if len(rows) == 1:
        parts = ", ".join(f"{_label(c)} {_render(rows[0].get(c))}" for c in columns[:4])
        return f"{parts}."

    numeric = [c for c in columns if _all_numeric(rows, c)]
    labels = [c for c in columns if c not in numeric]

    if labels and numeric:
        key, measure = labels[0], numeric[0]
        ranked = sorted(rows, key=lambda r: _as_float(r.get(measure)), reverse=True)
        top, bottom = ranked[0], ranked[-1]
        return (
            f"{count} rows. {_label(measure)} is highest for "
            f"{_render(top.get(key))} at {_render(top.get(measure))} and lowest for "
            f"{_render(bottom.get(key))} at {_render(bottom.get(measure))}."
        )

    if labels:
        key = labels[0]
        names = ", ".join(_render(r.get(key)) for r in rows[:MAX_LIST_NAMES])
        more = f" and {len(rows) - MAX_LIST_NAMES:,} more" if len(rows) > MAX_LIST_NAMES else ""
        return f"{count} rows, including {names}{more}."

    return f"{count} rows across {len(columns)} column(s): {', '.join(columns[:5])}."


def _label(column: str) -> str:
    return column.replace("_", " ").strip().capitalize()


def _all_numeric(rows: list[dict], column: str) -> bool:
    values = [r.get(column) for r in rows if r.get(column) is not None]
    return bool(values) and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in values
    )


def _as_float(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _render(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.2f}".rstrip("0").rstrip(".") if value % 1 else f"{int(value):,}"
    return str(value)
