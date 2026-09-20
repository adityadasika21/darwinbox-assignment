"""Key/value extraction for sheets that are forms rather than tables.

A payslip, a tax computation or a statutory certificate is not relational data. It is
a list of labelled figures laid out for a human to read: sparse rows, merged cells,
section headings, and a value floating somewhere to the right of its label.

Run through table detection such a sheet shatters into a dozen fragments with no
usable headers, which is worse than useless -- it produces confident nonsense. So a
sheet that looks like a form is instead read as what it is: one table of
(section, label, value) rows. That *is* queryable, which is the point. "What is the
total income?" becomes a filter over labels rather than an unanswerable question.
"""

from __future__ import annotations

import re

Grid = list[list[str | None]]

MIN_FORM_ROWS = 6
FORM_COLUMNS = ("section", "item", "value")
MAX_LABEL_CHARS = 90
FORM_MAX_DENSITY = 0.30
FORM_MIN_SPARSE_ROWS = 20
FORM_MIN_PLACEHOLDER = 0.5
# A value is the rightmost cell that reads as a figure, a date or a short token.
_FIGURE = re.compile(r"^[\s₹$£€]*-?[\d,]+(\.\d+)?\s*%?$")
_SHORT_TOKEN = re.compile(r"^[A-Za-z0-9][\w./@-]{0,30}$")


def looks_like_a_form(
    blocks_found: int, placeholder_ratio: float, density: float, rows: int
) -> bool:
    """Decide from the shape of what table detection produced.

    Two symptoms, either alone sufficient:

    * the sheet shattered into several fragments whose columns are mostly col_N. A
      fragment usually *does* latch onto some nearby label as a header, so "found no
      header" is a poor proxy -- what matters is that the names it produced carry no
      information.
    * or it came out as one long, very sparse block, with no column structure found
      across many rows. The row floor keeps a small sparse table out: a statutory
      certificate runs to dozens of rows, an eight-row table does not.
    """
    if rows < MIN_FORM_ROWS:
        return False
    # Either way, a usable header means this is a table. Sparsity alone is not a form:
    # a 60-column open-data extract with forty optional columns -- vehicle_defect,
    # towed_by, area_00_i .. area_06_i, nearly all empty -- is under any density
    # threshold worth setting, and reading it as (section, item, value) does not lose
    # a little precision, it destroys the table. What a form actually lacks is column
    # names, which is what the placeholder ratio measures.
    if placeholder_ratio < FORM_MIN_PLACEHOLDER:
        return False
    if blocks_found >= 3:
        return True
    return blocks_found <= 2 and density < FORM_MAX_DENSITY and rows >= FORM_MIN_SPARSE_ROWS


def extract(grid: Grid) -> list[list[str]]:
    """Return (section, label, value) rows for a form-shaped grid.

    The section is the most recent heading -- a row with a single populated cell --
    so that "Gross Salary" under "OLD REGIME" stays distinguishable from the same
    label elsewhere in the sheet.
    """
    out: list[list[str]] = []
    section = ""

    for row in grid:
        cells = [(i, c.strip()) for i, c in enumerate(row) if c and c.strip()]
        if not cells:
            continue

        heading = _as_heading(cells)
        if heading is not None:
            section = heading
            continue

        label, value = _split(cells)
        if label:
            out.append([section, label, value])

    return out


def _as_heading(cells: list[tuple[int, str]]) -> str | None:
    """A section heading: text on its own, or a numbered heading like "3  DEDUCTIONS".

    Forms number their sections, so a heading often arrives as two cells. Treating it
    as a label/value pair loses it, and every row beneath then inherits the wrong
    section -- which is exactly what "TOTAL DEDUCTIONS" filed under "COMPUTATION OF
    TOTAL INCOME" looked like.
    """
    texts = [t for _, t in cells]
    if len(texts) == 1:
        only = texts[0]
        return only if not _FIGURE.match(only) and len(only) <= MAX_LABEL_CHARS else None

    if len(texts) == 2 and _is_ordinal(texts[0]) and not _FIGURE.match(texts[1]):
        return texts[1][:MAX_LABEL_CHARS]
    return None


def _split(cells: list[tuple[int, str]]) -> tuple[str, str]:
    """Pick the label and its value from one row's populated cells."""
    # The value is the rightmost cell that reads like one; everything left of it that
    # is prose becomes the label, so numbering columns ("1", "(a)") fall away.
    value_index = None
    for index in range(len(cells) - 1, 0, -1):
        text = cells[index][1]
        if _FIGURE.match(text) or _SHORT_TOKEN.match(text):
            value_index = index
            break

    if value_index is None:
        value_index = len(cells) - 1

    # Figures left of the value belong to other columns -- a two-regime tax sheet puts
    # the old-regime amount beside the new one -- and must not end up inside the label,
    # which produced items like "Total Expected Income 587899".
    label_parts = [
        text
        for _, text in cells[:value_index]
        if not _is_ordinal(text) and not _FIGURE.match(text)
    ]
    label = " ".join(label_parts).strip(" :.-")
    value = cells[value_index][1]

    if not label:
        return "", ""
    return label[:MAX_LABEL_CHARS], value


def _is_ordinal(text: str) -> bool:
    """Row numbering a form uses for reference: 1, 2, (a), (iv), A."""
    stripped = text.strip("()[]. ")
    return bool(re.fullmatch(r"\d{1,3}|[a-zA-Z]|[ivxIVX]{1,4}", stripped))
