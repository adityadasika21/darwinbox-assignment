"""Grid -> TableBlock[].

The single hardest thing about real uploaded spreadsheets is that one sheet is not
one table. This module finds the rectangular regions that *are* tables, decides
which row inside each is the header, and drops title and footnote rows.

It knows nothing about pandas, DuckDB or the LLM: in, a grid of strings; out,
a TableBlock plus its data rows as strings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from darwinbox.ingest import forms
from darwinbox.models import SourceRef, TableBlock

Grid = list[list[str | None]]

MAX_HEADER_SCAN = 8
MAX_COLUMN_NAME = 48
MAX_HEADER_BAND = 3

_PLACEHOLDER_NAME = re.compile(r"col_\d+")
MIN_BLOCK_ROWS = 2
MIN_POPULATED_CELLS = 3
DTYPE_SAMPLE_ROWS = 50
MIN_ROWS_BELOW_HEADER = 2
HEADER_SCORE_THRESHOLD = 2

_JUNK_FIRST_CELL = re.compile(r"(?i)^(total|grand total|subtotal|sum|note|source|footnote)\b")
_NUMERIC_LIKE = re.compile(r"^[\s$€£₹(]*-?[\d,]+(\.\d+)?\s*[%)]?$")
_DATE_LIKE = re.compile(r"^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}")


@dataclass(frozen=True)
class Region:
    """Half-open rectangle in original-grid coordinates."""

    r0: int
    r1: int
    c0: int
    c1: int

    @property
    def n_rows(self) -> int:
        return self.r1 - self.r0

    @property
    def n_cols(self) -> int:
        return self.c1 - self.c0


@dataclass
class DetectedBlock:
    """A TableBlock plus the raw string rows beneath its header."""

    block: TableBlock
    rows: list[list[str | None]]


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def detect_blocks(
    filename: str,
    sheet: str | None,
    grid: Grid,
    taken_ids: set[str] | None = None,
    taken_aliases: set[str] | None = None,
) -> list[DetectedBlock]:
    """Split one grid into table blocks with headers resolved and junk trimmed.

    ``taken_ids`` / ``taken_aliases`` carry names already used elsewhere in the
    session so that table_id and alias stay globally unique.
    """
    taken_ids = taken_ids if taken_ids is not None else set()
    taken_aliases = taken_aliases if taken_aliases is not None else set()

    mask = _occupancy(grid)
    whole = Region(0, len(grid), 0, max((len(r) for r in grid), default=0))
    # Merge before filtering: a one-row continuation is not viable on its own and
    # would be discarded before it could be rejoined to the table it belongs to.
    regions = _merge_continuations(grid, _split(grid, mask, whole))
    viable = [r for r in regions if _is_viable(mask, r)]
    # Regions too small to be a table are still worth keeping when the sheet has a
    # real table elsewhere: they hold the invoice number, the claim period, the
    # approver. Dropped here, that information is simply gone.
    discarded = [r for r in regions if r not in viable]
    regions = viable

    out: list[DetectedBlock] = []
    orphans: list[list[str | None]] = [
        _row_values(grid, r, region)
        for region in discarded
        for r in range(region.r0, region.r1)
    ]
    for index, region in enumerate(regions):
        detected = _build_block(
            filename, sheet, grid, region, index, taken_ids, taken_aliases
        )
        if detected is not None:
            out.append(detected)
        else:
            # A one-row region has its only row taken as a header and is then left with
            # no data, so it disappears -- which is how an invoice lost its "Invoice
            # No / Date" line entirely. Keep the rows for the details table.
            orphans.extend(
                _row_values(grid, r, region) for r in range(region.r0, region.r1)
            )

    form = _as_form(filename, sheet, grid, out, taken_ids, taken_aliases)
    if form is not None:
        return [form]
    return _collapse_metadata(
        filename, sheet, grid, out, orphans, taken_ids, taken_aliases
    )


MIN_SUBSTANTIAL_ROWS = 3
MAX_FRAGMENT_ROWS = 2
TOTALS_BLOCK_SHARE = 0.6

_TOTALS_LABEL = re.compile(
    r"(?i)^(sub\s*total|total|grand\s*total|net|gross|balance|c?gst|sgst|igst|vat|tax|"
    r"discount|freight|round\s*off|amount\s+in\s+words)\b"
)


def _is_totals_block(detected: DetectedBlock) -> bool:
    """True for the Subtotal / GST / TOTAL block that trails an invoice's line items.

    Those rows are labelled figures, not another table of the same kind, and left
    alone they become a separate table of col_1 / col_2. Folding them into the
    document's details keeps the numbers a reader actually wants.
    """
    labelled = 0
    for row in detected.rows:
        first = next((v for v in row if v), None)
        if first and _TOTALS_LABEL.match(str(first)):
            labelled += 1
    return bool(detected.rows) and labelled / len(detected.rows) >= TOTALS_BLOCK_SHARE


def _collapse_metadata(
    filename: str,
    sheet: str | None,
    grid: Grid,
    blocks: list[DetectedBlock],
    orphans: list[list[str | None]],
    taken_ids: set[str],
    taken_aliases: set[str],
) -> list[DetectedBlock]:
    """Fold a sheet's header block into one form table beside its real table.

    An invoice, an expense claim or a bill of quantities opens with a few labelled
    fields -- claim number, period, approver -- and then gets on with the line items.
    Those opening rows are not tables: split on their blank columns they become a
    scatter of one-row fragments with names like "employee / ravi_kumar". Collapsing
    them into a single (section, item, value) table keeps the information and stops
    the sheet looking shattered.
    """
    substantial = [
        b
        for b in blocks
        if b.block.n_rows >= MIN_SUBSTANTIAL_ROWS and not _is_totals_block(b)
    ]
    fragments = [
        b
        for b in blocks
        if b.block.n_rows <= MAX_FRAGMENT_ROWS or _is_totals_block(b)
    ]
    if not substantial or (not fragments and not orphans):
        return blocks

    rows: list[list[str]] = list(forms.extract(orphans))
    for fragment in fragments:
        source = fragment.block.source
        first = source.header_row if source.header_row >= 0 else source.data_start_row
        span = [
            [_cell(grid, r, c) for c in range(source.col_start, source.col_end + 1)]
            for r in range(first, source.data_end_row + 1)
        ]
        rows.extend(forms.extract(span))

    if not rows:
        return blocks

    for fragment in fragments:
        taken_ids.discard(fragment.block.table_id)
        taken_aliases.discard(fragment.block.alias)

    kept = [b for b in blocks if b not in fragments]
    details = _form_block(filename, sheet, rows, taken_ids, taken_aliases, suffix="details")
    return [details, *kept]


def _as_form(
    filename: str,
    sheet: str | None,
    grid: Grid,
    blocks: list[DetectedBlock],
    taken_ids: set[str],
    taken_aliases: set[str],
) -> DetectedBlock | None:
    """Re-read the sheet as (section, label, value) when it is a form, not a table.

    Returning one queryable table beats returning a dozen fragments with no headers.
    The alternative -- leaving the fragments in place -- puts confident nonsense in
    front of the user and in front of the planner.
    """
    populated = sum(1 for row in grid for cell in row if cell)
    cells = sum(len(row) for row in grid) or 1
    names = [name for b in blocks for name in b.block.columns]
    placeholder = (
        sum(1 for n in names if _PLACEHOLDER_NAME.fullmatch(n)) / len(names) if names else 0.0
    )

    if not forms.looks_like_a_form(len(blocks), placeholder, populated / cells, len(grid)):
        return None

    rows = forms.extract(grid)
    if len(rows) < forms.MIN_FORM_ROWS:
        return None

    # The fragments are being replaced, so release the names they reserved.
    for block in blocks:
        taken_ids.discard(block.block.table_id)
        taken_aliases.discard(block.block.alias)

    return _form_block(filename, sheet, rows, taken_ids, taken_aliases)


def _form_block(
    filename: str,
    sheet: str | None,
    rows: list[list[str]],
    taken_ids: set[str],
    taken_aliases: set[str],
    suffix: str = "form",
) -> DetectedBlock:
    """Wrap extracted (section, item, value) rows as a table block."""
    columns = list(forms.FORM_COLUMNS)
    stem = _slug(filename.rsplit(".", 1)[0])
    table_id = _unique(f"{stem}__{_slug(sheet or 'csv')}__{suffix}", taken_ids)
    base = _derive_alias(filename, sheet, 0)
    alias = _unique(base if suffix == "form" else f"{base}_{suffix}", taken_aliases)
    source = SourceRef(
        filename=filename, sheet=sheet, block_index=0, header_row=-1,
        data_start_row=0, data_end_row=max(0, len(rows) - 1),
        col_start=0, col_end=len(columns) - 1,
    )
    block = TableBlock(
        table_id=table_id, alias=alias, source=source,
        columns=columns, raw_columns=list(columns), n_rows=len(rows),
    )
    return DetectedBlock(block=block, rows=[list(r) for r in rows])


# --------------------------------------------------------------------------- #
# Region finding
# --------------------------------------------------------------------------- #


def _occupancy(grid: Grid) -> list[list[bool]]:
    width = max((len(r) for r in grid), default=0)
    return [
        [bool(row[c]) if c < len(row) and row[c] is not None else False for c in range(width)]
        for row in grid
    ]


def _split(grid: Grid, mask: list[list[bool]], region: Region, depth: int = 0) -> list[Region]:
    """Recursively cut a region on fully empty rows, then on fully empty columns."""
    if depth > 6 or region.n_rows <= 0 or region.n_cols <= 0:
        return [region]

    bands = _cut_rows(mask, region)
    if len(bands) > 1:
        return [out for band in bands for out in _split(grid, mask, band, depth + 1)]

    columns = _cut_cols(mask, region)
    if len(columns) > 1:
        return [out for col in columns for out in _split(grid, mask, col, depth + 1)]

    return [_trim(mask, region)]


def _cut_rows(mask: list[list[bool]], region: Region) -> list[Region]:
    keep = [
        r
        for r in range(region.r0, region.r1)
        if any(mask[r][c] for c in range(region.c0, min(region.c1, len(mask[r]))))
    ]
    return [
        Region(a, b + 1, region.c0, region.c1) for a, b in _runs(keep)
    ]


def _cut_cols(mask: list[list[bool]], region: Region) -> list[Region]:
    keep = [
        c
        for c in range(region.c0, region.c1)
        if any(
            mask[r][c]
            for r in range(region.r0, region.r1)
            if c < len(mask[r])
        )
    ]
    return [
        Region(region.r0, region.r1, a, b + 1) for a, b in _runs(keep)
    ]


def _runs(values: list[int]) -> list[tuple[int, int]]:
    """Collapse a sorted index list into contiguous (start, end-inclusive) runs."""
    if not values:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = values[0]
    for v in values[1:]:
        if v == prev + 1:
            prev = v
            continue
        runs.append((start, prev))
        start = prev = v
    runs.append((start, prev))
    return runs


def _trim(mask: list[list[bool]], region: Region) -> Region:
    rows = _cut_rows(mask, region)
    cols = _cut_cols(mask, region)
    if not rows or not cols:
        return region
    return Region(rows[0].r0, rows[-1].r1, cols[0].c0, cols[-1].c1)


def _is_viable(mask: list[list[bool]], region: Region) -> bool:
    if region.n_rows < MIN_BLOCK_ROWS or region.n_cols < 1:
        return False
    populated = sum(
        1
        for r in range(region.r0, region.r1)
        for c in range(region.c0, min(region.c1, len(mask[r])))
        if mask[r][c]
    )
    return populated >= MIN_POPULATED_CELLS


def _merge_continuations(grid: Grid, regions: list[Region]) -> list[Region]:
    """Rejoin regions that a stray blank row split out of one table.

    A blank row separates two tables, which is why we cut on it -- but a blank row
    also appears *inside* one table, and then cutting is wrong: "id,name / 1,A / /
    2,B" becomes three regions and only the first keeps its header, losing every
    other row. The discriminator is that a real second table starts with a header
    and a continuation does not, so regions sharing a column span are rejoined when
    the lower one has no detectable header of its own.
    """
    if len(regions) < 2:
        return regions

    merged: list[Region] = [regions[0]]
    for region in regions[1:]:
        previous = merged[-1]
        same_span = previous.c0 == region.c0 and previous.c1 == region.c1
        if same_span and (
            _pick_header(grid, region)[0] < 0
            or _spans_a_split_header(grid, previous, region)
        ):
            merged[-1] = Region(previous.r0, region.r1, previous.c0, previous.c1)
        else:
            merged.append(region)
    return merged


def _spans_a_split_header(grid: Grid, upper: Region, lower: Region) -> bool:
    """True when a blank row fell *inside* a multi-row header, splitting it in two.

    Real sheets put a spacer between the group label and the field names -- the row
    that says "Name of Official | Designation | Pay Matrix" and the row beneath it
    carrying "Month/Year | Pay". The splitter cuts there, so the band logic never
    sees them together and the upper row is lost with the title block.
    """
    last = upper.r1 - 1
    first = lower.r0
    if last < upper.r0 or first >= lower.r1:
        return False

    upper_cells = [v for v in _row_values(grid, last, upper) if v]
    if len(upper_cells) < 2:
        return False  # a lone cell is a title, not a header
    return _is_group_label(grid, upper, last, first)


# --------------------------------------------------------------------------- #
# Header detection
# --------------------------------------------------------------------------- #


def _cell(grid: Grid, r: int, c: int) -> str | None:
    if r >= len(grid) or c >= len(grid[r]):
        return None
    return grid[r][c]


def _row_values(grid: Grid, r: int, region: Region) -> list[str | None]:
    return [_cell(grid, r, c) for c in range(region.c0, region.c1)]


def score_header_row(grid: Grid, region: Region, r: int) -> int:
    """Score how much row ``r`` looks like the header of ``region`` (see SPEC 5.3)."""
    values = _row_values(grid, r, region)
    populated = [v for v in values if v]
    if not populated:
        return -99

    score = 0

    non_numeric = sum(1 for v in populated if not _looks_numeric(v))
    if non_numeric / len(populated) >= 0.70:
        score += 2

    if len(populated) == len(values):
        score += 1

    if _consistent_dtypes_below(grid, region, r):
        score += 2

    # Extension to SPEC 5.3: a headerless numeric table also has consistent dtypes
    # below its first row, so "consistent below" alone cannot separate the two. The
    # discriminator is whether the candidate row looks unlike the data under it.
    if _looks_like_data_row(grid, region, r):
        score -= 3

    if len(populated) == 1 and region.n_cols >= 3:
        score -= 3  # a lone cell across a wide region is a title, not a header

    lowered = [v.strip().lower() for v in populated]
    if len(set(lowered)) != len(lowered):
        score -= 2

    return score


def _looks_numeric(value: str) -> bool:
    return bool(_NUMERIC_LIKE.match(value.strip())) or bool(_DATE_LIKE.match(value.strip()))


def _consistent_dtypes_below(grid: Grid, region: Region, header_row: int) -> bool:
    """True when each column below ``header_row`` holds one consistent kind of value."""
    start, end = header_row + 1, min(region.r1, header_row + 1 + DTYPE_SAMPLE_ROWS)
    if end - start < 2:
        return False

    consistent = 0
    considered = 0
    for c in range(region.c0, region.c1):
        kinds = {_kind(_cell(grid, r, c)) for r in range(start, end)}
        kinds.discard("empty")
        if not kinds:
            continue
        considered += 1
        if len(kinds) == 1:
            consistent += 1
    return considered > 0 and consistent / considered >= 0.75


def _looks_like_data_row(grid: Grid, region: Region, r: int) -> bool:
    """True when row r has the same per-column value kinds as the rows beneath it."""
    start, end = r + 1, min(region.r1, r + 1 + DTYPE_SAMPLE_ROWS)
    if end - start < 2:
        return False

    matched = considered = 0
    for c in range(region.c0, region.c1):
        kinds = {_kind(_cell(grid, rr, c)) for rr in range(start, end)}
        kinds.discard("empty")
        own = _kind(_cell(grid, r, c))
        if len(kinds) != 1 or own == "empty":
            continue
        # Only typed columns are evidence. A text column under a text candidate is
        # uninformative -- counting it would penalise the real header of a table
        # whose columns happen to all be strings.
        if next(iter(kinds)) == "text":
            continue
        considered += 1
        matched += own in kinds
    return considered > 0 and matched / considered >= 0.75


def _kind(value: str | None) -> str:
    if not value:
        return "empty"
    text = value.strip()
    if _DATE_LIKE.match(text):
        return "date"
    if _NUMERIC_LIKE.match(text):
        return "number"
    return "text"


def _pick_header(grid: Grid, region: Region) -> tuple[int, int]:
    """Return (header_row, data_start_row); header_row is -1 when none scores well."""
    limit = min(region.r0 + MAX_HEADER_SCAN, region.r1)
    limit = max(region.r0 + 1, min(limit, region.r1 - MIN_ROWS_BELOW_HEADER))
    best_row, best_score = -1, HEADER_SCORE_THRESHOLD - 1
    for r in range(region.r0, limit):
        score = score_header_row(grid, region, r)
        if score > best_score:
            best_row, best_score = r, score
    if best_row < 0:
        return -1, region.r0
    start = _header_band_start(grid, region, best_row)
    return start, _header_band_end(grid, region, best_row) + 1


def _header_band_start(grid: Grid, region: Region, best: int) -> int:
    """First row of a multi-row header ending at ``best``.

    The group label usually sits *above* the detail names -- "Room | Room | Finishes"
    over "No. | Name | Floor" -- and scores worse than them, so the search lands on the
    lower row and the upper one is lost. Walking back up recovers it.
    """
    start = best
    for r in range(best - 1, max(region.r0 - 1, best - MAX_HEADER_BAND - 1), -1):
        values = [v for v in _row_values(grid, r, region) if v]
        # A spacer between the group label and the field names is normal, and it is
        # what split the header in the first place; stepping over it is the whole
        # point of having merged the two regions back together.
        if not values:
            continue
        # A lone cell across the row is a title, not part of the header.
        if len(values) < 2 or not _is_group_label(grid, region, r, start):
            break
        start = r
    return start


def _is_group_label(grid: Grid, region: Region, r: int, below: int) -> bool:
    """True when row r labels groups of the columns named on row ``below``."""
    values = [v for v in _row_values(grid, r, region) if v]
    non_numeric = sum(1 for v in values if not _looks_numeric(v))
    if non_numeric / len(values) < 0.70:
        return False

    # A trailing colon marks a field label, not a column group: "Invoice No:" and
    # "Date:" head a metadata line, where "Pay Matrix" and "Earnings" head columns.
    # Without this an invoice's header line is swallowed into the line-item table and
    # its number and date are lost outright.
    if any(v.rstrip().endswith(":") for v in values):
        return False

    # Either the label was repeated across the columns it spans, or it was merged and
    # so covers only some of them while the row below covers the rest.
    if _has_adjacent_duplicates(grid, region, r):
        return True
    populated = _populated_columns(grid, region, r)
    return bool(populated) and populated < _populated_columns(grid, region, below)


def _header_band_end(grid: Grid, region: Region, start: int) -> int:
    """Last row of a multi-row header beginning at ``start``.

    Real forms spread a header over two or three rows -- a group label on one, the
    field names beneath it, and often a row of column numbers under that. Taking only
    one of them names half the columns after a title and leaves the rest as col_N.
    """
    covered = _populated_columns(grid, region, start)
    end = start
    for r in range(start + 1, min(region.r1 - MIN_ROWS_BELOW_HEADER, start + MAX_HEADER_BAND)):
        if not _is_header_continuation(grid, region, r, covered):
            break
        covered |= _populated_columns(grid, region, r)
        end = r
    return end


def _populated_columns(grid: Grid, region: Region, r: int) -> set[int]:
    return {i for i, v in enumerate(_row_values(grid, r, region)) if v}


def _is_header_continuation(grid: Grid, region: Region, r: int, covered: set[int]) -> bool:
    """True when row r completes the header above it rather than starting the data.

    The signal is that a continuation *fills the gaps*: a form writes a group label
    across some columns on one row and the field names for the remaining columns
    beneath it. A data row, by contrast, populates the same columns the header does.
    Testing "does this look like labels" instead was not enough -- in an all-text
    table the first data row looks exactly like a header.
    """
    values = [v for v in _row_values(grid, r, region) if v]
    if not values:
        return False

    # A row of consecutive small integers under a header is column numbering -- unless
    # it reads as data, which "1,2,3" above "4,5,6" does. This returned early, before
    # the data-row test below could object, and so ate the first row of any numeric
    # table whose values happened to start small and ascend. The row vanished from the
    # answer with nothing to show it ever existed, which is the worst way to be wrong.
    if _is_numbering_row(values) and not _looks_like_data_row(grid, region, r):
        return True

    non_numeric = sum(1 for v in values if not _looks_numeric(v))
    if non_numeric / len(values) < 0.70 or _looks_like_data_row(grid, region, r):
        return False

    # Case 1: the row above is a group label spanning columns it left empty, and this
    # row fills them in. That is what an unmerged Excel header looks like.
    populated = _populated_columns(grid, region, r)
    if populated - covered and len(populated & covered) <= len(populated) / 2:
        return True

    # Case 2: both rows are fully populated because the group label was repeated
    # rather than merged -- "Room | Room | Finishes | Finishes | Area", which is how a
    # merged header survives a CSV export. Adjacent duplicates above are the tell, and
    # they are what separates a stacked header from the first row of an all-text table.
    return _has_adjacent_duplicates(grid, region, r - 1)


def _has_adjacent_duplicates(grid: Grid, region: Region, r: int) -> bool:
    """True when row r repeats a value across neighbouring columns."""
    if r < region.r0:
        return False
    values = [v.strip().lower() if v else None for v in _row_values(grid, r, region)]
    return any(
        a is not None and a == b for a, b in zip(values, values[1:], strict=False)
    )


def _is_numbering_row(values: list[str]) -> bool:
    try:
        numbers = [int(v.strip()) for v in values]
    except ValueError:
        return False
    return (
        len(numbers) >= 3
        and numbers == sorted(numbers)
        and numbers[0] <= 2
        and all(b - a <= 2 for a, b in zip(numbers, numbers[1:], strict=False))
    )


# --------------------------------------------------------------------------- #
# Column names
# --------------------------------------------------------------------------- #


def normalize_column_names(raw: list[str | None]) -> list[str]:
    """lowercase, non-alphanumerics to _, collapse, strip leading digits, dedupe."""
    out: list[str] = []
    seen: dict[str, int] = {}
    for i, value in enumerate(raw):
        name = _normalize_token(value) or f"col_{i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        out.append(name)
    return out


def _normalize_token(value: str | None) -> str:
    if not value:
        return ""
    name = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip().lower())
    name = re.sub(r"_+", "_", name).strip("_")
    # Strip leading digits so the name is a usable identifier -- but a header that is
    # *only* digits is a real column (day 1, week 3, year 2024 across a register), and
    # erasing it turned every day of an attendance sheet into col_N.
    stripped = re.sub(r"^\d+", "", name).strip("_")
    name = stripped if stripped else (f"n{name}" if name else "")
    # A form puts whole sentences where a header belongs; an unbounded name is
    # unreadable in the UI and eats the prompt budget in the data dictionary.
    if len(name) > MAX_COLUMN_NAME:
        name = name[:MAX_COLUMN_NAME].rsplit("_", 1)[0] or name[:MAX_COLUMN_NAME]
    return name


def _slug(value: str) -> str:
    return _normalize_token(value) or "x"


def _unique(candidate: str, taken: set[str]) -> str:
    name, n = candidate, 2
    while name in taken:
        name = f"{candidate}_{n}"
        n += 1
    taken.add(name)
    return name


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def _build_block(
    filename: str,
    sheet: str | None,
    grid: Grid,
    region: Region,
    index: int,
    taken_ids: set[str],
    taken_aliases: set[str],
) -> DetectedBlock | None:
    header_row, data_start = _pick_header(grid, region)

    if header_row >= 0:
        raw_columns = _combine_header_rows(grid, region, header_row, data_start - 1)
    else:
        raw_columns = [f"col_{i + 1}" for i in range(region.n_cols)]

    columns = normalize_column_names(
        [v or None for v in raw_columns] if header_row >= 0 else raw_columns
    )

    rows = [_row_values(grid, r, region) for r in range(data_start, region.r1)]
    rows = _drop_trailing_junk(rows)
    rows = [r for r in rows if any(v for v in r)]
    if not rows:
        return None

    file_slug = _slug(filename.rsplit(".", 1)[0])
    sheet_slug = _slug(sheet) if sheet else "csv"
    table_id = _unique(f"{file_slug}__{sheet_slug}__b{index}", taken_ids)
    alias = _unique(_derive_alias(filename, sheet, index), taken_aliases)

    source = SourceRef(
        filename=filename,
        sheet=sheet,
        block_index=index,
        header_row=header_row,
        data_start_row=data_start,
        data_end_row=data_start + len(rows) - 1,
        col_start=region.c0,
        col_end=region.c1 - 1,
    )
    block = TableBlock(
        table_id=table_id,
        alias=alias,
        source=source,
        columns=columns,
        raw_columns=raw_columns,
        n_rows=len(rows),
    )
    return DetectedBlock(block=block, rows=rows)


def _combine_header_rows(grid: Grid, region: Region, start: int, end: int) -> list[str]:
    """Join a multi-row header into one name per column, skipping numbering rows."""
    rows = [
        _row_values(grid, r, region)
        for r in range(start, end + 1)
        if not _is_numbering_row([v for v in _row_values(grid, r, region) if v])
    ]
    if not rows:
        rows = [_row_values(grid, start, region)]

    out: list[str] = []
    for c in range(region.n_cols):
        parts: list[str] = []
        for row in rows:
            value = row[c] if c < len(row) else None
            if value and value not in parts:
                parts.append(value.strip())
        out.append(" ".join(parts))
    return out


def _derive_alias(filename: str, sheet: str | None, index: int) -> str:
    """Short human name for the LLM: prefer the sheet name, else the file stem."""
    base = _slug(sheet) if sheet and not _is_generic_sheet(sheet) else None
    base = base or _slug(filename.rsplit(".", 1)[0])
    return base if index == 0 else f"{base}_{index + 1}"


def _is_generic_sheet(sheet: str) -> bool:
    return bool(re.fullmatch(r"(?i)sheet\s*\d*", sheet.strip()))


def _drop_trailing_junk(rows: list[list[str | None]]) -> list[list[str | None]]:
    """Remove trailing 'Total:' / 'Source:' style rows (SPEC 5.4).

    Tradeoff: the spec says drop only when the first column alone is populated, but
    a real total row carries its total, so we also accept a junk-word first cell on a
    mostly-empty row. A genuine data row named "Total" keeps its other columns filled.
    """
    end = len(rows)
    while end > 0:
        row = rows[end - 1]
        populated = [i for i, v in enumerate(row) if v]
        if not populated:
            end -= 1
            continue
        first = row[0] or ""
        # In a one-column table every row is "only the first column populated", so
        # this rule would delete the entire table. It only means anything when there
        # are other columns that could have been filled.
        only_first = populated == [0] and len(row) > 1
        incomplete = len(populated) < len(row)
        if only_first or (_JUNK_FIRST_CELL.match(first) and incomplete):
            end -= 1
            continue
        break
    return rows[:end]
