"""Phase 1 acceptance criteria (SPEC 5). No LLM, no DuckDB."""

from __future__ import annotations

import contextlib
import pathlib

import pytest
from darwinbox.ingest.blocks import detect_blocks, normalize_column_names
from darwinbox.ingest.loader import load_file

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def load_blocks(name: str):
    data = (FIXTURES / name).read_bytes()
    out = []
    ids: set[str] = set()
    aliases: set[str] = set()
    for loaded in load_file(name, data):
        out.extend(detect_blocks(name, loaded.sheet, loaded.grid, ids, aliases))
    return out


# --------------------------------------------------------------------------- #
# clean.csv -> 1 block, header row 0
# --------------------------------------------------------------------------- #


def test_clean_csv_single_block_header_row_zero():
    blocks = load_blocks("clean.csv")
    assert len(blocks) == 1

    block = blocks[0].block
    assert block.source.header_row == 0
    assert block.source.data_start_row == 1
    assert block.columns == ["order_id", "order_date", "amount"]
    assert block.n_rows == 5
    assert blocks[0].rows[0] == ["ORD-1001", "2024-01-05", "120.50"]


# --------------------------------------------------------------------------- #
# no_header.csv -> 1 block, header_row == -1, synthesised names
# --------------------------------------------------------------------------- #


def test_no_header_csv_synthesises_names():
    blocks = load_blocks("no_header.csv")
    assert len(blocks) == 1

    block = blocks[0].block
    assert block.source.header_row == -1
    assert block.source.data_start_row == 0
    assert block.columns == ["col_1", "col_2", "col_3"]
    assert block.n_rows == 5  # no row was consumed as a header


# --------------------------------------------------------------------------- #
# two_tables_one_sheet.xlsx -> 2 blocks
# --------------------------------------------------------------------------- #


def test_two_tables_in_one_sheet_are_split():
    blocks = load_blocks("two_tables_one_sheet.xlsx")
    assert len(blocks) == 2

    first, second = blocks[0].block, blocks[1].block

    assert first.columns == ["order_id", "region", "amount"]
    assert first.source.header_row == 0
    assert first.n_rows == 4

    assert second.columns == ["emp_id", "dept"]
    assert second.source.header_row == 6
    assert second.source.data_start_row == 7
    assert second.n_rows == 3

    # The narrower second table must not inherit the first table's third column.
    assert second.source.col_end == 1
    assert first.table_id != second.table_id
    assert first.alias != second.alias


# --------------------------------------------------------------------------- #
# title_and_footnote.xlsx -> 1 block, header on row 3, footnote dropped
# --------------------------------------------------------------------------- #


def test_title_rows_skipped_and_footer_dropped():
    blocks = load_blocks("title_and_footnote.xlsx")
    assert len(blocks) == 1

    block = blocks[0].block
    assert block.source.header_row == 3
    assert block.source.data_start_row == 4
    assert block.columns == ["region", "quarter", "revenue"]

    assert block.n_rows == 4  # Total: and Source: rows removed
    first_cells = [row[0] for row in blocks[0].rows]
    assert first_cells == ["North", "South", "East", "West"]
    assert not any(str(c).lower().startswith(("total", "source")) for c in first_cells)


# --------------------------------------------------------------------------- #
# merged_header.xlsx -> 1 block, no null column names
# --------------------------------------------------------------------------- #


def test_merged_header_yields_no_null_column_names():
    blocks = load_blocks("merged_header.xlsx")
    assert len(blocks) == 1

    block = blocks[0].block
    assert block.source.header_row == 0
    assert len(block.columns) == 3
    assert all(c and c.strip() for c in block.columns)
    assert len(set(block.columns)) == 3
    assert block.n_rows == 4


# --------------------------------------------------------------------------- #
# Column name normalization (SPEC 5.5)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (["Order ID"], ["order_id"]),
        (["  Amount (USD) "], ["amount_usd"]),
        (["2024 Revenue"], ["revenue"]),
        (["a", "a", "a"], ["a", "a_2", "a_3"]),
        ([None, ""], ["col_1", "col_2"]),
        (["Employee//Name"], ["employee_name"]),
    ],
)
def test_normalize_column_names(raw, expected):
    assert normalize_column_names(raw) == expected


# --------------------------------------------------------------------------- #
# Encoding, delimiter and shape edge cases
#
# Each of these was a real bug found by feeding the loader adversarial files.
# All three failure modes were SILENT -- the file ingested and produced wrong
# data rather than an error, which is the worst outcome this system can have.
# --------------------------------------------------------------------------- #


def blocks_from_bytes(name: str, data: bytes):
    out = []
    ids: set[str] = set()
    aliases: set[str] = set()
    for loaded in load_file(name, data):
        out.extend(detect_blocks(name, loaded.sheet, loaded.grid, ids, aliases))
    return out


def test_utf16_is_not_silently_read_as_latin1():
    # latin-1 decodes every byte and never raises, so UTF-16 used to become
    # "i\x00d\x00,\x00n\x00a\x00m\x00e" and be ingested as real column names.
    data = "id,name,qty\n1,Alpha,5\n2,Beta,6\n3,Gamma,7\n".encode("utf-16")
    (block,) = blocks_from_bytes("utf16.csv", data)
    assert block.block.columns == ["id", "name", "qty"]
    assert block.rows[0] == ["1", "Alpha", "5"]


def test_utf16_without_a_bom_is_still_detected():
    data = "id,name\n1,Alpha\n2,Beta\n3,Gamma\n".encode("utf-16-le")
    (block,) = blocks_from_bytes("utf16le.csv", data)
    assert block.block.columns == ["id", "name"]


def test_latin1_accents_still_decode():
    data = "id,city\n1,Köln\n2,Zürich\n3,Málaga\n".encode("latin-1")
    (block,) = blocks_from_bytes("latin1.csv", data)
    assert [r[1] for r in block.rows] == ["Köln", "Zürich", "Málaga"]


@pytest.mark.parametrize("delimiter", [";", "\t", "|"])
def test_delimiters_are_sniffed(delimiter):
    text = delimiter.join(["id", "name", "amount"]) + "\n"
    text += "".join(delimiter.join([str(i), f"N{i}", f"{i}.5"]) + "\n" for i in range(1, 5))
    (block,) = blocks_from_bytes("d.csv", text.encode())
    assert block.block.columns == ["id", "name", "amount"]


def test_single_column_table_survives():
    # Every row of a one-column table is "only the first column populated", which
    # the trailing-junk rule used to read as a footnote -- deleting the whole table.
    data = b"employee_id\nEMP-1\nEMP-2\nEMP-3\nEMP-4\n"
    (block,) = blocks_from_bytes("onecol.csv", data)
    assert block.block.columns == ["employee_id"]
    assert block.block.n_rows == 4


def test_quoted_fields_with_embedded_newlines_and_commas():
    data = b'id,address\n1,"12 Main St\nApt 4"\n2,"5 Oak Rd, Unit 2"\n3,"9 Elm"\n'
    (block,) = blocks_from_bytes("quoted.csv", data)
    assert block.block.n_rows == 3
    assert block.rows[0][1] == "12 Main St\nApt 4"
    assert block.rows[1][1] == "5 Oak Rd, Unit 2"


def test_ragged_rows_are_padded_not_rejected():
    (block,) = blocks_from_bytes("ragged.csv", b"a,b,c\n1,2,3\n4,5\n6,7,8,9\n")
    assert len(block.block.columns) == 4


def test_empty_and_header_only_files_yield_no_table():
    assert blocks_from_bytes("empty.csv", b"") == []
    assert blocks_from_bytes("headers.csv", b"id,name,amount\n") == []


def test_duplicate_headers_are_deduplicated():
    (block,) = blocks_from_bytes("dupes.csv", b"id,name,name,id\n1,A,B,2\n3,C,D,4\n5,E,F,6\n")
    assert block.block.columns == ["id", "name", "name_2", "id_2"]


# --------------------------------------------------------------------------- #
# Encoding detection beyond the Latin fallbacks
#
# Limitation: single-byte non-Latin and CJK encodings need roughly a kilobyte of
# text before statistical detection is reliable. Below that the loader falls back
# to cp1252. These fixtures are sized accordingly.
# --------------------------------------------------------------------------- #


def test_cp1252_smart_punctuation_is_not_mangled():
    # latin-1 maps 0x80-0x9F to control characters, and charset detection reports
    # this file as cp775 (Baltic DOS). Both produce plausible-looking rubbish.
    text = "id,note\n" + "".join(f"{i},“quoted” € and – dash\n" for i in range(1, 30))
    (block,) = blocks_from_bytes("win.csv", text.encode("cp1252"))
    assert block.rows[0][1] == "“quoted” € and – dash"


@pytest.mark.parametrize(
    "encoding",
    ["koi8-r", "cp1251", "iso-8859-5", "utf-8"],
)
def test_cyrillic_survives_every_common_encoding(encoding):
    text = "id,gorod\n" + "".join(f"{i},Москва Санкт-Петербург\n" for i in range(1, 30))
    (block,) = blocks_from_bytes("ru.csv", text.encode(encoding))
    assert "Москва" in block.rows[0][1]


@pytest.mark.parametrize(
    ("encoding", "samples"),
    [
        ("koi8-r", ["Москва", "Питер", "Сочи"]),
        ("cp1251", ["Москва", "Питер", "Сочи"]),
        ("iso-8859-5", ["Москва", "Питер", "Сочи"]),
        ("cp1253", ["Αθήνα", "Πάτρα", "Ρόδος"]),
        ("shift_jis", ["東京", "大阪", "京都"]),
        ("gbk", ["北京", "上海", "广州"]),
    ],
)
def test_short_non_latin_files_are_decoded_correctly(encoding, samples):
    """Three rows is well below the ~1 KB frequency statistics need.

    The right encoding is still among the detector's candidates, just not ranked
    first -- a 32-byte KOI8-R file ranks mac_cyrillic above koi8_r, and its own chaos
    metric *prefers* the wrong one (0.071 vs 0.143). Candidates are therefore
    re-scored on script coherence: real words are letters of one script, while
    mojibake wedges quotation marks and currency signs inside them.

    Limit: a few *identical* CJK characters stay undecidable, because the same bytes
    are valid in Chinese and Korean encodings alike and every candidate scores a
    perfect 0.000 chaos. Varied content, as here and as in any real file, resolves it.
    """
    rows = "".join(f"{i},{s}\n" for i, s in enumerate(samples, start=1))
    (block,) = blocks_from_bytes("short.csv", f"id,name\n{rows}".encode(encoding))
    assert [r[1] for r in block.rows] == samples


def test_script_coherence_prefers_real_text_over_mojibake():
    from darwinbox.ingest.loader import _script_coherence

    assert _script_coherence("Москва Санкт Петербург") == 1.0
    # Cyrillic letters interleaved with typographic quotes: the mac_cyrillic misread.
    assert _script_coherence("нѕ\u201dЋ\u201eЅ рЅ\u2018\u2248\u201c") < 0.5


def test_utf16_big_endian_without_a_bom():
    # Guessing the wrong endianness does not raise, it yields CJK mojibake, so it
    # has to be decided from where the NUL bytes fall rather than tried.
    data = "id,city\n1,Alpha\n2,Beta\n3,Gamma\n".encode("utf-16-be")
    (block,) = blocks_from_bytes("be.csv", data)
    assert block.block.columns == ["id", "city"]
    assert block.rows[0] == ["1", "Alpha"]


def test_classic_mac_cr_line_endings():
    (block,) = blocks_from_bytes("mac.csv", b"id,name\r1,A\r2,B\r3,C\r")
    assert block.block.columns == ["id", "name"]
    assert block.block.n_rows == 3


def test_blank_row_inside_a_table_does_not_split_it():
    # A blank row separates two tables, but it also appears inside one. A real second
    # table starts with a header; a continuation does not.
    (block,) = blocks_from_bytes("gappy.csv", b"id,name\n1,A\n\n2,B\n\n\n3,C\n")
    assert block.block.columns == ["id", "name"]
    assert block.block.n_rows == 3


def test_two_genuine_tables_are_still_split_by_a_blank_row():
    blocks = blocks_from_bytes(
        "two.csv", b"id,name\n1,A\n2,B\n\nsku,qty\nS1,5\nS2,6\n"
    )
    assert len(blocks) == 2
    assert blocks[0].block.columns == ["id", "name"]
    assert blocks[1].block.columns == ["sku", "qty"]


def test_side_by_side_tables_are_split_on_an_empty_column():
    blocks = blocks_from_bytes(
        "side.csv", b"id,name,,pid,part\n1,A,,P1,bolt\n2,B,,P2,nut\n3,C,,P3,screw\n"
    )
    assert len(blocks) == 2
    assert blocks[0].block.columns == ["id", "name"]
    assert blocks[1].block.columns == ["pid", "part"]


# --------------------------------------------------------------------------- #
# Forms: sheets that are not tables at all
#
# A payslip, a tax computation or a statutory certificate is a list of labelled
# figures, not relational data. Run through table detection one real tax workbook
# shattered into 10 fragments with placeholder column names -- confident nonsense.
# --------------------------------------------------------------------------- #

def _tax_form() -> bytes:
    """A tax computation shaped like the real one: sections, blank rows, sparse cells.

    Built at realistic length on purpose. The form heuristics key off a sheet
    shattering into several headerless fragments, which a six-line toy never does --
    tuning the thresholds down to fit a toy would defeat what they are for.
    """
    lines = [
        "Income Tax Calculation",
        ",,,Name:,Test Taxpayer",
        ",,PAN,AAAPZ0000A",
        ",,DOB,1990-01-01",
        "",
        "COMPUTATION OF TOTAL INCOME",
        "1,INCOME FROM SALARY:",
        ",Total Expected Income,,,587899",
        ",,LESS:STANDARD DEDUCTION,,50000",
        ",,,INCOME FROM SALARY,,537899",
        "",
        "2,INCOME FROM HOUSE PROPERTY",
        ",,HRA EXEMPTION,,0",
        ",,LESS:PROFESSIONAL TAX,,2400",
        "",
        "3,DEDUCTIONS",
        ",,DEDUCTION U/S SEC 80C,,150000",
        ",,DEDUCTION U/S 80D,,25000",
        ",,LESS: NPS DEDUCTION U/S 80CCD (1B),,50000",
        ",,,TOTAL DEDUCTIONS,,175000",
        "",
        "4,TAX PAYABLE",
        ",,TOTAL INCOME,,362899",
        ",,TAX ON TOTAL INCOME,,5645",
        ",,EDUCATION CESS,,226",
        ",,,TOTAL TAX PAYABLE,,5871",
    ]
    return ("\n".join(lines) + "\n").encode()


TAX_FORM = _tax_form()


def test_a_form_becomes_one_queryable_table():
    (block,) = blocks_from_bytes("tax.csv", TAX_FORM)

    assert block.block.columns == ["section", "item", "value"]
    items = {row[1]: row[2] for row in block.rows}
    assert items["Total Expected Income"] == "587899"
    assert items["LESS:STANDARD DEDUCTION"] == "50000"
    assert items["PAN"] == "AAAPZ0000A"


def test_a_form_keeps_the_section_each_row_sits_under():
    (block,) = blocks_from_bytes("tax.csv", TAX_FORM)
    sections = {row[1]: row[0] for row in block.rows}
    assert sections["DEDUCTION U/S SEC 80C"] == "DEDUCTIONS"
    assert sections["Total Expected Income"] == "INCOME FROM SALARY:"


def test_a_figure_beside_the_label_does_not_end_up_inside_it():
    # A two-regime sheet puts the old amount beside the new one; joining every cell
    # left of the value produced items like "Total Expected Income 587899".
    rows = blocks_from_bytes("tax.csv", TAX_FORM)[0].rows
    items = {r[1]: r[2] for r in rows}
    assert "Total Expected Income" in items, list(items)
    assert items["Total Expected Income"] == "587899"


def test_a_real_table_is_never_read_as_a_form():
    for name in ["clean.csv", "two_tables_one_sheet.xlsx", "title_and_footnote.xlsx"]:
        for detected in load_blocks(name):
            assert detected.block.columns != ["section", "item", "value"], name


def test_two_different_forms_are_not_fused_together():
    # Both extract to (section, item, value); that is the reader's shape, not a shared
    # schema, and fusing merged a tax computation with a statutory certificate.
    import duckdb
    from darwinbox.profile.registry import Registry

    registry = Registry(duckdb.connect(":memory:"))
    ids: set[str] = set()
    aliases: set[str] = set()
    for name in ("tax_a.csv", "tax_b.csv"):
        for loaded in load_file(name, TAX_FORM):
            for detected in detect_blocks(name, loaded.sheet, loaded.grid, ids, aliases):
                registry.register(detected)

    assert len(registry.table_ids) == 2
    assert registry.merge_identical_tables() == []


# --------------------------------------------------------------------------- #
# Multi-row headers and mixed form/table sheets
#
# Both patterns are everywhere in real spreadsheets: a group label spanning several
# columns above the field names, and a document that opens with a few labelled
# fields before getting on with its line items.
# --------------------------------------------------------------------------- #


def test_group_label_above_the_field_names_is_kept():
    # "Room | Room | Finishes | Finishes | Area" over "No. | Name | Floor | Wall | sqm".
    # The group row scores worse than the detail row, so the search lands on the lower
    # one; without walking back up, "Room" and "Area" are lost entirely.
    data = (
        b"Room,Room,Finishes,Area\n"
        b"No.,Name,Floor,sqm\n"
        b"301,Lobby,Granite,42.5\n"
        b"302,Office A,Vinyl,28.0\n"
        b"303,Office B,Vinyl,28.0\n"
    )
    (block,) = blocks_from_bytes("rooms.csv", data)
    assert block.block.columns == ["room_no", "room_name", "finishes_floor", "area_sqm"]
    assert block.block.n_rows == 3


def test_merged_group_label_leaving_gaps_is_also_kept():
    # The same header as Excel actually stores it: value in the first cell of the
    # merged range, empty in the rest.
    data = (
        b"Earnings,,,Deductions,\n"
        b"Basic,HRA,Allow,PF,Tax\n"
        b"45000,18000,6000,5400,4200\n"
        b"38000,15200,5000,4560,3100\n"
        b"52000,20800,7000,6240,5800\n"
    )
    (block,) = blocks_from_bytes("pay.csv", data)
    assert block.block.columns[0] == "earnings_basic"
    assert block.block.columns[3] == "deductions_pf"


def test_an_all_text_table_does_not_lose_its_first_row():
    # The guard against over-reaching: with no group label above, the first data row
    # of an all-text table must not be swallowed as a second header row.
    data = b"code,description\nAC,Air conditioning\nEL,Electrical\nPL,Plumbing\n"
    (block,) = blocks_from_bytes("codes.csv", data)
    assert block.block.columns == ["code", "description"]
    assert block.block.n_rows == 3


def test_a_form_header_above_a_table_becomes_its_own_details_table():
    data = (
        b"EXPENSE CLAIM\n"
        b"\n"
        b"Employee:,Ravi Kumar,,Claim No:,EXP-2291\n"
        b"Department:,Sales,,Period:,Feb 2026\n"
        b"\n"
        b"Date,Description,Category,Amount\n"
        b"2026-02-03,Client lunch,Meals,2400\n"
        b"2026-02-07,Taxi,Travel,850\n"
        b"2026-02-11,Hotel,Lodging,11200\n"
    )
    blocks = blocks_from_bytes("claim.csv", data)

    assert len(blocks) == 2, [b.block.columns for b in blocks]
    details = next(b for b in blocks if b.block.columns == ["section", "item", "value"])
    items = {row[1]: row[2] for row in details.rows}
    assert items["Claim No"] == "EXP-2291"
    assert items["Employee"] == "Ravi Kumar"

    lines = next(b for b in blocks if b.block.columns != ["section", "item", "value"])
    assert lines.block.columns == ["date", "description", "category", "amount"]
    assert lines.block.n_rows == 3


def test_a_column_headed_only_with_digits_survives():
    # An attendance register heads its day columns "1".."31". Stripping leading digits
    # to make a usable identifier erased them entirely, turning every day into col_N.
    data = (
        b"Emp ID,Name,1,2,3,Present\n"
        b"E001,Asha,P,P,A,2\n"
        b"E002,Ravi,P,L,P,3\n"
        b"E003,Lin,A,A,P,1\n"
    )
    (block,) = blocks_from_bytes("attendance.csv", data)
    assert block.block.columns == ["emp_id", "name", "n1", "n2", "n3", "present"]


def test_a_year_prefixed_name_still_loses_its_digits():
    assert normalize_column_names(["2024 Revenue"]) == ["revenue"]


def test_an_invoice_totals_block_folds_into_the_details():
    # Subtotal / GST / TOTAL rows are labelled figures trailing the line items, not
    # another table; left alone they became a separate table of col_1 / col_2.
    data = (
        b"TAX INVOICE\n"
        b"\n"
        b"Invoice No:,INV-2026-0441,,Date:,2026-04-12\n"
        b"\n"
        b"#,Item,Qty,Rate,Amount\n"
        b"1,Steel bracket,120,45.0,5400.0\n"
        b"2,Anchor bolt,500,12.5,6250.0\n"
        b"3,Labour,1,8000.0,8000.0\n"
        b"\n"
        b",,,Subtotal,19650.0\n"
        b",,,CGST 9%,1768.5\n"
        b",,,TOTAL,23187.0\n"
    )
    blocks = blocks_from_bytes("invoice.csv", data)

    assert len(blocks) == 2, [b.block.columns for b in blocks]
    details = next(b for b in blocks if b.block.columns == ["section", "item", "value"])
    items = {row[1]: row[2] for row in details.rows}
    assert items["Invoice No"] == "INV-2026-0441"
    assert items["TOTAL"] == "23187.0"

    lines = next(b for b in blocks if b.block.columns != ["section", "item", "value"])
    assert lines.block.n_rows == 3


def test_a_blank_row_inside_a_header_band_does_not_lose_the_group_label():
    # Real sheets put a spacer between the group label and the field names. The
    # splitter cuts there, so the band logic never saw them together and the upper
    # row was lost with the title block.
    data = (
        b"PAY REGISTER\n"
        b"\n"
        b"S.No,Name,Pay Matrix,Pay Matrix\n"
        b"\n"
        b",,Level,Cell\n"
        b"1,Test Person A,02,03\n"
        b"2,Asha Menon,05,01\n"
        b"3,Lin Wei,04,02\n"
    )
    blocks = blocks_from_bytes("pay.csv", data)
    columns = blocks[-1].block.columns
    assert "pay_matrix_level" in columns, columns
    assert "pay_matrix_cell" in columns, columns


# --------------------------------------------------------------------------- #
# Malformed input must fail as a typed error, never as a crash
#
# Everything here reaches the API as an upload. UnreadableFileError maps to a 422
# naming the file; anything else falls through to the catch-all handler and becomes
# a 500, which tells the user nothing and looks like the service broke.
# --------------------------------------------------------------------------- #

from darwinbox.ingest.loader import UnreadableFileError  # noqa: E402


def test_a_cell_larger_than_the_csv_field_limit_is_read_not_rejected():
    """The csv module caps a field at 128 KB; a long free-text column exceeds it.

    This arrived as a bare _csv.Error, so one oversized notes cell turned a perfectly
    good file into a 500. The real limit is the upload size, and a field cannot be
    bigger than the file holding it.
    """
    data = b"note,n\n" + b"x" * 200_000 + b",1\n"
    (grid,) = load_file("big.csv", data)
    assert len(grid.grid) == 2
    assert len(grid.grid[1][0]) == 200_000


def test_an_unterminated_quote_is_a_typed_error():
    with pytest.raises(UnreadableFileError):
        load_file("bad.csv", b'a,b\n"never closed,2\n' + b"y" * 200_000_000)


@pytest.mark.parametrize(
    ("name", "data"),
    [
        ("fake.xlsx", b"not really a zip"),
        ("empty.csv", b""),
        ("only_header.csv", b"a,b,c\n"),
        ("nul.csv", b"a,b\n1,2\x00\n3,4\n"),
        ("ragged.csv", b"a,b,c\n1,2\n3,4,5,6,7\n8\n"),
        ("dupes.csv", b"id,id,id\n1,2,3\n"),
        ("blank_headers.csv", b",,\n1,2,3\n"),
        ("cr_only.csv", b"a,b\r1,2\r3,4\r"),
        ("formula.csv", b"a,b\n=1+1,@SUM(A1)\n"),
        ("injection.csv", b"name,note\n'; DROP TABLE x; --,\"1' OR '1'='1\"\n"),
    ],
)
def test_malformed_files_never_raise_something_untyped(name, data):
    # Either it reads, or it says it cannot. Both are fine; a third outcome is not.
    with contextlib.suppress(UnreadableFileError):
        load_file(name, data)


def test_a_wide_sparse_table_is_not_mistaken_for_a_form():
    """Sparsity is not formness; a missing header is.

    A real open-data extract (traffic crashes, 71 columns) carries forty optional
    columns -- vehicle_defect, towed_by, area_00_i..area_06_i -- that are almost
    always empty. That puts it under any density threshold worth setting, and it was
    being read as (section, item, value): not a loss of precision, a destroyed table
    whose every column name became a hash of its values.
    """
    header = ",".join(f"col_{i}" for i in range(40))
    # Only the first three columns carry data; the rest are present but empty.
    row = "a,1,2" + "," * 37
    data = (header + "\n" + "\n".join(row for _ in range(60)) + "\n").encode()

    blocks = blocks_from_bytes("wide.csv", data)
    assert len(blocks) == 1
    block = blocks[0].block
    assert not block.table_id.endswith("form"), "a headed table must not become a form"
    assert len(block.columns) == 40
    assert block.columns[0] == "col_0"


def test_a_sparse_sheet_with_no_usable_headers_is_still_a_form():
    """The other side of the same rule, so the fix does not simply disable forms.

    A statutory computation lays a label in the first column and its figure several
    columns to the right, with nothing in between -- wide, sparse, and with no row
    that reads as a header.
    """
    labels = [
        "Gross Salary", "Less: Standard Deduction", "House Rent Allowance",
        "Professional Tax", "Chapter VI-A Deductions", "Total Income",
        "Tax Payable", "Cess", "Net Tax", "Advance Tax Paid", "Balance Payable",
        "TDS Deducted", "Refund Due", "Rebate 87A", "Surcharge",
        "Interest 234A", "Interest 234B", "Relief 89", "Net Payable", "Rounded Off",
        "Bank Account", "IFSC Code",
    ]
    rows = ["COMPUTATION OF TOTAL INCOME" + "," * 7]
    rows += [f"{label}" + "," * 5 + f"{(i + 1) * 1000}," for i, label in enumerate(labels)]
    data = ("\n".join(rows) + "\n").encode()

    blocks = blocks_from_bytes("tax.csv", data)
    assert blocks, "a form-shaped sheet must still produce a table"
    assert any(b.block.table_id.endswith("form") for b in blocks), [
        b.block.table_id for b in blocks
    ]
