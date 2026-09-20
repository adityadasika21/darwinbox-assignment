"""The CSV matrix: a systematic sweep of encodings, dialects, shapes, locales and
hostile input, run against the real ingestion path.

This exists because the README used to claim "42 cases, all pass" on the strength of
a one-off script that was never committed. A number nobody can reproduce is not
evidence, so every case is here as a test that fails loudly when it regresses.

Each case asserts what the file *means* -- the round-tripped text, the column count,
the coerced value -- rather than that loading merely did not raise. "It did not
crash" is the weakest possible claim about an ingester and it is not what is being
promised.
"""

from __future__ import annotations

import pandas as pd
import pytest
from darwinbox.ingest.blocks import detect_blocks
from darwinbox.ingest.loader import load_file
from darwinbox.profile.columns import coerce_series


def blocks(name: str, data: bytes):
    out = []
    ids: set[str] = set()
    aliases: set[str] = set()
    for loaded in load_file(name, data):
        out.extend(detect_blocks(name, loaded.sheet, loaded.grid, ids, aliases))
    return out


def one_block(name: str, data: bytes):
    found = blocks(name, data)
    assert len(found) == 1, f"expected exactly one table, got {len(found)}"
    return found[0]


def coerce(values: list[str]) -> pd.Series:
    return coerce_series(pd.Series(values, dtype="string"))


# --------------------------------------------------------------------------- #
# Encoding
#
# latin-1 decodes every possible byte and never raises, so a wrong guess here is
# silent: the file ingests, the columns look plausible, and every value is mojibake.
# Each case therefore asserts the text came back, not that the read succeeded.
# --------------------------------------------------------------------------- #

# Varied vocabulary, not one word repeated. Detection is statistical: koi8_r,
# cp1251, iso8859_5 and mac_cyrillic all turn a short repeated token into coherent
# Cyrillic and all score identically, so what resolves them is the letter
# distribution across *different* words. Real files have that; a synthetic file of
# "Москва" twelve times does not, and no amount of length rescues it.
CYRILLIC = ["Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург"]
GREEK = ["Αθήνα", "Θεσσαλονίκη", "Πάτρα", "Ηράκλειο"]
JAPANESE = ["東京都", "大阪府", "名古屋市", "札幌市"]
CHINESE = ["北京市", "上海市", "广州市", "深圳市"]
LATIN = ["Málaga", "Köln", "Zürich", "Genève"]


def encoded(words: list[str], encoding: str, bom: bytes = b"") -> bytes:
    rows = "\n".join(f"{w},{i * 1000}" for i, w in enumerate(words))
    return bom + f"city,population\n{rows}\n".encode(encoding)


ENCODINGS = [
    ("utf-8", encoded(CYRILLIC, "utf-8"), CYRILLIC[0]),
    ("utf-8-bom", encoded(CYRILLIC, "utf-8", b"\xef\xbb\xbf"), CYRILLIC[0]),
    ("utf-16-le-bom", encoded(CYRILLIC, "utf-16-le", b"\xff\xfe"), CYRILLIC[0]),
    ("utf-16-be-bom", encoded(CYRILLIC, "utf-16-be", b"\xfe\xff"), CYRILLIC[0]),
    # No BOM: detected from NUL positions, so these carry Latin text. The
    # non-Latin case is a documented limit, asserted below.
    ("utf-16-le-no-bom", encoded(LATIN, "utf-16-le"), LATIN[0]),
    ("utf-16-be-no-bom", encoded(LATIN, "utf-16-be"), LATIN[0]),
    ("utf-32-bom", encoded(CYRILLIC, "utf-32"), CYRILLIC[0]),
    ("latin-1", encoded(LATIN, "latin-1"), LATIN[0]),
    ("cp1252", encoded(["Café—“x”", "Köln", "Zürich", "Genève"], "cp1252"), "Café—“x”"),
    ("shift-jis", encoded(JAPANESE, "shift_jis"), JAPANESE[0]),
    ("euc-jp", encoded(JAPANESE, "euc_jp"), JAPANESE[0]),
    ("gbk", encoded(CHINESE, "gbk"), CHINESE[0]),
    ("koi8-r", encoded(CYRILLIC, "koi8_r"), CYRILLIC[0]),
    ("cp1251", encoded(CYRILLIC, "cp1251"), CYRILLIC[0]),
    ("iso-8859-5", encoded(CYRILLIC, "iso8859_5"), CYRILLIC[0]),
    ("cp1253-greek", encoded(GREEK, "cp1253"), GREEK[0]),
    ("iso-8859-7-greek", encoded(GREEK, "iso8859_7"), GREEK[0]),
    ("emoji-4-byte", encoded(["Tokyo 🗼", "Osaka 🎌", "Kyoto ⛩", "Nara 🦌"], "utf-8"), "Tokyo 🗼"),
]


@pytest.mark.parametrize(("name", "data", "expected"), ENCODINGS, ids=[c[0] for c in ENCODINGS])
def test_encoding_round_trips_the_actual_characters(name, data, expected):
    block = one_block(f"{name}.csv", data)
    assert block.rows[0][0] == expected, f"{name} decoded to {block.rows[0][0]!r}"


def test_corrupt_bytes_mid_file_do_not_lose_the_rest():
    """One bad byte must not take the file with it."""
    data = b"city,n\nParis,1\n" + b"\xff\xfe\n" + b"Berlin,3\n"
    cities = [r[0] for r in one_block("corrupt.csv", data).rows]
    assert "Paris" in cities and "Berlin" in cities


@pytest.mark.parametrize("endianness", ["utf-16-le", "utf-16-be"])
def test_bom_less_utf16_of_non_latin_text_is_a_known_limit(endianness):
    """Asserted, not hidden, so nobody discovers it from a support ticket.

    BOM-less UTF-16 is found by counting NULs, which works because the format is in
    practice used for mostly-ASCII text -- anything writing UTF-16 normally writes a
    BOM. Russian in UTF-16 has a quarter the NULs, its high byte being 0x04, and falls
    through. The rule that would catch it also fires on "a,b,c / 1,2,3 / 4,5,6", so
    the narrow rule is the deliberate choice.
    """
    block = one_block("cyr.csv", encoded(CYRILLIC, endianness))
    assert block.rows[0][0] != CYRILLIC[0]


def test_a_bom_rescues_non_latin_utf16():
    """Which is why this is a narrow limit rather than a real gap: a BOM settles it."""
    bom = b"\xff\xfe"
    block = one_block("cyr_bom.csv", encoded(CYRILLIC, "utf-16-le", bom))
    assert block.rows[0][0] == CYRILLIC[0]


# --------------------------------------------------------------------------- #
# Dialect
# --------------------------------------------------------------------------- #

DIALECTS = [
    ("comma", b"a,b,c\n1,2,3\n4,5,6\n", 3),
    ("semicolon", b"a;b;c\n1;2;3\n4;5;6\n", 3),
    ("tab", b"a\tb\tc\n1\t2\t3\n4\t5\t6\n", 3),
    ("pipe", b"a|b|c\n1|2|3\n4|5|6\n", 3),
    ("quoted-delimiter", b'a,b\n"x,y",2\n"p,q",3\n', 2),
    ("quoted-newline", b'a,b\n"line\none",2\n"other",3\n', 2),
    ("escaped-quotes", b'a,b\n"he said ""hi""",2\n"x",3\n', 2),
    ("no-trailing-newline", b"a,b\n1,2\n3,4", 2),
    ("crlf", b"a,b\r\n1,2\r\n3,4\r\n", 2),
    ("classic-mac-cr", b"a,b\r1,2\r3,4\r", 2),
    ("whitespace-padding", b"a , b \n 1 , 2 \n 3 , 4 \n", 2),
]


@pytest.mark.parametrize(("name", "data", "cols"), DIALECTS, ids=[c[0] for c in DIALECTS])
def test_dialect_yields_the_right_column_count(name, data, cols):
    block = one_block(f"{name}.csv", data).block
    assert len(block.columns) == cols
    assert block.n_rows == 2


def test_a_quoted_delimiter_stays_inside_its_cell():
    block = one_block("q.csv", b'a,b\n"x,y",2\n"p,q",3\n')
    assert block.rows[0][0] == "x,y"


def test_an_escaped_quote_becomes_one_quote():
    block = one_block("e.csv", b'a,b\n"he said ""hi""",2\n"x",3\n')
    assert block.rows[0][0] == 'he said "hi"'


# --------------------------------------------------------------------------- #
# Shape
# --------------------------------------------------------------------------- #


def test_an_empty_file_yields_no_table():
    assert blocks("empty.csv", b"") == []


def test_a_header_with_no_rows_yields_no_table():
    assert blocks("headeronly.csv", b"a,b,c\n") == []


def test_a_single_column_file_is_a_table():
    block = one_block("single.csv", b"name\nAda\nGrace\nAlan\n").block
    assert block.columns == ["name"] and block.n_rows == 3


def test_a_single_data_row_is_a_table():
    block = one_block("onerow.csv", b"a,b,c\n1,2,3\n").block
    assert block.n_rows == 1


def test_ragged_rows_are_padded_not_rejected():
    """Short rows are padded to the header width rather than dropped."""
    found = one_block("ragged.csv", b"a,b,c\n1,2\n3,4,5\n7,8,9\n")
    assert len(found.block.columns) == 3 and found.block.n_rows == 3
    assert found.rows[0] == ["1", "2", None]


def test_a_trailing_one_cell_row_is_treated_as_a_footer():
    """"6" alone under a three-column table is a footnote, not a data row."""
    block = one_block("footer.csv", b"a,b,c\n1,2,3\n4,5,6\n6\n").block
    assert block.n_rows == 2


def test_duplicate_headers_are_made_unique():
    block = one_block("dupes.csv", b"id,id,id\n1,2,3\n4,5,6\n").block
    assert len(set(block.columns)) == 3, block.columns


def test_blank_headers_still_produce_named_columns():
    block = one_block("blank.csv", b",,\n1,2,3\n4,5,6\n").block
    assert len(block.columns) == 3
    assert all(c and c.strip() for c in block.columns)


def test_an_entirely_blank_column_does_not_break_the_others():
    block = one_block("blankcol.csv", b"a,b,c\n1,,3\n4,,6\n").block
    assert len(block.columns) == 3


def test_two_tables_side_by_side_are_split():
    data = b"a,b,,x,y\n1,2,,7,8\n3,4,,9,10\n"
    assert len(blocks("side.csv", data)) == 2


def test_six_hundred_columns():
    header = ",".join(f"c{i}" for i in range(600)).encode()
    row = ",".join(str(i) for i in range(600)).encode()
    block = one_block("wide.csv", header + b"\n" + row + b"\n" + row + b"\n").block
    assert len(block.columns) == 600


def test_a_blank_row_inside_a_table_does_not_split_it():
    """A stray empty row is a typo, not a table boundary; two blank rows are."""
    data = b"a,b\n1,2\n\n3,4\n5,6\n"
    assert len(blocks("gap.csv", data)) == 1


def test_a_five_kilobyte_cell_survives():
    big = "x" * 5000
    block = one_block("big.csv", f"note,n\n{big},1\n{big},2\n".encode())
    assert len(block.rows[0][0]) == 5000


_UNICODE_HEADERS = "événement,город\n1,2\n3,4\n".encode()

COLUMN_NAMES = [
    ("unicode-names", _UNICODE_HEADERS),
    ("sql-keyword-names", b"select,from,where,table\n1,2,3,4\n5,6,7,8\n"),
    ("numeric-names", b"1,2,3\n4,5,6\n7,8,9\n"),
]


@pytest.mark.parametrize(("name", "data"), COLUMN_NAMES, ids=[c[0] for c in COLUMN_NAMES])
def test_awkward_column_names_still_produce_usable_identifiers(name, data):
    block = one_block(f"{name}.csv", data).block
    assert len(set(block.columns)) == len(block.columns)  # unique
    assert all(c and not c[0].isdigit() for c in block.columns), block.columns


# --------------------------------------------------------------------------- #
# Locale
#
# Every convention is decided per column, never per value: a uniform mistake is
# recoverable and a mixed one is not.
# --------------------------------------------------------------------------- #


def test_european_decimals():
    out = coerce(["1.234,56", "2.000,00", "3,50"])
    assert out.iloc[0] == pytest.approx(1234.56) and out.iloc[2] == pytest.approx(3.50)


def test_anglo_decimals():
    out = coerce(["1,234.56", "2,000.00", "3.50"])
    assert out.iloc[0] == pytest.approx(1234.56)


def test_day_first_dates():
    out = coerce(["25/12/2024", "13/01/2024", "01/02/2024"])
    assert out.iloc[0].month == 12 and out.iloc[0].day == 25


def test_month_first_dates():
    out = coerce(["12/25/2024", "01/13/2024", "02/01/2024"])
    assert out.iloc[0].month == 12 and out.iloc[0].day == 25


def test_iso_dates():
    out = coerce(["2024-01-05", "2024-02-06", "2024-03-07"])
    assert out.iloc[0].year == 2024 and out.iloc[0].month == 1


def test_timezone_aware_timestamps():
    out = coerce(["2024-01-05T10:00:00+05:30", "2024-01-06T11:00:00+05:30"])
    assert pd.api.types.is_datetime64_any_dtype(out)


def test_parenthesised_negatives():
    out = coerce(["(50)", "100", "(25.5)"])
    assert out.iloc[0] == pytest.approx(-50) and out.iloc[2] == pytest.approx(-25.5)


def test_percentages():
    out = coerce(["12%", "7.5%", "100%"])
    assert pd.api.types.is_numeric_dtype(out)


def test_currency_symbols():
    out = coerce(["₹1,234.56", "₹2,000", "₹3"])
    assert out.iloc[0] == pytest.approx(1234.56)


def test_scientific_notation():
    out = coerce(["1e3", "2.5e-2", "3E2"])
    assert out.iloc[0] == pytest.approx(1000.0)


def test_leading_zeros_are_kept_as_text():
    """A zip code or phone number is an identifier, not a small number."""
    out = coerce(["01234", "00987", "05678"])
    assert out.iloc[0] == "01234"


def test_an_integer_too_big_for_int64_is_kept_exactly():
    """astype("Int64") does not raise on overflow -- it wraps.

    A 30-digit reference number became -9223372036854775808 with no warning. Float64
    would round it instead, which breaks a join just as thoroughly, so a value that
    large is treated as the identifier it almost certainly is.
    """
    out = coerce(["123456789012345678901234567890", "1", "2"])
    assert out.iloc[0] == "123456789012345678901234567890"


def test_an_integer_that_still_fits_int64_stays_numeric():
    out = coerce(["9223372036854775806", "1", "2"])
    assert pd.api.types.is_numeric_dtype(out)


NULL_TOKENS = ["NA", "N/A", "#N/A", "null", "-", "NaN", "none"]


@pytest.mark.parametrize("token", NULL_TOKENS)
def test_null_tokens_become_missing_not_text(token):
    out = coerce(["10", token, "30"])
    assert pd.isna(out.iloc[1]), f"{token!r} survived as {out.iloc[1]!r}"
    assert pd.api.types.is_numeric_dtype(out)


def test_a_mixed_column_stays_text_rather_than_guessing():
    out = coerce(["1", "two", "3.5", "2024-01-01"])
    assert not pd.api.types.is_numeric_dtype(out)


# --------------------------------------------------------------------------- #
# Hostile input
# --------------------------------------------------------------------------- #


def test_formula_injection_is_data_not_a_formula():
    block = one_block("f.csv", b'a,b\n"=cmd|\' /c calc\'!A1",1\n"@SUM(1)",2\n')
    assert block.rows[0][0].startswith("=cmd")  # kept verbatim, never evaluated


def test_sql_injection_in_a_value_is_just_a_value():
    block = one_block("s.csv", b"name,note\n'; DROP TABLE x; --,1\n\"1' OR '1'='1\",2\n")
    assert "DROP TABLE" in block.rows[0][0]


def test_sql_injection_in_a_column_name_is_neutralised():
    block = one_block("c.csv", b'id,"); DROP TABLE users; --"\n1,2\n3,4\n').block
    assert all(";" not in c and "(" not in c for c in block.columns), block.columns


@pytest.mark.parametrize(
    "filename",
    ["../../etc/passwd.csv", "..\\..\\windows\\system32.csv", "/tmp/evil.csv"],
)
def test_path_traversal_in_a_filename_cannot_escape_the_table_name(filename):
    found = blocks(filename, b"a,b\n1,2\n3,4\n")
    assert len(found) == 1
    table_id = found[0].block.table_id
    assert "/" not in table_id and "\\" not in table_id and ".." not in table_id
