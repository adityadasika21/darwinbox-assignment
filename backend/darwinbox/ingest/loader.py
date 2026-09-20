"""File bytes -> raw grids.

A "grid" is ``list[list[str | None]]`` with no interpretation applied: no header
row, no types, no trimming. Everything downstream works off grids, so this is the
only module that knows what a CSV or an .xlsx is.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass

from openpyxl import load_workbook

Grid = list[list[str | None]]

# Byte-order marks are checked before the fallback chain, because latin-1 decodes
# every possible byte and so never raises -- a UTF-16 file would silently become
# mojibake ("i\x00d\x00,\x00n\x00a\x00m\x00e") and be ingested as real columns.
_BOMS: tuple[tuple[bytes, str], ...] = (
    (b"\x00\x00\xfe\xff", "utf-32-be"),
    (b"\xff\xfe\x00\x00", "utf-32-le"),
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xfe\xff", "utf-16-be"),
    (b"\xff\xfe", "utf-16-le"),
)
# cp1252 before latin-1: latin-1 maps bytes 0x80-0x9F to control characters, so a
# Windows export's curly quotes and euro sign decode to invisible junk instead of
# failing. cp1252 rejects the few bytes it lacks, and latin-1 then catches those.
_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
_MAX_CANDIDATES = 8
_CSV_EXT = {".csv", ".tsv", ".txt"}
# The csv module caps a single field at 128 KB, which a long free-text cell exceeds
# for real reasons -- a notes or description column in an exported ticket dump. The
# cap that actually matters is the upload limit, so the field limit is raised to it:
# a field cannot be larger than the file containing it, and nothing new is admitted.
csv.field_size_limit(100 * 1024 * 1024)
_XL_EXT = {".xlsx", ".xlsm", ".xltx", ".xltm"}


class UnreadableFileError(Exception):
    """Raised when a file cannot be decoded or parsed at all (API maps to 422)."""


@dataclass(frozen=True)
class LoadedGrid:
    """One grid plus the sheet it came from (None for CSV: a CSV is one grid)."""

    sheet: str | None
    grid: Grid


def load_file(filename: str, data: bytes) -> list[LoadedGrid]:
    """Parse an uploaded file into one grid per sheet.

    CSVs yield exactly one grid; a workbook yields one grid per worksheet.
    """
    ext = _extension(filename)
    if ext in _XL_EXT:
        return _load_excel(data)
    if ext in _CSV_EXT:
        return [LoadedGrid(sheet=None, grid=_load_csv(data))]
    # Unknown extension: sniff. Workbooks are zip archives and start with "PK".
    if data[:2] == b"PK":
        return _load_excel(data)
    return [LoadedGrid(sheet=None, grid=_load_csv(data))]


def _extension(filename: str) -> str:
    _, _, tail = filename.rpartition(".")
    return f".{tail.lower()}" if tail and tail != filename else ""


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #


def _load_csv(data: bytes) -> Grid:
    text = _decode(data)
    # Classic-Mac files use a bare CR as the record separator, which the csv module
    # refuses ("new-line character seen in unquoted field").
    if "\r" in text and "\r\n" not in text and "\n" not in text:
        text = text.replace("\r", "\n")
    delimiter = _sniff_delimiter(text)
    # Tradeoff: the csv module rather than pandas.read_csv(header=None, dtype=str).
    # pandas raises on ragged rows, and ragged rows are the normal case in a sheet
    # holding two stacked tables of different widths -- exactly what we must survive.
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    try:
        rows: Grid = [[_clean(cell) for cell in row] for row in reader]
    except csv.Error as exc:
        # An unterminated quote makes the rest of the file one enormous field, and a
        # free-text notes column can genuinely run long. Either way this is a bad
        # file, not a bug, and it has to arrive as the typed error the API maps to a
        # 422 -- csv.Error reaches the catch-all handler and becomes a 500.
        raise UnreadableFileError(f"malformed CSV: {exc}") from exc
    return _pad_rectangular(rows)


def _wide_encoding(data: bytes) -> str | None:
    """Detect BOM-less UTF-16 from where the NUL bytes fall.

    ASCII text encoded UTF-16-LE puts its NULs on odd byte offsets and UTF-16-BE on
    even ones. Guessing wrong does not raise -- it yields CJK mojibake -- so the
    endianness has to be decided rather than tried.

    Limit, accepted deliberately: this finds BOM-less UTF-16 that is *mostly ASCII*,
    which is what the format is used for in practice, since anything that writes
    UTF-16 normally writes a BOM with it. A BOM-less UTF-16 file of Russian carries a
    quarter the NULs -- its high byte is 0x04, not 0x00 -- and falls through to the
    statistical detector. Generalising the rule to "one side has few distinct values"
    does catch it, and also catches "a,b,c / 1,2,3 / 4,5,6", an ordinary CSV whose odd
    bytes happen to hold two values. Corrupting common input to rescue rare input is
    the wrong trade, so the narrow rule stays.
    """
    head = data[:4096]
    if not head or head.count(0) <= len(head) // 4:
        return None
    even = sum(1 for i in range(0, len(head), 2) if head[i] == 0)
    odd = sum(1 for i in range(1, len(head), 2) if head[i] == 0)
    if odd > even:
        return "utf-16-le"
    return "utf-16-be" if even > odd else None


def _mostly_non_latin(text: str) -> bool:
    """True when the decoded letters are predominantly outside the Latin scripts."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    # U+0370 is past Latin Extended-B: Greek, Cyrillic, Hebrew, Arabic, CJK all sit above.
    return sum(1 for c in letters if ord(c) > 0x370) > len(letters) * 0.3


# Rough Unicode script ranges, enough to tell "one script" from "several".
_SCRIPTS: tuple[tuple[int, int, str], ...] = (
    (0x0000, 0x024F, "latin"),
    (0x0370, 0x03FF, "greek"),
    (0x0400, 0x04FF, "cyrillic"),
    (0x0590, 0x05FF, "hebrew"),
    (0x0600, 0x06FF, "arabic"),
    (0x0900, 0x097F, "devanagari"),
    (0x0E00, 0x0E7F, "thai"),
    (0x3040, 0x30FF, "kana"),
    (0x3400, 0x9FFF, "han"),
    (0xAC00, 0xD7AF, "hangul"),
)
# Characters that legitimately appear inside a word.
_IN_WORD = set("-_.'’/&+")


def _script_of(char: str) -> str | None:
    code = ord(char)
    for low, high, name in _SCRIPTS:
        if low <= code <= high:
            return name
    return None


def _script_coherence(text: str) -> float:
    """Fraction of words written in a single script with no symbols wedged inside.

    This is the signal that separates a correct decoding from a plausible wrong one
    on input too short for frequency statistics. Real words are letters of one
    script; mojibake sprays quotation marks, currency signs and multiplication signs
    through the middle of them -- KOI8-R "Москва" read as mac_cyrillic comes out as
    "нѕ\u201dЋ\u201eЅ", Cyrillic letters interleaved with typographic quotes.
    """
    words = [w for w in re.split(r"[\s,;|\t]+", text) if any(c.isalpha() for c in w)]
    if not words:
        return 0.0

    good = 0
    for word in words:
        scripts = {_script_of(c) for c in word if c.isalpha()}
        intruders = any(
            not c.isalnum() and c not in _IN_WORD and not c.isspace() for c in word
        )
        if len(scripts) == 1 and None not in scripts and not intruders:
            good += 1
    return good / len(words)


def _sniff_encoding(data: bytes) -> str | None:
    """Ask charset-normalizer, for encodings no byte pattern gives away.

    Shift-JIS, GBK and KOI8-R are all valid latin-1 byte sequences, so no fallback
    chain can distinguish them -- statistical detection is the only option. Optional
    import: the loader must keep working if the dependency is absent.

    The verdict is accepted only when it yields non-Latin script. Latin single-byte
    encodings are near-indistinguishable from each other by statistics, and the
    detector genuinely picks wrong: a Windows-1252 file of curly quotes and euro
    signs is reported as cp775 (Baltic DOS) and decodes to plausible-looking rubbish.
    Since cp1252 is overwhelmingly the common case for Western CSV exports, it wins
    any Latin-vs-Latin disagreement.
    """
    try:
        from charset_normalizer import from_bytes
    except ImportError:
        return None

    # The detector ranks by frequency statistics, which need roughly a kilobyte to
    # settle. On a short file the right answer is usually still among the candidates
    # but not first -- a 32-byte KOI8-R sample ranks mac_cyrillic above koi8_r -- so
    # the shortlist is re-scored on how much like real text each decoding reads.
    candidates = [m.encoding for m in from_bytes(data[:65536])][:_MAX_CANDIDATES]
    # An empty candidate list is left empty on purpose. Below a few dozen bytes the
    # detector returns nothing, and supplying a shortlist here does not help: koi8_r,
    # cp1251, iso8859_5 and mac_cyrillic all decode a short Cyrillic sample into
    # coherent Cyrillic and all score 1.0, so the choice between them would be
    # arbitrary. Guessing yields confident wrong words; declining yields visible
    # mojibake, which is the failure a human can actually see and correct.
    head = data[:8192]

    best_encoding, best_score = None, 0.0
    for encoding in candidates:
        try:
            decoded = head.decode(encoding, errors="replace")
        except LookupError:
            continue
        if not _mostly_non_latin(decoded):
            continue
        score = _script_coherence(decoded)
        if score > best_score:
            best_encoding, best_score = encoding, score
    return best_encoding


def _decode(data: bytes) -> str:
    for bom, encoding in _BOMS:
        if data.startswith(bom):
            try:
                return data.decode(encoding)
            except UnicodeDecodeError as exc:
                raise UnreadableFileError(f"{encoding} BOM but undecodable: {exc}") from exc

    wide = _wide_encoding(data)
    if wide:
        try:
            return data.decode(wide)
        except UnicodeDecodeError:
            pass

    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass

    sniffed = _sniff_encoding(data)
    if sniffed:
        try:
            return data.decode(sniffed)
        except (UnicodeDecodeError, LookupError):
            pass

    for enc in _ENCODINGS:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise UnreadableFileError("file could not be decoded in any known encoding")


def _sniff_delimiter(text: str) -> str:
    sample = text[:16384]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        # Sniffer fails on single-column files and on files whose first rows are
        # title lines. Fall back to whichever candidate appears most often.
        counts = {d: sample.count(d) for d in ",;\t|"}
        best = max(counts, key=lambda d: counts[d])
        return best if counts[best] else ","


def _pad_rectangular(rows: Grid) -> Grid:
    width = max((len(r) for r in rows), default=0)
    return [r + [None] * (width - len(r)) for r in rows]


# --------------------------------------------------------------------------- #
# Excel
# --------------------------------------------------------------------------- #


def _load_excel(data: bytes) -> list[LoadedGrid]:
    try:
        wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    except Exception as exc:  # openpyxl raises a wide variety of parse errors
        raise UnreadableFileError(f"could not open workbook: {exc}") from exc

    out: list[LoadedGrid] = []
    try:
        for ws in wb.worksheets:
            # openpyxl already materialises a merged range as value-in-top-left and
            # None in the remaining cells, which is precisely the unmerge rule. Blank
            # header cells left behind are named col_N during normalization.
            grid: Grid = [
                [_clean(_stringify(cell)) for cell in row]
                for row in ws.iter_rows(values_only=True)
            ]
            out.append(LoadedGrid(sheet=ws.title, grid=_pad_rectangular(grid)))
    finally:
        wb.close()

    if not out:
        raise UnreadableFileError("workbook contains no worksheets")
    return out


def _stringify(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _clean(cell: object) -> str | None:
    """Normalize a raw cell to a stripped string, or None when it is empty."""
    if cell is None:
        return None
    text = cell if isinstance(cell, str) else str(cell)
    text = text.strip()
    return text or None
