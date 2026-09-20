"""Sample datasets bundled with the app.

Loading a file from disk is the same operation as an upload, so this reuses the exact
ingestion path rather than a shortcut -- a sample session is byte-for-byte what the
user would get by dragging the same files in.
"""

from __future__ import annotations

import pathlib

from darwinbox.models import SampleSet

SAMPLES_DIR = pathlib.Path(__file__).resolve().parents[2] / "demo"

SAMPLES: list[SampleSet] = [
    SampleSet(
        name="sales",
        title="Sales & HR (messy)",
        description=(
            "Orders whose customer key was renamed on one side, a payroll file whose "
            "keys carry case and punctuation noise, plus an unrelated weather file."
        ),
        files=[
            "orders_renamed.csv",
            "customers.csv",
            "employees.csv",
            "payroll_noisy.csv",
            "weather.csv",
        ],
    ),
    SampleSet(
        name="one_sheet_two_tables",
        title="One sheet, two tables",
        description="A single worksheet holding two unrelated tables separated by blank rows.",
        files=["stacked_tables.xlsx"],
    ),
]


class UnknownSampleError(Exception):
    """Raised when a sample set name is not recognised (API maps to 404)."""


def get(name: str) -> SampleSet:
    for sample in SAMPLES:
        if sample.name == name:
            return sample
    raise UnknownSampleError(name)


def read(sample: SampleSet) -> list[tuple[str, bytes]]:
    """Read a sample set off disk as (filename, bytes), ready for Session.add_files."""
    out: list[tuple[str, bytes]] = []
    for name in sample.files:
        path = SAMPLES_DIR / name
        if path.exists():
            out.append((path.name, path.read_bytes()))
    return out
