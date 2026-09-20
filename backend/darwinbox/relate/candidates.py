"""Candidate relationship generation: containment, name similarity, composites.

The ordering here is the central design claim of the system. Statistics propose;
the LLM only ever adjudicates a narrow uncertainty band that the statistics have
already surfaced. The model can never introduce an edge from column names alone.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import warnings
from dataclasses import dataclass, field

import pandas as pd

from darwinbox.models import ColumnProfile, Relationship, TableProfile
from darwinbox.profile.registry import Registry
from darwinbox.relate.derive import (
    ALNUM_ONLY,
    DIGITS_ONLY,
    GRAINS,
    LOWER_TRIM,
    RAW,
    Grain,
    Projection,
    project_values,
    render_series,
)

VALUE_CAP = 50_000

_DIGITS = re.compile(r"-?\d+(\.\d+)?")

FK_ACCEPT = 0.75
LLM_BAND_LOW = 0.50
COMPOSITE_POOL_LOW = 0.20
COMPOSITE_ACCEPT = 0.80
COMPOSITE_UNIQUE_MIN = 0.95
PARENT_UNIQUE_MIN = 0.95
COVERAGE_MIN = 0.95
MIN_UNCORROBORATED_DISTINCT = 20
CORROBORATION_NAME_MIN = 0.34
NAME_IDENTICAL = 0.99
COMPOSITE_NAME_MIN = 0.50

# Tokens nearly every key column carries, so they say nothing about whether two
# columns mean the same thing. Without stripping them "id" and "industry_code" score
# 0.5 similarity purely because both expand to "identifier".
GENERIC_TOKENS = {"identifier", "number"}
AUTO_CONFIRM = 0.90
MIN_DISTINCT = 2
MAX_LLM_CALLS = 8
MAX_COMPOSITE_POOL_FOR_TRIPLES = 4
COMPOSITE_POOL_MAX = 6

CONTAINMENT_WEIGHT = 0.7
NAME_WEIGHT = 0.3
SIGNATURE_BONUS = 0.1

# Which semantic types may plausibly hold the same real-world key.
_COMPATIBLE: dict[str, set[str]] = {
    "id": {"id", "numeric", "text", "category"},
    "numeric": {"numeric", "id"},
    "text": {"text", "category", "id"},
    "category": {"category", "text", "id"},
    "date": {"date"},
    "boolean": {"boolean"},
}


# --------------------------------------------------------------------------- #
# Primitive measures
# --------------------------------------------------------------------------- #


def containment(left: set[str], right: set[str]) -> float:
    """|A n B| / min(|A|, |B|).

    Containment rather than Jaccard: a 50k-row fact table joining a 50-row dimension
    is the normal case, and Jaccard would score that near zero and discard it.
    """
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def name_similarity(left: list[str], right: list[str]) -> float:
    """Token Jaccard over abbreviation-expanded name tokens, minus generic key words.

    Only the content words carry information. "cust_ref" and "customer_id" both reduce
    to {customer} and score 1.0, which is right; "id" and "industry_code" reduce to {}
    and {industry} and score 0.0, which is also right -- they share nothing but the
    fact that both are identifiers.
    """
    a = set(left) - GENERIC_TOKENS
    b = set(right) - GENERIC_TOKENS
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def score_pair(contain: float, name_sim: float, signature_match: bool) -> float:
    base = CONTAINMENT_WEIGHT * contain + NAME_WEIGHT * name_sim
    return round(min(1.0, base + (SIGNATURE_BONUS if signature_match else 0.0)), 4)


# --------------------------------------------------------------------------- #
# Value-set cache
# --------------------------------------------------------------------------- #


class ValueSets:
    """Distinct values per (table, column, projection), sampled and memoised."""

    def __init__(self, registry: Registry, cap: int = VALUE_CAP) -> None:
        self._registry = registry
        self._cap = cap
        self._cache: dict[tuple[str, str, str], set[str]] = {}
        self._rows: dict[tuple[str, str, str], list[str]] = {}
        self._rendered: dict[tuple[str, str], list[str]] = {}

    def get(self, table_id: str, column: str, projection: Projection = RAW) -> set[str]:
        key = (table_id, column, projection.name)
        if key not in self._cache:
            series = self._registry.frame(table_id)[column]
            if len(series) > self._cap:
                series = series.sample(self._cap, random_state=42)
            self._cache[key] = project_values(series, projection)
        return self._cache[key]

    def rows(self, table_id: str, column: str, projection: Projection = RAW) -> list[str]:
        """Every non-null row value, not just the distinct set, for coverage checks."""
        key = (table_id, column, projection.name)
        if key not in self._rows:
            series = self._registry.frame(table_id)[column]
            if len(series) > self._cap:
                series = series.sample(self._cap, random_state=42)
            rendered = render_series(series)
            self._rows[key] = [v for v in (projection.apply(x) for x in rendered) if v]
        return self._rows[key]

    def _column_strings(self, table_id: str, column: str) -> list[str]:
        """Row-aligned, case-folded strings for one column, computed once.

        Composite search asks for many column combinations from the same table, so
        rendering per combination repeats the same pandas work thousands of times.
        """
        key = (table_id, column)
        if key not in self._rendered:
            frame = self._registry.frame(table_id)
            if len(frame) > self._cap:
                frame = frame.sample(self._cap, random_state=42)
            self._rendered[key] = _render_column(frame[column])
        return self._rendered[key]

    def tuple_values(self, table_id: str, columns: list[str]) -> list[tuple[str, ...]]:
        """Row-wise tuples for composite-key containment (order matters, nulls dropped)."""
        if not columns:
            return []
        cols = [self._column_strings(table_id, c) for c in columns]
        return [t for t in zip(*cols, strict=True) if all(v not in ("", "nan") for v in t)]


def _render_column(series: pd.Series) -> list[str]:
    """Render one column to comparable strings; composites compare case-insensitively.

    Nulls become "" so the result stays row-aligned with the frame; tuple_values drops
    tuples containing one. Dropping rows here instead would misalign the columns.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.strftime("%Y-%m-%d").fillna("").tolist()
    if pd.api.types.is_float_dtype(series):
        return [
            "" if pd.isna(v) else (str(int(v)) if float(v).is_integer() else str(v))
            for v in series
        ]
    return series.astype(str).str.strip().str.lower().fillna("").tolist()


# --------------------------------------------------------------------------- #
# Single-column candidates
# --------------------------------------------------------------------------- #


@dataclass
class PairMeasure:
    """One scored column pair, before it becomes a Relationship."""

    left_table: str
    left_column: str
    right_table: str
    right_column: str
    containment: float
    name_similarity: float
    signature_match: bool
    projection: Projection
    score: float = 0.0
    grain: Grain | None = None
    identifier_join: bool = True
    min_distinct: int = 0
    numeric_values: bool = True

    def __post_init__(self) -> None:
        self.score = score_pair(self.containment, self.name_similarity, self.signature_match)


def _compatible(a: ColumnProfile, b: ColumnProfile) -> bool:
    return (
        b.semantic_type in _COMPATIBLE[a.semantic_type]
        and a.semantic_type in _COMPATIBLE[b.semantic_type]
    )


def _joinable(col: ColumnProfile) -> bool:
    return col.n_distinct >= MIN_DISTINCT


_STRINGISH = {"id", "text", "category"}


def _digit_like(col: ColumnProfile) -> bool:
    return col.semantic_type == "numeric" or bool(
        col.regex_signature and col.regex_signature.startswith(r"^\d")
    )


_MAX_IDENTIFIER_CHARS = 24


def _identifier_shaped(col: ColumnProfile) -> bool:
    """True when the values read as codes rather than as prose.

    Digits-only exists to match "EMP-0042" against a bare 42. Offered to free text
    that merely *contains* a number it invents joins: reduced to its digits the street
    address "204 Lisha Kill Rd Colonie" becomes 204, which duly appears in a column
    counting paperback loans, and two open-data sets from different cities acquire a
    relationship. A code is one short token; a description has spaces and length.

    A regex signature settles it outright. Failing that the evidence is the five stored
    samples plus the lexicographic extremes, which between them are likely to surface a
    long value in a column that has any. That is evidence rather than proof -- a column
    whose sampled values are short but whose prose sits in the middle would still pass
    -- so this is narrower than the bug it fixes, deliberately.
    """
    if col.regex_signature:
        return True

    evidence = [
        s.strip() for s in [*col.samples, col.min_value, col.max_value] if s and s.strip()
    ]
    if not evidence:
        return False
    return all(" " not in s and len(s) <= _MAX_IDENTIFIER_CHARS for s in evidence)


def _digits_only_applies(lc: ColumnProfile, rc: ColumnProfile) -> bool:
    """Exactly one side bare digits, and the other side an identifier rather than text."""
    if _digit_like(lc) == _digit_like(rc):
        return False
    other = rc if _digit_like(lc) else lc
    return _identifier_shaped(other)


def applicable_projections(lc: ColumnProfile, rc: ColumnProfile) -> list[Projection]:
    """Which normalizations could legitimately rescue this pair (SPEC 7.2).

    Digits-only is deliberately narrow. Between two prefixed-ID columns it is actively
    wrong -- it makes orders.order_id "O-1" match customers.code "C-1" at containment
    1.0 by throwing away the very prefix that distinguishes them. Its real use is
    "EMP-0042" against a bare 42, so it fires only when exactly one side is bare digits
    *and* the other is shaped like a code.
    """
    if lc.semantic_type not in _STRINGISH or rc.semantic_type not in _STRINGISH:
        return [DIGITS_ONLY] if _digits_only_applies(lc, rc) else []

    out = [LOWER_TRIM, ALNUM_ONLY]
    if _digits_only_applies(lc, rc):
        out.insert(0, DIGITS_ONLY)
    return out


def coverage(registry: Registry, values: ValueSets, m: PairMeasure) -> float:
    """Fraction of CHILD rows that actually find a parent under this projection.

    Distinct-set containment can read 1.0 while most rows still fail to join. In the
    noisy-key fixture some payroll rows keep their clean key, so every employee value
    is present somewhere in payroll and containment is 1.0 -- yet three quarters of
    payroll's rows carry case or punctuation noise and drop silently out of the JOIN.

    The direction matters and cannot be maxed over: measuring parent rows against the
    child set gives 1.0 here precisely because the noise *adds* variants rather than
    removing them. Only the child's row-level hit rate sees the problem.
    """
    left = registry.profile(m.left_table)
    right = registry.profile(m.right_table)
    left_ratio = _distinct_ratio(left, m.left_column)
    right_ratio = _distinct_ratio(right, m.right_column)

    # The parent is the side closer to unique; the child is the other one.
    if left_ratio >= right_ratio:
        child_table, child_column = m.right_table, m.right_column
        parent_table, parent_column = m.left_table, m.left_column
    else:
        child_table, child_column = m.left_table, m.left_column
        parent_table, parent_column = m.right_table, m.right_column

    child_rows = values.rows(child_table, child_column, m.projection)
    parent_set = values.get(parent_table, parent_column, m.projection)
    if not child_rows or not parent_set:
        return 0.0
    return sum(1 for v in child_rows if v in parent_set) / len(child_rows)


def is_corroborated(m: PairMeasure) -> bool:
    """A single-column FK needs agreement on meaning, not just on values.

    The discriminator is what the *overlapping values* look like. Integers collide by
    chance: at 42 tables a bus-ridership surrogate key holding 1..600 "contained" every
    count, score, percentage and code column in range across the corpus, and 63 of 102
    discovered edges traced to that one column. Structured strings do not collide --
    two tables independently containing "C-88" and "C-104" is not a coincidence.

    So a join whose shared values are purely numeric must also agree on a content word
    or share a value format. A join on structured strings stands on containment alone,
    which is what keeps the noisy-key case working: lowercase-and-punctuation noise
    destroys the regex signature, so a signature requirement there would lose a real key.

    containment divides by min(|A|, |B|), which is what lets a 50-row dimension match a
    50k-row fact table. The same denominator makes a two-value column trivially
    "contained" in anything: real FDIC data paired failures.id against a Treasury
    record_calendar_year holding only {2025, 2026} and scored it 1.00, because both
    years appear somewhere among 1500 row numbers. Bank names stripped to digits
    matched a fiscal-quarter column of {1,2,3,4} the same way.

    Above a reasonable key cardinality containment stands on its own. Below it, the
    pair must also agree on name or share a value format -- which every genuine key in
    the test corpus does, and none of the coincidences do.
    """
    if not m.numeric_values:
        return True
    if not (m.signature_match or m.name_similarity > 0):
        return False
    if m.min_distinct >= MIN_UNCORROBORATED_DISTINCT:
        return True
    # Small numeric sets are contained in anything, so demand real name agreement.
    return m.signature_match or m.name_similarity >= CORROBORATION_NAME_MIN


def is_identifier_join(lc: ColumnProfile, rc: ColumnProfile) -> bool:
    """A single-column FK must join on an identifier, not on a measure.

    Without this, payments.paid and orders.amount match at containment 1.0 simply
    because every payment equals its order total, and the graph grows an edge that is
    a coincidence of values rather than a key. Composite and derived keys are exempt:
    those are matched on tuples and truncated dates, where the same risk does not arise.
    """
    return "id" in {lc.semantic_type, rc.semantic_type}


def _distinct_ratio(profile: TableProfile, column: str) -> float:
    col = profile.column(column)
    if col is None or not profile.n_rows:
        return 0.0
    return col.n_distinct / profile.n_rows


def has_key_side(registry: Registry, m: PairMeasure) -> bool:
    """True when one side is near-unique on the column, i.e. it can be the parent key.

    Containment alone is not enough for a single-column FK. Two tables that both carry
    a `region` column share all four values, scoring containment 1.0, but region is a
    shared dimension attribute, not a key. Without this gate those pairs are emitted as
    bogus FKs and -- worse -- the composite search never runs, because it only fires
    when a table pair has no accepted FK. This is what makes drop_fk work.
    """
    left = registry.profile(m.left_table)
    right = registry.profile(m.right_table)
    return (
        max(
            _distinct_ratio(left, m.left_column),
            _distinct_ratio(right, m.right_column),
        )
        >= PARENT_UNIQUE_MIN
    )


def measure_pair(
    registry: Registry,
    values: ValueSets,
    left: TableProfile,
    lc: ColumnProfile,
    right: TableProfile,
    rc: ColumnProfile,
) -> PairMeasure | None:
    """Score one column pair, retrying normalized projections if raw containment is low."""
    if not _compatible(lc, rc) or not _joinable(lc) or not _joinable(rc):
        return None

    signature_match = bool(
        lc.regex_signature and lc.regex_signature == rc.regex_signature
    )
    name_sim = name_similarity(lc.normalized_name_tokens, rc.normalized_name_tokens)

    identifier_join = is_identifier_join(lc, rc)
    best = PairMeasure(
        left_table=left.table_id,
        left_column=lc.name,
        right_table=right.table_id,
        right_column=rc.name,
        containment=containment(
            values.get(left.table_id, lc.name), values.get(right.table_id, rc.name)
        ),
        name_similarity=name_sim,
        signature_match=signature_match,
        projection=RAW,
        identifier_join=identifier_join,
    )

    def measure(projection: Projection) -> PairMeasure:
        return PairMeasure(
            left_table=left.table_id,
            left_column=lc.name,
            right_table=right.table_id,
            right_column=rc.name,
            containment=containment(
                values.get(left.table_id, lc.name, projection),
                values.get(right.table_id, rc.name, projection),
            ),
            name_similarity=name_sim,
            signature_match=signature_match,
            projection=projection,
            identifier_join=identifier_join,
        )

    # SPEC 7.2 retries projections when raw containment fails. We also retry when raw
    # containment succeeds but row coverage does not: that is the noisy-key case, where
    # the edge is found but the JOIN written from it would quietly lose most rows.
    needs_retry = best.containment < LLM_BAND_LOW or (
        best.containment >= FK_ACCEPT
        and identifier_join
        and coverage(registry, values, best) < COVERAGE_MIN
    )

    if needs_retry:
        best_coverage = coverage(registry, values, best)
        for projection in applicable_projections(lc, rc):
            candidate = measure(projection)
            candidate_coverage = coverage(registry, values, candidate)
            better = (candidate_coverage, candidate.containment) > (
                best_coverage, best.containment
            )
            if better:
                best, best_coverage = candidate, candidate_coverage
            if best.containment >= FK_ACCEPT and best_coverage >= COVERAGE_MIN:
                break

    left_set = values.get(left.table_id, lc.name, best.projection)
    right_set = values.get(right.table_id, rc.name, best.projection)
    best.min_distinct = min(len(left_set), len(right_set))
    shared = left_set & right_set
    best.numeric_values = bool(shared) and all(_DIGITS.fullmatch(v) for v in shared)
    return best


# --------------------------------------------------------------------------- #
# Composite keys (SPEC 7.3)
# --------------------------------------------------------------------------- #


def composite_candidates(
    values: ValueSets, pool: list[PairMeasure]
) -> list[tuple[list[str], list[str], float]]:
    """Combine mid-scoring column pairs into (left_cols, right_cols, containment)."""
    if len(pool) < 2:
        return []

    # A 116-column table yields a pool in the hundreds, and combinations() over that is
    # quadratic -- it was 76% of total discovery time on a 42-file corpus. The best
    # composite is built from the best-scoring columns, so the tail cannot help.
    pool = sorted(pool, key=lambda m: -m.containment)[:COMPOSITE_POOL_MAX]

    sizes = [2, 3] if len(pool) <= MAX_COMPOSITE_POOL_FOR_TRIPLES else [2]
    out: list[tuple[list[str], list[str], float]] = []

    for size in sizes:
        for combo in itertools.combinations(pool, size):
            left_cols = [p.left_column for p in combo]
            right_cols = [p.right_column for p in combo]
            if len(set(left_cols)) != size or len(set(right_cols)) != size:
                continue

            left_rows = values.tuple_values(combo[0].left_table, left_cols)
            right_rows = values.tuple_values(combo[0].right_table, right_cols)
            left_set, right_set = set(left_rows), set(right_rows)

            contain = containment(left_set, right_set)
            if contain < COMPOSITE_ACCEPT:
                continue

            left_unique = len(left_set) / len(left_rows) if left_rows else 0.0
            right_unique = len(right_set) / len(right_rows) if right_rows else 0.0
            if max(left_unique, right_unique) < COMPOSITE_UNIQUE_MIN:
                continue

            out.append((left_cols, right_cols, contain))

    # Prefer the tightest composite per table pair; 2-column keys before 3-column ones.
    out.sort(key=lambda t: (len(t[0]), -t[2]))
    return out[:1]


# --------------------------------------------------------------------------- #
# Relationship assembly
# --------------------------------------------------------------------------- #


def build_relationship_id(
    left_table: str, left_cols: list[str], right_table: str, right_cols: list[str]
) -> str:
    """Stable id for a column pair, so a user decision survives re-discovery."""
    raw = f"{left_table}:{'+'.join(left_cols)}~{right_table}:{'+'.join(right_cols)}"
    return "rel_" + hashlib.md5(raw.encode()).hexdigest()[:10]


def _parent_side(
    left: TableProfile, left_cols: list[str], right: TableProfile, right_cols: list[str]
) -> str | None:
    """The parent is the side whose key is closer to unique (SPEC 7.1.6)."""
    lp = [left.column(c) for c in left_cols]
    rp = [right.column(c) for c in right_cols]
    if any(c is None for c in lp) or any(c is None for c in rp):
        return None

    left_unique = all(c.is_unique for c in lp)
    right_unique = all(c.is_unique for c in rp)
    if left_unique != right_unique:
        return "left" if left_unique else "right"

    lr = min(c.n_distinct / left.n_rows for c in lp) if left.n_rows else 0.0
    rr = min(c.n_distinct / right.n_rows for c in rp) if right.n_rows else 0.0
    if abs(lr - rr) < 1e-9:
        return None
    return "left" if lr > rr else "right"


def _evidence(
    registry: Registry,
    left_table: str,
    left_cols: list[str],
    right_table: str,
    right_cols: list[str],
    contain: float,
    name_sim: float,
    signature_match: bool,
    projection: Projection,
    grain: Grain | None,
) -> str:
    la, ra = registry.alias_of(left_table), registry.alias_of(right_table)
    left_ref = f"{la}.{'+'.join(left_cols)}"
    right_ref = f"{ra}.{'+'.join(right_cols)}"

    parts = [f"{contain:.0%} of {left_ref} values appear in {right_ref}"]
    if grain is not None:
        parts.append(f"after truncating dates to {grain.unit}")
    elif projection.name != "raw":
        parts.append(f"after normalizing values ({projection.name.replace('_', ' ')})")
    if signature_match:
        parts.append("both columns share one value format")
    if name_sim >= 0.5:
        parts.append("names agree")
    return "; ".join(parts) + "."


def _build(
    registry: Registry,
    *,
    left_table: str,
    left_cols: list[str],
    right_table: str,
    right_cols: list[str],
    kind: str,
    contain: float,
    name_sim: float,
    signature_match: bool,
    projection: Projection = RAW,
    grain: Grain | None = None,
) -> Relationship:
    left = registry.profile(left_table)
    right = registry.profile(right_table)
    score = score_pair(contain, name_sim, signature_match)

    if grain is not None:
        derivation = grain.sql(f"{left_table}.{grain.source_column}")
    else:
        derivation = projection.express(f"{left_table}.{left_cols[0]}")

    auto = score >= AUTO_CONFIRM and contain >= FK_ACCEPT
    return Relationship(
        id=build_relationship_id(left_table, left_cols, right_table, right_cols),
        left_table=left_table,
        left_columns=left_cols,
        right_table=right_table,
        right_columns=right_cols,
        kind=kind,  # type: ignore[arg-type]
        containment=round(contain, 4),
        name_similarity=round(name_sim, 4),
        score=score,
        parent_side=_parent_side(left, left_cols, right, right_cols),  # type: ignore[arg-type]
        evidence=_evidence(
            registry, left_table, left_cols, right_table, right_cols,
            contain, name_sim, signature_match, projection, grain,
        ),
        status="confirmed" if auto else "proposed",
        derivation=derivation,
        projection=None if projection.name == "raw" else projection.name,
    )


# --------------------------------------------------------------------------- #
# Top-level discovery
# --------------------------------------------------------------------------- #


@dataclass
class Discovery:
    """Everything statistics could determine, plus what needs adjudication."""

    relationships: list[Relationship] = field(default_factory=list)
    undecided: list[PairMeasure] = field(default_factory=list)


def discover(registry: Registry) -> Discovery:
    """Statistical relationship discovery over every table pair. No LLM involved."""
    values = ValueSets(registry)
    profiles = [registry.profile(t) for t in registry.data_table_ids]
    out = Discovery()

    for left, right in itertools.combinations(profiles, 2):
        measures = [
            m
            for lc in left.columns
            for rc in right.columns
            if (m := measure_pair(registry, values, left, lc, right, rc)) is not None
        ]

        accepted = [
            m
            for m in measures
            if m.containment >= FK_ACCEPT
            and m.identifier_join
            and is_corroborated(m)
            and has_key_side(registry, m)
        ]
        for m in sorted(accepted, key=lambda m: -m.score):
            out.relationships.append(
                _build(
                    registry,
                    left_table=m.left_table, left_cols=[m.left_column],
                    right_table=m.right_table, right_cols=[m.right_column],
                    kind="fk", contain=m.containment, name_sim=m.name_similarity,
                    signature_match=m.signature_match, projection=m.projection,
                )
            )

        # A pair whose normalized names are identical and whose values already overlap
        # substantially does not need a third signal -- the statistics are the evidence.
        # Sending it to the model is actively harmful: asked about branches.cert vs
        # institutions.cert (the real FDIC foreign key, same name, 58% overlap) a 7B
        # answers "different tables and columns" and vetoes it. Statistics propose, and
        # the model must not be able to overrule them; it only breaks genuine ties.
        strong = [
            m
            for m in measures
            if LLM_BAND_LOW <= m.containment < FK_ACCEPT
            and m.identifier_join
            and m.name_similarity >= NAME_IDENTICAL
            and has_key_side(registry, m)
        ]
        for m in sorted(strong, key=lambda m: -m.score):
            out.relationships.append(
                _build(
                    registry,
                    left_table=m.left_table, left_cols=[m.left_column],
                    right_table=m.right_table, right_cols=[m.right_column],
                    kind="fk", contain=m.containment, name_sim=m.name_similarity,
                    signature_match=m.signature_match, projection=m.projection,
                )
            )
        strong_ids = {(m.left_column, m.right_column) for m in strong}

        out.undecided.extend(
            m
            for m in measures
            if LLM_BAND_LOW <= m.containment < FK_ACCEPT
            and m.identifier_join
            and has_key_side(registry, m)
            # Band candidates are weak on containment by definition, so they must be
            # strong on something else before spending one of the 8 model calls.
            # Without this, "failures.id" and "record_calendar_day" -- 52% overlap,
            # nothing in common by name -- crowd out the real key waiting behind them.
            and (m.signature_match or m.name_similarity > 0)
            and (m.left_column, m.right_column) not in strong_ids
        )

        if not accepted:
            out.relationships.extend(_composites(registry, values, left, right, measures))
            out.relationships.extend(_derived(registry, values, left, right))

    return out


def _composites(
    registry: Registry, values: ValueSets, left: TableProfile, right: TableProfile,
    measures: list[PairMeasure],
) -> list[Relationship]:
    # High-containment pairs that failed the key-side gate belong here too: they are
    # precisely the (month, region) style columns a composite key is built from.
    pool = [m for m in measures if m.containment >= COMPOSITE_POOL_LOW]
    out = []
    for left_cols, right_cols, contain in composite_candidates(values, pool):
        per_pair = [
            name_similarity(
                left.column(lc).normalized_name_tokens, right.column(rc).normalized_name_tokens
            )
            for lc, rc in zip(left_cols, right_cols, strict=True)
        ]
        # Every part must agree, not just the best one. A real composite is the same
        # key on both sides -- (month, region) = (month, region). Taking the max let
        # "crash_date+unit_no" pair with "record_date+src_line_nbr" on the strength of
        # the shared word "date" alone.
        if min(per_pair) < COMPOSITE_NAME_MIN:
            continue
        name_sim = max(per_pair)
        out.append(
            _build(
                registry,
                left_table=left.table_id, left_cols=left_cols,
                right_table=right.table_id, right_cols=right_cols,
                kind="composite", contain=contain, name_sim=name_sim, signature_match=False,
            )
        )
    return out


def _derived(
    registry: Registry, values: ValueSets, left: TableProfile, right: TableProfile
) -> list[Relationship]:
    """SPEC 7.4: daily on one side, monthly on the other, so truncate and retry."""
    left_dates = [c.name for c in left.columns if c.semantic_type == "date"]
    right_dates = [c.name for c in right.columns if c.semantic_type == "date"]
    if not left_dates or not right_dates:
        return []

    for lc, rc in itertools.product(left_dates, right_dates):
        for unit in GRAINS:
            lg, rg = Grain(unit, lc), Grain(unit, rc)
            left_set = _grain_values(registry, values, left.table_id, lg)
            right_set = _grain_values(registry, values, right.table_id, rg)
            contain = containment(left_set, right_set)
            if contain < FK_ACCEPT:
                continue

            # A truncated date is only a join key if it discriminates. Two tables that
            # merely cover the same period match at containment 1.0 -- three attendance
            # sheets all from March produce a single month each and "join" perfectly to
            # each other and to a holidays sheet. Require both sides to span more than
            # one period, and one side to be a genuine per-period row (a month table).
            if len(left_set) < MIN_DISTINCT or len(right_set) < MIN_DISTINCT:
                continue
            left_ratio = len(left_set) / left.n_rows if left.n_rows else 0.0
            right_ratio = len(right_set) / right.n_rows if right.n_rows else 0.0
            if max(left_ratio, right_ratio) < PARENT_UNIQUE_MIN:
                continue

            registry.add_derived_column(
                left.table_id, lg.column, lg.sql(f'"{left.table_id}"."{lc}"')
            )
            registry.add_derived_column(
                right.table_id, rg.column, rg.sql(f'"{right.table_id}"."{rc}"')
            )
            return [
                _build(
                    registry,
                    left_table=left.table_id, left_cols=[lg.column],
                    right_table=right.table_id, right_cols=[rg.column],
                    kind="derived", contain=contain,
                    name_sim=name_similarity(
                        left.column(lc).normalized_name_tokens,
                        right.column(rc).normalized_name_tokens,
                    ),
                    signature_match=False, grain=lg,
                )
            ]
    return []


def _grain_values(registry: Registry, values: ValueSets, table_id: str, grain: Grain) -> set[str]:
    series = registry.frame(table_id)[grain.source_column]
    truncated = grain.apply(series).dropna()
    return set(truncated.dt.strftime("%Y-%m-%d")) if len(truncated) else set()


# --------------------------------------------------------------------------- #
# LLM adjudication (SPEC 7.5) -- the third signal, never the first
# --------------------------------------------------------------------------- #

ADJUDICATION_SYSTEM = (
    "You decide whether two database columns hold the same kind of real-world "
    "identifier, so that joining on them would be meaningful. They come from "
    "different tables and may have different names; that is expected and is not a "
    "reason to say no. Judge the values, not the column names. Answer with JSON only."
)


def adjudicate(
    registry: Registry, discovery: Discovery, client, max_calls: int = MAX_LLM_CALLS
) -> list[Relationship]:
    """Ask the model about the 0.5-0.75 band only, highest-scoring candidates first.

    The model can confirm or deny an edge that containment already surfaced. It is
    never shown a pair the statistics rejected, so it cannot invent a join.
    """
    if client is None:
        return []

    # Deterministic ordering, so which candidates get the 8 calls is reproducible
    # across eval runs rather than dependent on dict iteration order.
    ranked = sorted(
        discovery.undecided, key=lambda m: (-m.score, m.left_table, m.left_column)
    )[:max_calls]

    out: list[Relationship] = []
    for m in ranked:
        left = registry.profile(m.left_table)
        right = registry.profile(m.right_table)
        lc, rc = left.column(m.left_column), right.column(m.right_column)
        if lc is None or rc is None:
            continue

        user = (
            "Two columns from different tables. Do they refer to the same "
            "real-world entity?\n"
            f'A: table "{left.alias}", column "{lc.name}", samples: {", ".join(lc.samples[:3])}\n'
            f'B: table "{right.alias}", column "{rc.name}", samples: {", ".join(rc.samples[:3])}\n'
            'Answer with JSON only: {"same_entity": true|false, "why": "<8 words>"}'
        )
        try:
            reply = client.complete_json(ADJUDICATION_SYSTEM, user, max_tokens=80)
        except Exception as exc:  # a dead model must not sink the whole upload
            _warn(f"adjudication skipped for {lc.name}~{rc.name}: {exc}")
            continue

        if not isinstance(reply, dict) or not reply.get("same_entity"):
            continue

        relationship = _build(
            registry,
            left_table=m.left_table, left_cols=[m.left_column],
            right_table=m.right_table, right_cols=[m.right_column],
            kind="fk", contain=m.containment, name_sim=m.name_similarity,
            signature_match=m.signature_match, projection=m.projection,
        )
        why = str(reply.get("why", "")).strip()
        suffix = f": {why}" if why else ""
        relationship.evidence += f" Model agrees these are the same entity{suffix}."
        relationship.status = "proposed"  # an LLM-assisted edge is never auto-confirmed
        out.append(relationship)

    return out


def _warn(message: str) -> None:
    """Adjudication failures surface as warnings rather than being swallowed."""
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def measures_as_json(measures: list[PairMeasure]) -> str:
    """Debug helper used by the eval harness to dump why a pair scored as it did."""
    return json.dumps(
        [
            {
                "left": f"{m.left_table}.{m.left_column}",
                "right": f"{m.right_table}.{m.right_column}",
                "containment": round(m.containment, 3),
                "name_similarity": round(m.name_similarity, 3),
                "projection": m.projection.name,
                "score": m.score,
            }
            for m in measures
        ],
        indent=2,
    )
