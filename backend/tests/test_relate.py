"""Phase 3 acceptance: the relationship discovery layer (SPEC 7).

Each test isolates one of the corruption cases from SPEC 11.1, so a regression names
the real-world failure it reintroduces.
"""

from __future__ import annotations

import duckdb
import pytest
from darwinbox.ingest.blocks import detect_blocks
from darwinbox.ingest.loader import load_file
from darwinbox.profile.registry import Registry
from darwinbox.relate.candidates import (
    Discovery,
    adjudicate,
    containment,
    discover,
    name_similarity,
)
from darwinbox.relate.graph import RelationshipGraph, RelationshipNotFoundError


def build(files: dict[str, str]) -> tuple[Registry, Discovery, RelationshipGraph]:
    """Load CSV text into a fresh session and run statistical discovery."""
    registry = Registry(duckdb.connect(":memory:"))
    ids: set[str] = set()
    aliases: set[str] = set()
    for name, text in files.items():
        for loaded in load_file(name, text.encode()):
            for detected in detect_blocks(name, loaded.sheet, loaded.grid, ids, aliases):
                registry.register(detected)

    discovery = discover(registry)
    graph = RelationshipGraph(registry)
    graph.extend(discovery.relationships)
    return registry, discovery, graph


def edge_between(graph: RelationshipGraph, a: str, b: str):
    for edge in graph.all:
        if {edge.left_table, edge.right_table} == {a, b}:
            return edge
    return None


def only_table(registry: Registry, needle: str) -> str:
    return next(t for t in registry.table_ids if needle in t)


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #


def test_containment_beats_jaccard_for_fact_to_dimension():
    fact = {f"C-{i % 50}" for i in range(50)}
    dimension = {f"C-{i}" for i in range(50)} | {f"C-{i}" for i in range(50, 500)}
    # Every fact key is present in the dimension: containment 1.0, Jaccard only 0.1.
    assert containment(fact, dimension) == 1.0


def test_name_similarity_uses_expanded_tokens():
    assert name_similarity(["customer", "identifier"], ["customer", "identifier"]) == 1.0
    # "identifier" is generic -- nearly every key expands to it -- so it must not make
    # two unrelated keys look similar. Only the content words count.
    assert name_similarity(["customer", "identifier"], ["product", "identifier"]) == 0.0
    assert name_similarity(["row", "identifier"], ["grade", "7"]) == 0.0


# --------------------------------------------------------------------------- #
# rename_key: the FK survives being renamed on one side
# --------------------------------------------------------------------------- #

CUSTOMERS = (
    "code,customer_name,city\n"
    "C-88,Acme,Pune\nC-91,Borealis,Delhi\nC-90,Cirrus,Mumbai\nC-104,Delta,Chennai\n"
)
ORDERS_RENAMED = (
    "order_id,cust_ref,amount\n"
    "ORD-1,C-88,120\nORD-2,C-91,240\nORD-3,C-88,90\nORD-4,C-104,310\nORD-5,C-90,55\n"
)


def test_renamed_key_is_still_discovered():
    registry, _, graph = build({"customers.csv": CUSTOMERS, "orders.csv": ORDERS_RENAMED})
    edge = edge_between(
        graph, only_table(registry, "customers"), only_table(registry, "orders")
    )

    assert edge is not None
    assert edge.kind == "fk"
    assert edge.containment == 1.0
    assert sorted(edge.left_columns + edge.right_columns) == ["code", "cust_ref"]
    # customers.code is the unique side, so it is the parent.
    parent = edge.left_table if edge.parent_side == "left" else edge.right_table
    assert parent == only_table(registry, "customers")
    # Renamed on one side, so the names no longer agree once the generic "identifier"
    # token is discounted. It is found on containment plus a shared value format, but
    # it is surfaced for a human rather than auto-confirmed -- which is the right
    # outcome for a join whose two columns are not called the same thing.
    assert edge.status == "proposed"
    assert edge in graph.active
    assert "appear in" in edge.evidence


# --------------------------------------------------------------------------- #
# noisy_keys: case / whitespace / punctuation noise
# --------------------------------------------------------------------------- #

ORDERS_NOISY = (
    "order_id,cust_ref,amount\n"
    "ORD-1, c-88 ,120\nORD-2,C_91,240\nORD-3,c-88,90\nORD-4, C-104,310\nORD-5,c_90,55\n"
)


def test_noisy_keys_are_rescued_by_a_normalized_projection():
    registry, _, graph = build({"customers.csv": CUSTOMERS, "orders.csv": ORDERS_NOISY})
    edge = edge_between(
        graph, only_table(registry, "customers"), only_table(registry, "orders")
    )

    assert edge is not None
    assert edge.containment >= 0.75
    assert edge.derivation is not None  # the projection is recorded for SQL
    assert "normalizing" in edge.evidence


# --------------------------------------------------------------------------- #
# drop_fk: no ID at all, only (region, month) links the two
# --------------------------------------------------------------------------- #

SALES = (
    "month,region,revenue\n"
    "2024-01,North,1000\n2024-01,South,900\n2024-02,North,1100\n2024-02,South,950\n"
    "2024-03,North,1200\n2024-03,South,1050\n"
)
TARGETS = (
    "month,region,target\n"
    "2024-01,North,1050\n2024-01,South,880\n2024-02,North,1150\n2024-02,South,900\n"
    "2024-03,North,1250\n2024-03,South,1000\n"
)


def test_composite_key_found_when_no_single_column_is_a_key():
    registry, _, graph = build({"sales.csv": SALES, "targets.csv": TARGETS})
    edge = edge_between(graph, only_table(registry, "sales"), only_table(registry, "targets"))

    assert edge is not None, "no edge found between sales and targets"
    assert edge.kind == "composite"
    assert sorted(edge.left_columns) == ["month", "region"]
    assert sorted(edge.right_columns) == ["month", "region"]
    assert edge.containment >= 0.8


def test_shared_dimension_columns_are_not_emitted_as_single_column_fks():
    # region matches at containment 1.0 on both sides but is a key on neither.
    _, _, graph = build({"sales.csv": SALES, "targets.csv": TARGETS})
    single = [e for e in graph.all if e.kind == "fk" and len(e.left_columns) == 1]
    assert single == []


# --------------------------------------------------------------------------- #
# change_granularity: daily one side, monthly the other
# --------------------------------------------------------------------------- #

DAILY = (
    "txn_date,store,amount\n"
    "2024-01-04,S1,100\n2024-01-19,S1,150\n2024-02-08,S1,200\n2024-02-22,S1,120\n"
    "2024-03-03,S1,180\n2024-03-27,S1,210\n"
)
MONTHLY = (
    "period,store,budget\n2024-01-01,S1,300\n2024-02-01,S1,350\n2024-03-01,S1,400\n"
)


def test_derived_granularity_key_links_daily_to_monthly():
    registry, _, graph = build({"daily.csv": DAILY, "monthly.csv": MONTHLY})
    edge = edge_between(graph, only_table(registry, "daily"), only_table(registry, "monthly"))

    assert edge is not None, "daily and monthly were not linked"
    assert edge.kind == "derived"
    assert edge.derivation is not None and "date_trunc" in edge.derivation
    assert "month" in edge.evidence

    # The derived column must be real, so generated SQL can name it directly.
    daily_id = only_table(registry, "daily")
    assert registry.has_column(daily_id, "txn_date__month")
    rows = registry.conn.execute(
        f'SELECT count(DISTINCT txn_date__month) FROM "{daily_id}"'
    ).fetchone()[0]
    assert rows == 3


# --------------------------------------------------------------------------- #
# unrelated_pair: two datasets with nothing in common
# --------------------------------------------------------------------------- #

WEATHER = "city,rainfall_mm\nOslo,88\nLima,3\nCairo,1\nTokyo,140\n"
PARTS = "part_no,weight_kg\nP-1,2.5\nP-2,7.25\nP-3,0.5\nP-4,11.0\n"


def test_unrelated_files_form_separate_components():
    registry, _, graph = build({"weather.csv": WEATHER, "parts.csv": PARTS})
    components = graph.components()

    assert len(components) == 2, f"expected 2 components, got {components}"
    assert all(len(c) == 1 for c in components)
    assert graph.join_path(*registry.table_ids) is None


# --------------------------------------------------------------------------- #
# Graph mechanics
# --------------------------------------------------------------------------- #

PAYMENTS = (
    "payment_id,order_ref,paid\n"
    "P-1,ORD-1,120\nP-2,ORD-2,240\nP-3,ORD-3,90\nP-4,ORD-4,310\nP-5,ORD-5,55\n"
)


def test_join_path_spans_three_tables():
    registry, _, graph = build(
        {"customers.csv": CUSTOMERS, "orders.csv": ORDERS_RENAMED, "payments.csv": PAYMENTS}
    )
    customers = only_table(registry, "customers")
    payments = only_table(registry, "payments")

    path = graph.join_path(customers, payments)
    assert path is not None
    assert len(path) == 2  # customers -> orders -> payments
    assert len(graph.components()) == 1


def test_rejected_edges_are_excluded_from_traversal_and_hints():
    registry, _, graph = build({"customers.csv": CUSTOMERS, "orders.csv": ORDERS_RENAMED})
    edge = graph.all[0]
    graph.set_status(edge.id, "confirmed")  # as a user would, before querying

    assert graph.to_sql_hints() != []
    graph.set_status(edge.id, "rejected")

    assert graph.active == []
    assert graph.to_sql_hints() == []
    assert len(graph.components()) == 2
    assert graph.join_path(*registry.table_ids) is None


def test_unknown_relationship_id_raises():
    _, _, graph = build({"customers.csv": CUSTOMERS, "orders.csv": ORDERS_RENAMED})
    with pytest.raises(RelationshipNotFoundError):
        graph.get("rel_nope")


def test_sql_hints_use_aliases_not_table_ids():
    registry, _, graph = build({"customers.csv": CUSTOMERS, "orders.csv": ORDERS_RENAMED})
    graph.set_status(graph.all[0].id, "confirmed")
    (hint,) = graph.to_sql_hints()

    aliases = {registry.alias_of(t) for t in registry.table_ids}
    assert any(hint.startswith(f"{a}.") for a in aliases)
    assert "__b0" not in hint


# --------------------------------------------------------------------------- #
# LLM adjudication is a third signal, never a first one
# --------------------------------------------------------------------------- #


class RecordingClient:
    def __init__(self, answer: bool) -> None:
        self.answer = answer
        self.calls: list[str] = []

    def complete_json(self, system: str, user: str, max_tokens: int = 700) -> dict:
        self.calls.append(user)
        return {"same_entity": self.answer, "why": "same customer code format"}


def test_adjudication_is_never_offered_a_pair_statistics_rejected():
    registry, discovery, _ = build({"weather.csv": WEATHER, "parts.csv": PARTS})
    client = RecordingClient(answer=True)

    added = adjudicate(registry, discovery, client)

    assert client.calls == [], "the model was asked about an unrelated pair"
    assert added == []


def test_adjudicated_edges_are_proposed_not_auto_confirmed():
    # Half of orders.cust_ref is missing from customers: containment lands in the band.
    customers = "code,name\nC-1,A\nC-2,B\nC-3,C\nC-4,D\nC-5,E\n"
    orders = (
        "order_id,cust_ref,amount\n"
        "O-1,C-1,10\nO-2,C-2,20\nO-3,C-3,30\nO-4,X-9,40\nO-5,X-8,50\nO-6,X-7,60\n"
    )
    registry, discovery, _ = build({"customers.csv": customers, "orders.csv": orders})

    assert discovery.undecided, "expected a candidate in the 0.5-0.75 band"

    client = RecordingClient(answer=True)
    added = adjudicate(registry, discovery, client)

    assert len(client.calls) == 1
    assert len(added) == 1
    assert added[0].status == "proposed"
    assert "Model agrees" in added[0].evidence


def test_model_denial_drops_the_edge():
    customers = "code,name\nC-1,A\nC-2,B\nC-3,C\nC-4,D\nC-5,E\n"
    orders = (
        "order_id,cust_ref,amount\n"
        "O-1,C-1,10\nO-2,C-2,20\nO-3,C-3,30\nO-4,X-9,40\nO-5,X-8,50\nO-6,X-7,60\n"
    )
    registry, discovery, _ = build({"customers.csv": customers, "orders.csv": orders})

    assert adjudicate(registry, discovery, RecordingClient(answer=False)) == []


def test_adjudication_is_capped_and_deterministic():
    registry, discovery, _ = build({"customers.csv": CUSTOMERS, "orders.csv": ORDERS_RENAMED})
    client = RecordingClient(answer=True)

    adjudicate(registry, discovery, client, max_calls=8)
    assert len(client.calls) <= 8

    second = RecordingClient(answer=True)
    adjudicate(registry, discovery, second, max_calls=8)
    assert client.calls == second.calls


# --------------------------------------------------------------------------- #
# split_sheets: one logical table spread across several sheets
# --------------------------------------------------------------------------- #

WEEK_COLUMNS = "employee_id,work_date,hours\n"
WEEK1 = WEEK_COLUMNS + "EMP-1,2024-03-01,8\nEMP-2,2024-03-01,7\nEMP-3,2024-03-01,9\n"
WEEK2 = WEEK_COLUMNS + "EMP-1,2024-03-08,6\nEMP-2,2024-03-08,8\nEMP-3,2024-03-08,7\n"
EMPLOYEES = "employee_id,department\nEMP-1,Sales\nEMP-2,Ops\nEMP-3,Sales\n"


def test_identical_schemas_are_fused_into_one_table():
    registry, _, _ = build({"week1.csv": WEEK1, "week2.csv": WEEK2})
    merged = registry.merge_identical_tables()

    assert len(merged) == 1
    merged_id, members = merged[0]
    assert len(members) == 2

    # The fragments are gone; one table with every row takes their place.
    assert registry.table_ids == [merged_id]
    assert registry.profile(merged_id).n_rows == 6
    count = registry.conn.execute(f'SELECT count(*) FROM "{merged_id}"').fetchone()[0]
    assert count == 6


def test_tables_with_different_schemas_are_never_fused():
    # A wrong union silently doubles rows in every later answer, so the signature
    # match must be exact.
    registry, _, _ = build({"week1.csv": WEEK1, "employees.csv": EMPLOYEES})
    assert registry.merge_identical_tables() == []
    assert len(registry.table_ids) == 2


def test_fusion_happens_before_discovery_so_the_graph_sees_one_edge():
    from darwinbox.llm.client import FakeLLMClient
    from darwinbox.session import Session

    session = Session(FakeLLMClient())
    session.add_files([
        ("week1.csv", WEEK1.encode()),
        ("week2.csv", WEEK2.encode()),
        ("employees.csv", EMPLOYEES.encode()),
    ])

    assert len(session.tables) == 2, [t.alias for t in session.tables]
    assert len(session.graph.active) == 1


def test_fusion_is_reported_as_an_upload_warning():
    from darwinbox.llm.client import FakeLLMClient
    from darwinbox.session import Session

    session = Session(FakeLLMClient())
    outcome = session.add_files([
        ("week1.csv", WEEK1.encode()),
        ("week2.csv", WEEK2.encode()),
    ])

    assert any("combined into" in w for w in outcome.warnings), outcome.warnings
    assert len(outcome.tables) == 1


# --------------------------------------------------------------------------- #
# Digits-only must not be offered to prose
#
# Found on 42 real open-data CSVs: farmers_markets joined library_circulation on
# market_location = young_adult_mass_market_paperback_books. The columns share no
# values whatsoever -- the edge came from the digits-only projection reducing the
# street address "204 Lisha Kill Rd Colonie" to 204, which duly appears in a column
# counting paperback loans.
# --------------------------------------------------------------------------- #

from darwinbox.models import ColumnProfile  # noqa: E402
from darwinbox.relate.candidates import DIGITS_ONLY, applicable_projections  # noqa: E402


def column(name: str, semantic_type: str, samples: list[str]) -> ColumnProfile:
    return ColumnProfile(
        name=name,
        semantic_type=semantic_type,
        dtype="string",
        n_distinct=len(samples),
        n_null=0,
        null_rate=0.0,
        is_unique=True,
        samples=samples,
    )


def test_digits_only_is_offered_for_a_prefixed_code_against_a_bare_number():
    """The case it exists for: EMP-0042 against 42."""
    code = column("employee_code", "id", ["EMP-0042", "EMP-0043", "EMP-0101"])
    bare = column("employee_id", "numeric", ["42", "43", "101"])
    assert DIGITS_ONLY in applicable_projections(code, bare)


def test_digits_only_is_refused_for_free_text_that_merely_contains_a_number():
    address = column(
        "market_location",
        "text",
        ["204 Lisha Kill Rd Colonie", "6654 Dunnsville Rd Altamont", "2479 Western Ave"],
    )
    count = column("paperback_books", "numeric", ["0", "2", "11"])
    assert DIGITS_ONLY not in applicable_projections(address, count)


def test_a_long_single_token_is_still_not_a_code():
    """No spaces is necessary but not sufficient; a code is also short."""
    slug = column("slug", "text", ["north-region-quarterly-summary-2024-final-v2-001"])
    count = column("n", "numeric", ["1", "2", "3"])
    assert DIGITS_ONLY not in applicable_projections(slug, count)


def test_two_numeric_columns_get_no_digits_projection():
    """Digits-only between two bare-number columns throws away nothing and proves nothing."""
    left = column("a", "numeric", ["1", "2", "3"])
    right = column("b", "numeric", ["1", "2", "3"])
    assert DIGITS_ONLY not in applicable_projections(left, right)
