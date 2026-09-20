"""Shared fixtures: a small three-table session built without FastAPI or Ollama."""

from __future__ import annotations

import duckdb
import pytest
from darwinbox.ingest.blocks import detect_blocks
from darwinbox.ingest.loader import load_file
from darwinbox.llm.router import TableRouter
from darwinbox.profile.registry import Registry
from darwinbox.relate.candidates import discover
from darwinbox.relate.graph import RelationshipGraph

CUSTOMERS = (
    "code,customer_name,region\n"
    "C-88,Acme,North\nC-91,Borealis,South\nC-90,Cirrus,North\nC-104,Delta,East\n"
)
ORDERS = (
    "order_id,cust_ref,order_date,amount\n"
    "ORD-1,C-88,2024-01-05,120.50\n"
    "ORD-2,C-91,2024-01-17,240.00\n"
    "ORD-3,C-88,2024-02-02,90.25\n"
    "ORD-4,C-104,2024-02-19,310.00\n"
    "ORD-5,C-90,2024-03-08,55.75\n"
    "ORD-6,C-91,2024-03-22,410.00\n"
)
HEADCOUNT = (
    "employee_id,department,salary,joined_on\n"
    "EMP-01,Sales,82000,2021-04-01\n"
    "EMP-02,Ops,74000,2022-07-15\n"
    "EMP-03,Sales,91000,2020-01-20\n"
    "EMP-04,Finance,68000,2023-03-11\n"
)

FILES = {"customers.csv": CUSTOMERS, "orders.csv": ORDERS, "headcount.csv": HEADCOUNT}


def load_session(files: dict[str, str]):
    """Registry + graph + router for a set of CSV texts."""
    registry = Registry(duckdb.connect(":memory:"))
    ids: set[str] = set()
    aliases: set[str] = set()
    for name, text in files.items():
        for loaded in load_file(name, text.encode()):
            for detected in detect_blocks(name, loaded.sheet, loaded.grid, ids, aliases):
                registry.register(detected)

    registry.refresh_catalog()
    graph = RelationshipGraph(registry)
    graph.extend(discover(registry).relationships)
    return registry, graph, TableRouter(registry)


@pytest.fixture
def demo():
    """(registry, graph, router) over customers + orders + an unrelated headcount file."""
    return load_session(FILES)


@pytest.fixture
def registry(demo):
    return demo[0]


@pytest.fixture
def graph(demo):
    return demo[1]


@pytest.fixture
def router(demo):
    return demo[2]


def alias_for(registry, needle: str) -> str:
    return registry.alias_of(next(t for t in registry.table_ids if needle in t))
