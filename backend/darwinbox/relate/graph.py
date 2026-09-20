"""The relationship graph: connected components, join paths, SQL hints.

Undirected with typed edges. Connected components are the unit that matters to the
product: two tables in different components genuinely cannot be joined, and the UI
says so out loud rather than letting the model improvise a join.
"""

from __future__ import annotations

from collections import deque

from darwinbox.models import Relationship
from darwinbox.profile.registry import Registry
from darwinbox.relate.derive import PROJECTIONS


class RelationshipNotFoundError(Exception):
    """Raised for an unknown relationship id (API maps to 404)."""


class RelationshipGraph:
    """Every discovered or user-added edge, plus the queries the rest of the app needs."""

    def __init__(self, registry: Registry) -> None:
        self._registry = registry
        self._edges: dict[str, Relationship] = {}

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #

    def add(self, relationship: Relationship) -> Relationship:
        """Add an edge, keeping the higher-scoring one if the pair already exists."""
        existing = self._edges.get(relationship.id)
        if existing is not None and existing.score >= relationship.score:
            return existing
        self._edges[relationship.id] = relationship
        return relationship

    def extend(self, relationships: list[Relationship]) -> None:
        for relationship in relationships:
            self.add(relationship)

    def set_status(self, relationship_id: str, status: str) -> Relationship:
        relationship = self.get(relationship_id)
        relationship.status = status  # type: ignore[assignment]
        return relationship

    def get(self, relationship_id: str) -> Relationship:
        try:
            return self._edges[relationship_id]
        except KeyError as exc:
            raise RelationshipNotFoundError(relationship_id) from exc

    # ------------------------------------------------------------------ #
    # Views
    # ------------------------------------------------------------------ #

    @property
    def all(self) -> list[Relationship]:
        """Every edge, best first, so the UI shows the strongest evidence at the top."""
        return sorted(self._edges.values(), key=lambda r: (-r.score, r.id))

    @property
    def active(self) -> list[Relationship]:
        """Confirmed and proposed edges: exactly what the LLM is allowed to see."""
        return [r for r in self.all if r.status != "rejected"]

    @property
    def confirmed(self) -> list[Relationship]:
        return [r for r in self.all if r.status == "confirmed"]

    # ------------------------------------------------------------------ #
    # Traversal
    # ------------------------------------------------------------------ #

    def _adjacency(self, edges: list[Relationship]) -> dict[str, list[tuple[str, Relationship]]]:
        adjacency: dict[str, list[tuple[str, Relationship]]] = {
            t: [] for t in self._registry.data_table_ids
        }
        for edge in edges:
            adjacency.setdefault(edge.left_table, []).append((edge.right_table, edge))
            adjacency.setdefault(edge.right_table, []).append((edge.left_table, edge))
        return adjacency

    def components(self) -> list[list[str]]:
        """Connected components over active edges; isolated tables are singletons."""
        adjacency = self._adjacency(self.active)
        seen: set[str] = set()
        out: list[list[str]] = []

        for start in self._registry.data_table_ids:
            if start in seen:
                continue
            group: list[str] = []
            queue = deque([start])
            seen.add(start)
            while queue:
                node = queue.popleft()
                group.append(node)
                for neighbour, _ in adjacency.get(node, []):
                    if neighbour not in seen:
                        seen.add(neighbour)
                        queue.append(neighbour)
            out.append(sorted(group))

        return sorted(out, key=lambda g: (-len(g), g[0]))

    def join_path(self, left: str, right: str) -> list[Relationship] | None:
        """Shortest chain of active edges connecting two tables, or None."""
        if left == right:
            return []
        adjacency = self._adjacency(self.active)
        previous: dict[str, tuple[str, Relationship]] = {}
        seen = {left}
        queue = deque([left])

        while queue:
            node = queue.popleft()
            for neighbour, edge in adjacency.get(node, []):
                if neighbour in seen:
                    continue
                seen.add(neighbour)
                previous[neighbour] = (node, edge)
                if neighbour == right:
                    return _unwind(previous, left, right)
                queue.append(neighbour)

        return None

    def neighbours(self, table_id: str, confirmed_only: bool = True) -> list[str]:
        """Tables one confirmed hop away, used to expand router results (SPEC 8.2)."""
        edges = self.confirmed if confirmed_only else self.active
        out = []
        for edge in edges:
            if edge.left_table == table_id:
                out.append(edge.right_table)
            elif edge.right_table == table_id:
                out.append(edge.left_table)
        return sorted(set(out))

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def to_sql_hints(self, edges: list[Relationship] | None = None) -> list[str]:
        """Join predicates in alias space, e.g. 'orders.cust_ref = customers.code'."""
        chosen = self.confirmed if edges is None else edges
        return [hint for edge in chosen for hint in self.hints_for(edge)]

    def hints_for(self, edge: Relationship) -> list[str]:
        """Join predicates, with any normalization applied to BOTH sides.

        An edge discovered under a projection must be joined under it too. Emitting a
        bare equality for a noisy key produces SQL that parses, runs, and silently
        drops every row whose key differs by case or punctuation -- a wrong answer
        with no error, which is the worst failure this system can have.
        """
        left_alias = self._registry.alias_of(edge.left_table)
        right_alias = self._registry.alias_of(edge.right_table)
        wrap = _projection_sql(edge.projection)
        return [
            f"{wrap(f'{left_alias}.{lc}')} = {wrap(f'{right_alias}.{rc}')}"
            for lc, rc in zip(edge.left_columns, edge.right_columns, strict=True)
        ]

    def describe(self, edge: Relationship) -> str:
        """One prompt line: the predicate, its status and its strength."""
        predicate = " AND ".join(self.hints_for(edge))
        return f"{predicate}    ({edge.status}, {edge.containment:.0%} value overlap)"


def _projection_sql(name: str | None):
    """Return the SQL wrapper for a projection name, or identity when there is none."""
    if not name:
        return lambda column: column
    for projection in PROJECTIONS:
        if projection.name == name:
            return projection.sql
    return lambda column: column


def _unwind(
    previous: dict[str, tuple[str, Relationship]], start: str, end: str
) -> list[Relationship]:
    path: list[Relationship] = []
    node = end
    while node != start:
        node, edge = previous[node]
        path.append(edge)
    path.reverse()
    return path
