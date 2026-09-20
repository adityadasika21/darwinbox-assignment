"""BM25 table routing (SPEC 8.2).

With 8 GB of VRAM the whole schema does not fit in the prompt, so retrieval is not
an optimisation here -- it is what makes the system work at all on more than a
handful of tables. Two stages: BM25 over table documents, then graph expansion along
confirmed relationships so a join partner is never left out of the prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from darwinbox.profile.columns import normalize_name_tokens
from darwinbox.profile.registry import Registry
from darwinbox.relate.graph import RelationshipGraph

TOP_K = 4
MAX_EXPANSION = 2
MAX_CATEGORY_VALUES = 20
ALIAS_WEIGHT = 3

_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is", "are", "was",
    "were", "by", "with", "what", "which", "how", "many", "much", "show", "me", "list",
    "give", "get", "per", "each", "total", "all", "from", "that", "this", "it", "do",
    "does", "did", "be", "been", "has", "have", "had", "at", "as", "we", "our",
}


@dataclass
class RoutingResult:
    """Which tables were chosen, and the terms that earned them their place."""

    table_ids: list[str] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)
    expanded: list[str] = field(default_factory=list)
    matched_terms: dict[str, list[str]] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)

    def summary(self, registry: Registry) -> str:
        aliases = ", ".join(registry.alias_of(t) for t in self.table_ids)
        if not self.expanded:
            return f"Routed to {aliases}"
        added = ", ".join(registry.alias_of(t) for t in self.expanded)
        return f"Routed to {aliases} (pulled in {added} via a confirmed relationship)"


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens with abbreviations expanded, so 'cust' matches 'customer'."""
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9]*|\d+", text.lower())
    out: list[str] = []
    for word in words:
        if word in _STOPWORDS:
            continue
        out.extend(normalize_name_tokens(word) or [word])
    return out


def table_document(registry: Registry, table_id: str) -> list[str]:
    """The BM25 document for one table: names, not data.

    Category values are included because questions name them ("revenue in the North
    region") far more often than they name the column that holds them. Only low
    cardinality columns qualify, so this never leaks a meaningful amount of data.
    """
    profile = registry.profile(table_id)
    # The alias is repeated because BM25 normalises by document length: a narrow side
    # table matching one incidental column ("pay_month") was outranking the table the
    # question names outright ("order amount"), purely for having fewer columns.
    terms: list[str] = list(tokenize(profile.alias)) * ALIAS_WEIGHT

    # The catalog answers questions phrased about the upload itself rather than about
    # any column in it, so its document carries the words people actually use.
    if profile.kind == "catalog":
        # Deliberately NOT its column names. "n_distinct" matched "how many distinct
        # customers", and the catalog then hijacked a question about the data --
        # answering it without the data tables even reaching the prompt.
        return terms + tokenize(
            "schema catalog file files table tables column columns "
            "present contain contains inside overview describe structure uploaded"
        )

    terms.extend(tokenize(profile.source.filename))
    if profile.source.sheet:
        terms.extend(tokenize(profile.source.sheet))

    for column in profile.columns:
        terms.extend(tokenize(column.name))
        terms.extend(column.normalized_name_tokens)
        if column.semantic_type == "category":
            for value in column.samples[:MAX_CATEGORY_VALUES]:
                terms.extend(tokenize(value))

    return terms


class TableRouter:
    """BM25 index over table documents, rebuilt whenever the schema changes."""

    def __init__(self, registry: Registry) -> None:
        self._registry = registry
        self._table_ids: list[str] = list(registry.table_ids)
        self._documents = [table_document(registry, t) for t in self._table_ids]
        self._index = BM25Okapi(self._documents) if self._documents else None

    def route(
        self,
        question: str,
        graph: RelationshipGraph,
        top_k: int = TOP_K,
        max_expansion: int = MAX_EXPANSION,
    ) -> RoutingResult:
        """Retrieve the top-k tables, then expand along confirmed relationships."""
        result = RoutingResult()
        if self._index is None or not self._table_ids:
            return result

        terms = tokenize(question)
        scores = self._index.get_scores(terms) if terms else [0.0] * len(self._table_ids)

        ranked = sorted(
            zip(self._table_ids, scores, strict=True), key=lambda p: (-p[1], p[0])
        )
        # A question that matches nothing still needs tables to reason over; falling
        # back to the largest tables beats refusing before the model has seen a schema.
        retrieved = [t for t, s in ranked if s > 0][:top_k]
        if not retrieved:
            retrieved = [t for t, _ in ranked][:top_k]

        retrieved = self._resolve_catalog(retrieved)
        result.retrieved = retrieved
        result.scores = {t: round(float(s), 3) for t, s in ranked if t in set(retrieved)}
        result.matched_terms = {
            t: self._matched(t, terms) for t in retrieved
        }

        chosen = list(retrieved)
        for table_id in retrieved:
            for neighbour in graph.neighbours(table_id, confirmed_only=True):
                if len(result.expanded) >= max_expansion:
                    break
                if neighbour not in chosen:
                    chosen.append(neighbour)
                    result.expanded.append(neighbour)

        result.table_ids = chosen
        return result

    def _resolve_catalog(self, retrieved: list[str]) -> list[str]:
        """The catalog is all or nothing: a schema question, or not one at all.

        Mixing them crowds the prompt. A question about the data was being routed to
        the catalog *and* its tables, and the model -- shown a schema_columns listing
        alongside real columns -- would return no SQL at all. The catalog ranks first
        for a genuine schema question and below the data otherwise, so rank decides.
        """
        if not retrieved:
            return retrieved
        is_catalog = [self._registry.profile(t).kind == "catalog" for t in retrieved]
        if is_catalog[0]:
            return [t for t, cat in zip(retrieved, is_catalog, strict=True) if cat]
        return [t for t, cat in zip(retrieved, is_catalog, strict=True) if not cat]

    def _matched(self, table_id: str, terms: list[str]) -> list[str]:
        document = set(self._documents[self._table_ids.index(table_id)])
        seen: list[str] = []
        for term in terms:
            if term in document and term not in seen:
                seen.append(term)
        return seen
