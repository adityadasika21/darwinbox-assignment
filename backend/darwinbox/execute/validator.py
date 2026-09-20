"""sqlglot guards over generated SQL (SPEC 8.5).

Nothing the model writes reaches DuckDB unvalidated. The validator also rewrites
aliases to table_ids: the model is shown short human names ("orders") and never the
mangled physical name, so translation has to happen somewhere, and doing it on the
parsed tree is safer than string substitution.

It knows nothing about prompts or HTTP -- it takes SQL and a registry, and returns
either rewritten SQL or a typed error whose message is fit to show a human.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import duckdb
import sqlglot
from sqlglot import exp

from darwinbox.profile.registry import Registry

DIALECT = "duckdb"

FORBIDDEN_NODES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.Command,
)

FORBIDDEN_KEYWORDS = (
    "insert", "update", "delete", "create", "drop", "alter",
    "attach", "copy", "install", "load", "pragma", "truncate", "grant", "vacuum",
)


class ValidationError(Exception):
    """A generated statement that must not be executed."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class ValidatedSQL:
    """Rewritten, executable SQL plus what it touches."""

    sql: str
    tables: list[str] = field(default_factory=list)
    original: str = ""


def validate(sql: str, registry: Registry, explain: bool = True) -> ValidatedSQL:
    """Run every guard in SPEC 8.5 and return SQL safe to execute."""
    text = (sql or "").strip().rstrip(";").strip()
    if not text:
        raise ValidationError("EMPTY_SQL", "the model returned no SQL")

    statements = _parse(text)
    if len(statements) != 1:
        raise ValidationError(
            "MULTIPLE_STATEMENTS",
            f"expected exactly one statement, got {len(statements)}",
        )

    tree = statements[0]
    _reject_non_select(tree)
    _reject_forbidden(tree, text)

    tables = _rewrite_tables(tree, registry)
    _check_columns(tree, registry, tables)

    rewritten = tree.sql(dialect=DIALECT)
    if explain:
        _explain(rewritten, registry)

    return ValidatedSQL(sql=rewritten, tables=sorted(set(tables)), original=text)


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #


def _parse(text: str) -> list[exp.Expression]:
    try:
        parsed = sqlglot.parse(text, dialect=DIALECT)
    except Exception as exc:
        raise ValidationError("PARSE_ERROR", f"SQL did not parse: {exc}") from exc
    statements = [s for s in parsed if s is not None]
    if not statements:
        raise ValidationError("PARSE_ERROR", "SQL did not parse")
    return statements


def _reject_non_select(tree: exp.Expression) -> None:
    root = tree
    if isinstance(root, exp.Subquery):
        root = root.this
    if not isinstance(root, exp.Select | exp.Union | exp.Except | exp.Intersect):
        raise ValidationError(
            "NOT_A_SELECT", f"only SELECT is allowed, got {type(tree).__name__.upper()}"
        )


def _reject_forbidden(tree: exp.Expression, text: str) -> None:
    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES):
            raise ValidationError(
                "WRITE_STATEMENT",
                f"{type(node).__name__.upper()} is not allowed; this system is read-only",
            )

    # Belt and braces: a keyword sqlglot parsed into something benign still must not
    # reach DuckDB. Only bare words count, so a column named "loaded" stays legal.
    words = {w.strip("(),;").lower() for w in text.replace("\n", " ").split()}
    banned = words & set(FORBIDDEN_KEYWORDS)
    if banned:
        raise ValidationError(
            "WRITE_STATEMENT",
            f"{sorted(banned)[0].upper()} is not allowed; this system is read-only",
        )


def _cte_names(tree: exp.Expression) -> set[str]:
    return {
        cte.alias_or_name.lower()
        for cte in tree.find_all(exp.CTE)
        if cte.alias_or_name
    }


def _rewrite_tables(tree: exp.Expression, registry: Registry) -> list[str]:
    """Map every real table reference from alias to table_id, in place."""
    ctes = _cte_names(tree)
    touched: list[str] = []

    for node in tree.find_all(exp.Table):
        name = node.name
        if not name or name.lower() in ctes:
            continue
        try:
            table_id = registry.resolve(name)
        except Exception as exc:
            known = ", ".join(sorted(registry.alias_of(t) for t in registry.table_ids))
            raise ValidationError(
                "TABLE_NOT_FOUND",
                f'unknown table "{name}"; available tables are: {known}',
            ) from exc

        node.set("this", exp.to_identifier(table_id, quoted=True))
        node.set("db", None)
        node.set("catalog", None)
        # Keep the name the model used as an explicit alias. Columns are very often
        # qualified by the table name -- "orders.cust_ref = customers.code" -- and
        # those qualifiers are column nodes, not table nodes, so rewriting the table
        # alone leaves every one of them dangling and DuckDB rejects valid SQL.
        if not node.alias:
            node.set("alias", exp.to_identifier(name))
        touched.append(table_id)

    if not touched:
        raise ValidationError("NO_TABLES", "the query does not reference any known table")
    return touched


def _check_columns(tree: exp.Expression, registry: Registry, tables: list[str]) -> None:
    """Every column must exist on some referenced table, a CTE, or a select alias."""
    allowed: set[str] = set()
    for table_id in set(tables):
        allowed.update(c.name.lower() for c in registry.profile(table_id).columns)
    allowed.update(
        key.split(".", 1)[1].lower()
        for key in registry.derived
        if key.split(".", 1)[0] in set(tables)
    )

    # Output aliases are legal in ORDER BY / GROUP BY / HAVING in DuckDB.
    for alias in tree.find_all(exp.Alias):
        if alias.alias:
            allowed.add(alias.alias.lower())
    # Anything projected out of a CTE or subquery is opaque to us; trust EXPLAIN there.
    opaque = _cte_names(tree) | {
        sub.alias_or_name.lower() for sub in tree.find_all(exp.Subquery) if sub.alias_or_name
    }
    if opaque:
        return

    for column in tree.find_all(exp.Column):
        name = column.name.lower()
        if not name or name == "*":
            continue
        if name not in allowed:
            close = _closest(name, allowed)
            hint = f'; did you mean "{close}"?' if close else ""
            raise ValidationError(
                "COLUMN_NOT_FOUND", f'unknown column "{column.name}"{hint}'
            )


def _closest(name: str, allowed: set[str]) -> str | None:
    """Cheap suggestion for the repair message: shared tokens, no fuzzy matching.

    Deliberately not Levenshtein -- fuzzy matching is out of scope for this build and
    a wrong suggestion in the repair prompt is worse than none.
    """
    parts = set(name.split("_"))
    best, best_overlap = None, 0
    for candidate in sorted(allowed):
        overlap = len(parts & set(candidate.split("_")))
        if overlap > best_overlap:
            best, best_overlap = candidate, overlap
    return best


def _explain(sql: str, registry: Registry) -> None:
    try:
        registry.conn.execute(f"EXPLAIN {sql}")
    except duckdb.Error as exc:
        raise ValidationError("EXPLAIN_FAILED", _clean_duckdb_error(exc)) from exc


def _clean_duckdb_error(exc: Exception) -> str:
    """First line only: DuckDB appends a candidate list and a caret diagram."""
    message = str(exc).strip().splitlines()
    return message[0] if message else "DuckDB rejected the query"
