"""Check that a result's column names are grounded in the schema.

Asked for "the average temperature for each customer region" against files that hold no
temperature at all, the planner wrote ``AVG(w.rainfall_mm) AS avg_temperature`` and the
answer read "The average temperature ... North (244.5)". Every layer did its job: the
join was real, the SQL was valid, the arithmetic was right. The lie was in the label.

Refusing whenever an alias names something absent from the schema is too blunt --
``sum(amount) AS total_revenue`` is the same shape and is perfectly reasonable, and the
eval's own ground truth is written that way. A synonym is legitimate; an invention is
not, and no amount of string comparison separates them.

So the substitution is reported rather than blocked. The answer still comes back, with a
sentence naming the column that actually produced it. That cannot cause a false refusal,
and it makes the one failure mode this system is built to avoid -- a confident number
under a label nobody can check -- visible in the answer itself.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from darwinbox.profile.columns import normalize_name_tokens

# Words that describe an aggregation rather than name a thing in the data.
AGGREGATION_TOKENS = {
    "average", "avg", "mean", "total", "sum", "count", "number", "minimum", "min",
    "maximum", "max", "highest", "lowest", "distinct", "percent", "percentage",
    "ratio", "rate", "per", "each", "overall", "value", "result", "amount",
}


def ungrounded_labels(sql: str, columns: list[str], tables: list[str]) -> dict[str, str]:
    """Map each output alias to the source column it renames, where the name is new.

    Only aliases over a real column are considered: a bare ``count(*) AS n`` renames
    nothing and can invent nothing.
    """
    try:
        statement = sqlglot.parse_one(sql, dialect="duckdb")
    except Exception:  # noqa: BLE001 - the validator already parsed this; never fail here
        return {}

    known = set()
    for name in [*columns, *tables]:
        known.update(normalize_name_tokens(name))
    known -= AGGREGATION_TOKENS

    out: dict[str, str] = {}
    select = statement.find(exp.Select)
    if select is None:
        return {}

    for projection in select.expressions:
        if not isinstance(projection, exp.Alias):
            continue
        alias = projection.alias
        sources = [c.name for c in projection.find_all(exp.Column) if c.name]
        if not alias or not sources:
            continue

        novel = [
            token
            for token in normalize_name_tokens(alias)
            if token not in known and token not in AGGREGATION_TOKENS
        ]
        if novel:
            out[alias] = sources[0]
    return out


def caveat(labels: dict[str, str]) -> str:
    """One sentence naming what the numbers actually are, or empty when all is well."""
    if not labels:
        return ""
    parts = [f'"{alias}" is {source}' for alias, source in sorted(labels.items())]
    joined = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + f" and {parts[-1]}"
    return (
        f"No column in these files is named for that, so {joined} — "
        f"check that is the measure you meant."
    )
