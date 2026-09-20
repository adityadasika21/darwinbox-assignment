import { useMemo, useState } from "react";
import type { Relationship, RelationshipStatus, TableProfile } from "../api";

interface Props {
  tables: TableProfile[];
  relationships: Relationship[];
  components: string[][];
  onSetStatus: (id: string, status: RelationshipStatus) => void;
  onAdd: (body: {
    left_table: string;
    left_columns: string[];
    right_table: string;
    right_columns: string[];
  }) => void;
  busy: boolean;
}

/**
 * The centre pane, and the point of the product: every join the system believes in,
 * with the evidence for it and a way to overrule it.
 */
export default function RelationshipEditor({
  tables,
  relationships,
  components,
  onSetStatus,
  onAdd,
  busy,
}: Props) {
  const aliasOf = useMemo(() => {
    const map = new Map<string, string>();
    tables.forEach((t) => map.set(t.table_id, t.alias));
    return map;
  }, [tables]);

  const live = relationships.filter((r) => r.status !== "rejected");
  const rejected = relationships.filter((r) => r.status === "rejected");

  return (
    <section className="pane">
      <div className="pane-head">
        <h2>Relationships</h2>
        <div className="sub">
          {relationships.length
            ? `${live.length} join${live.length === 1 ? "" : "s"} the system will use · statistics first, model only as a tiebreak`
            : "How your files connect to each other"}
        </div>
      </div>

      <div className="pane-body">
        {!tables.length && <div className="empty">Upload files to see how they relate.</div>}

        {tables.length > 0 && !relationships.length && (
          <div className="empty">
            No relationships found between these files. You can add one by hand below.
          </div>
        )}

        {live.map((rel) => (
          <RelationshipRow
            key={rel.id}
            rel={rel}
            aliasOf={aliasOf}
            onSetStatus={onSetStatus}
            busy={busy}
          />
        ))}

        {components.length > 1 && (
          <div className="disconnected">
            <h3>These files are not related to each other</h3>
            <p>
              A question spanning two groups cannot be answered — the system will ask
              instead of guessing a join.
            </p>
            {components.map((group, i) => (
              <div key={i} style={{ marginBottom: 4 }}>
                {group.map((id) => (
                  <span className="group-chip" key={id}>
                    {aliasOf.get(id) ?? id}
                  </span>
                ))}
              </div>
            ))}
          </div>
        )}

        {tables.length >= 2 && <AddRelationship tables={tables} onAdd={onAdd} busy={busy} />}

        {rejected.length > 0 && (
          <details className="disclosure" style={{ marginTop: 14 }}>
            <summary>{rejected.length} rejected</summary>
            <div className="disclosure-body">
              {rejected.map((rel) => (
                <RelationshipRow
                  key={rel.id}
                  rel={rel}
                  aliasOf={aliasOf}
                  onSetStatus={onSetStatus}
                  busy={busy}
                />
              ))}
            </div>
          </details>
        )}
      </div>
    </section>
  );
}

function RelationshipRow({
  rel,
  aliasOf,
  onSetStatus,
  busy,
}: {
  rel: Relationship;
  aliasOf: Map<string, string>;
  onSetStatus: (id: string, status: RelationshipStatus) => void;
  busy: boolean;
}) {
  const left = aliasOf.get(rel.left_table) ?? rel.left_table;
  const right = aliasOf.get(rel.right_table) ?? rel.right_table;
  const predicate = rel.left_columns
    .map((lc, i) => `${left}.${lc} = ${right}.${rel.right_columns[i]}`)
    .join("  AND  ");

  return (
    <div className={`rel ${rel.status}`}>
      <div className="predicate">{predicate}</div>
      <div className="evidence">{rel.evidence}</div>
      <div className="meta">
        <span className={`badge kind-${rel.kind}`}>{rel.kind}</span>
        <span className={`badge status-${rel.status}`}>{rel.status}</span>
        <span className="badge">{Math.round(rel.containment * 100)}% overlap</span>
        <span className="actions">
          {rel.status !== "confirmed" && (
            <button disabled={busy} onClick={() => onSetStatus(rel.id, "confirmed")}>
              Confirm
            </button>
          )}
          {rel.status !== "rejected" ? (
            <button disabled={busy} onClick={() => onSetStatus(rel.id, "rejected")}>
              Reject
            </button>
          ) : (
            <button disabled={busy} onClick={() => onSetStatus(rel.id, "proposed")}>
              Restore
            </button>
          )}
        </span>
      </div>
    </div>
  );
}

function AddRelationship({
  tables,
  onAdd,
  busy,
}: {
  tables: TableProfile[];
  onAdd: Props["onAdd"];
  busy: boolean;
}) {
  const [leftTable, setLeftTable] = useState(tables[0]?.table_id ?? "");
  const [leftColumn, setLeftColumn] = useState("");
  const [rightTable, setRightTable] = useState(tables[1]?.table_id ?? "");
  const [rightColumn, setRightColumn] = useState("");

  const columnsOf = (id: string) => tables.find((t) => t.table_id === id)?.columns ?? [];
  const ready = leftTable && leftColumn && rightTable && rightColumn && leftTable !== rightTable;

  return (
    <div className="add-rel">
      <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 8 }}>
        + Add a relationship the statistics missed
      </div>

      <div className="row">
        <select
          value={leftTable}
          onChange={(e) => {
            setLeftTable(e.target.value);
            setLeftColumn("");
          }}
        >
          {tables.map((t) => (
            <option key={t.table_id} value={t.table_id}>
              {t.alias}
            </option>
          ))}
        </select>
        <select value={leftColumn} onChange={(e) => setLeftColumn(e.target.value)}>
          <option value="">column…</option>
          {columnsOf(leftTable).map((c) => (
            <option key={c.name} value={c.name}>
              {c.name}
            </option>
          ))}
        </select>
      </div>

      <div style={{ textAlign: "center", color: "var(--faint)", fontSize: 12, margin: "2px 0" }}>
        =
      </div>

      <div className="row">
        <select
          value={rightTable}
          onChange={(e) => {
            setRightTable(e.target.value);
            setRightColumn("");
          }}
        >
          {tables.map((t) => (
            <option key={t.table_id} value={t.table_id}>
              {t.alias}
            </option>
          ))}
        </select>
        <select value={rightColumn} onChange={(e) => setRightColumn(e.target.value)}>
          <option value="">column…</option>
          {columnsOf(rightTable).map((c) => (
            <option key={c.name} value={c.name}>
              {c.name}
            </option>
          ))}
        </select>
      </div>

      <button
        disabled={!ready || busy}
        style={{ marginTop: 6 }}
        onClick={() => {
          onAdd({
            left_table: leftTable,
            left_columns: [leftColumn],
            right_table: rightTable,
            right_columns: [rightColumn],
          });
          setLeftColumn("");
          setRightColumn("");
        }}
      >
        Add relationship
      </button>
    </div>
  );
}
