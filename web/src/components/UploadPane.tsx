import { useRef, useState } from "react";
import type { SampleSet, TableProfile } from "../api";

interface Props {
  tables: TableProfile[];
  warnings: string[];
  busy: boolean;
  samples: SampleSet[];
  onFiles: (files: File[]) => void;
  onSample: (name: string) => void;
}

/** Left pane: drop files, then see the tables discovered inside each one. */
export default function UploadPane({
  tables,
  warnings,
  busy,
  samples,
  onFiles,
  onSample,
}: Props) {
  const [over, setOver] = useState(false);
  const [open, setOpen] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);

  // Grouping by filename is the point: it shows that one file yielded two tables.
  const byFile = new Map<string, TableProfile[]>();
  for (const table of tables) {
    const list = byFile.get(table.source.filename) ?? [];
    list.push(table);
    byFile.set(table.source.filename, list);
  }

  return (
    <section className="pane">
      <div className="pane-head">
        <h2>Files &amp; Schema</h2>
        <div className="sub">
          {tables.length
            ? `${tables.length} table${tables.length === 1 ? "" : "s"} across ${byFile.size} file${byFile.size === 1 ? "" : "s"}`
            : "CSV or Excel, several at once"}
        </div>
      </div>

      <div className="pane-body">
        <div
          className={`dropzone${over ? " over" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setOver(true);
          }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setOver(false);
            const files = Array.from(e.dataTransfer.files);
            if (files.length) onFiles(files);
          }}
          onClick={() => input.current?.click()}
        >
          {busy ? (
            <>
              <span className="spinner" /> <strong>Reading files…</strong>
              <small>Detecting tables and discovering relationships</small>
            </>
          ) : (
            <>
              <strong>Drop CSV or Excel files here</strong>
              <small>or click to choose — multiple files, one session</small>
            </>
          )}
          <input
            ref={input}
            type="file"
            multiple
            accept=".csv,.tsv,.txt,.xlsx,.xlsm"
            style={{ display: "none" }}
            onChange={(e) => {
              const files = Array.from(e.target.files ?? []);
              if (files.length) onFiles(files);
              e.target.value = "";
            }}
          />
        </div>

        {!tables.length && samples.length > 0 && (
          <div className="samples">
            <div className="samples-title">or start with a bundled dataset</div>
            {samples.map((sample) => (
              <button
                key={sample.name}
                className="sample"
                disabled={busy}
                onClick={() => onSample(sample.name)}
              >
                <span className="sample-name">{sample.title}</span>
                <span className="sample-desc">{sample.description}</span>
              </button>
            ))}
          </div>
        )}

        {warnings.map((warning) => (
          <div className="warnings" key={warning}>
            ⚠ {warning}
          </div>
        ))}

        {[...byFile.entries()].map(([filename, group]) => (
          <div className="file-group" key={filename}>
            <div className="filename">{filename}</div>
            {group.map((table) => (
              <TableCard
                key={table.table_id}
                table={table}
                open={open === table.table_id}
                onToggle={() => setOpen(open === table.table_id ? null : table.table_id)}
              />
            ))}
          </div>
        ))}
      </div>
    </section>
  );
}

function TableCard({
  table,
  open,
  onToggle,
}: {
  table: TableProfile;
  open: boolean;
  onToggle: () => void;
}) {
  const where = table.source.sheet
    ? `${table.source.sheet}, rows ${table.source.data_start_row + 1}–${table.source.data_end_row + 1}`
    : `rows ${table.source.data_start_row + 1}–${table.source.data_end_row + 1}`;

  return (
    <div className="table-card">
      <button onClick={onToggle}>
        <span className="alias">{table.alias}</span>
        <span style={{ color: "var(--muted)", fontSize: 12 }}>
          {table.n_rows.toLocaleString()} rows
        </span>
        <span className="where">{where}</span>
      </button>
      {open && (
        <div className="columns">
          {table.columns.map((column) => (
            <div className="column-row" key={column.name}>
              <span className={`type-tag type-${column.semantic_type}`}>
                {column.semantic_type}
              </span>
              <span className="name">{column.name}</span>
              <span className="samples">{column.samples.slice(0, 3).join(", ")}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
