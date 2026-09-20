import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ChartSpec, QueryResult } from "../api";

const AXIS = { stroke: "#626b7d", fontSize: 11 };
const GRID = "#262b36";
const SERIES = "#6ea8fe";

const TOOLTIP_STYLE = {
  contentStyle: {
    background: "#1c2029",
    border: "1px solid #262b36",
    borderRadius: 6,
    fontSize: 12,
  },
  labelStyle: { color: "#8b93a3" },
};

export default function ResultView({ result }: { result: QueryResult }) {
  const { answer_rows: rows, answer_columns: columns, chart } = result;

  if (!rows.length) {
    return (
      <div className="empty">{result.summary || "That query returned no rows."}</div>
    );
  }

  const plotted = chart && chart.type !== "table" && chart.type !== "big_number";

  return (
    <>
      {result.summary && <p className="answer-summary">{result.summary}</p>}

      {chart?.type === "big_number" && <BigNumber result={result} />}

      {/* The chart leads. It is the answer at a glance; the table is the backing
          detail. Rendering the table first pushed the chart below the fold on any
          result with more than a handful of rows. */}
      {plotted && (
        <figure className="chart-wrap">
          <Chart spec={chart} rows={rows} />
          <figcaption>
            {chart.y}
            {chart.x ? ` by ${chart.x}` : ""} · {rows.length.toLocaleString()} rows
          </figcaption>
        </figure>
      )}

      {chart?.type !== "big_number" &&
        (plotted ? (
          <details className="disclosure">
            <summary>Data behind this chart · {rows.length.toLocaleString()} rows</summary>
            <div className="disclosure-body">
              <DataTable columns={columns} rows={rows} />
            </div>
          </details>
        ) : (
          <DataTable columns={columns} rows={rows} />
        ))}
    </>
  );
}

function BigNumber({ result }: { result: QueryResult }) {
  const column = result.chart?.y ?? result.answer_columns[0];
  const value = result.answer_rows[0]?.[column];
  return (
    <div className="big-number">
      {format(value)}
      <span className="unit">{column}</span>
    </div>
  );
}

export function DataTable({
  columns,
  rows,
  max = 100,
}: {
  columns: string[];
  rows: Record<string, unknown>[];
  max?: number;
}) {
  const shown = rows.slice(0, max);
  return (
    <>
      <div className="data-table-wrap">
        <table className="data">
          <thead>
            <tr>
              {columns.map((c) => (
                <th key={c}>{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {shown.map((row, i) => (
              <tr key={i}>
                {columns.map((c) => (
                  <td key={c} className={typeof row[c] === "number" ? "num" : ""}>
                    {format(row[c])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {rows.length > shown.length && (
        <div style={{ color: "var(--faint)", fontSize: 12, marginTop: 6 }}>
          showing {shown.length} of {rows.length.toLocaleString()} rows
        </div>
      )}
    </>
  );
}

function Chart({ spec, rows }: { spec: ChartSpec; rows: Record<string, unknown>[] }) {
  const x = spec.x ?? "";
  const y = spec.y ?? "";
  const data = rows.slice(0, 500);

  if (spec.type === "line") {
    return (
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={data} margin={{ top: 4, right: 12, bottom: 4, left: 0 }}>
          <CartesianGrid stroke={GRID} strokeDasharray="2 4" vertical={false} />
          <XAxis dataKey={x} tick={AXIS} tickLine={false} axisLine={{ stroke: GRID }} />
          <YAxis tick={AXIS} tickLine={false} axisLine={false} width={52} />
          <Tooltip {...TOOLTIP_STYLE} />
          <Line type="monotone" dataKey={y} stroke={SERIES} strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    );
  }

  if (spec.type === "scatter") {
    return (
      <ResponsiveContainer width="100%" height={200}>
        <ScatterChart margin={{ top: 4, right: 12, bottom: 4, left: 0 }}>
          <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
          <XAxis dataKey={x} type="number" tick={AXIS} tickLine={false} axisLine={{ stroke: GRID }} />
          <YAxis dataKey={y} type="number" tick={AXIS} tickLine={false} axisLine={false} width={52} />
          <Tooltip {...TOOLTIP_STYLE} />
          <Scatter data={data} fill={SERIES} />
        </ScatterChart>
      </ResponsiveContainer>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data} margin={{ top: 4, right: 12, bottom: 4, left: 0 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="2 4" vertical={false} />
        <XAxis dataKey={x} tick={AXIS} tickLine={false} axisLine={{ stroke: GRID }} />
        <YAxis tick={AXIS} tickLine={false} axisLine={false} width={52} />
        <Tooltip {...TOOLTIP_STYLE} cursor={{ fill: "rgba(110,168,254,0.08)" }} />
        <Bar dataKey={y} fill={SERIES} radius={[3, 3, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

function format(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    return Number.isInteger(value) ? value.toLocaleString() : value.toLocaleString(undefined, {
      maximumFractionDigits: 2,
    });
  }
  return String(value);
}
