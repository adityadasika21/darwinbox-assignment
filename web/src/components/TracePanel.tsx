import type { TraceEvent } from "../api";

/** "How this was answered" — stays visible after the answer lands. */
export default function TracePanel({
  trace,
  live,
  sql,
}: {
  trace: TraceEvent[];
  live: boolean;
  sql?: string;
}) {
  if (!trace.length) return null;

  return (
    <details className="disclosure" open={live}>
      <summary>
        {live && <span className="spinner" style={{ marginRight: 6 }} />}
        How this was answered · {trace.length} step{trace.length === 1 ? "" : "s"}
      </summary>
      <div className="disclosure-body">
        {trace.map((event, i) => (
          <div className="trace-row" key={i}>
            <span className={`trace-stage ${event.stage}`}>{event.stage}</span>
            <span className="trace-msg">{event.message}</span>
            <span className="trace-ms">{event.ms}ms</span>
          </div>
        ))}
        {sql && <div className="sql-block">{sql}</div>}
      </div>
    </details>
  );
}
