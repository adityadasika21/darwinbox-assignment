import { useState } from "react";
import type { QueryResult, TraceEvent } from "../api";
import ResultView, { DataTable } from "./ResultView";
import TracePanel from "./TracePanel";

export interface Exchange {
  question: string;
  trace: TraceEvent[];
  result: QueryResult | null;
  error: string | null;
}

interface Props {
  ready: boolean;
  asking: boolean;
  exchange: Exchange | null;
  onAsk: (question: string) => void;
}

const SUGGESTIONS = [
  "What is the total by category?",
  "Show the trend over time",
  "Which rows have no match in the other file?",
];

/** Right pane: ask, watch it think, read the answer and what it rests on. */
export default function ChatPane({ ready, asking, exchange, onAsk }: Props) {
  const [text, setText] = useState("");

  const submit = () => {
    const question = text.trim();
    if (!question || asking) return;
    onAsk(question);
    setText("");
  };

  return (
    <section className="pane">
      <div className="pane-head">
        <h2>Ask</h2>
        <div className="sub">
          {ready
            ? "Plain English, across every file in this session"
            : "Upload a file to start asking"}
        </div>
      </div>

      <div className="ask">
        <textarea
          value={text}
          placeholder={ready ? "e.g. total revenue by region" : "Waiting for files…"}
          disabled={!ready || asking}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
        />
        <button disabled={!ready || asking || !text.trim()} onClick={submit}>
          {asking ? <span className="spinner" /> : "Ask"}
        </button>
      </div>

      <div className="pane-body">
        {!exchange && ready && (
          <div className="empty">
            Ask a question about your files. Every answer comes with the SQL, the rows
            behind it, and the steps taken to get there.
            <div className="chips">
              {SUGGESTIONS.map((s) => (
                <button className="chip" key={s} onClick={() => setText(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {exchange && (
          <div className="answer-block">
            <div className="question-echo">{exchange.question}</div>

            {exchange.error && <div className="error-box">{exchange.error}</div>}

            {exchange.result?.clarification && (
              <div className="clarify">
                <div className="label">I need to ask</div>
                {exchange.result.clarification}
              </div>
            )}

            {exchange.result && !exchange.result.clarification && (
              <ResultView result={exchange.result} />
            )}

            <TracePanel
              trace={exchange.result?.trace.length ? exchange.result.trace : exchange.trace}
              live={asking}
              sql={exchange.result?.sql}
            />

            {exchange.result && exchange.result.evidence_rows.length > 0 && (
              <details className="disclosure">
                <summary>
                  Rows behind this answer · {exchange.result.evidence_rows.length}
                </summary>
                <div className="disclosure-body">
                  <DataTable
                    columns={Object.keys(exchange.result.evidence_rows[0])}
                    rows={exchange.result.evidence_rows}
                  />
                </div>
              </details>
            )}

            {exchange.result && exchange.result.followups.length > 0 && (
              <div className="chips">
                {exchange.result.followups.map((f) => (
                  <button className="chip" key={f} onClick={() => setText(f)}>
                    {f}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
