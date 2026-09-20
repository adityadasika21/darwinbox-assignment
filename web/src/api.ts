// Typed client for the Darwinbox Assignment API.
//
// The query endpoint streams Server-Sent Events over POST, which EventSource
// cannot do (it is GET-only), so the stream is parsed by hand off fetch().

export type SemanticType = "id" | "numeric" | "date" | "category" | "text" | "boolean";

export interface SourceRef {
  filename: string;
  sheet: string | null;
  block_index: number;
  header_row: number;
  data_start_row: number;
  data_end_row: number;
  col_start: number;
  col_end: number;
}

export interface ColumnProfile {
  name: string;
  semantic_type: SemanticType;
  dtype: string;
  n_distinct: number;
  n_null: number;
  null_rate: number;
  is_unique: boolean;
  min_value: string | null;
  max_value: string | null;
  samples: string[];
  regex_signature: string | null;
  normalized_name_tokens: string[];
}

export interface TableProfile {
  table_id: string;
  alias: string;
  source: SourceRef;
  n_rows: number;
  columns: ColumnProfile[];
}

export type RelationshipKind = "fk" | "composite" | "derived" | "unrelated";
export type RelationshipStatus = "proposed" | "confirmed" | "rejected";

export interface Relationship {
  id: string;
  left_table: string;
  left_columns: string[];
  right_table: string;
  right_columns: string[];
  kind: RelationshipKind;
  containment: number;
  name_similarity: number;
  score: number;
  parent_side: "left" | "right" | null;
  evidence: string;
  status: RelationshipStatus;
  derivation: string | null;
}

export interface SchemaResponse {
  tables: TableProfile[];
  relationships: Relationship[];
  components: string[][];
}

export interface SampleSet {
  name: string;
  title: string;
  description: string;
  files: string[];
}

export interface UploadResponse {
  tables: TableProfile[];
  warnings: string[];
}

export interface TraceEvent {
  stage: string;
  message: string;
  detail: Record<string, unknown> | null;
  ms: number;
}

export interface ChartSpec {
  type: "line" | "bar" | "scatter" | "big_number" | "table";
  x: string | null;
  y: string | null;
  series: string | null;
  title: string;
}

export interface QueryResult {
  summary: string;
  answer_rows: Record<string, unknown>[];
  answer_columns: string[];
  evidence_rows: Record<string, unknown>[];
  sql: string;
  chart: ChartSpec | null;
  followups: string[];
  trace: TraceEvent[];
  clarification: string | null;
}

/**
 * Where the API lives.
 *
 * Empty in development: Vite proxies /api to the local server, so requests stay
 * same-origin. The hosted build sets VITE_API_BASE to the tunnel that fronts the
 * machine actually running DuckDB and the local model -- no cloud runtime has the
 * GPU this depends on, so the backend cannot move.
 */
// Optional chaining because import.meta.env is a Vite construct: the module is also
// imported by node:test, where it does not exist at all.
const API_BASE = (import.meta.env?.VITE_API_BASE ?? "").replace(/\/$/, "");

export function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

/** localtunnel shows an interstitial unless this header is present; Cloudflare does not. */
function withTunnelHeader(init?: RequestInit): RequestInit {
  if (!API_BASE.includes("loca.lt")) return init ?? {};
  return {
    ...init,
    headers: { ...(init?.headers ?? {}), "bypass-tunnel-reminder": "1" },
  };
}

export class ApiError extends Error {
  // Written out rather than as constructor parameter properties, so this module
  // runs under Node's type stripping and api.test.ts can import it directly.
  code: string;
  status: number;

  constructor(code: string, message: string, status: number) {
    super(message);
    this.code = code;
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(apiUrl(path), withTunnelHeader(init));
  if (!response.ok) {
    let code = "HTTP_ERROR";
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      code = body?.error?.code ?? code;
      message = body?.error?.message ?? message;
    } catch {
      // A non-JSON error body is still an error; keep the status-based message.
    }
    throw new ApiError(code, message, response.status);
  }
  if (response.status === 204) return undefined as T;
  return response.json();
}

export const api = {
  createSession: () => request<{ session_id: string }>("/api/sessions", { method: "POST" }),

  uploadFiles: (sessionId: string, files: File[]) => {
    const form = new FormData();
    files.forEach((file) => form.append("files", file));
    return request<UploadResponse>(`/api/sessions/${sessionId}/files`, {
      method: "POST",
      body: form,
    });
  },

  listSamples: () => request<SampleSet[]>("/api/samples"),

  loadSample: (sessionId: string, name: string) =>
    request<UploadResponse>(`/api/sessions/${sessionId}/samples`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    }),

  getSchema: (sessionId: string) => request<SchemaResponse>(`/api/sessions/${sessionId}/schema`),

  setRelationshipStatus: (sessionId: string, id: string, status: RelationshipStatus) =>
    request<Relationship>(`/api/sessions/${sessionId}/relationships/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status }),
    }),

  addRelationship: (
    sessionId: string,
    body: {
      left_table: string;
      left_columns: string[];
      right_table: string;
      right_columns: string[];
    },
  ) =>
    request<Relationship>(`/api/sessions/${sessionId}/relationships`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
};

/**
 * Split an SSE buffer into complete frames plus the incomplete remainder.
 *
 * Per the spec a blank line may be CRLFCRLF, LFLF or CRCR. sse-starlette emits
 * CRLF, and splitting on "\n\n" alone never matches that, so every event was
 * silently buffered forever and the UI never received one.
 */
export function splitFrames(buffer: string): { frames: string[]; rest: string } {
  const parts = buffer.split(/\r\n\r\n|\n\n|\r\r/);
  const rest = parts.pop() ?? "";
  return { frames: parts, rest };
}

/** Parse one SSE frame into its event name and data payload, tolerating CRLF. */
export function parseFrame(frame: string): { name: string; data: string } | null {
  let name = "message";
  const data: string[] = [];
  for (const raw of frame.split("\n")) {
    const line = raw.endsWith("\r") ? raw.slice(0, -1) : raw;
    if (line.startsWith("event:")) name = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trim());
  }
  return data.length ? { name, data: data.join("\n") } : null;
}

/**
 * Ask a question, calling onTrace as each stage lands and onResult at the end.
 * Returns an abort handle so a new question cancels the one in flight.
 */
export function askQuestion(
  sessionId: string,
  question: string,
  handlers: {
    onTrace: (event: TraceEvent) => void;
    onResult: (result: QueryResult) => void;
    onError: (message: string) => void;
  },
): () => void {
  const controller = new AbortController();

  (async () => {
    try {
      const response = await fetch(
        apiUrl(`/api/sessions/${sessionId}/queries`),
        withTunnelHeader({
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question }),
          signal: controller.signal,
        }),
      );

      if (!response.ok || !response.body) {
        const body = await response.json().catch(() => null);
        handlers.onError(body?.error?.message ?? `Request failed (${response.status})`);
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      const dispatch = (frame: string) => {
        const parsed = parseFrame(frame);
        if (!parsed) return;
        const payload = JSON.parse(parsed.data);
        if (parsed.name === "trace") handlers.onTrace(payload as TraceEvent);
        else if (parsed.name === "result") handlers.onResult(payload as QueryResult);
        else if (parsed.name === "error") {
          handlers.onError(payload?.error?.message ?? "Query failed");
        }
      };

      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const { frames, rest } = splitFrames(buffer);
        buffer = rest;
        for (const frame of frames) dispatch(frame);
      }

      // A final frame with no trailing blank line would otherwise be dropped.
      if (buffer.trim()) dispatch(buffer);
    } catch (error) {
      if ((error as Error).name !== "AbortError") {
        handlers.onError((error as Error).message || "Could not reach the server");
      }
    }
  })();

  return () => controller.abort();
}
