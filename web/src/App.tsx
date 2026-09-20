import { useCallback, useEffect, useRef, useState } from "react";
import {
  ApiError,
  api,
  askQuestion,
  type Relationship,
  type RelationshipStatus,
  type SampleSet,
  type SchemaResponse,
} from "./api";
import ChatPane, { type Exchange } from "./components/ChatPane";
import RelationshipEditor from "./components/RelationshipEditor";
import UploadPane from "./components/UploadPane";

const EMPTY: SchemaResponse = { tables: [], relationships: [], components: [] };

export default function App() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [schema, setSchema] = useState<SchemaResponse>(EMPTY);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [asking, setAsking] = useState(false);
  const [exchange, setExchange] = useState<Exchange | null>(null);
  const [samples, setSamples] = useState<SampleSet[]>([]);
  const [fatal, setFatal] = useState<string | null>(null);
  const abortRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    api
      .createSession()
      .then((r) => setSessionId(r.session_id))
      .catch((e: ApiError) => setFatal(e.message));
    api.listSamples().then(setSamples).catch(() => setSamples([]));
  }, []);

  const refreshSchema = useCallback(async (id: string) => {
    setSchema(await api.getSchema(id));
  }, []);

  // The server evicts idle sessions to bound its memory, so a tab left open overnight
  // comes back to a session that no longer exists. That is entirely recoverable -- a
  // new one is a single request -- and a dead screen demanding a reload is the wrong
  // answer to it. The uploaded data is genuinely gone, so the notice says so rather
  // than pretending nothing happened.
  const handleError = useCallback(async (error: unknown) => {
    if (error instanceof ApiError && error.code === "SESSION_NOT_FOUND") {
      try {
        const fresh = await api.createSession();
        setSchema(EMPTY);
        setExchange(null);
        setSessionId(fresh.session_id);
        setWarnings(["That session expired, so a new one was started. Add your files again."]);
        return;
      } catch {
        // Could not even start a fresh one; fall through and say so plainly.
      }
    }
    setFatal(error instanceof ApiError ? error.message : String(error));
  }, []);

  const handleFiles = useCallback(
    async (files: File[]) => {
      if (!sessionId) return;
      setBusy(true);
      setFatal(null);
      try {
        const response = await api.uploadFiles(sessionId, files);
        setWarnings(response.warnings);
        await refreshSchema(sessionId);
      } catch (error) {
        await handleError(error);
      } finally {
        setBusy(false);
      }
    },
    [sessionId, refreshSchema, handleError],
  );

  const handleSample = useCallback(
    async (name: string) => {
      if (!sessionId) return;
      setBusy(true);
      setFatal(null);
      try {
        const response = await api.loadSample(sessionId, name);
        setWarnings(response.warnings);
        await refreshSchema(sessionId);
      } catch (error) {
        await handleError(error);
      } finally {
        setBusy(false);
      }
    },
    [sessionId, refreshSchema, handleError],
  );

  // Relationship edits update optimistically, then reconcile with the server, so
  // confirm/reject feels instant on a list the user is scanning.
  const patchLocal = (updated: Relationship) =>
    setSchema((prev) => ({
      ...prev,
      relationships: prev.relationships.map((r) => (r.id === updated.id ? updated : r)),
    }));

  const handleSetStatus = useCallback(
    async (id: string, status: RelationshipStatus) => {
      if (!sessionId) return;
      const before = schema.relationships.find((r) => r.id === id);
      if (before) patchLocal({ ...before, status });
      try {
        patchLocal(await api.setRelationshipStatus(sessionId, id, status));
        await refreshSchema(sessionId); // components change when an edge flips
      } catch (error) {
        if (before) patchLocal(before);
        await handleError(error);
      }
    },
    [sessionId, schema.relationships, refreshSchema, handleError],
  );

  const handleAdd = useCallback(
    async (body: Parameters<typeof api.addRelationship>[1]) => {
      if (!sessionId) return;
      try {
        await api.addRelationship(sessionId, body);
        await refreshSchema(sessionId);
      } catch (error) {
        await handleError(error);
      }
    },
    [sessionId, refreshSchema, handleError],
  );

  const handleAsk = useCallback(
    (question: string) => {
      if (!sessionId) return;
      abortRef.current?.();
      setAsking(true);
      setExchange({ question, trace: [], result: null, error: null });

      abortRef.current = askQuestion(sessionId, question, {
        onTrace: (event) =>
          setExchange((prev) => (prev ? { ...prev, trace: [...prev.trace, event] } : prev)),
        onResult: (result) => {
          setExchange((prev) => (prev ? { ...prev, result } : prev));
          setAsking(false);
        },
        onError: (message) => {
          setExchange((prev) => (prev ? { ...prev, error: message } : prev));
          setAsking(false);
        },
      });
    },
    [sessionId],
  );

  return (
    <div className="app">
      <header className="topbar">
        <h1>Darwinbox Assignment</h1>
        <span className="tagline">
          Ask questions across messy spreadsheets — with the joins shown, not assumed
        </span>
        <span className="spacer" />
        {fatal && <span style={{ color: "var(--bad)", fontSize: 12 }}>{fatal}</span>}
        {!sessionId && !fatal && <span className="spinner" />}
      </header>

      <div className="panes">
        <UploadPane
          tables={schema.tables}
          warnings={warnings}
          busy={busy}
          samples={samples}
          onFiles={handleFiles}
          onSample={handleSample}
        />
        <RelationshipEditor
          tables={schema.tables}
          relationships={schema.relationships}
          components={schema.components}
          onSetStatus={handleSetStatus}
          onAdd={handleAdd}
          busy={busy}
        />
        <ChatPane
          ready={Boolean(sessionId && schema.tables.length)}
          asking={asking}
          exchange={exchange}
          onAsk={handleAsk}
        />
      </div>
    </div>
  );
}
