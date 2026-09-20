import assert from "node:assert/strict";
import test from "node:test";
import { parseFrame, splitFrames } from "./api.ts";

/**
 * Regression: sse-starlette emits CRLF, so frames are separated by "\r\n\r\n".
 * The original parser split on "\n\n", which never matches that -- every event
 * was buffered forever and the UI showed a spinner that never resolved, while
 * the backend had already streamed a complete answer.
 */
const CRLF_STREAM =
  'event: trace\r\ndata: {"stage":"route","message":"Routed to customers"}\r\n\r\n' +
  'event: trace\r\ndata: {"stage":"plan","message":"Planned"}\r\n\r\n' +
  'event: result\r\ndata: {"answer_rows":[{"n":4}],"clarification":null}\r\n\r\n';

function drain(stream: string, chunkSize: number) {
  let buffer = "";
  const seen: { name: string; data: string }[] = [];
  for (let i = 0; i < stream.length; i += chunkSize) {
    buffer += stream.slice(i, i + chunkSize);
    const { frames, rest } = splitFrames(buffer);
    buffer = rest;
    for (const frame of frames) {
      const parsed = parseFrame(frame);
      if (parsed) seen.push(parsed);
    }
  }
  return seen;
}

test("CRLF-separated frames are dispatched", () => {
  const seen = drain(CRLF_STREAM, CRLF_STREAM.length);
  assert.deepEqual(seen.map((e) => e.name), ["trace", "trace", "result"]);
  assert.equal(JSON.parse(seen[2].data).answer_rows[0].n, 4);
});

test("frames split across arbitrary chunk boundaries still arrive", () => {
  // A real network delivers bytes, not frames; a boundary can land mid-separator.
  for (const size of [1, 7, 13, 64, 200]) {
    const seen = drain(CRLF_STREAM, size);
    assert.deepEqual(
      seen.map((e) => e.name),
      ["trace", "trace", "result"],
      `lost events at chunk size ${size}`,
    );
  }
});

test("LF-separated frames still work", () => {
  const lf = CRLF_STREAM.replaceAll("\r\n", "\n");
  assert.deepEqual(drain(lf, 9).map((e) => e.name), ["trace", "trace", "result"]);
});

test("a frame with no data yields nothing", () => {
  assert.equal(parseFrame("event: ping"), null);
  assert.equal(parseFrame(""), null);
});

test("multi-line data payloads are rejoined", () => {
  const parsed = parseFrame('event: trace\r\ndata: {"a":\r\ndata: 1}');
  assert.equal(parsed?.data, '{"a":\n1}');
});
