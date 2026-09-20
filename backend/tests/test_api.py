"""Phase 5 acceptance: every endpoint in SPEC 9, exercised with FakeLLMClient.

REST discipline is being graded, so these assert status codes and the error shape
as hard as they assert the payloads.
"""

from __future__ import annotations

import json

import duckdb
import pytest
from conftest import CUSTOMERS, HEADCOUNT, ORDERS
from darwinbox.api.app import create_app
from darwinbox.llm.client import FakeLLMClient
from fastapi.testclient import TestClient

PLAN = {
    "reasoning": "Join orders to customers and sum by region.",
    "tables_used": ["orders", "customers"],
    "answer_sql": (
        "SELECT c.region, round(sum(o.amount), 2) AS total "
        "FROM orders o JOIN customers c ON o.cust_ref = c.code "
        "GROUP BY c.region ORDER BY c.region"
    ),
    "evidence_sql": "SELECT order_id, amount FROM orders LIMIT 100",
    "chart_intent": "comparison",
    "followups": ["By month?", "Top customer?", "Average order?"],
    "clarification_needed": None,
}


@pytest.fixture
def client():
    app = create_app(client_factory=lambda: FakeLLMClient(default=PLAN))
    with TestClient(app) as test_client:
        yield test_client


def new_session(client) -> str:
    response = client.post("/api/sessions")
    assert response.status_code == 201
    return response.json()["session_id"]


def upload(client, session_id: str, files: dict[str, str]):
    return client.post(
        f"/api/sessions/{session_id}/files",
        files=[("files", (name, text.encode(), "text/csv")) for name, text in files.items()],
    )


@pytest.fixture
def loaded(client):
    """A session with customers + orders + an unrelated headcount file."""
    session_id = new_session(client)
    response = upload(
        client,
        session_id,
        {"customers.csv": CUSTOMERS, "orders.csv": ORDERS, "headcount.csv": HEADCOUNT},
    )
    assert response.status_code == 201
    return session_id


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #


def test_create_session_returns_201_and_an_id(client):
    response = client.post("/api/sessions")
    assert response.status_code == 201
    assert response.json()["session_id"]


def test_delete_session_returns_204_then_404(client):
    session_id = new_session(client)

    assert client.delete(f"/api/sessions/{session_id}").status_code == 204
    assert client.get(f"/api/sessions/{session_id}/schema").status_code == 404


def test_unknown_session_returns_the_error_shape(client):
    response = client.get("/api/sessions/does-not-exist/schema")

    assert response.status_code == 404
    body = response.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message", "detail"}
    assert body["error"]["code"] == "SESSION_NOT_FOUND"
    assert "Traceback" not in json.dumps(body)


# --------------------------------------------------------------------------- #
# Upload
# --------------------------------------------------------------------------- #


def test_multi_file_upload_returns_201_and_profiles(client):
    session_id = new_session(client)
    response = upload(client, session_id, {"customers.csv": CUSTOMERS, "orders.csv": ORDERS})

    assert response.status_code == 201
    body = response.json()
    assert {t["alias"] for t in body["tables"]} == {"customers", "orders"}
    assert body["warnings"] == []

    orders = next(t for t in body["tables"] if t["alias"] == "orders")
    assert orders["n_rows"] == 6
    assert {c["name"] for c in orders["columns"]} == {
        "order_id", "cust_ref", "order_date", "amount"
    }


def test_upload_accumulates_across_calls(client):
    session_id = new_session(client)
    upload(client, session_id, {"customers.csv": CUSTOMERS})
    upload(client, session_id, {"orders.csv": ORDERS})

    schema = client.get(f"/api/sessions/{session_id}/schema").json()
    assert len(schema["tables"]) == 2
    # The relationship must be found even though the files arrived separately.
    assert len(schema["relationships"]) >= 1


def test_unparseable_file_returns_422(client):
    session_id = new_session(client)
    response = client.post(
        f"/api/sessions/{session_id}/files",
        files=[("files", ("broken.xlsx", b"PK\x03\x04not-a-real-zip", "application/vnd.ms-excel"))],
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "UNPARSEABLE_FILE"


def test_upload_to_unknown_session_returns_404(client):
    response = upload(client, "nope", {"customers.csv": CUSTOMERS})
    assert response.status_code == 404


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


def test_schema_returns_tables_relationships_and_components(client, loaded):
    body = client.get(f"/api/sessions/{loaded}/schema").json()

    assert len(body["tables"]) == 3
    assert len(body["relationships"]) >= 1
    # headcount is unrelated, so it must be its own component.
    assert len(body["components"]) == 2

    edge = body["relationships"][0]
    assert set(edge) >= {
        "id", "left_table", "left_columns", "right_table", "right_columns",
        "kind", "containment", "score", "evidence", "status",
    }
    assert edge["evidence"]


def test_source_location_is_reported_for_each_table(client, loaded):
    body = client.get(f"/api/sessions/{loaded}/schema").json()
    source = body["tables"][0]["source"]
    assert source["filename"].endswith(".csv")
    assert source["data_start_row"] >= 0


# --------------------------------------------------------------------------- #
# Relationship editing
# --------------------------------------------------------------------------- #


def test_patch_relationship_status(client, loaded):
    schema = client.get(f"/api/sessions/{loaded}/schema").json()
    edge_id = schema["relationships"][0]["id"]

    response = client.patch(
        f"/api/sessions/{loaded}/relationships/{edge_id}", json={"status": "rejected"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"

    after = client.get(f"/api/sessions/{loaded}/schema").json()
    assert next(r for r in after["relationships"] if r["id"] == edge_id)["status"] == "rejected"
    # A rejected edge disconnects the graph, which the components must show.
    assert len(after["components"]) == 3


def test_patch_unknown_relationship_returns_404(client, loaded):
    response = client.patch(
        f"/api/sessions/{loaded}/relationships/rel_nope", json={"status": "confirmed"}
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "RELATIONSHIP_NOT_FOUND"


def test_patch_with_an_invalid_status_returns_400(client, loaded):
    schema = client.get(f"/api/sessions/{loaded}/schema").json()
    edge_id = schema["relationships"][0]["id"]

    response = client.patch(
        f"/api/sessions/{loaded}/relationships/{edge_id}", json={"status": "maybe"}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "BAD_REQUEST"


def test_user_added_relationship_is_created_and_confirmed(client, loaded):
    response = client.post(
        f"/api/sessions/{loaded}/relationships",
        json={
            "left_table": "orders",
            "left_columns": ["cust_ref"],
            "right_table": "customers",
            "right_columns": ["code"],
        },
    )

    assert response.status_code == 201
    edge = response.json()
    assert edge["status"] == "confirmed"
    assert "Added by you" in edge["evidence"]


def test_user_added_relationship_with_a_bad_column_returns_404(client, loaded):
    response = client.post(
        f"/api/sessions/{loaded}/relationships",
        json={
            "left_table": "orders",
            "left_columns": ["not_a_column"],
            "right_table": "customers",
            "right_columns": ["code"],
        },
    )
    assert response.status_code == 404


def test_user_added_relationship_to_itself_returns_400(client, loaded):
    response = client.post(
        f"/api/sessions/{loaded}/relationships",
        json={
            "left_table": "orders",
            "left_columns": ["cust_ref"],
            "right_table": "orders",
            "right_columns": ["cust_ref"],
        },
    )
    assert response.status_code == 400


# --------------------------------------------------------------------------- #
# Queries (SSE)
# --------------------------------------------------------------------------- #


def read_sse(response) -> list[tuple[str, dict]]:
    """Parse an SSE body into (event, payload) pairs."""
    events: list[tuple[str, dict]] = []
    name = None
    for line in response.text.splitlines():
        if line.startswith("event:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("data:") and name:
            events.append((name, json.loads(line.split(":", 1)[1].strip())))
    return events


def test_query_streams_trace_then_a_final_result(client, loaded):
    response = client.post(
        f"/api/sessions/{loaded}/queries", json={"question": "total revenue by region"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = read_sse(response)
    kinds = [name for name, _ in events]
    assert kinds.count("result") == 1
    assert kinds[-1] == "result", "the final event must be the QueryResult"
    assert "trace" in kinds

    _, result = events[-1]
    assert result["clarification"] is None
    assert {r["region"] for r in result["answer_rows"]} == {"East", "North", "South"}
    assert result["chart"]["type"] == "bar"
    assert len(result["evidence_rows"]) == 6
    assert result["trace"]


def test_trace_events_name_the_stages(client, loaded):
    response = client.post(
        f"/api/sessions/{loaded}/queries", json={"question": "total revenue by region"}
    )
    stages = [payload["stage"] for name, payload in read_sse(response) if name == "trace"]
    assert stages[0] == "route"
    assert "plan" in stages and "validate" in stages and "execute" in stages


def test_query_refuses_rather_than_inventing_a_join(client, loaded):
    app = create_app(
        client_factory=lambda: FakeLLMClient(
            default={**PLAN, "answer_sql": "", "clarification_needed": "Which file has salaries?"}
        )
    )
    with TestClient(app) as isolated:
        session_id = new_session(isolated)
        upload(isolated, session_id, {"customers.csv": CUSTOMERS, "headcount.csv": HEADCOUNT})

        response = isolated.post(
            f"/api/sessions/{session_id}/queries",
            json={"question": "compare salaries to revenue"},
        )
        _, result = read_sse(response)[-1]

    assert result["clarification"] == "Which file has salaries?"
    assert result["answer_rows"] == []


def test_empty_question_returns_400(client, loaded):
    response = client.post(f"/api/sessions/{loaded}/queries", json={"question": "   "})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "BAD_REQUEST"


def test_missing_question_field_returns_400(client, loaded):
    response = client.post(f"/api/sessions/{loaded}/queries", json={})
    assert response.status_code == 400


def test_query_on_unknown_session_returns_404(client):
    response = client.post("/api/sessions/nope/queries", json={"question": "hi"})
    assert response.status_code == 404


# --------------------------------------------------------------------------- #
# Meta
# --------------------------------------------------------------------------- #


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_method_not_allowed_keeps_the_error_shape(client):
    response = client.put("/api/sessions")
    assert response.status_code == 405
    assert "error" in response.json()


# --------------------------------------------------------------------------- #
# Session lifetime
#
# A session holds a DuckDB connection and every table ingested into it. Only an
# explicit DELETE used to free one, and a browser does not send one when its tab
# closes -- so every visitor leaked a session for the life of the process. On the
# single hosted instance that is the service falling over after a few dozen people.
# --------------------------------------------------------------------------- #

from darwinbox import session as session_module  # noqa: E402
from darwinbox.session import SessionNotFoundError, SessionStore  # noqa: E402


def make_store() -> SessionStore:
    return SessionStore(lambda: FakeLLMClient())


def test_an_idle_session_is_reclaimed(monkeypatch):
    monkeypatch.setattr(session_module, "SESSION_TTL_SECONDS", 0.05)
    store = make_store()
    stale = store.create()

    import time

    time.sleep(0.06)
    store.create()  # eviction runs on create

    with pytest.raises(SessionNotFoundError):
        store.get(stale.id)


def test_a_session_still_in_use_is_not_reclaimed(monkeypatch):
    """Eviction is by idleness, so touching it has to be what keeps it alive."""
    monkeypatch.setattr(session_module, "SESSION_TTL_SECONDS", 0.05)
    store = make_store()
    busy = store.create()

    import time

    for _ in range(4):
        time.sleep(0.03)
        store.get(busy.id)  # each get refreshes it
    store.create()

    assert store.get(busy.id) is busy


def test_the_number_of_live_sessions_is_bounded(monkeypatch):
    monkeypatch.setattr(session_module, "MAX_SESSIONS", 5)
    store = make_store()
    made = [store.create() for _ in range(20)]

    assert len(store) <= 5
    # The survivors must be the most recent ones, not an arbitrary five.
    assert made[-1].id in {s for s in store._sessions}
    with pytest.raises(SessionNotFoundError):
        store.get(made[0].id)


def test_an_evicted_session_releases_its_connection(monkeypatch):
    monkeypatch.setattr(session_module, "MAX_SESSIONS", 2)
    store = make_store()
    first = store.create()
    store.create()
    store.create()

    # DuckDB raises once the connection is closed, which is the point: the memory
    # behind it is gone rather than merely unreferenced.
    with pytest.raises(duckdb.Error):
        first.conn.execute("SELECT 1")


def test_eviction_survives_a_connection_that_will_not_close(monkeypatch):
    monkeypatch.setattr(session_module, "MAX_SESSIONS", 2)
    store = make_store()
    broken = store.create()
    monkeypatch.setattr(type(broken), "close", lambda self: 1 / 0)

    store.create()
    store.create()  # must not raise

    assert len(store) <= 2


def test_deleting_a_session_twice_is_a_clean_404():
    store = make_store()
    s = store.create()
    store.delete(s.id)
    with pytest.raises(SessionNotFoundError):
        store.delete(s.id)


def test_a_session_serving_a_request_is_never_evicted(monkeypatch):
    """The ceiling must not close a connection out from under a running request.

    Under load this happened for real: eviction measured idleness from when a request
    started, picked a session whose own ingest was still going, and closed its DuckDB
    connection mid-flight. A valid upload came back as
    'ConnectionException: Connection already closed'.
    """
    monkeypatch.setattr(session_module, "MAX_SESSIONS", 2)
    store = make_store()
    busy = store.create()

    with store.acquire(busy.id):
        for _ in range(10):
            store.create()  # would previously have reaped `busy`
        # Still registered, and its connection still works.
        assert store.get(busy.id) is busy
        assert busy.conn.execute("SELECT 1").fetchone() == (1,)


def test_the_ceiling_gives_way_rather_than_break_live_requests(monkeypatch):
    """If every session is busy there is nothing safe to evict, so the cap yields."""
    monkeypatch.setattr(session_module, "MAX_SESSIONS", 2)
    store = make_store()
    a, b = store.create(), store.create()

    with store.acquire(a.id), store.acquire(b.id):
        store.create()  # must not hang, and must not kill a or b
        assert store.get(a.id) is a
        assert store.get(b.id) is b


def test_a_released_session_becomes_evictable_again(monkeypatch):
    monkeypatch.setattr(session_module, "MAX_SESSIONS", 2)
    store = make_store()
    done = store.create()
    with store.acquire(done.id):
        pass

    for _ in range(5):
        store.create()

    with pytest.raises(SessionNotFoundError):
        store.get(done.id)


def test_acquire_on_an_unknown_session_is_a_404():
    store = make_store()
    with pytest.raises(SessionNotFoundError), store.acquire("nope"):
        pass


def test_the_in_use_count_is_released_even_when_the_request_fails():
    store = make_store()
    s = store.create()
    with pytest.raises(ValueError), store.acquire(s.id):
        raise ValueError("the route blew up")
    assert s.in_use == 0
