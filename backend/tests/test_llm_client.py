"""The hosted model path.

The local Ollama client is exercised end to end by the eval harness against a real
GPU. This covers the client that runs where there is no GPU -- the one the deployed
service depends on -- without making a network call.
"""

from __future__ import annotations

import httpx
import pytest
from darwinbox.llm import client as client_module
from darwinbox.llm.client import GeminiClient, LLMError, OllamaClient, client_from_env


class Recorder:
    """Stands in for httpx.post and remembers what it was asked to send."""

    def __init__(self, body: dict, status: int = 200) -> None:
        self.body = body
        self.status = status
        self.url: str = ""
        self.payload: dict = {}
        self.headers: dict = {}

    def __call__(self, url, json=None, headers=None, timeout=None):
        self.url, self.payload, self.headers = url, json or {}, headers or {}
        return httpx.Response(self.status, json=self.body, request=httpx.Request("POST", url))


def reply(text: str) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


@pytest.fixture
def post(monkeypatch):
    def install(body, status=200):
        recorder = Recorder(body, status)
        monkeypatch.setattr(client_module.httpx, "post", recorder)
        return recorder

    return install


# --------------------------------------------------------------------------- #
# Routing and credentials
# --------------------------------------------------------------------------- #


def test_an_api_key_goes_to_the_public_endpoint(post):
    recorder = post(reply('{"sql": "SELECT 1"}'))
    GeminiClient(api_key="k-123").complete_json("sys", "user")

    assert recorder.url.startswith("https://generativelanguage.googleapis.com/")
    assert recorder.headers["x-goog-api-key"] == "k-123"
    assert "Authorization" not in recorder.headers


def test_vertex_mode_uses_the_regional_endpoint_and_a_bearer_token(post, monkeypatch):
    monkeypatch.setattr(client_module, "_vertex_token", lambda: "tok-abc")
    recorder = post(reply("{}"))
    GeminiClient(project="proj", location="europe-west4").complete_json("sys", "user")

    assert "europe-west4-aiplatform.googleapis.com" in recorder.url
    assert "projects/proj/locations/europe-west4" in recorder.url
    assert recorder.headers["Authorization"] == "Bearer tok-abc"


def test_a_client_with_no_credentials_at_all_is_refused_up_front():
    with pytest.raises(LLMError):
        GeminiClient()


# --------------------------------------------------------------------------- #
# The request body
# --------------------------------------------------------------------------- #


def test_the_request_asks_for_deterministic_json(post):
    recorder = post(reply("{}"))
    GeminiClient(api_key="k").complete_json("the system prompt", "the question", max_tokens=321)

    config = recorder.payload["generationConfig"]
    assert config["temperature"] == 0
    assert config["responseMimeType"] == "application/json"
    assert config["maxOutputTokens"] == 321
    assert recorder.payload["systemInstruction"]["parts"][0]["text"] == "the system prompt"
    assert recorder.payload["contents"][0]["parts"][0]["text"] == "the question"


def test_thinking_is_switched_off(post):
    """Left on, hidden reasoning eats the token budget and the answer comes back empty."""
    recorder = post(reply("{}"))
    GeminiClient(api_key="k").complete_json("s", "u", max_tokens=700)

    assert recorder.payload["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 0


# --------------------------------------------------------------------------- #
# Reading the reply
# --------------------------------------------------------------------------- #


def test_the_plan_is_decoded(post):
    post(reply('{"sql": "SELECT 1", "tables_used": ["a"]}'))
    assert GeminiClient(api_key="k").complete_json("s", "u") == {
        "sql": "SELECT 1",
        "tables_used": ["a"],
    }


def test_a_fenced_reply_is_still_decoded(post):
    post(reply('```json\n{"sql": "SELECT 1"}\n```'))
    assert GeminiClient(api_key="k").complete_json("s", "u")["sql"] == "SELECT 1"


def test_no_candidates_is_a_typed_error(post):
    post({"promptFeedback": {"blockReason": "SAFETY"}})
    with pytest.raises(LLMError):
        GeminiClient(api_key="k").complete_json("s", "u")


def test_an_empty_candidate_names_the_reason(post):
    """MAX_TOKENS and a blocked prompt look identical without it."""
    post({"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}]})
    with pytest.raises(LLMError, match="MAX_TOKENS"):
        GeminiClient(api_key="k").complete_json("s", "u")


def test_a_transport_failure_becomes_an_llm_error(post):
    def boom(url, json=None, headers=None, timeout=None):
        raise httpx.ConnectError("no route to host")

    post(reply("{}"))
    import darwinbox.llm.client as mod

    mod.httpx.post = boom
    with pytest.raises(LLMError, match="could not reach Gemini"):
        GeminiClient(api_key="k").complete_json("s", "u")


# --------------------------------------------------------------------------- #
# Selection from the environment
# --------------------------------------------------------------------------- #


def test_ollama_is_the_default(monkeypatch):
    monkeypatch.delenv("DARWINBOX_LLM", raising=False)
    assert isinstance(client_from_env(), OllamaClient)


def test_gemini_is_selected_with_a_key(monkeypatch):
    monkeypatch.setenv("DARWINBOX_LLM", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.delenv("VERTEX_PROJECT", raising=False)
    built = client_from_env()
    assert isinstance(built, GeminiClient)
    assert built.api_key == "k"


def test_gemini_is_selected_with_a_vertex_project(monkeypatch):
    monkeypatch.setenv("DARWINBOX_LLM", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("VERTEX_PROJECT", "proj")
    built = client_from_env()
    assert built.project == "proj" and built.api_key is None


# --------------------------------------------------------------------------- #
# Rate limiting
#
# The free tier allows roughly ten requests a minute and one question costs two.
# Running the eval against Gemini, exactly five questions answered and the next
# fifteen failed -- reported to the user, wrongly, as "Is Ollama running?".
# --------------------------------------------------------------------------- #


class Sequence:
    """Returns the given statuses in order, so a retry can be observed."""

    def __init__(self, statuses, body=None, headers=None):
        self.statuses = list(statuses)
        self.body = body or reply('{"ok": 1}')
        self.headers = headers or {}
        self.calls = 0

    def __call__(self, url, json=None, headers=None, timeout=None):
        status = self.statuses[min(self.calls, len(self.statuses) - 1)]
        self.calls += 1
        payload = self.body if status == 200 else {
            "error": {"details": [{"retryDelay": "0.01s"}]}
        }
        return httpx.Response(
            status, json=payload, headers=self.headers,
            request=httpx.Request("POST", url),
        )


def test_a_rate_limited_call_is_retried_not_failed(monkeypatch):
    monkeypatch.setattr(client_module.time, "sleep", lambda _: None)
    seq = Sequence([429, 429, 200])
    monkeypatch.setattr(client_module.httpx, "post", seq)

    assert GeminiClient(api_key="k").complete_json("s", "u") == {"ok": 1}
    assert seq.calls == 3


def test_service_unavailable_is_also_waited_out(monkeypatch):
    monkeypatch.setattr(client_module.time, "sleep", lambda _: None)
    seq = Sequence([503, 200])
    monkeypatch.setattr(client_module.httpx, "post", seq)

    GeminiClient(api_key="k").complete_json("s", "u")
    assert seq.calls == 2


def test_persistent_rate_limiting_eventually_gives_up(monkeypatch):
    monkeypatch.setattr(client_module.time, "sleep", lambda _: None)
    seq = Sequence([429])
    monkeypatch.setattr(client_module.httpx, "post", seq)

    with pytest.raises(LLMError, match="rate-limiting"):
        GeminiClient(api_key="k").complete_json("s", "u")
    assert seq.calls == client_module.RATE_LIMIT_RETRIES + 1


def test_the_servers_own_retry_delay_is_preferred(monkeypatch):
    slept = []
    monkeypatch.setattr(client_module.time, "sleep", slept.append)
    seq = Sequence([429, 200], headers={"retry-after": "7"})
    monkeypatch.setattr(client_module.httpx, "post", seq)

    GeminiClient(api_key="k").complete_json("s", "u")
    assert slept == [7.0]


def test_a_404_is_not_retried(monkeypatch):
    """Only 429 and 503 are transient; a bad model name must fail immediately."""
    monkeypatch.setattr(client_module.time, "sleep", lambda _: None)
    seq = Sequence([404])
    monkeypatch.setattr(client_module.httpx, "post", seq)

    with pytest.raises(LLMError):
        GeminiClient(api_key="k").complete_json("s", "u")
    assert seq.calls == 1


def test_the_failure_message_names_the_backend_in_use(monkeypatch):
    from darwinbox.llm.client import unavailable_message

    monkeypatch.setenv("DARWINBOX_LLM", "gemini")
    assert "Ollama" not in unavailable_message().split("locally")[0]
    assert "free tier" in unavailable_message()

    monkeypatch.setenv("DARWINBOX_LLM", "ollama")
    assert "Ollama" in unavailable_message()


def quota_body(quota_id: str) -> dict:
    return {
        "error": {
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [{"quotaId": quota_id, "quotaMetric": "generate_content"}],
                },
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "50s"},
            ],
        }
    }


class QuotaResponse:
    def __init__(self, quota_id):
        self.quota_id = quota_id
        self.calls = 0

    def __call__(self, url, json=None, headers=None, timeout=None):
        self.calls += 1
        return httpx.Response(
            429, json=quota_body(self.quota_id), request=httpx.Request("POST", url)
        )


def test_a_daily_quota_fails_immediately(monkeypatch):
    """Retrying a quota that resets tomorrow only makes the user wait to fail."""
    monkeypatch.setattr(client_module.time, "sleep", lambda _: None)
    stub = QuotaResponse("GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    monkeypatch.setattr(client_module.httpx, "post", stub)

    with pytest.raises(LLMError, match="daily quota"):
        GeminiClient(api_key="k").complete_json("s", "u")
    assert stub.calls == 1, "a daily quota must not be retried"


def test_a_per_minute_quota_is_still_retried(monkeypatch):
    monkeypatch.setattr(client_module.time, "sleep", lambda _: None)
    stub = QuotaResponse("GenerateRequestsPerMinutePerProjectPerModel-FreeTier")
    monkeypatch.setattr(client_module.httpx, "post", stub)

    with pytest.raises(LLMError, match="rate-limiting"):
        GeminiClient(api_key="k").complete_json("s", "u")
    assert stub.calls == client_module.RATE_LIMIT_RETRIES + 1


def test_a_429_with_no_quota_detail_is_retried(monkeypatch):
    """Absent evidence it is the daily cap, treat it as transient."""
    monkeypatch.setattr(client_module.time, "sleep", lambda _: None)
    seq = Sequence([429, 200])
    monkeypatch.setattr(client_module.httpx, "post", seq)

    GeminiClient(api_key="k").complete_json("s", "u")
    assert seq.calls == 2


def test_the_hosted_failure_message_explains_the_daily_cap(monkeypatch):
    from darwinbox.llm.client import unavailable_message

    monkeypatch.setenv("DARWINBOX_LLM", "gemini")
    message = unavailable_message()
    assert "daily" in message and "locally" in message
