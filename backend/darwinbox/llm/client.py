"""The only place in the codebase that talks to a model.

Everything else depends on the ``LLMClient`` protocol, so every module stays
importable and testable without Ollama and without a GPU.
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Protocol, runtime_checkable

import httpx

DEFAULT_MODEL = "qwen2.5-coder:7b-instruct-q4_K_M"
DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT = 120.0
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
# The free tier allows roughly ten requests a minute, and one question costs two (a
# plan and a summary). Without backoff a handful of questions in quick succession --
# an eval run, or two people using the demo at once -- turns into a wall of failures.
# Measured against the free tier: every question the model reached it answered
# correctly, but an eval run asking twenty in a row exhausts the quota and twelve come
# back as refusals. Interactive use -- a person asking a question every half minute --
# sits well inside the limit. Bulk runs raise the budget rather than the tier.
RATE_LIMIT_RETRIES = int(os.environ.get("DARWINBOX_RATE_LIMIT_RETRIES", "5"))
RATE_LIMIT_BASE_DELAY = 4.0
RATE_LIMIT_MAX_DELAY = float(os.environ.get("DARWINBOX_RATE_LIMIT_MAX_DELAY", "45"))
DEFAULT_LOCATION = "us-central1"


class LLMError(Exception):
    """Raised when the model is unreachable or returns unusable output."""


@runtime_checkable
class LLMClient(Protocol):
    def complete_json(self, system: str, user: str, max_tokens: int = 700) -> dict: ...


class OllamaClient:
    """Ollama /api/chat with JSON-mode decoding.

    Options are pinned for reproducibility: temperature 0 and a fixed seed, because
    the eval harness reports spread across repeated runs and drift there must come
    from the model, not from our sampling settings.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout: float = DEFAULT_TIMEOUT,
        num_ctx: int = 8192,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.num_ctx = num_ctx

    def complete_json(self, system: str, user: str, max_tokens: int = 700) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "seed": 42,
                "num_ctx": self.num_ctx,
                "num_predict": max_tokens,
            },
        }

        try:
            response = httpx.post(
                f"{self.host}/api/chat", json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPError as exc:
            raise LLMError(f"could not reach Ollama at {self.host}: {exc}") from exc

        content = (body.get("message") or {}).get("content", "")
        return _parse_json(content)


def _parse_json(content: str) -> dict:
    """Decode JSON-mode output, tolerating a stray code fence around it."""
    text = content.strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text.lstrip("`")
        text = text.removeprefix("json").strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise LLMError(f"model did not return JSON: {content[:200]!r}") from exc
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as inner:
            raise LLMError(f"model did not return JSON: {content[:200]!r}") from inner

    if not isinstance(parsed, dict):
        raise LLMError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


class FakeLLMClient:
    """Canned replies so the API and planner suites run without a GPU.

    Replies are matched by substring against the user prompt, in insertion order.
    """

    def __init__(self, replies: dict[str, dict] | None = None, default: dict | None = None) -> None:
        self.replies = replies or {}
        self.default = default if default is not None else {}
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system: str, user: str, max_tokens: int = 700) -> dict:
        self.calls.append((system, user))
        for needle, reply in self.replies.items():
            if needle.lower() in user.lower():
                return dict(reply)
        return dict(self.default)


class GeminiClient:
    """Gemini, for the deployment that has no GPU behind it.

    The local Ollama path is the design: a 7B on 8 GB of consumer VRAM, nothing
    leaving the machine. But a hosted demo cannot depend on a laptop being awake, and
    no serverless runtime ships an RTX 5060, so the same protocol is spoken to Gemini
    instead. Nothing above this class changes -- the planner, the validator and the
    computed-facts guard are all model-agnostic, which is the point of the protocol.

    Two ways in, because the deployment and the desk want different things:

    * an API key, which works anywhere and is what a reviewer can reproduce with;
    * Vertex, where Cloud Run's own service account is the credential and there is no
      key to leak into an environment variable.

    Thinking is switched off explicitly. Flash spends its budget on hidden reasoning
    tokens before it writes anything, so a 700-token cap that is ample for the answer
    can be consumed entirely by the thinking and return an empty candidate -- a
    failure that looks like a model outage rather than a misconfiguration.
    """

    def __init__(
        self,
        model: str = DEFAULT_GEMINI_MODEL,
        api_key: str | None = None,
        project: str | None = None,
        location: str = DEFAULT_LOCATION,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        if not api_key and not project:
            raise LLMError("GeminiClient needs either an API key or a Vertex project")
        self.model = model
        self.api_key = api_key
        self.project = project
        self.location = location
        self.timeout = timeout

    def complete_json(self, system: str, user: str, max_tokens: int = 700) -> dict:
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }

        for attempt in range(RATE_LIMIT_RETRIES + 1):
            try:
                response = httpx.post(
                    self._endpoint(),
                    json=payload,
                    headers=self._headers(),
                    timeout=self.timeout,
                )
            except httpx.HTTPError as exc:
                raise LLMError(f"could not reach Gemini: {exc}") from exc

            # 429 is the expected steady state on a free tier, not an error condition,
            # so it is waited out rather than reported. 503 is Google shedding load and
            # behaves the same way.
            if response.status_code in (429, 503):
                # A per-minute throttle is worth waiting out; a per-day quota is not.
                # Retrying the latter makes someone watch a spinner for four minutes
                # to reach a failure that was certain at the first response.
                if _is_daily_quota(response):
                    raise LLMError(
                        "the free-tier daily quota for this model is exhausted; "
                        "it resets at midnight Pacific"
                    )
                if attempt < RATE_LIMIT_RETRIES:
                    time.sleep(_retry_delay(response, attempt))
                    continue
                raise LLMError(
                    f"Gemini is rate-limiting: still {response.status_code} after "
                    f"{RATE_LIMIT_RETRIES} retries"
                )

            try:
                response.raise_for_status()
                body = response.json()
            except (httpx.HTTPError, ValueError) as exc:
                raise LLMError(f"Gemini returned {response.status_code}: {exc}") from exc

            return _parse_json(_first_candidate(body))

        raise LLMError("Gemini retry loop exited without a response")  # unreachable

    def _endpoint(self) -> str:
        if self.api_key:
            return (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{self.model}:generateContent"
            )
        return (
            f"https://{self.location}-aiplatform.googleapis.com/v1/projects/"
            f"{self.project}/locations/{self.location}/publishers/google/models/"
            f"{self.model}:generateContent"
        )

    def _headers(self) -> dict[str, str]:
        if self.api_key:
            return {"x-goog-api-key": self.api_key}
        return {"Authorization": f"Bearer {_vertex_token()}"}


def _is_daily_quota(response) -> bool:
    """True when the 429 is the daily allowance rather than the per-minute rate.

    Gemini reports both through the same status code; the QuotaFailure violation names
    which one, and they want opposite handling.
    """
    try:
        for detail in response.json().get("error", {}).get("details", []):
            if not str(detail.get("@type", "")).endswith("QuotaFailure"):
                continue
            for violation in detail.get("violations", []):
                haystack = f"{violation.get('quotaId', '')} {violation.get('quotaMetric', '')}"
                if "perday" in haystack.lower().replace("_", "").replace("-", ""):
                    return True
    except Exception:  # noqa: BLE001 - an unreadable body is not a daily quota
        return False
    return False


def _retry_delay(response, attempt: int) -> float:
    """How long to wait before retrying, preferring what the server asked for.

    Gemini puts a RetryInfo in the error body and sometimes a Retry-After header.
    Either beats guessing. The fallback is exponential with jitter, because several
    requests rejected at the same moment must not all come back at the same moment.
    """
    header = response.headers.get("retry-after")
    if header:
        try:
            return min(float(header), RATE_LIMIT_MAX_DELAY)
        except ValueError:
            pass

    try:
        for detail in response.json().get("error", {}).get("details", []):
            delay = detail.get("retryDelay")
            if isinstance(delay, str) and delay.endswith("s"):
                return min(float(delay[:-1]), RATE_LIMIT_MAX_DELAY)
    except Exception:  # noqa: BLE001 - a malformed error body must not mask the 429
        pass

    backoff = min(RATE_LIMIT_BASE_DELAY * (2**attempt), RATE_LIMIT_MAX_DELAY)
    return backoff * (0.5 + random.random() / 2)


def unavailable_message() -> str:
    """What to tell the user when the model cannot be reached, for this backend.

    "Is Ollama running?" is unhelpful and wrong when the deployment is talking to
    Gemini, which is what the hosted one does.
    """
    if os.environ.get("DARWINBOX_LLM", "").lower() == "gemini":
        return (
            "The model is not responding. The hosted demo runs on a free tier with a "
            "daily cap, which may be spent — it resets at midnight Pacific. Running it "
            "locally with Ollama has no such limit."
        )
    return "The local model is not responding. Is Ollama running?"


def _first_candidate(body: dict) -> str:
    """Pull the text out, and say why it is missing when it is.

    A blocked prompt and an exhausted token budget both arrive as a candidate with no
    parts. Reporting finishReason turns a bare "model did not return JSON" into
    something that names the actual cause.
    """
    candidates = body.get("candidates") or []
    if not candidates:
        raise LLMError(f"Gemini returned no candidates: {str(body)[:200]}")

    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts)
    if not text.strip():
        reason = candidates[0].get("finishReason")
        raise LLMError(f"Gemini returned no text (finishReason={reason})")
    return text


def _vertex_token() -> str:
    """An access token from the ambient credentials.

    On Cloud Run this is the metadata server and the service account attached to the
    revision; on a workstation it is whatever gcloud wrote. Imported here rather than
    at module scope so the API-key path, and every test, runs without google-auth
    installed at all.
    """
    try:
        import google.auth
        import google.auth.transport.requests
    except ImportError as exc:  # pragma: no cover - only hit without the extra
        raise LLMError("Vertex mode needs google-auth: pip install '.[cloud]'") from exc

    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(google.auth.transport.requests.Request())
    return credentials.token


def client_from_env() -> LLMClient:
    """Build the configured client.

    Ollama is the default because the local path is the design. DARWINBOX_LLM=fake
    keeps the API usable with no GPU at all, and =gemini is what the hosted deployment
    runs, where there is no GPU to have.
    """
    backend = os.environ.get("DARWINBOX_LLM", "").lower()

    if backend == "fake":
        return FakeLLMClient()

    if backend == "gemini":
        return GeminiClient(
            model=os.environ.get("DARWINBOX_MODEL", DEFAULT_GEMINI_MODEL),
            api_key=os.environ.get("GEMINI_API_KEY") or None,
            project=os.environ.get("VERTEX_PROJECT") or None,
            location=os.environ.get("VERTEX_LOCATION", DEFAULT_LOCATION),
        )

    return OllamaClient(
        model=os.environ.get("DARWINBOX_MODEL", DEFAULT_MODEL),
        host=os.environ.get("OLLAMA_HOST", DEFAULT_HOST),
    )
