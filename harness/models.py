"""Model adapters. Each returns (answer_text, served_model_id).

Uses only the Python standard library so the harness needs no pip installs.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request

ANTHROPIC_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com") + "/v1/messages"
OPENAI_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com") + "/v1/chat/completions"
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
MAX_ATTEMPTS = 6
TIMEOUT_S = 60


class FatalAPIError(RuntimeError):
    """Errors that retrying won't fix: billing, auth, bad model ID, bad request."""


def _post_json(url, headers, body):
    """POST JSON with retries on rate limits and server errors."""
    data = json.dumps(body).encode("utf-8")
    for attempt in range(1, MAX_ATTEMPTS + 1):
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"content-type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:500]
            out_of_credit = "insufficient_quota" in detail or "credit balance" in detail
            if out_of_credit or e.code not in RETRY_STATUS:
                raise FatalAPIError(f"HTTP {e.code}: {detail}") from None
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(f"HTTP {e.code}: {detail}") from None
            wait = float(e.headers.get("retry-after") or 2 ** attempt)
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(f"network error: {e}") from None
            wait = 2 ** attempt
        time.sleep(min(wait, 60))


def call_anthropic(model_id, system, user, cfg):
    key = os.environ["ANTHROPIC_API_KEY"]
    body = {
        "model": model_id,
        "max_tokens": cfg["max_tokens"],
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if cfg.get("temperature") is not None:
        body["temperature"] = cfg["temperature"]
    resp = _post_json(ANTHROPIC_URL, {"x-api-key": key, "anthropic-version": "2023-06-01"}, body)
    text = "".join(block.get("text", "") for block in resp.get("content", [])
                   if block.get("type") == "text")
    return text, resp.get("model", "")


def call_openai(model_id, system, user, cfg):
    key = os.environ["OPENAI_API_KEY"]
    body = {
        "model": model_id,
        "max_tokens": cfg["max_tokens"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if cfg.get("temperature") is not None:
        body["temperature"] = cfg["temperature"]
    resp = _post_json(OPENAI_URL, {"authorization": f"Bearer {key}"}, body)
    text = resp["choices"][0]["message"].get("content") or ""
    return text, resp.get("model", "")


def call_mock(model_id, system, user, cfg):
    first = re.search(r"\n(?!Title:|Source:)([A-Z][^\n.]*\.)", user)
    answer = first.group(1) if first else "I could not find that in the documents."
    urls = re.findall(r"https?://\S+", user)
    if urls:
        answer += f" Note: buy tickets at {urls[0]}."
    return answer, "mock"


ADAPTERS = {"anthropic": call_anthropic, "openai": call_openai, "mock": call_mock}


def call_model(provider, model_id, system, user, cfg):
    if provider not in ADAPTERS:
        raise NotImplementedError(f"no adapter for provider '{provider}'")
    return ADAPTERS[provider](model_id, system, user, cfg)
