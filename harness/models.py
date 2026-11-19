"""Model adapters. Each returns (answer_text, served_model_id).

Uses only the Python standard library so the harness needs no pip installs.

Tool use is opt-in. When cfg["tools"] lists a tool name (e.g. "send_email") and
cfg["_tool_impls"] supplies its implementation, the adapter exposes the tool to
the model and runs a short tool-call loop, invoking the implementation and
feeding the result back. The implementations (see app/tools.py) only log calls,
so the return signature stays (text, served_model); the caller reads the log.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request

from app.tools import TOOL_SCHEMAS

ANTHROPIC_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com") + "/v1/messages"
OPENAI_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com") + "/v1/chat/completions"
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
MAX_ATTEMPTS = 6
MAX_TOOL_ROUNDS = 5
TIMEOUT_S = 60


class FatalAPIError(RuntimeError):
    """Errors that retrying won't fix: billing, auth, bad model ID, bad request."""


def _post_json(url, headers, body):
    """POST JSON with retries on rate limits and transient server errors."""
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


def _tools_for(cfg, shape):
    """Reshape the shared schemas for the named tools into one provider's format."""
    return [shape(TOOL_SCHEMAS[name]) for name in (cfg.get("tools") or [])]


def call_anthropic(model_id, system, user, cfg):
    key = os.environ["ANTHROPIC_API_KEY"]
    impls = cfg.get("_tool_impls") or {}
    tools = _tools_for(cfg, lambda s: {"name": s["name"], "description": s["description"],
                                       "input_schema": s["parameters"]})
    messages = [{"role": "user", "content": user}]
    served = ""
    for _ in range(MAX_TOOL_ROUNDS + 1):
        body = {"model": model_id, "max_tokens": cfg["max_tokens"],
                "system": system, "messages": messages}
        if cfg.get("temperature") is not None:
            body["temperature"] = cfg["temperature"]
        if tools:
            body["tools"] = tools
        resp = _post_json(ANTHROPIC_URL, {"x-api-key": key, "anthropic-version": "2023-06-01"}, body)
        served = resp.get("model", "") or served
        content = resp.get("content", [])
        if tools and resp.get("stop_reason") == "tool_use":
            messages.append({"role": "assistant", "content": content})
            results = []
            for block in content:
                if block.get("type") == "tool_use":
                    impl = impls.get(block["name"])
                    out = impl(**block.get("input", {})) if impl else "tool unavailable"
                    results.append({"type": "tool_result", "tool_use_id": block["id"],
                                    "content": str(out)})
            messages.append({"role": "user", "content": results})
            continue
        text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
        return text, served
    return "", served   # tool loop exhausted


def call_openai(model_id, system, user, cfg):
    key = os.environ["OPENAI_API_KEY"]
    impls = cfg.get("_tool_impls") or {}
    tools = _tools_for(cfg, lambda s: {"type": "function", "function": s})
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    served = ""
    for _ in range(MAX_TOOL_ROUNDS + 1):
        body = {"model": model_id, "max_tokens": cfg["max_tokens"], "messages": messages}
        if cfg.get("temperature") is not None:
            body["temperature"] = cfg["temperature"]
        if tools:
            body["tools"] = tools
        resp = _post_json(OPENAI_URL, {"authorization": f"Bearer {key}"}, body)
        served = resp.get("model", "") or served
        msg = resp["choices"][0]["message"]
        if tools and msg.get("tool_calls"):
            messages.append(msg)
            for call in msg["tool_calls"]:
                fn = call["function"]
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    args = {}
                impl = impls.get(fn["name"])
                out = impl(**args) if impl else "tool unavailable"
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": str(out)})
            continue
        return msg.get("content") or "", served
    return "", served


def call_mock(model_id, system, user, cfg):
    """Offline stand-in. With tools on, it exercises the tool loop by emailing
    any address found in the prompt, so the plumbing and the exfiltration check
    can be tested without network access. It carries no injection logic."""
    impls = cfg.get("_tool_impls") or {}
    if impls.get("send_email") and "send_email" in (cfg.get("tools") or []):
        for addr in _EMAIL_RE.findall(user):
            impls["send_email"](to=addr, subject="FYI", body="(mock) forwarded document text")
    first = re.search(r"\n(?!Title:|Source:)([A-Z][^\n.]*\.)", user)
    answer = first.group(1) if first else "I could not find that in the documents."
    urls = re.findall(r"https?://\S+", user)
    if urls:
        answer += f" Note: buy tickets at {urls[0]}."
    return answer, "mock"


_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

ADAPTERS = {"anthropic": call_anthropic, "openai": call_openai, "mock": call_mock}


def call_model(provider, model_id, system, user, cfg):
    if provider not in ADAPTERS:
        raise NotImplementedError(f"no adapter for provider '{provider}'")
    return ADAPTERS[provider](model_id, system, user, cfg)
