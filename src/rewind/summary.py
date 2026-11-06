"""One-line, human-readable summaries of a step's request and response.

ASCII only on purpose: these go to terminals, including Windows consoles on cp1252, where a
stray emoji is a UnicodeEncodeError rather than a pretty glyph.
"""

from __future__ import annotations

import json
from typing import Any

from .sse import reassemble
from .store import Step


def _clip(text: str, n: int = 70) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 3] + "..."


def response_body(step: Step) -> dict[str, Any] | None:
    try:
        if step.streamed:
            return reassemble(step.provider, step.response)
        obj = json.loads(step.response)
        return obj if isinstance(obj, dict) else None
    except (ValueError, KeyError, TypeError):
        return None


def request_body(step: Step) -> dict[str, Any] | None:
    try:
        obj = json.loads(step.request)
    except (ValueError, UnicodeDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if not isinstance(p, dict):
                continue
            if p.get("type") == "text":
                parts.append(p.get("text", ""))
            elif p.get("type") == "tool_result":
                parts.append(f"[tool_result] {_content_text(p.get('content'))}")
            elif p.get("type") == "tool_use":
                parts.append(f"[tool_use {p.get('name')}]")
        return " ".join(parts)
    return ""


def describe_request(step: Step) -> str:
    body = request_body(step)
    if not body:
        return f"{step.method} {step.path}"
    msgs = body.get("messages") or []
    if not msgs:
        return f"{body.get('model', '?')} {step.path}"
    last = msgs[-1]
    role = last.get("role", "?")
    return f"{body.get('model', '?')} [{len(msgs)} msgs] {role}: {_clip(_content_text(last.get('content')), 50)}"


def tool_calls(provider: str, body: dict[str, Any]) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []
    if provider == "anthropic":
        for b in body.get("content") or []:
            if b.get("type") == "tool_use":
                calls.append((b.get("name", "?"), json.dumps(b.get("input", {}))))
    else:
        for ch in body.get("choices") or []:
            for tc in (ch.get("message") or {}).get("tool_calls") or []:
                fn = tc.get("function") or {}
                calls.append((fn.get("name", "?"), fn.get("arguments", "")))
    return calls


def reply_text(provider: str, body: dict[str, Any]) -> str:
    if provider == "anthropic":
        return " ".join(b.get("text", "") for b in body.get("content") or [] if b.get("type") == "text")
    choices = body.get("choices") or []
    return (choices[0].get("message") or {}).get("content") or "" if choices else ""


def describe_response(step: Step) -> str:
    if step.status >= 400:
        try:
            err = json.loads(step.response).get("error")
            msg = err.get("message") if isinstance(err, dict) else err
        except (ValueError, AttributeError):
            msg = step.response[:80].decode("utf-8", "replace")
        return f"HTTP {step.status}: {_clip(msg or '', 60)}"
    body = response_body(step)
    if not body:
        return f"{len(step.response)} bytes"
    calls = tool_calls(step.provider, body)
    if calls:
        return "; ".join(f"tool {n}({_clip(a, 40)})" for n, a in calls)
    return f'"{_clip(reply_text(step.provider, body))}"'
