"""Server-sent events for the two wire formats agents actually use.

Two directions, and both are needed:
  * reassemble(provider, sse_bytes) -> the equivalent non-streamed JSON body. Used to *read* a
    streamed step (summaries, diffs, `fork --edit` opens it in your editor).
  * to_sse(provider, body) -> a valid stream. Used to *serve* an edited or synthetic reply to an
    agent that asked for `stream: true`.

Covered: OpenAI Chat Completions (text + tool calls) and Anthropic Messages (text, tool_use,
thinking). Other endpoints (embeddings, OpenAI Responses API) still record and replay byte for
byte — they just are not summarised or editable yet.
"""

from __future__ import annotations

import json
from typing import Any


def parse_sse(data: bytes) -> list[tuple[str | None, str]]:
    """Split a stream into (event, data) pairs. Tolerates \\r\\n and multi-line data fields."""
    events: list[tuple[str | None, str]] = []
    text = data.decode("utf-8", "replace").replace("\r\n", "\n")
    for block in text.split("\n\n"):
        name: str | None = None
        lines: list[str] = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                lines.append(line[5:].lstrip(" ") if line[5:6] == " " else line[5:])
        if lines:
            events.append((name, "\n".join(lines)))
    return events


# --------------------------------------------------------------------------- OpenAI chat
def _openai_reassemble(data: bytes) -> dict[str, Any]:
    head: dict[str, Any] = {}
    choices: dict[int, dict[str, Any]] = {}
    usage = None
    for _, payload in parse_sse(data):
        if payload.strip() == "[DONE]":
            break
        chunk = json.loads(payload)
        if not head:
            head = {k: chunk.get(k) for k in ("id", "created", "model")}
        if chunk.get("usage"):
            usage = chunk["usage"]
        for c in chunk.get("choices") or []:
            st = choices.setdefault(
                c.get("index", 0), {"content": None, "tools": {}, "finish": None}
            )
            delta = c.get("delta") or {}
            if delta.get("content") is not None:
                st["content"] = (st["content"] or "") + delta["content"]
            for tc in delta.get("tool_calls") or []:
                t = st["tools"].setdefault(
                    tc.get("index", 0), {"id": None, "name": "", "arguments": ""}
                )
                t["id"] = tc.get("id") or t["id"]
                fn = tc.get("function") or {}
                t["name"] += fn.get("name") or ""
                t["arguments"] += fn.get("arguments") or ""
            if c.get("finish_reason"):
                st["finish"] = c["finish_reason"]
    out_choices = []
    for i in sorted(choices):
        st = choices[i]
        msg: dict[str, Any] = {"role": "assistant", "content": st["content"]}
        if st["tools"]:
            msg["tool_calls"] = [
                {"id": t["id"], "type": "function",
                 "function": {"name": t["name"], "arguments": t["arguments"]}}
                for _, t in sorted(st["tools"].items())
            ]
        out_choices.append({"index": i, "message": msg, "finish_reason": st["finish"]})
    body = {**head, "object": "chat.completion", "choices": out_choices}
    if usage:
        body["usage"] = usage
    return body


def _openai_to_sse(body: dict[str, Any]) -> bytes:
    base = {"id": body.get("id", "rewind"), "object": "chat.completion.chunk",
            "created": body.get("created", 0), "model": body.get("model", "")}
    out: list[str] = []

    def emit(index: int, delta: dict[str, Any], finish: str | None = None) -> None:
        out.append("data: " + json.dumps(
            {**base, "choices": [{"index": index, "delta": delta, "finish_reason": finish}]}
        ) + "\n\n")

    for ch in body.get("choices") or []:
        i = ch.get("index", 0)
        msg = ch.get("message") or {}
        emit(i, {"role": "assistant", "content": ""})
        if msg.get("content"):
            emit(i, {"content": msg["content"]})
        for n, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            emit(i, {"tool_calls": [{"index": n, "id": tc.get("id"), "type": "function",
                                     "function": {"name": fn.get("name", ""),
                                                  "arguments": fn.get("arguments", "")}}]})
        emit(i, {}, ch.get("finish_reason") or "stop")
    if body.get("usage"):
        out.append("data: " + json.dumps({**base, "choices": [], "usage": body["usage"]}) + "\n\n")
    out.append("data: [DONE]\n\n")
    return "".join(out).encode()


# ----------------------------------------------------------------------- Anthropic messages
def _anthropic_reassemble(data: bytes) -> dict[str, Any]:
    msg: dict[str, Any] = {}
    blocks: dict[int, dict[str, Any]] = {}
    partial_json: dict[int, str] = {}
    for _, payload in parse_sse(data):
        ev = json.loads(payload)
        kind = ev.get("type")
        if kind == "message_start":
            msg = dict(ev["message"])
        elif kind == "content_block_start":
            blocks[ev["index"]] = dict(ev["content_block"])
        elif kind == "content_block_delta":
            b, d = blocks[ev["index"]], ev["delta"]
            if d["type"] == "text_delta":
                b["text"] = b.get("text", "") + d["text"]
            elif d["type"] == "input_json_delta":
                partial_json[ev["index"]] = partial_json.get(ev["index"], "") + d["partial_json"]
            elif d["type"] == "thinking_delta":
                b["thinking"] = b.get("thinking", "") + d["thinking"]
            elif d["type"] == "signature_delta":
                b["signature"] = d["signature"]
        elif kind == "content_block_stop":
            i = ev["index"]
            if blocks[i].get("type") == "tool_use":
                raw = partial_json.get(i, "")
                blocks[i]["input"] = json.loads(raw) if raw.strip() else {}
        elif kind == "message_delta":
            msg.update(ev.get("delta") or {})
            if ev.get("usage"):
                msg["usage"] = {**(msg.get("usage") or {}), **ev["usage"]}
    msg["content"] = [blocks[i] for i in sorted(blocks)]
    return msg


def _anthropic_to_sse(body: dict[str, Any]) -> bytes:
    out: list[str] = []

    def emit(name: str, payload: dict[str, Any]) -> None:
        out.append(f"event: {name}\ndata: {json.dumps(payload)}\n\n")

    start = {k: v for k, v in body.items() if k != "content"}
    start.update(content=[], stop_reason=None, stop_sequence=None)
    emit("message_start", {"type": "message_start", "message": start})
    for i, b in enumerate(body.get("content") or []):
        t = b.get("type")
        if t == "text":
            emit("content_block_start", {"type": "content_block_start", "index": i,
                                         "content_block": {"type": "text", "text": ""}})
            emit("content_block_delta", {"type": "content_block_delta", "index": i,
                                         "delta": {"type": "text_delta", "text": b.get("text", "")}})
        elif t == "tool_use":
            emit("content_block_start", {"type": "content_block_start", "index": i,
                                         "content_block": {"type": "tool_use", "id": b.get("id"),
                                                           "name": b.get("name"), "input": {}}})
            emit("content_block_delta", {"type": "content_block_delta", "index": i,
                                         "delta": {"type": "input_json_delta",
                                                   "partial_json": json.dumps(b.get("input", {}))}})
        elif t == "thinking":
            emit("content_block_start", {"type": "content_block_start", "index": i,
                                         "content_block": {"type": "thinking", "thinking": ""}})
            emit("content_block_delta", {"type": "content_block_delta", "index": i,
                                         "delta": {"type": "thinking_delta",
                                                   "thinking": b.get("thinking", "")}})
            if b.get("signature"):
                emit("content_block_delta", {"type": "content_block_delta", "index": i,
                                             "delta": {"type": "signature_delta",
                                                       "signature": b["signature"]}})
        else:  # unknown block: send it whole rather than drop it
            emit("content_block_start", {"type": "content_block_start", "index": i,
                                         "content_block": b})
        emit("content_block_stop", {"type": "content_block_stop", "index": i})
    emit("message_delta", {"type": "message_delta",
                           "delta": {"stop_reason": body.get("stop_reason", "end_turn"),
                                     "stop_sequence": body.get("stop_sequence")},
                           "usage": {"output_tokens": (body.get("usage") or {}).get("output_tokens", 0)}})
    emit("message_stop", {"type": "message_stop"})
    return "".join(out).encode()


# ---------------------------------------------------------------------------------- public
def reassemble(provider: str, data: bytes) -> dict[str, Any]:
    return _anthropic_reassemble(data) if provider == "anthropic" else _openai_reassemble(data)


def to_sse(provider: str, body: dict[str, Any]) -> bytes:
    return _anthropic_to_sse(body) if provider == "anthropic" else _openai_to_sse(body)


def synth_text_reply(provider: str, request: dict[str, Any], text: str) -> dict[str, Any]:
    """A minimal, well-formed non-streamed reply that says `text`, in the provider's shape."""
    model = request.get("model", "rewind")
    if provider == "anthropic":
        return {"id": "msg_rewind_override", "type": "message", "role": "assistant",
                "model": model, "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0}}
    return {"id": "chatcmpl-rewind-override", "object": "chat.completion", "created": 0,
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
