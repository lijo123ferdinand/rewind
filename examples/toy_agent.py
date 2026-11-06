"""A real (tiny) tool-using agent loop, standard library only, for either provider.

    python examples/toy_agent.py --question "what is 17 + 25?" [--provider anthropic] [--stream]

It reads OPENAI_BASE_URL / ANTHROPIC_BASE_URL like every SDK does, which is the only reason
`rewind record -- python examples/toy_agent.py` works without changing a line here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

TOOLS_OPENAI = [{"type": "function", "function": {
    "name": "add", "description": "Add two integers",
    "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                   "required": ["a", "b"]}}}]
TOOLS_ANTHROPIC = [{"name": "add", "description": "Add two integers",
                    "input_schema": TOOLS_OPENAI[0]["function"]["parameters"]}]


def post(url: str, body: dict, headers: dict) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
        if "text/event-stream" in r.headers.get("Content-Type", ""):
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
            from rewind.sse import reassemble  # the example reuses rewind's SSE parser
            return reassemble("anthropic" if "/messages" in url else "openai", raw)
        return json.loads(raw)


def add(a: int, b: int) -> int:
    return a + b


def run_openai(q: str, stream: bool) -> str:
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    hdr = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"}
    msgs = [{"role": "user", "content": q}]
    for _ in range(6):
        res = post(f"{base}/chat/completions",
                   {"model": "gpt-4o-mini", "messages": msgs, "tools": TOOLS_OPENAI, "stream": stream}, hdr)
        msg = res["choices"][0]["message"]
        msgs.append(msg)
        if not msg.get("tool_calls"):
            return msg.get("content") or ""
        for tc in msg["tool_calls"]:
            out = add(**json.loads(tc["function"]["arguments"]))
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": str(out)})
    return "gave up"


def run_anthropic(q: str, stream: bool) -> str:
    base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
    hdr = {"x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""), "anthropic-version": "2023-06-01"}
    msgs: list[dict] = [{"role": "user", "content": q}]
    for _ in range(6):
        res = post(f"{base}/v1/messages", {"model": "claude-haiku", "max_tokens": 512, "messages": msgs,
                                           "tools": TOOLS_ANTHROPIC, "stream": stream}, hdr)
        msgs.append({"role": "assistant", "content": res["content"]})
        uses = [b for b in res["content"] if b["type"] == "tool_use"]
        if not uses:
            return " ".join(b.get("text", "") for b in res["content"] if b["type"] == "text")
        msgs.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": u["id"], "content": str(add(**u["input"]))} for u in uses]})
    return "gave up"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default="what is 17 + 25?")
    ap.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    ap.add_argument("--stream", action="store_true")
    a = ap.parse_args()
    try:
        answer = (run_anthropic if a.provider == "anthropic" else run_openai)(a.question, a.stream)
    except urllib.error.HTTPError as e:
        print(f"agent error: HTTP {e.code} {e.read().decode()[:200]}", file=sys.stderr)
        sys.exit(1)
    print(f"final: {answer}")
