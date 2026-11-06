"""A deliberately NON-deterministic fake model server speaking both wire formats.

Every reply carries a fresh random nonce and every tool call a fresh random id. If replay were not
truly serving recordings, two runs could never print the same thing — which is what makes the
replay tests meaningful.

Its SSE is written by hand here (not with rewind.sse), so streaming tests check rewind against an
independent implementation of the formats.
"""

from __future__ import annotations

import json
import re
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _numbers(text: str) -> list[int]:
    return [int(n) for n in re.findall(r"-?\d+", text)][:2] or [1, 1]


class FakeLLM:
    def __init__(self) -> None:
        self.hits = 0
        self.auth_seen: list[str] = []
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a): pass

            def do_POST(self):
                outer.hits += 1
                outer.auth_seen.append(self.headers.get("Authorization") or self.headers.get("x-api-key") or "")
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                anthropic = self.path.endswith("/messages")
                data, ctype = (outer._anthropic if anthropic else outer._openai)(body)
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    # ---------------------------------------------------------------- openai
    def _openai(self, body):
        msgs = body["messages"]
        nonce = secrets.token_hex(3)
        if msgs[-1]["role"] == "tool":
            msg = {"role": "assistant", "content": f"The answer is {msgs[-1]['content']} (nonce {nonce})"}
            finish = "stop"
        else:
            a, b = _numbers(msgs[-1]["content"])
            msg = {"role": "assistant", "content": None, "tool_calls": [{
                "id": f"call_{nonce}", "type": "function",
                "function": {"name": "add", "arguments": json.dumps({"a": a, "b": b})}}]}
            finish = "tool_calls"
        if not body.get("stream"):
            return json.dumps({"id": f"cc-{nonce}", "object": "chat.completion", "created": 1,
                               "model": body["model"],
                               "choices": [{"index": 0, "message": msg, "finish_reason": finish}]}).encode(), "application/json"
        base = {"id": f"cc-{nonce}", "object": "chat.completion.chunk", "created": 1, "model": body["model"]}
        chunks = [{"role": "assistant", "content": ""}]
        if msg.get("content"):
            text = msg["content"]
            chunks += [{"content": text[i:i + 5]} for i in range(0, len(text), 5)]
        for tc in msg.get("tool_calls") or []:
            args = tc["function"]["arguments"]
            chunks.append({"tool_calls": [{"index": 0, "id": tc["id"], "type": "function",
                                           "function": {"name": "add", "arguments": ""}}]})
            chunks += [{"tool_calls": [{"index": 0, "function": {"arguments": args[i:i + 4]}}]}
                       for i in range(0, len(args), 4)]
        lines = [f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': d, 'finish_reason': None}]})}\n\n"
                 for d in chunks]
        lines.append(f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}]})}\n\n")
        lines.append("data: [DONE]\n\n")
        return "".join(lines).encode(), "text/event-stream"

    # ------------------------------------------------------------- anthropic
    def _anthropic(self, body):
        msgs = body["messages"]
        nonce = secrets.token_hex(3)
        last = msgs[-1]["content"]
        if isinstance(last, list) and last and last[0].get("type") == "tool_result":
            content = [{"type": "text", "text": f"The answer is {last[0]['content']} (nonce {nonce})"}]
            stop = "end_turn"
        else:
            a, b = _numbers(last if isinstance(last, str) else "")
            content = [{"type": "text", "text": "Let me add."},
                       {"type": "tool_use", "id": f"toolu_{nonce}", "name": "add", "input": {"a": a, "b": b}}]
            stop = "tool_use"
        msg = {"id": f"msg_{nonce}", "type": "message", "role": "assistant", "model": body["model"],
               "content": content, "stop_reason": stop, "stop_sequence": None,
               "usage": {"input_tokens": 10, "output_tokens": 5}}
        if not body.get("stream"):
            return json.dumps(msg).encode(), "application/json"
        ev = []

        def e(name, payload):
            ev.append(f"event: {name}\r\ndata: {json.dumps(payload)}\r\n\r\n")  # CRLF on purpose

        e("message_start", {"type": "message_start", "message": {**msg, "content": [], "stop_reason": None}})
        e("ping", {"type": "ping"})
        for i, blk in enumerate(content):
            if blk["type"] == "text":
                e("content_block_start", {"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}})
                for j in range(0, len(blk["text"]), 4):
                    e("content_block_delta", {"type": "content_block_delta", "index": i,
                                              "delta": {"type": "text_delta", "text": blk["text"][j:j + 4]}})
            else:
                e("content_block_start", {"type": "content_block_start", "index": i,
                                          "content_block": {"type": "tool_use", "id": blk["id"], "name": "add", "input": {}}})
                raw = json.dumps(blk["input"])
                for j in range(0, len(raw), 3):
                    e("content_block_delta", {"type": "content_block_delta", "index": i,
                                              "delta": {"type": "input_json_delta", "partial_json": raw[j:j + 3]}})
            e("content_block_stop", {"type": "content_block_stop", "index": i})
        e("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                            "usage": {"output_tokens": 5}})
        e("message_stop", {"type": "message_stop"})
        return "".join(ev).encode(), "text/event-stream"
