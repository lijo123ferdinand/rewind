"""The recording / replaying proxy.

The agent is pointed at this proxy instead of the model provider (`OPENAI_BASE_URL`,
`ANTHROPIC_BASE_URL`). Every model call then passes through `Handler._handle`, which decides,
per call, where the answer comes from:

    replaying this step?  --hash matches a recorded call-->  serve the recording   (cache)
            |                                   no match -->  DIVERGENCE: fail (409) or go live
            no
    the fork step with an override?  ------------------->  serve the edited reply (override)
            no
    --offline?  ---------------------------------------->  refuse (409)            (blocked)
            no
    forward to the real provider, stream it through, record it                     (live)

Why an HTTP proxy rather than an SDK wrapper: it works for every language and framework with no
code change. LangGraph, the OpenAI Agents SDK, the Claude Agent SDK, a 40-line raw loop and a
TypeScript agent all honour the base-URL variables.

Why match by hash *and* position: agents can issue calls concurrently, so the k-th request of a
replay is not guaranteed to be the k-th recorded one. Each request takes the recorded step at its
own position if the hash agrees, otherwise the first unused recorded step with that hash. A
request with no unused match is the divergence.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from http.client import HTTPConnection, HTTPSConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import SplitResult, urlsplit

from .canon import request_hash, wants_stream
from .sse import synth_text_reply, to_sse
from .store import Step, Store

# Hop-by-hop headers (RFC 9110 §7.6.1) plus the ones we must recompute ourselves.
_DROP_REQUEST = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
                 "trailers", "transfer-encoding", "upgrade", "host", "content-length",
                 "accept-encoding"}
# Upstream response headers worth passing through live (SDK retry logic reads these).
_PASS_RESPONSE = ("retry-after", "x-request-id", "request-id", "openai-", "anthropic-",
                  "x-ratelimit-")


@dataclass
class Override:
    """What to answer at the fork step: plain text (synthesised per provider) or a full body."""
    text: str | None = None
    body: dict[str, Any] | None = None


@dataclass
class Divergence:
    idx: int
    req_hash: str
    expected: Step | None
    request: bytes


@dataclass
class Session:
    store: Store
    run_id: str
    upstreams: dict[str, SplitResult]
    parent_steps: Sequence[Step] = ()
    fork_at: int | None = None
    override: Override | None = None
    on_divergence: str = "fail"            # "fail" | "live"
    offline: bool = False
    masks: Sequence[re.Pattern[str]] = ()
    divergences: list[Divergence] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    _next: int = 1
    _used: set[int] = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def claim_index(self) -> int:
        with self._lock:
            idx, self._next = self._next, self._next + 1
            return idx

    def replaying(self, idx: int) -> bool:
        return bool(self.parent_steps) and (self.fork_at is None or idx < self.fork_at)

    def match(self, idx: int, h: str) -> Step | None:
        with self._lock:
            by_pos = next((s for s in self.parent_steps if s.idx == idx), None)
            if by_pos and by_pos.req_hash == h and by_pos.idx not in self._used:
                hit = by_pos
            else:
                hit = next((s for s in self.parent_steps
                            if s.req_hash == h and s.idx not in self._used
                            and s.source != "divergence"), None)
            if hit:
                self._used.add(hit.idx)
            return hit

    def expected_at(self, idx: int) -> Step | None:
        return next((s for s in self.parent_steps if s.idx == idx), None)

    def bump(self, source: str) -> None:
        with self._lock:
            self.counts[source] = self.counts.get(source, 0) + 1


def detect_provider(headers: Any, path: str) -> str:
    # The Anthropic SDKs always send `anthropic-version`; that beats guessing from the path,
    # because both providers serve e.g. /v1/models.
    if headers.get("anthropic-version") or headers.get("x-api-key"):
        return "anthropic"
    return "anthropic" if path.startswith("/v1/messages") else "openai"


def join_upstream(base: SplitResult, path: str) -> str:
    """Map the agent's path onto the upstream base, tolerating bases that already end in /v1
    (OPENAI_BASE_URL=https://openrouter.ai/api/v1 -> /api/v1/chat/completions)."""
    root = base.path.rstrip("/")
    if root.endswith("/v1") and path.startswith("/v1"):
        root = root[:-3]
    return root + path


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    session: Session


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _Server

    def do_POST(self) -> None:  # noqa: N802 (http.server naming)
        self._handle()

    do_GET = do_PUT = do_DELETE = do_PATCH = do_POST

    def log_message(self, format: str, *args: Any) -> None:  # silence per-request stderr noise
        pass

    # ------------------------------------------------------------------ helpers
    def _read_body(self) -> bytes:
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            out = bytearray()
            while True:
                size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                if size == 0:
                    self.rfile.readline()
                    return bytes(out)
                out += self.rfile.read(size)
                self.rfile.readline()
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _send(self, status: int, ctype: str, data: bytes, extra: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)
        self.wfile.flush()

    def _error(self, status: int, message: str) -> bytes:
        # Shaped like both providers' errors, so SDKs surface the message instead of crashing.
        data = json.dumps({"type": "error", "error": {"type": "rewind_error", "message": message}}).encode()
        self._send(status, "application/json", data)
        return data

    # --------------------------------------------------------------------- core
    def _handle(self) -> None:
        s = self.server.session
        body = self._read_body()
        path = self.path
        provider = detect_provider(self.headers, path)
        h = request_hash(self.command, path, body, s.masks)
        idx = s.claim_index()
        stream = wants_stream(body)
        t0 = time.perf_counter()

        def record(status: int, ctype: str, data: bytes, streamed: bool, source: str) -> None:
            s.bump(source)
            s.store.add_step(Step(s.run_id, idx, self.command, path, provider, h, body, status,
                                  ctype, data, streamed, source,
                                  round((time.perf_counter() - t0) * 1000, 2)))

        if s.replaying(idx):
            hit = s.match(idx, h)
            if hit:
                self._send(hit.status, hit.content_type, hit.response)
                record(hit.status, hit.content_type, hit.response, hit.streamed, "cache")
                return
            s.divergences.append(Divergence(idx, h, s.expected_at(idx), body))
            if s.on_divergence == "fail":
                data = self._error(409, f"rewind: the agent diverged from the recording at step {idx} "
                                        "(its request differs). Run `rewind diff` to see how, or "
                                        "replay with --on-divergence live.")
                record(409, "application/json", data, False, "divergence")
                return
        elif s.fork_at == idx and s.override is not None:
            try:
                req = json.loads(body) if body else {}
            except ValueError:
                req = {}
            reply = s.override.body or synth_text_reply(provider, req, s.override.text or "")
            if stream:
                data, ctype = to_sse(provider, reply), "text/event-stream"
            else:
                data, ctype = json.dumps(reply).encode(), "application/json"
            self._send(200, ctype, data)
            record(200, ctype, data, stream, "override")
            return

        if s.offline:
            data = self._error(409, f"rewind: step {idx} needs a live model call, but --offline is set")
            record(409, "application/json", data, False, "blocked")
            return
        self._forward(s, provider, body, record)

    def _forward(self, s: Session, provider: str, body: bytes, record: Any) -> None:
        up = s.upstreams[provider]
        conn_cls = HTTPSConnection if up.scheme == "https" else HTTPConnection
        conn = conn_cls(up.hostname or "", up.port, timeout=600)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in _DROP_REQUEST}
        headers["Host"] = up.netloc
        headers["Accept-Encoding"] = "identity"   # store plain bytes, not gzip
        headers["Content-Length"] = str(len(body))
        try:
            conn.request(self.command, join_upstream(up, self.path), body=body, headers=headers)
            resp = conn.getresponse()
        except OSError as e:
            data = self._error(502, f"rewind: upstream {up.netloc} unreachable: {e}")
            record(502, "application/json", data, False, "live")
            return
        ctype = resp.getheader("Content-Type", "application/octet-stream")
        passthru = {k: v for k, v in resp.getheaders()
                    if k.lower().startswith(_PASS_RESPONSE)}
        try:
            if "text/event-stream" in ctype:
                # Stream through as it arrives (the agent sees tokens live) AND keep a copy.
                self.send_response(resp.status)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                for k, v in passthru.items():
                    self.send_header(k, v)
                self.end_headers()
                buf = bytearray()
                client_alive = True
                while chunk := resp.read1(65536):
                    buf += chunk
                    if client_alive:
                        try:
                            self.wfile.write(b"%X\r\n%s\r\n" % (len(chunk), chunk))
                            self.wfile.flush()
                        except OSError:
                            client_alive = False   # keep draining so the recording is complete
                if client_alive:
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                record(resp.status, ctype, bytes(buf), True, "live")
            else:
                data = resp.read()
                self._send(resp.status, ctype, data, passthru)
                record(resp.status, ctype, data, False, "live")
        finally:
            conn.close()


def start_proxy(session: Session, port: int = 0, host: str = "127.0.0.1") -> tuple[_Server, str]:
    server = _Server((host, port), Handler)
    server.session = session
    threading.Thread(target=server.serve_forever, name="rewind-proxy", daemon=True).start()
    return server, f"http://{host}:{server.server_address[1]}"


def resolve_upstreams(env: dict[str, str]) -> dict[str, SplitResult]:
    """Where live calls go. Honours a base URL the agent was ALREADY using (OpenRouter, Ollama,
    a corporate gateway) — read before we overwrite those variables to point at ourselves."""
    return {
        "openai": urlsplit(env.get("REWIND_OPENAI_UPSTREAM") or env.get("OPENAI_BASE_URL")
                           or "https://api.openai.com/v1"),
        "anthropic": urlsplit(env.get("REWIND_ANTHROPIC_UPSTREAM") or env.get("ANTHROPIC_BASE_URL")
                              or "https://api.anthropic.com"),
    }
