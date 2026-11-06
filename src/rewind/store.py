"""SQLite trace store: runs and the model calls (steps) inside them.

One file, WAL mode, a single connection guarded by a lock. The proxy serves requests on
several threads, but writes are tiny (one row per model call), so a lock is simpler and faster
than a connection pool.

What is stored, and what is deliberately NOT:
  * the request BODY and the response BODY, byte for byte — that is what replay needs;
  * never request HEADERS. API keys travel in `Authorization` / `x-api-key`; by never writing
    headers, a trace file is safe to attach to a bug report. (`tests/test_rewind.py` checks it.)
"""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    kind        TEXT NOT NULL,          -- record | replay | fork | serve | imported
    parent      TEXT,                   -- the run this one replays or forks
    fork_at     INTEGER,
    command     TEXT,                   -- JSON list, so `rewind replay last` can re-run it
    exit_code   INTEGER,
    note        TEXT
);
CREATE TABLE IF NOT EXISTS steps (
    run_id       TEXT NOT NULL,
    idx          INTEGER NOT NULL,      -- 1-based: "step 4" means the 4th model call
    method       TEXT NOT NULL,
    path         TEXT NOT NULL,
    provider     TEXT NOT NULL,         -- openai | anthropic
    req_hash     TEXT NOT NULL,
    request      BLOB NOT NULL,
    status       INTEGER NOT NULL,
    content_type TEXT NOT NULL,
    response     BLOB NOT NULL,
    streamed     INTEGER NOT NULL,
    source       TEXT NOT NULL,         -- live | cache | override | divergence | blocked
    latency_ms   REAL NOT NULL,
    PRIMARY KEY (run_id, idx)
);
"""


def default_db_path() -> Path:
    return Path(os.environ.get("REWIND_DB") or Path.cwd() / ".rewind" / "rewind.db")


@dataclass(frozen=True)
class Step:
    run_id: str
    idx: int
    method: str
    path: str
    provider: str
    req_hash: str
    request: bytes
    status: int
    content_type: str
    response: bytes
    streamed: bool
    source: str
    latency_ms: float


@dataclass(frozen=True)
class Run:
    id: str
    created_at: str
    kind: str
    parent: str | None
    fork_at: int | None
    command: list[str] | None
    exit_code: int | None
    note: str | None


class Store:
    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path) if path else default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # isolation_level=None = autocommit; every write is one statement anyway.
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ------------------------------------------------------------------ runs
    def new_run(
        self,
        kind: str,
        *,
        parent: str | None = None,
        fork_at: int | None = None,
        command: list[str] | None = None,
        note: str | None = None,
    ) -> str:
        now = datetime.now(timezone.utc)
        # Sortable and readable: you will type these. The random suffix makes two runs started in
        # the same second distinct.
        run_id = f"{now:%y%m%d-%H%M%S}-{secrets.token_hex(2)}"
        with self._lock:
            self._db.execute(
                "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?)",
                (run_id, now.isoformat(timespec="seconds"), kind, parent, fork_at,
                 json.dumps(command) if command else None, None, note),
            )
        return run_id

    def finish_run(self, run_id: str, exit_code: int | None) -> None:
        with self._lock:
            self._db.execute("UPDATE runs SET exit_code=? WHERE id=?", (exit_code, run_id))

    def runs(self, limit: int = 50) -> list[Run]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM runs ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._run(r) for r in rows]

    def run(self, ref: str) -> Run:
        """Resolve `last`, a full id, or a unique id prefix/suffix. Raises KeyError."""
        with self._lock:
            if ref == "last":
                rows = self._db.execute(
                    "SELECT * FROM runs ORDER BY created_at DESC, rowid DESC LIMIT 1"
                ).fetchall()
            else:
                rows = self._db.execute(
                    "SELECT * FROM runs WHERE id = ? OR id LIKE ? OR id LIKE ?",
                    (ref, f"{ref}%", f"%{ref}"),
                ).fetchall()
        exact = [r for r in rows if r[0] == ref]
        if exact:
            rows = exact
        if not rows:
            raise KeyError(f"no run matches {ref!r} (try `rewind ls`)")
        if len(rows) > 1:
            raise KeyError(f"{ref!r} is ambiguous: {', '.join(r[0] for r in rows[:5])}")
        return self._run(rows[0])

    @staticmethod
    def _run(r: tuple) -> Run:
        return Run(r[0], r[1], r[2], r[3], r[4], json.loads(r[5]) if r[5] else None, r[6], r[7])

    # ----------------------------------------------------------------- steps
    def add_step(self, s: Step) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO steps VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (s.run_id, s.idx, s.method, s.path, s.provider, s.req_hash, s.request,
                 s.status, s.content_type, s.response, int(s.streamed), s.source, s.latency_ms),
            )

    def steps(self, run_id: str) -> list[Step]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM steps WHERE run_id=? ORDER BY idx", (run_id,)
            ).fetchall()
        return [
            Step(r[0], r[1], r[2], r[3], r[4], r[5], bytes(r[6]), r[7], r[8], bytes(r[9]),
                 bool(r[10]), r[11], r[12])
            for r in rows
        ]

    def step_count(self, run_id: str) -> int:
        with self._lock:
            return self._db.execute(
                "SELECT COUNT(*) FROM steps WHERE run_id=?", (run_id,)
            ).fetchone()[0]
