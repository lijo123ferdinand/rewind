"""Run an agent command under the proxy: record, replay, or fork."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field

from .proxy import Divergence, Override, Session, resolve_upstreams, start_proxy
from .store import Store


@dataclass
class RunResult:
    run_id: str
    exit_code: int
    counts: dict[str, int] = field(default_factory=dict)
    divergences: list[Divergence] = field(default_factory=list)


def agent_env(base: dict[str, str], proxy_url: str, run_id: str, *, replaying: bool) -> dict[str, str]:
    env = dict(base)
    # OPENAI_API_BASE: older LangChain / LiteLLM read this spelling.
    env.update(OPENAI_BASE_URL=f"{proxy_url}/v1", OPENAI_API_BASE=f"{proxy_url}/v1",
               ANTHROPIC_BASE_URL=proxy_url, REWIND_PROXY_URL=proxy_url, REWIND_RUN_ID=run_id)
    if replaying:
        # SDKs refuse to start without a key even when no request will reach a provider.
        # Only fill a placeholder when none is set — a fork that goes live needs the real one.
        env.setdefault("OPENAI_API_KEY", "rewind-replay-placeholder")
        env.setdefault("ANTHROPIC_API_KEY", "rewind-replay-placeholder")
    return env


def run_agent(
    store: Store,
    kind: str,
    command: Sequence[str],
    *,
    parent: str | None = None,
    fork_at: int | None = None,
    override: Override | None = None,
    on_divergence: str = "fail",
    offline: bool = False,
    masks: Sequence[str] = (),
    port: int = 0,
    env: dict[str, str] | None = None,
) -> RunResult:
    base_env = dict(os.environ if env is None else env)
    run_id = store.new_run(kind, parent=parent, fork_at=fork_at, command=list(command))
    session = Session(
        store=store,
        run_id=run_id,
        upstreams=resolve_upstreams(base_env),
        parent_steps=store.steps(parent) if parent else (),
        fork_at=fork_at,
        override=override,
        on_divergence=on_divergence,
        offline=offline,
        masks=[re.compile(m) for m in masks],
    )
    server, url = start_proxy(session, port)
    try:
        try:
            proc = subprocess.run(list(command), env=agent_env(base_env, url, run_id, replaying=bool(parent)))
            code = proc.returncode
        except FileNotFoundError:
            print(f"rewind: command not found: {command[0]}", file=sys.stderr)
            code = 127  # what a shell returns for the same mistake
    finally:
        server.shutdown()
        server.server_close()
    store.finish_run(run_id, code)
    return RunResult(run_id, code, dict(session.counts), list(session.divergences))
