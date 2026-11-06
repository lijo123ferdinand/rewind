"""`rewind` command line.

    rewind record -- python agent.py          record every model call (command after `--`)
    rewind replay last                        re-run it offline, deterministically, $0
    rewind fork last --at 3 --edit            change the model's reply at step 3, run on from there
    rewind diff <run-a> <run-b>               where (and how) two runs parted ways
    rewind ls | show <run> | export | import | serve
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import __version__
from .diff import diff_runs, render
from .proxy import Override, Session, resolve_upstreams, start_proxy
from .runner import RunResult, run_agent
from .store import Step, Store
from .summary import describe_request, describe_response, response_body


def say(msg: str) -> None:
    print(f"rewind: {msg}", file=sys.stderr, flush=True)


def _command(args: argparse.Namespace, store: Store, parent_ref: str | None = None) -> list[str]:
    cmd = list(args.command or [])
    if not cmd and parent_ref:
        cmd = store.run(parent_ref).command or []
    if not cmd:
        raise SystemExit("rewind: no command to run (put it after `--`)")
    return cmd


def _report(store: Store, res: RunResult, verb: str) -> int:
    counts = ", ".join(f"{n} {k}" for k, n in sorted(res.counts.items())) or "no model calls"
    say(f"{verb} run {res.run_id} - {store.step_count(res.run_id)} steps ({counts}) - agent exit {res.exit_code}")
    for d in res.divergences:
        exp = describe_request(d.expected) if d.expected else "nothing (the recording ended here)"
        say(f"DIVERGED at step {d.idx}: the recording expected {exp}")
    if res.divergences:
        say(f"see how: rewind diff {store.run(res.run_id).parent} {res.run_id}")
        return 3  # "behaviour changed" - distinct from the agent's own failure codes, for CI
    if verb == "recorded" and store.step_count(res.run_id):
        say(f"replay it free and offline: rewind replay {res.run_id}")
    return res.exit_code


# ---------------------------------------------------------------------------- commands
def cmd_record(a: argparse.Namespace, store: Store) -> int:
    res = run_agent(store, "record", _command(a, store), masks=a.mask, port=a.port)
    return _report(store, res, "recorded")


def cmd_replay(a: argparse.Namespace, store: Store) -> int:
    parent = store.run(a.run)
    res = run_agent(store, "replay", _command(a, store, parent.id), parent=parent.id,
                    on_divergence=a.on_divergence, offline=a.on_divergence == "fail",
                    masks=a.mask, port=a.port)
    return _report(store, res, "replayed")


def _edit_reply(store: Store, run_id: str, at: int) -> Override:
    step = next((s for s in store.steps(run_id) if s.idx == at), None)
    if step is None:
        raise SystemExit(f"rewind: run {run_id} has no step {at} to edit")
    body = response_body(step)
    if body is None:
        raise SystemExit(f"rewind: step {at} is not an editable chat reply")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(body, f, indent=2, ensure_ascii=False)
        path = f.name
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
    subprocess.run([*shlex.split(editor), path], check=True)
    edited = json.loads(Path(path).read_text(encoding="utf-8"))
    os.unlink(path)
    return Override(body=edited)


def cmd_fork(a: argparse.Namespace, store: Store) -> int:
    parent = store.run(a.run)
    if a.at < 1:
        raise SystemExit("rewind: --at is 1-based (step 1 is the first model call)")
    override = None
    if a.reply_text is not None:
        override = Override(text=a.reply_text)
    elif a.reply_json:
        override = Override(body=json.loads(Path(a.reply_json).read_text(encoding="utf-8")))
    elif a.edit:
        override = _edit_reply(store, parent.id, a.at)
    what = "your reply" if override else "a fresh live sample"
    say(f"forking {parent.id}: steps 1-{a.at - 1} from the recording, step {a.at} = {what}, then "
        f"{'offline' if a.offline else 'live'}")
    res = run_agent(store, "fork", _command(a, store, parent.id), parent=parent.id, fork_at=a.at,
                    override=override, on_divergence="fail", offline=a.offline, masks=a.mask,
                    port=a.port)
    return _report(store, res, "forked")


def cmd_serve(a: argparse.Namespace, store: Store) -> int:
    parent = store.run(a.replay).id if a.replay else None
    run_id = store.new_run("serve" if not parent else "replay", parent=parent)
    session = Session(store=store, run_id=run_id, upstreams=resolve_upstreams(dict(os.environ)),
                      parent_steps=store.steps(parent) if parent else (),
                      on_divergence="live" if a.live else "fail", offline=bool(parent) and not a.live)
    server, url = start_proxy(session, a.port, a.host)
    say(f"run {run_id} {'replaying ' + parent if parent else 'recording'} on {url}")
    say(f"point your agent at it:  OPENAI_BASE_URL={url}/v1  ANTHROPIC_BASE_URL={url}   (Ctrl-C to stop)")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        store.finish_run(run_id, 0)
    say(f"stopped - {store.step_count(run_id)} steps in run {run_id}")
    return 0


def cmd_ls(a: argparse.Namespace, store: Store) -> int:
    runs = store.runs(a.limit)
    if not runs:
        say("no runs yet - start with: rewind record -- <your agent command>")
        return 0
    print(f"{'RUN':<20} {'KIND':<8} {'STEPS':>5} {'EXIT':>4}  FROM / COMMAND")
    for r in runs:
        origin = f"{r.parent}@{r.fork_at}" if r.fork_at else (r.parent or " ".join(r.command or [])[:50])
        print(f"{r.id:<20} {r.kind:<8} {store.step_count(r.id):>5} "
              f"{'' if r.exit_code is None else r.exit_code:>4}  {origin}")
    return 0


def cmd_show(a: argparse.Namespace, store: Store) -> int:
    run = store.run(a.run)
    steps = store.steps(run.id)
    if a.step:
        step = next((s for s in steps if s.idx == a.step), None)
        if not step:
            raise SystemExit(f"rewind: no step {a.step} in {run.id}")
        print(json.dumps({"request": json.loads(step.request or b"null"),
                          "response": response_body(step) or step.response.decode("utf-8", "replace")},
                         indent=2, ensure_ascii=False))
        return 0
    print(f"run {run.id} ({run.kind}{', from ' + run.parent if run.parent else ''}"
          f"{', fork at ' + str(run.fork_at) if run.fork_at else ''}) - {' '.join(run.command or [])}")
    for s in steps:
        print(f"{s.idx:>4} {s.source:<10} {s.provider:<9} {s.latency_ms:>8.0f}ms  {describe_request(s)}")
        print(f"{'':>34}-> {describe_response(s)}")
    return 0


def cmd_diff(a: argparse.Namespace, store: Store) -> int:
    ra, rb = store.run(a.a), store.run(a.b)
    d = diff_runs(store.steps(ra.id), store.steps(rb.id))
    print(render(d, ra.id, rb.id))
    return 0 if d.first_divergence is None else 1


def cmd_export(a: argparse.Namespace, store: Store) -> int:
    run = store.run(a.run)
    with open(a.output, "w", encoding="utf-8") as f:
        f.write(json.dumps({"rewind_export": 1, "run": run.__dict__}) + "\n")
        for s in store.steps(run.id):
            d = dict(s.__dict__)
            d["request"] = base64.b64encode(s.request).decode()
            d["response"] = base64.b64encode(s.response).decode()
            f.write(json.dumps(d) + "\n")
    say(f"exported {run.id} to {a.output} (bodies only - no API keys or headers)")
    return 0


def cmd_import(a: argparse.Namespace, store: Store) -> int:
    with open(a.file, encoding="utf-8") as f:
        head = json.loads(f.readline())
        if head.get("rewind_export") != 1:
            raise SystemExit("rewind: not a rewind export file")
        src = head["run"]
        run_id = store.new_run("imported", command=src.get("command"), note=f"imported from {src['id']}")
        for line in f:
            d = json.loads(line)
            d.update(run_id=run_id, request=base64.b64decode(d["request"]),
                     response=base64.b64decode(d["response"]))
            store.add_step(Step(**d))
    say(f"imported {src['id']} as {run_id} - replay it: rewind replay {run_id}")
    return 0


# ------------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rewind", description="Record, replay and fork AI agent runs.")
    p.add_argument("--version", action="version", version=f"rewind {__version__}")
    p.add_argument("--db", help="trace database (default ./.rewind/rewind.db or $REWIND_DB)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def runnable(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--mask", action="append", default=[], metavar="REGEX",
                        help="ignore text matching REGEX when matching requests (timestamps, uuids)")
        sp.add_argument("--port", type=int, default=0, help="proxy port (default: any free port)")

    sp = sub.add_parser("record", help="run an agent and record every model call")
    runnable(sp)
    sp.set_defaults(fn=cmd_record)

    sp = sub.add_parser("replay", help="re-run a recorded agent offline and deterministically")
    sp.add_argument("run")
    sp.add_argument("--on-divergence", choices=["fail", "live"], default="fail",
                    help="when the agent asks something new: refuse (default) or call the model")
    runnable(sp)
    sp.set_defaults(fn=cmd_replay)

    sp = sub.add_parser("fork", help="replay up to a step, change it, run on from there")
    sp.add_argument("run")
    sp.add_argument("--at", type=int, required=True, help="the step to change (1-based)")
    g = sp.add_mutually_exclusive_group()
    g.add_argument("--reply-text", help="answer step N with this text")
    g.add_argument("--reply-json", help="answer step N with this provider-format JSON body")
    g.add_argument("--edit", action="store_true", help="edit the recorded reply in $EDITOR")
    sp.add_argument("--offline", action="store_true", help="refuse live calls after the fork")
    runnable(sp)
    sp.set_defaults(fn=cmd_fork)

    sp = sub.add_parser("serve", help="run the proxy standalone (agents in Docker, notebooks)")
    sp.add_argument("--port", type=int, default=8787)
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--replay", metavar="RUN", help="serve this recording instead of recording")
    sp.add_argument("--live", action="store_true", help="with --replay: go live on divergence")
    sp.set_defaults(fn=cmd_serve)

    sp = sub.add_parser("ls", help="list runs")
    sp.add_argument("--limit", type=int, default=20)
    sp.set_defaults(fn=cmd_ls)

    sp = sub.add_parser("show", help="the steps of a run")
    sp.add_argument("run")
    sp.add_argument("--step", type=int, help="print one step's full request and reply")
    sp.set_defaults(fn=cmd_show)

    sp = sub.add_parser("diff", help="where two runs diverged")
    sp.add_argument("a")
    sp.add_argument("b")
    sp.set_defaults(fn=cmd_diff)

    sp = sub.add_parser("export", help="write a run to a shareable file")
    sp.add_argument("run")
    sp.add_argument("-o", "--output", required=True)
    sp.set_defaults(fn=cmd_export)

    sp = sub.add_parser("import", help="load a run someone exported")
    sp.add_argument("file")
    sp.set_defaults(fn=cmd_import)
    return p


def main(argv: list[str] | None = None) -> int:
    # Split at the first `--` ourselves. argparse.REMAINDER after a positional (`replay RUN`)
    # swallows every later option too, so `replay last --mask X -- cmd` would run "--mask".
    argv = list(sys.argv[1:] if argv is None else argv)
    command: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, command = argv[:cut], argv[cut + 1:]
    args = build_parser().parse_args(argv)
    args.command = command
    store = Store(args.db)
    try:
        return args.fn(args, store)
    except KeyError as e:
        say(str(e.args[0]))
        return 2
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
