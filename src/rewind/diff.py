"""Where did two runs of an agent part ways, and how?

Steps are aligned by position. The first step whose request hash differs is the divergence; the
useful detail is *what changed in the conversation the agent sent* — usually one tool result or
one system-prompt line — so the message list is diffed from the first differing message on.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass

from .store import Step
from .summary import describe_request, describe_response, request_body


@dataclass
class RunDiff:
    rows: list[tuple[int, Step | None, Step | None, bool]]
    first_divergence: int | None


def diff_runs(a: list[Step], b: list[Step]) -> RunDiff:
    by_a = {s.idx: s for s in a}
    by_b = {s.idx: s for s in b}
    rows = []
    first = None
    for i in range(1, max([0, *by_a, *by_b]) + 1):
        sa, sb = by_a.get(i), by_b.get(i)
        same = bool(sa and sb and sa.req_hash == sb.req_hash)
        rows.append((i, sa, sb, same))
        if not same and first is None:
            first = i
    return RunDiff(rows, first)


def message_diff(a: Step | None, b: Step | None, context: int = 2, limit: int = 60) -> list[str]:
    ma = (request_body(a) or {}).get("messages", []) if a else []
    mb = (request_body(b) or {}).get("messages", []) if b else []
    start = 0
    while start < min(len(ma), len(mb)) and ma[start] == mb[start]:
        start += 1

    def lines(msgs: list) -> list[str]:
        return json.dumps(msgs[start:], indent=1, ensure_ascii=False, sort_keys=True).splitlines()

    out = list(difflib.unified_diff(lines(ma), lines(mb), "run A", "run B", n=context, lineterm=""))
    if not out and a and b:  # messages equal: the difference is elsewhere (model, tools, params)
        ra = {k: v for k, v in (request_body(a) or {}).items() if k != "messages"}
        rb = {k: v for k, v in (request_body(b) or {}).items() if k != "messages"}
        out = list(difflib.unified_diff(
            json.dumps(ra, indent=1, sort_keys=True).splitlines(),
            json.dumps(rb, indent=1, sort_keys=True).splitlines(),
            "run A (params)", "run B (params)", n=context, lineterm=""))
    if start:
        out.insert(0, f"(first {start} message(s) identical)")
    return out[:limit] + ([f"... ({len(out) - limit} more lines)"] if len(out) > limit else [])


def render(d: RunDiff, a_id: str, b_id: str) -> str:
    out = [f"A = {a_id}", f"B = {b_id}", ""]
    for i, sa, sb, same in d.rows:
        mark = "==" if same else "!="
        left = describe_response(sa) if sa else "(no step)"
        right = describe_response(sb) if sb else "(no step)"
        out.append(f"{i:>4} {mark}  A: {left}")
        if not same:
            out.append(f"         B: {right}")
    if d.first_divergence is None:
        out += ["", "identical: every request matches."]
        return "\n".join(out)
    i = d.first_divergence
    _, sa, sb, _ = d.rows[i - 1]
    out += ["", f"first divergence at step {i}:",
            f"  A sent: {describe_request(sa) if sa else '(nothing)'}",
            f"  B sent: {describe_request(sb) if sb else '(nothing)'}", ""]
    out += message_diff(sa, sb)
    return "\n".join(out)
