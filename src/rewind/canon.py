"""Request identity: when do two model calls count as "the same call"?

Replay serves a recorded response when the agent sends the same request again. "Same" has to
ignore what is noise and keep what changes the model's answer:

  * JSON key order is noise -> canonical JSON (sorted keys, no whitespace).
  * `user` and `metadata` are tracking fields that never change the completion -> dropped.
  * Everything else — model, messages, tools, temperature, `stream` — stays. `stream` matters
    because it changes the response *format*, so a streamed recording cannot answer a
    non-streamed call.

Agents often stamp prompts with the time or a fresh UUID; that makes every replay diverge at step
one. `--mask REGEX` replaces matches with `<masked>` before hashing, so those fields stop counting.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable

VOLATILE_KEYS = frozenset({"user", "metadata"})


def canonical(body: bytes, masks: Iterable[re.Pattern[str]] = ()) -> bytes:
    try:
        obj = json.loads(body) if body else None
    except (ValueError, UnicodeDecodeError):
        text = body.decode("utf-8", "replace")
    else:
        if isinstance(obj, dict):
            obj = {k: v for k, v in obj.items() if k not in VOLATILE_KEYS}
        text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    for m in masks:
        text = m.sub("<masked>", text)
    return text.encode("utf-8")


def request_hash(
    method: str, path: str, body: bytes, masks: Iterable[re.Pattern[str]] = ()
) -> str:
    h = hashlib.sha256()
    h.update(f"{method} {path}\n".encode())
    h.update(canonical(body, masks))
    return h.hexdigest()[:16]


def wants_stream(body: bytes) -> bool:
    try:
        obj = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return False
    return isinstance(obj, dict) and obj.get("stream") is True
