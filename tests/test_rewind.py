"""End-to-end tests: a real agent process, the real CLI, a random fake model.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_llm import FakeLLM  # noqa: E402
from rewind.canon import request_hash  # noqa: E402
from rewind.sse import reassemble, to_sse  # noqa: E402

AGENT = [sys.executable, str(ROOT / "examples" / "toy_agent.py")]


class E2E(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "t.db")
        self.fake = FakeLLM()
        self.env = {**os.environ, "PYTHONPATH": str(ROOT / "src"),
                    "OPENAI_BASE_URL": self.fake.url + "/v1", "ANTHROPIC_BASE_URL": self.fake.url,
                    "OPENAI_API_KEY": "sk-test-SUPERSECRET-0001", "ANTHROPIC_API_KEY": "sk-ant-SUPERSECRET-0002",
                    "PYTHONUTF8": "1"}
        self.fake_open = True

    def tearDown(self) -> None:
        if self.fake_open:
            self.fake.close()
        self.tmp.cleanup()

    def stop_model(self) -> None:
        """Take the 'provider' offline: replays must not need it."""
        self.fake.close()
        self.fake_open = False

    def rewind(self, *args: str, db: str | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, "-m", "rewind", "--db", db or self.db, *args],
                              env=self.env, capture_output=True, text=True, timeout=60)

    # ------------------------------------------------------------------ replay
    def test_replay_is_identical_and_needs_no_model(self) -> None:
        for provider in ("openai", "anthropic"):
            for stream in (False, True):
                with self.subTest(provider=provider, stream=stream):
                    self.setUp()
                    extra = ["--provider", provider] + (["--stream"] if stream else [])
                    rec = self.rewind("record", "--", *AGENT, *extra)
                    self.assertEqual(rec.returncode, 0, rec.stderr)
                    self.assertIn("final: The answer is 42 (nonce", rec.stdout)
                    self.assertEqual(self.fake.hits, 2)   # tool call, then the answer
                    self.stop_model()
                    rep = self.rewind("replay", "last")
                    self.assertEqual(rep.returncode, 0, rep.stderr)
                    # The fake's nonce is random per call: equality proves the reply came from tape.
                    self.assertEqual(rep.stdout, rec.stdout)
                    self.assertIn("2 cache", rep.stderr)
                    self.tearDown()
        self.setUp()  # leave a fresh fixture for the outer tearDown

    def test_divergence_is_caught_and_explained(self) -> None:
        rec = self.rewind("record", "--", *AGENT, "--question", "what is 17 + 25?")
        self.assertEqual(rec.returncode, 0, rec.stderr)
        rep = self.rewind("replay", "last", "--", *AGENT, "--question", "what is 2 + 2?")
        self.assertEqual(rep.returncode, 3, rep.stderr)       # 3 = diverged
        self.assertIn("DIVERGED at step 1", rep.stderr)
        self.assertEqual(self.fake.hits, 2)                   # strict replay never went live
        runs = [ln.split()[0] for ln in self.rewind("ls").stdout.splitlines()[1:]]
        diff = self.rewind("diff", runs[1], runs[0])
        self.assertEqual(diff.returncode, 1)
        self.assertIn("first divergence at step 1", diff.stdout)
        self.assertIn('-  "content": "what is 17 + 25?"', diff.stdout)
        self.assertIn('+  "content": "what is 2 + 2?"', diff.stdout)

    def test_live_on_divergence_records_the_new_path(self) -> None:
        self.rewind("record", "--", *AGENT)
        rep = self.rewind("replay", "last", "--on-divergence", "live", "--", *AGENT, "--question", "5 + 6")
        self.assertEqual(rep.returncode, 3)
        self.assertIn("final: The answer is 11", rep.stdout)

    def test_mask_ignores_volatile_prompt_text(self) -> None:
        q1, q2 = "what is 17 + 25? asked at 09:00:01", "what is 17 + 25? asked at 17:45:59"
        rec = self.rewind("record", "--mask", r"asked at [\d:]+", "--", *AGENT, "--question", q1)
        self.stop_model()
        rep = self.rewind("replay", "last", "--mask", r"asked at [\d:]+", "--", *AGENT, "--question", q2)
        self.assertEqual(rep.returncode, 0, rep.stderr)
        self.assertEqual(rep.stdout, rec.stdout)

    # -------------------------------------------------------------------- fork
    def test_fork_with_an_edited_reply(self) -> None:
        for provider in ("openai", "anthropic"):
            for stream in (False, True):
                with self.subTest(provider=provider, stream=stream):
                    extra = ["--provider", provider] + (["--stream"] if stream else [])
                    self.rewind("record", "--", *AGENT, *extra)
                    hits = self.fake.hits
                    fork = self.rewind("fork", "last", "--at", "2", "--reply-text", "it is 99, trust me",
                                       "--offline")
                    self.assertEqual(fork.returncode, 0, fork.stderr)
                    self.assertEqual(fork.stdout.strip(), "final: it is 99, trust me")
                    self.assertEqual(self.fake.hits, hits)     # step 1 from tape, step 2 overridden
                    self.assertIn("1 cache", fork.stderr)
                    self.assertIn("1 override", fork.stderr)

    def test_fork_without_override_resamples_live(self) -> None:
        rec = self.rewind("record", "--", *AGENT)
        fork = self.rewind("fork", "last", "--at", "2")
        self.assertEqual(fork.returncode, 0, fork.stderr)
        self.assertEqual(self.fake.hits, 3)                    # only step 2 went to the model
        self.assertNotEqual(fork.stdout, rec.stdout)           # a fresh sample: new nonce
        self.assertIn("final: The answer is 42", fork.stdout)

    def test_fork_with_edited_json_changes_the_tool_call(self) -> None:
        self.rewind("record", "--", *AGENT)
        step1 = json.loads(self.rewind("show", "last", "--step", "1").stdout)["response"]
        step1["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = '{"a": 100, "b": 1}'
        path = os.path.join(self.tmp.name, "edit.json")
        Path(path).write_text(json.dumps(step1))
        fork = self.rewind("fork", "last", "--at", "1", "--reply-json", path)
        self.assertEqual(fork.returncode, 0, fork.stderr)
        self.assertIn("final: The answer is 101", fork.stdout)  # the agent ran the edited call

    # -------------------------------------------------------------- safety, sharing
    def test_api_keys_never_reach_the_trace(self) -> None:
        for provider in ("openai", "anthropic"):
            self.rewind("record", "--", *AGENT, "--provider", provider)
        self.assertIn("Bearer sk-test-SUPERSECRET-0001", self.fake.auth_seen)   # forwarded...
        self.assertIn("sk-ant-SUPERSECRET-0002", self.fake.auth_seen)
        raw = b"".join(p.read_bytes() for p in Path(self.tmp.name).glob("t.db*"))
        self.assertNotIn(b"SUPERSECRET", raw)                                   # ...never stored
        out = os.path.join(self.tmp.name, "run.rewind")
        self.rewind("export", "last", "-o", out)
        self.assertNotIn("SUPERSECRET", Path(out).read_text())

    def test_export_import_replays_on_another_machine(self) -> None:
        rec = self.rewind("record", "--", *AGENT, "--stream")
        out = os.path.join(self.tmp.name, "bug.rewind")
        self.assertEqual(self.rewind("export", "last", "-o", out).returncode, 0)
        self.stop_model()
        other = os.path.join(self.tmp.name, "other.db")
        self.assertEqual(self.rewind("import", out, db=other).returncode, 0)
        rep = self.rewind("replay", "last", db=other)
        self.assertEqual(rep.returncode, 0, rep.stderr)
        self.assertEqual(rep.stdout, rec.stdout)


class Units(unittest.TestCase):
    def test_hash_ignores_key_order_and_tracking_fields(self) -> None:
        a = json.dumps({"model": "m", "messages": [], "user": "u1"}).encode()
        b = json.dumps({"messages": [], "model": "m", "metadata": {"x": 1}}).encode()
        c = json.dumps({"messages": [], "model": "m", "stream": True}).encode()
        self.assertEqual(request_hash("POST", "/v1/x", a), request_hash("POST", "/v1/x", b))
        self.assertNotEqual(request_hash("POST", "/v1/x", a), request_hash("POST", "/v1/x", c))

    def test_openai_stream_reassembles_split_tool_arguments(self) -> None:
        sse = (b'data: {"id":"c","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}\n\n'
               b'data: {"id":"c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"add","arguments":""}}]}}]}\n\n'
               b'data: {"id":"c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"a\\":"}}]}}]}\n\n'
               b'data: {"id":"c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}}]}\n\n'
               b'data: {"id":"c","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n'
               b"data: [DONE]\n\n")
        body = reassemble("openai", sse)
        tc = body["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual((tc["id"], tc["function"]["name"], tc["function"]["arguments"]), ("call_1", "add", '{"a":1}'))
        self.assertEqual(body["choices"][0]["finish_reason"], "tool_calls")

    def test_round_trip_both_providers(self) -> None:
        oa = {"id": "x", "created": 1, "model": "m", "object": "chat.completion",
              "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                  "role": "assistant", "content": "hi",
                  "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}}]}
        self.assertEqual(reassemble("openai", to_sse("openai", oa)), oa)
        an = {"id": "m1", "type": "message", "role": "assistant", "model": "m",
              "content": [{"type": "thinking", "thinking": "hmm", "signature": "s"},
                          {"type": "text", "text": "ok"},
                          {"type": "tool_use", "id": "t1", "name": "f", "input": {"q": [1, 2]}}],
              "stop_reason": "tool_use", "stop_sequence": None, "usage": {"input_tokens": 3, "output_tokens": 4}}
        self.assertEqual(reassemble("anthropic", to_sse("anthropic", an)), an)


if __name__ == "__main__":
    unittest.main()
