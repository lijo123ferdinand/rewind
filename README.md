# rewind

**Record, replay and fork AI agent runs.** Deterministic, offline, $0 per replay, and it works with
any agent framework in any language, with no code changes.

```
$ rewind record -- python my_agent.py          # run your agent once, for real
final: The answer is 42 (nonce b769c3)
rewind: recorded run 261106-165217-8910 - 2 steps (2 live) - agent exit 0

$ rewind replay last                           # model server OFF, no API key needed
final: The answer is 42 (nonce b769c3)         # byte-identical
rewind: replayed run 261106-165218-6b1d - 2 steps (2 cache) - agent exit 0

$ rewind fork last --at 2 --reply-text "It is 99."   # "what if the model had said..."
final: It is 99.
rewind: forked run 261106-165219-c22e - 2 steps (1 cache, 1 override) - agent exit 0
```

## Why

Agent bugs are the worst kind to debug:

- **You can't reproduce them.** The model samples differently every run, so the failure you saw
  this morning is gone by the time you add a print statement.
- **Every attempt costs money and minutes.** Re-running a 30-step agent to test a one-line fix
  means paying for 30 model calls again.
- **You can't ask "what if".** What if the model had picked a different tool at step 4? Would
  your agent have recovered? Today the only way to find out is to wait until it happens in production.

Tracing tools show you *what happened*. Rewind lets you **run it again**: exactly, for free, and
with any step changed.

## What it does

| | |
|---|---|
| `rewind record -- <cmd>` | Runs your agent behind a local proxy and records every model call. |
| `rewind replay <run>` | Re-runs the agent and answers each model call from the recording. Offline, deterministic, no API key. |
| **divergence detection** | If your code changed and the agent now asks the model something it never asked before, replay stops at that step and tells you where. Exit code `3`, so it works as a **CI regression test for agent behaviour**. |
| `rewind diff <a> <b>` | Shows the step where two runs parted ways, and the exact message that differed. |
| `rewind fork <run> --at N` | Steps 1..N-1 from the recording; step N gets **your** reply (`--reply-text`, `--reply-json`, or `--edit` in `$EDITOR`) or a fresh live sample; then it runs on. |
| `rewind export` / `import` | Turns a run into a file you can attach to a bug report. The maintainer replays your exact failure without your API key. |
| `rewind serve` | The proxy on its own, for agents in Docker, notebooks or other hosts. |
| `--mask REGEX` | Ignores volatile prompt text (timestamps, UUIDs) when matching calls. |

## How it works

```
 your agent ──(OPENAI_BASE_URL / ANTHROPIC_BASE_URL)──▶ rewind proxy ──▶ OpenAI / Anthropic / any compatible API
   (any SDK,                                             │      ▲
    any language)                                        ▼      │ replay: answer from the recording
                                                   .rewind/rewind.db (SQLite)
```

Every SDK honours a base-URL variable, so `rewind` sets them to point at a local proxy and
launches your command. The proxy forwards and records (streaming passes through token by token),
or answers from the recording. A call is matched by a canonical hash of its request body: key
order and tracking fields are ignored, and the model, messages, tools and parameters are kept.
Matching uses hash plus position, so concurrent calls still replay correctly.

If you already point your agent at OpenRouter, Ollama, vLLM or a company gateway, rewind reads
your existing `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` and forwards there.

**Verified with the real vendor SDKs** (openai 3.46.0, anthropic 2.6.0, unmodified), streamed and
non-streamed, and with a raw-HTTP agent: see `tests/`. The test model server returns random text
and random tool-call ids on every call, so an identical replay can only come from the recording.

## Safety

- **API keys are never stored.** Only request and response *bodies* are written; headers
  (`Authorization`, `x-api-key`) are forwarded and dropped. A test asserts the key bytes appear
  nowhere in the database or in an export.
- Strict by default: a replay never silently calls a paid API. `--on-divergence live` is opt-in.
- Standard library only. Nothing to audit but this repository.

## Install

```
pip install git+https://github.com/lijo123ferdinand/rewind
rewind --help
```

Python 3.11+. Tested on 3.12, 3.13 and 3.14 locally; CI covers 3.11–3.14 on Linux, macOS and Windows.

## Limits (v0.1)

- Rewind records **model calls**. Your tools run for real during a replay. If a tool's output
  changes (it reads the clock, the web, a database), the next model call differs and rewind
  reports a divergence. That is often exactly the bug you are hunting; for known-noisy fields use
  `--mask`.
- Summaries, `diff` detail and `fork --edit` understand OpenAI Chat Completions and Anthropic
  Messages. Other endpoints (embeddings, the OpenAI Responses API) are recorded and replayed byte
  for byte, but are not yet editable.

## Roadmap

- **MCP tool recording**: proxy MCP servers too, so a replay is fully hermetic, tools included.
- A **web timeline**: click a step, edit the reply, fork from the browser.
- A `pytest` plugin: `@rewind.cassette("checkout_flow")` as an agent regression test.
- Token and cost accounting per run and per fork ("this fix saves 41% of tokens on the recorded suite").

## License

MIT
