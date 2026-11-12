#!/usr/bin/env python3
"""Live check of the Amp provider against the real CLI: which models answer, call tools, and how fast.

Spends a few Amp credits per model (two tiny turns each). Usage:

    python scripts/live_models.py claude-haiku-5-5 gpt-6.1-terra kimi-k3 [--keep-threads]

Prints one line per model. Run it with Hermes importable only if you want the real OpenAI tool-call types;
otherwise the bridge falls back to plain namespaces.
"""

from __future__ import annotations

import concurrent.futures as cf
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import amp_client  # noqa: E402

TOOLS = [{"type": "function", "function": {
    "name": "read_file", "description": "Read a text file and return its contents",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]


def check(model: str) -> str:
    client = amp_client.AmpClient()
    try:
        sysmsg = {"role": "system", "content": "You are Hermes. Be brief."}
        convo = [sysmsg, {"role": "user", "content": "Use the tool to read /tmp/token.txt, then reply with its contents reversed."}]
        t0 = time.time()
        first = client.chat.completions.create(model=model, messages=convo, tools=TOOLS)
        cold = time.time() - t0
        msg = first.choices[0].message
        if not msg.tool_calls:
            return f"{model:28s} NO TOOL CALL ({cold:.1f}s): {(msg.content or '')[:80]!r}"
        call = msg.tool_calls[0]
        convo += [{"role": "assistant", "content": msg.content or "",
                   "tool_calls": [{"id": call.id, "type": "function",
                                   "function": {"name": call.function.name, "arguments": call.function.arguments}}]},
                  {"role": "tool", "name": call.function.name, "tool_call_id": call.id, "content": "abc123"}]
        t1 = time.time()
        second = client.chat.completions.create(model=model, messages=convo, tools=TOOLS)
        warm = time.time() - t1
        answer = (second.choices[0].message.content or "").replace(" ", "")
        ok = "321cba" in answer
        u = first.usage
        toks = f"{u.prompt_tokens}in/{u.completion_tokens}out" if u else "no-usage"
        return (f"{model:28s} {'OK ' if ok else 'BAD'} cold {cold:5.1f}s warm {warm:4.1f}s first-turn {toks:>14s}"
                f"  tool args={call.function.arguments}  answer={answer[:30]!r}")
    except Exception as exc:  # report, do not abort the sweep
        return f"{model:28s} ERROR {type(exc).__name__}: {str(exc)[:140]}"
    finally:
        client.close()


def main() -> int:
    models = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not models:
        print(__doc__)
        return 2
    os.environ.setdefault("HERMES_HOME", tempfile.mkdtemp(prefix="hermes-amp-live-"))
    os.environ["HERMES_AMP_THREADS"] = "keep" if "--keep-threads" in sys.argv else "delete"
    with cf.ThreadPoolExecutor(min(4, len(models))) as pool:
        for line in pool.map(check, models):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
