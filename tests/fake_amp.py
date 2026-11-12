#!/usr/bin/env python3
"""A stand-in for the `amp` CLI, just enough of it for the tests: stream-json in and out, scripted by markers.

Behaviour is chosen by words in the user message:
  CALL_TOOL  reply with a flat-shape <tool_call> block      THINK  add a thinking block
  DIE        exit without answering (a stale warm thread)   HANG   never answer
  SLOW       answer after 1.5 s                              ERR    emit an error result and exit
  EFFORT     if the plugin sets an effort other than low/max, fail like Amp does for unsupported efforts
Everything the process sees is appended as JSON lines to $FAKE_AMP_LOG.
"""
import json
import os
import re
import sys
import time

LOG = os.environ.get("FAKE_AMP_LOG")


def log(**kw):
    if LOG:
        with open(LOG, "a") as fh:
            fh.write(json.dumps(dict(pid=os.getpid(), **kw)) + "\n")


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main(argv):
    if "threads" in argv:
        log(event="threads", argv=argv)
        return 0
    if "nonexistent-model-xyz" in argv:  # the catalog probe: Amp's ZodError, JSON inside JSON
        log(event="probe")
        values = ["puck", "low", "medium", "high", "ultra", "anthropic/claude-opus-5-5", "anthropic/claude-haiku-5-5",
                  "openai/gpt-6.1-terra", "zhipuai/glm-5.3-flash", "moonshotai/kimi-k3", "xai/grok-4.7"]
        inner = json.dumps([{"code": "invalid_union", "errors": [[{"code": "invalid_value", "values": values}]]}])
        print("Error: Unexpected error inside Amp CLI.\nCause: ThreadClientConnectionError: 400 "
              + json.dumps({"error": {"name": "ZodError", "message": inner}}))
        return 1
    key = argv[argv.index("-m") + 1]
    cwd = os.getcwd()
    plugin = os.path.join(cwd, ".amp", "plugins", "agent.ts")
    plugin_text = open(plugin).read() if os.path.exists(plugin) else None
    log(event="start", argv=argv, cwd=cwd, plugin=plugin_text)
    sid = f"T-fake-{os.getpid()}"
    first = True
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        text = json.loads(raw)["message"]["content"][0]["text"]
        log(event="message", text=text)
        if first:
            emit({"type": "system", "subtype": "init", "cwd": cwd, "session_id": sid, "tools": [],
                  "mcp_servers": [], "agent_mode": key, "subagents": []})
            first = False
        emit({"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": text}]},
              "parent_tool_use_id": None, "session_id": sid})
        if "DIE" in text:
            return 3
        if "HANG" in text:
            time.sleep(60)
        if "SLOW" in text:
            time.sleep(1.5)
        if "ERR" in text:
            emit({"type": "result", "subtype": "error_during_execution", "is_error": True, "num_turns": 0,
                  "error": "Claude Fable is disabled for your workspace.", "session_id": sid})
            return 1
        m = re.search(r'reasoningEffort: "(\w+)"', plugin_text or "")
        if "EFFORT" in text and (not m or m.group(1) not in ("low", "max")):
            asked = m.group(1) if m else "medium"
            emit({"type": "result", "subtype": "error_during_execution", "is_error": True, "num_turns": 0,
                  "error": f'Reasoning effort "{asked}" is not supported by model "fake/model". '
                           "Available efforts: low, max", "session_id": sid})
            return 1
        content = []
        if "THINK" in text:
            content.append({"type": "thinking", "thinking": "pondering the question"})
        if "CALL_TOOL" in text:
            content.append({"type": "text", "text": 'I will look. <tool_call>{"name": "read_file", '
                            '"arguments": {"path": "/tmp/x"}}</tool_call>'})
        else:
            content.append({"type": "text", "text": "echo: " + text[-60:]})
        emit({"type": "assistant", "message": {"type": "message", "role": "assistant", "content": content,
              "stop_reason": "end_turn", "usage": {"input_tokens": 3, "cache_creation_input_tokens": 100,
                                                    "cache_read_input_tokens": 50, "output_tokens": 7}},
              "parent_tool_use_id": None, "session_id": sid})
    emit({"type": "result", "subtype": "success", "is_error": False, "num_turns": 1, "result": "", "session_id": sid})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
