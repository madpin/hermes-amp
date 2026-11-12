"""Tests for the Amp provider. Run (Hermes on the path for the provider registry):

    PYTHONPATH=~/.hermes/hermes-agent uv run --no-project --python 3.14 --with openai --with pytest \
        --with ruamel.yaml python -m pytest tests -q -o addopts=

Client tests drive tests/fake_amp.py, a scripted stand-in for the CLI. Live tests (real `amp`, spends a few
credits) run only with AMP_LIVE=1.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import stat
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import amp_agent  # noqa: E402
import amp_bridge  # noqa: E402
import amp_client  # noqa: E402
import amp_models  # noqa: E402

FAKE = ROOT / "tests" / "fake_amp.py"


# ------------------------------------------------------------------------------------------- helpers
def user(text):
    return {"role": "user", "content": text}


def assistant(text="", calls=None):
    msg = {"role": "assistant", "content": text}
    if calls:
        msg["tool_calls"] = [
            {"id": cid, "type": "function", "function": {"name": name, "arguments": args}} for cid, name, args in calls
        ]
    return msg


def tool_result(name, text, cid="amp_call_1"):
    return {"role": "tool", "name": name, "tool_call_id": cid, "content": text}


TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file",
                                           "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]


@pytest.fixture
def fake(tmp_path, monkeypatch):
    FAKE.chmod(FAKE.stat().st_mode | stat.S_IXUSR)
    log = tmp_path / "fake.log"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("FAKE_AMP_LOG", str(log))
    monkeypatch.setenv("HERMES_AMP_THREADS", "keep")
    monkeypatch.delenv("HERMES_AMP_EFFORT", raising=False)
    monkeypatch.delenv("HERMES_AMP_TIMEOUT", raising=False)
    client = amp_client.AmpClient(command=str(FAKE))
    yield client, log
    client.close()


def events(log: Path, kind: str | None = None):
    if not log.exists():
        return []
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    return [r for r in rows if kind is None or r["event"] == kind]


def ask(client, messages, **kw):
    return client.chat.completions.create(model="claude-haiku-5-5", messages=messages, tools=kw.pop("tools", None), **kw)


# ============================================================================================== agent
def test_mode_key_fits_amp_limit():
    key = amp_agent.mode_key("A1B2-c3d4" * 6)
    assert len(key) <= 24 and re.fullmatch(r"hermes-[a-z0-9]+", key)


def test_plugin_is_bare_and_escapes_any_prompt():
    nasty = 'He said "hi"\nback\\slash </script> unicode é—✓ line\u2028sep ${not_a_template} `tick`'
    src = amp_agent.render_plugin(amp_agent.AgentSpec("hermes-abc", "anthropic/claude-opus-5-5", nasty, "high"))
    assert 'omitDefaultSystemPrompt: true' in src
    assert 'exclude: ["code_exec", "tool_search"]' in src and "include: []" in src
    assert '// @amp-agent-mode {"key": "hermes-abc", "label": "hermes-abc"}' in src
    assert 'reasoningEffort: "high"' in src
    literal = re.search(r"instructions: (\".*\"),\n", src).group(1)
    assert json.loads(literal) == nasty  # round-trips exactly
    assert "\u2028" not in src  # would end a JS string literal early if left raw
    assert "reasoningEffort" not in amp_agent.render_plugin(amp_agent.AgentSpec("k", "high", "x"))


def test_session_dir_has_plugin_and_private_settings(tmp_path):
    d = amp_agent.write_session_dir(tmp_path, amp_agent.AgentSpec("hermes-z", "low", "sys"))
    assert (d / ".amp" / "plugins" / "agent.ts").read_text().startswith("// @amp-agent-mode")
    assert json.loads((d / "settings.json").read_text()) == amp_agent.SETTINGS
    assert "amp.mcpServers" not in (d / "settings.json").read_text()


@pytest.mark.parametrize("asked,avail,want", [
    ("medium", "low, high, max", "low"),   # tie: earlier wins
    ("xhigh", "low, high", "high"),
    ("high", "low, max", "low"),
    ("none", "low, high", "low"),
])
def test_nearest_supported_effort(asked, avail, want):
    err = f'Reasoning effort "{asked}" is not supported by model "zhipuai/glm-5.3-flash". Available efforts: {avail}'
    assert amp_agent.nearest_supported_effort(err) == want
    assert amp_agent.nearest_supported_effort("some other failure") is None


# ============================================================================================= models
def zod_error(values):
    inner = json.dumps([{"errors": [[{"code": "invalid_value", "values": values}]]}])
    return "Cause: 400 " + json.dumps({"error": {"name": "ZodError", "message": inner}})


def test_parse_accepted_values_from_nested_json():
    vals = ["puck", "low", "anthropic/claude-opus-5-5", "openai/gpt-6.1-sol", "minimax/MiniMax-M3.1-Flash"]
    assert amp_models.parse_accepted_values(zod_error(vals)) == vals
    assert amp_models.parse_accepted_values("garbage") == []


def test_resolve_accepts_full_bare_modes_and_placeholders():
    models = list(amp_models.PINNED)
    assert amp_models.resolve("anthropic/claude-opus-5-5", models).extends == "anthropic/claude-opus-5-5"
    assert amp_models.resolve("claude-opus-5-5", models).extends == "anthropic/claude-opus-5-5"
    assert amp_models.resolve("CLAUDE-OPUS-5-5", models).extends == "anthropic/claude-opus-5-5"
    assert amp_models.resolve("high", models).extends == "high"
    assert amp_models.resolve("amp", models).extends == amp_models.DEFAULT_MODEL
    assert amp_models.resolve(None, models).extends == amp_models.DEFAULT_MODEL


def test_resolve_rejects_unknown_and_ambiguous_instead_of_guessing():
    with pytest.raises(amp_models.UnknownModel, match="opus"):
        amp_models.resolve("opus-9", ["anthropic/claude-opus-5-5", "anthropic/claude-opus-5"])
    with pytest.raises(amp_models.UnknownModel, match="did you mean|Did you mean"):
        amp_models.resolve("claude-opus", ["anthropic/claude-opus-5-5"])
    with pytest.raises(amp_models.UnknownModel, match="ambiguous"):
        amp_models.resolve("shared", ["a/shared", "b/shared"])


def test_picker_ids_prefer_bare_names_and_add_modes():
    ids = amp_models.picker_ids(["anthropic/claude-opus-5-5", "a/shared", "b/shared"])
    assert "claude-opus-5-5" in ids and "a/shared" in ids and "b/shared" in ids and "shared" not in ids
    assert ids[-3:] == ["high", "medium", "low"]


def test_catalog_probes_once_then_uses_cache(fake):
    client, log = fake
    first = amp_models.catalog(str(FAKE))
    assert "anthropic/claude-opus-5-5" in first and "puck" not in first  # bare aliases are not models
    again = amp_models.catalog(str(FAKE))
    assert again == first and len(events(log, "probe")) == 1


def test_catalog_falls_back_to_pinned_when_probe_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert amp_models.catalog("/nonexistent/amp") == list(amp_models.PINNED)


# ============================================================================================= bridge
def test_extract_both_shapes_and_prose_mentions():
    text = (
        'To act I emit a <tool_call> block now: '
        '<tool_call>{"id": "c1", "type": "function", "function": {"name": "a", "arguments": "{\\"x\\": 1}"}}</tool_call>'
        ' and also <tool_call>{"name": "b", "arguments": {"y": 2}}</tool_call> done'
    )
    calls, cleaned = amp_bridge.extract_tool_calls(text)
    assert [(c.id, c.function.name) for c in calls] == [("c1", "a"), ("amp_call_2", "b")]
    assert json.loads(calls[0].function.arguments) == {"x": 1}
    assert json.loads(calls[1].function.arguments) == {"y": 2}  # dict arguments become a JSON string
    assert "<tool_call>" in cleaned.split("now:")[0] + "<tool_call>"  # the prose mention survives
    assert '"name": "b"' not in cleaned


def test_extract_ignores_malformed_and_makes_ids_unique():
    assert amp_bridge.extract_tool_calls("<tool_call>{not json}</tool_call>")[0] == []
    assert amp_bridge.extract_tool_calls("no calls here")[1] == "no calls here"
    text = '<tool_call>{"id": "call_1", "name": "a", "arguments": {}}</tool_call>'
    calls, _ = amp_bridge.extract_tool_calls(text, used_ids={"call_1"})
    assert calls[0].id == "call_1_2"


def test_render_turn_shapes():
    assert amp_bridge.render_turn([user("hello")]) == "hello"
    delta = amp_bridge.render_turn([tool_result("read_file", "file body")])
    assert delta == "Tool result (read_file):\nfile body"
    hist = amp_bridge.render_turn([user("q"), assistant("a", [("c1", "t", "{}")]), user("next")], fresh_history=True)
    assert hist.startswith("Conversation so far:") and '"name": "t"' in hist and hist.endswith("continue as the Assistant.")


def test_instructions_carry_system_prompt_and_tools_only_when_given():
    with_tools = amp_bridge.build_instructions("Be kind.", TOOLS, None)
    assert "Be kind." in with_tools and "read_file" in with_tools and "<tool_call>" in with_tools
    assert "Available tools" not in amp_bridge.build_instructions("Be kind.", None, None)


def test_canonical_ignores_whitespace_ids_and_arg_formatting():
    ours = {"role": "assistant", "content": "I'll look.", "tool_calls": [
        amp_bridge.build_tool_call("amp_call_1", "read_file", '{"path":"/x"}')]}
    theirs = assistant("I'll   look.\n", [("different-id", "read_file", '{ "path": "/x" }')])
    assert amp_bridge.canonical(ours) == amp_bridge.canonical(theirs)
    assert amp_bridge.canonical(assistant("other")) != amp_bridge.canonical(theirs)


def test_split_system_takes_leading_system_and_developer_messages():
    sys_text, rest = amp_bridge.split_system([{"role": "system", "content": "A"}, {"role": "developer", "content": "B"},
                                              user("hi"), {"role": "system", "content": "late"}])
    assert sys_text == "A\n\nB" and [m["role"] for m in rest] == ["user", "system"]


# ============================================================================================= client
def test_single_turn_spawns_a_bare_agent_with_hermes_prompt(fake):
    client, log = fake
    out = ask(client, [{"role": "system", "content": "You are Hermes. Secret: kiwi."}, user("hello there")], tools=TOOLS)
    msg = out.choices[0].message
    assert msg.content == "echo: hello there" and not msg.tool_calls and out.choices[0].finish_reason == "stop"
    assert out.model == "claude-haiku-5-5"
    u = out.usage
    assert (u.prompt_tokens, u.completion_tokens, u.total_tokens, u.prompt_tokens_details.cached_tokens) == (153, 7, 160, 50)
    start = events(log, "start")[0]
    argv = start["argv"]
    for flag in ("-x", "--stream-json", "--stream-json-input", "--stream-json-thinking", "--no-ide", "--settings-file"):
        assert flag in argv
    assert argv[argv.index("--visibility") + 1] == "private"
    plugin = start["plugin"]
    assert "omitDefaultSystemPrompt: true" in plugin and 'extends: "anthropic/claude-haiku-5-5"' in plugin
    assert "Secret: kiwi." in plugin and "read_file" in plugin  # Hermes' prompt and tool schema are the instructions
    assert Path(start["cwd"]).parent == client.sessions_dir  # private cwd, not the user's repo


def test_second_turn_reuses_the_process_and_sends_only_the_delta(fake):
    client, log = fake
    convo = [{"role": "system", "content": "S"}, user("first question")]
    first = ask(client, convo)
    convo += [{"role": "assistant", "content": first.choices[0].message.content}, user("second question")]
    second = ask(client, convo)
    assert second.choices[0].message.content == "echo: second question"
    assert len(events(log, "start")) == 1  # one process
    texts = [e["text"] for e in events(log, "message")]
    assert texts == ["first question", "second question"]  # the second message is just the new turn


def test_tool_call_roundtrip(fake):
    client, log = fake
    convo = [{"role": "system", "content": "S"}, user("please CALL_TOOL")]
    first = ask(client, convo, tools=TOOLS)
    m = first.choices[0].message
    assert first.choices[0].finish_reason == "tool_calls"
    assert [(c.function.name, json.loads(c.function.arguments)) for c in m.tool_calls] == [("read_file", {"path": "/tmp/x"})]
    assert m.content == "I will look."
    # Hermes stores the assistant message and appends the tool result
    convo += [assistant(m.content, [(m.tool_calls[0].id, "read_file", m.tool_calls[0].function.arguments)]),
              tool_result("read_file", "contents of x")]
    second = ask(client, convo, tools=TOOLS)
    assert second.choices[0].message.content.startswith("echo: ")
    assert len(events(log, "start")) == 1
    assert [e["text"] for e in events(log, "message")][1] == "Tool result (read_file):\ncontents of x"


def test_diverged_history_opens_a_fresh_thread_and_replays(fake):
    client, log = fake
    a = ask(client, [user("alpha")])
    ask(client, [user("alpha"), {"role": "assistant", "content": a.choices[0].message.content}, user("beta")])
    # Hermes compacted / edited earlier history: the prefix no longer matches
    ask(client, [user("alpha EDITED"), {"role": "assistant", "content": "x"}, user("gamma")])
    pids = {e["pid"] for e in events(log, "start")}
    assert len(pids) == 2
    last = events(log, "message")[-1]["text"]
    assert last.startswith("Conversation so far:") and "alpha EDITED" in last and "gamma" in last


def test_stale_warm_thread_is_replaced_transparently(fake):
    client, log = fake
    convo = [user("one")]
    first = ask(client, convo)
    convo += [{"role": "assistant", "content": first.choices[0].message.content}, user("two DIE")]
    # the warm child exits without answering (DIE); the retry opens a new thread, replays, and DIE kills that one too
    with pytest.raises(amp_client.AmpError, match="exited"):
        ask(client, convo)
    assert len({e["pid"] for e in events(log, "start")}) == 2  # it did retry on a fresh process
    # without the marker the same conversation succeeds on a fresh thread
    convo[-1] = user("two")
    out = ask(client, convo)
    assert out.choices[0].message.content.endswith("continue as the Assistant.")
    sent = events(log, "message")[-1]["text"]
    assert sent.startswith("Conversation so far:") and "one" in sent and "two" in sent


def test_error_result_surfaces_amps_message_and_drops_the_session(fake):
    client, log = fake
    with pytest.raises(amp_client.AmpError, match="Claude Fable is disabled for your workspace"):
        ask(client, [user("please ERR")])
    assert not client._sessions


def test_unsupported_effort_downgrades_and_retries(fake, monkeypatch):
    client, log = fake
    monkeypatch.setenv("HERMES_AMP_EFFORT", "high")
    out = ask(client, [user("EFFORT please")])
    assert out.choices[0].message.content.startswith("echo:")
    plugins = [e["plugin"] for e in events(log, "start")]
    assert 'reasoningEffort: "high"' in plugins[0] and 'reasoningEffort: "low"' in plugins[1]


def test_effort_downgrade_is_remembered_for_the_next_conversation(fake, monkeypatch):
    client, log = fake
    monkeypatch.setenv("HERMES_AMP_EFFORT", "high")
    ask(client, [user("EFFORT one")])
    starts_after_first = len(events(log, "start"))  # failed attempt + successful retry
    ask(client, [user("EFFORT two, an unrelated conversation")])
    assert len(events(log, "start")) == starts_after_first + 1  # no second failed spawn
    assert 'reasoningEffort: "low"' in events(log, "start")[-1]["plugin"]


def test_session_files_are_private(fake):
    client, log = fake
    ask(client, [user("hello")])
    sess = next(iter(client._sessions.values()))
    plugin = sess.dir / ".amp" / "plugins" / "agent.ts"
    assert oct(plugin.stat().st_mode & 0o777) == "0o600" and oct(sess.dir.stat().st_mode & 0o777) == "0o700"


def test_effort_comes_from_reasoning_effort_or_extra_body(fake):
    client, log = fake
    ask(client, [user("a")], reasoning_effort="low")
    ask(client, [user("b")], extra_body={"reasoning": {"effort": "max"}})
    plugins = [e["plugin"] for e in events(log, "start")]
    assert 'reasoningEffort: "low"' in plugins[0] and 'reasoningEffort: "max"' in plugins[1]


def test_thinking_becomes_reasoning(fake):
    client, _ = fake
    msg = ask(client, [user("THINK about it")]).choices[0].message
    assert msg.reasoning == msg.reasoning_content == "pondering the question"


def test_timeout_raises_and_kills_the_child(fake):
    client, log = fake
    t0 = time.time()
    with pytest.raises(amp_client.AmpError, match="did not answer"):
        ask(client, [user("HANG")], timeout=1.5)
    assert time.time() - t0 < 15 and not client._sessions


def test_busy_session_gets_a_private_one_shot_thread(fake):
    client, log = fake
    first = ask(client, [user("warm up")])
    convo = [user("warm up"), {"role": "assistant", "content": first.choices[0].message.content}, user("SLOW answer")]
    results = {}

    def slow():
        results["slow"] = ask(client, list(convo)).choices[0].message.content

    th = threading.Thread(target=slow)
    th.start()
    time.sleep(0.5)  # the slow request now holds the warm thread
    results["other"] = ask(client, [user("independent question")]).choices[0].message.content
    th.join(10)
    assert results["slow"].startswith("echo:") and results["other"] == "echo: independent question"
    assert len({e["pid"] for e in events(log, "start")}) == 2


def test_unknown_model_fails_loudly(fake):
    client, _ = fake
    with pytest.raises(amp_client.AmpError, match="does not list model"):
        client.chat.completions.create(model="no-such-model", messages=[user("hi")])


def test_close_stops_children_removes_dirs_and_archives_threads(tmp_path, monkeypatch):
    FAKE.chmod(FAKE.stat().st_mode | stat.S_IXUSR)
    log = tmp_path / "fake.log"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("FAKE_AMP_LOG", str(log))
    monkeypatch.setenv("HERMES_AMP_THREADS", "archive")
    client = amp_client.AmpClient(command=str(FAKE))
    ask(client, [user("hi")])
    sess = next(iter(client._sessions.values()))
    proc, workdir = sess.proc.proc, sess.dir
    assert workdir.exists()
    client.close()
    assert proc.poll() is not None and not workdir.exists() and client.is_closed
    deadline = time.time() + 5
    while time.time() < deadline and not events(log, "threads"):
        time.sleep(0.1)
    arch = events(log, "threads")
    assert arch and arch[0]["argv"][:2] == ["threads", "archive"] and arch[0]["argv"][2].startswith("T-fake-")


def test_stream_true_returns_chunks_with_content_and_usage(fake):
    pytest.importorskip("agent.acp_openai_bridge")
    client, _ = fake
    chunks = client.chat.completions.create(model="claude-haiku-5-5", messages=[user("hi")], stream=True)
    assert chunks[0].choices[0].delta.content == "echo: hi" and chunks[-1].usage.completion_tokens == 7


def test_bad_command_gives_an_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    client = amp_client.AmpClient(command="/nonexistent/amp")
    try:
        with pytest.raises(amp_client.AmpError, match="Could not start the Amp CLI"):
            client.chat.completions.create(model="claude-haiku-5-5", messages=[user("hi")])
    finally:
        client.close()


# ========================================================================================== provider
def _load_as_package(name="amp_pkg_under_test"):
    spec = importlib.util.spec_from_file_location(name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_provider_registers_as_an_external_process_profile():
    pytest.importorskip("providers")
    mod = _load_as_package()
    assert mod.amp.name == "amp"
    from providers import get_provider_profile

    # (may resolve to a copy installed under $HERMES_HOME/plugins: assert on fields, not identity)
    prof = get_provider_profile("amp")
    assert prof.name == "amp" and prof.auth_type == "external_process" and prof.process_command == "amp"
    assert get_provider_profile("ampcode") is prof and get_provider_profile("amp-cli") is prof
    assert prof.supports_streaming is False and prof.base_url == "process://amp"
    client = prof.create_client(command=str(FAKE))  # package-relative imports worked if this builds
    assert type(client).__name__ == "AmpClient" and client.HERMES_SKIP_TRANSPORT_WRAP and client.command == str(FAKE)
    client.close()
    assert prof.get_model_context_length("anything") == 200_000
    assert "claude-opus-5-5" in prof.fetch_models()


# ================================================================================================ live
@pytest.mark.skipif(os.environ.get("AMP_LIVE") != "1", reason="spends Amp credits; set AMP_LIVE=1")
def test_live_roundtrip_with_a_tool_and_a_warm_second_turn(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_AMP_THREADS", "delete")
    client = amp_client.AmpClient()
    model = os.environ.get("AMP_LIVE_MODEL", "claude-haiku-5-5")
    try:
        sys_msg = {"role": "system", "content": "You are Hermes. The codeword is pelican-7."}
        convo = [sys_msg, user("Read the file /tmp/token.txt using the tool and reply with its contents reversed.")]
        t0 = time.time()
        first = client.chat.completions.create(model=model, messages=convo, tools=TOOLS)
        cold = time.time() - t0
        m = first.choices[0].message
        assert m.tool_calls and m.tool_calls[0].function.name == "read_file", m.content
        convo += [assistant(m.content, [(m.tool_calls[0].id, "read_file", m.tool_calls[0].function.arguments)]),
                  tool_result("read_file", "abc123")]
        t1 = time.time()
        second = client.chat.completions.create(model=model, messages=convo, tools=TOOLS)
        warm = time.time() - t1
        assert "321cba" in second.choices[0].message.content.replace(" ", "")
        print(f"cold {cold:.1f}s warm {warm:.1f}s")
        assert warm < cold
    finally:
        client.close()
