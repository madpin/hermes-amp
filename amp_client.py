"""OpenAI-client-shaped facade that runs Hermes turns on Amp.

One long-lived ``amp -x --stream-json --stream-json-input`` child per conversation keeps the Amp thread (and so
the provider-side prompt cache) alive between Hermes requests: the first request pays Amp's start-up (about
3 s), every later one only sends what is new (a user message, or the tool results Hermes just produced).

Each child runs a throw-away plugin agent (amp_agent.py): Hermes' system prompt is its whole prompt, it has no
tools and no MCP servers. A request whose history does not extend what the thread already holds (compaction,
edited history, a different system prompt or model) opens a fresh thread and replays the transcript.
"""

from __future__ import annotations

import collections
import hashlib
import json
import logging
import os
import queue
import shutil
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:  # loaded as a package by Hermes, or as flat modules by tests and scripts
    from . import amp_agent, amp_bridge, amp_models
except ImportError:  # pragma: no cover
    import amp_agent
    import amp_bridge
    import amp_models

logger = logging.getLogger("hermes_amp")

DEFAULT_TIMEOUT = 900.0
MARKER_BASE_URL = "process://amp"
_STOP_DONE = {"end_turn", "max_tokens", "stop_sequence", "refusal", "pause_turn"}  # anything but tool_use / None


class AmpError(RuntimeError):
    """A failed Amp request. ``stale`` marks errors a fresh thread may not have (the child died or hung up)."""

    def __init__(self, message: str, *, stale: bool = False):
        super().__init__(message)
        self.stale = stale


class _EffortRetry(Exception):
    def __init__(self, level: str):
        super().__init__(level)
        self.level = level


@dataclass
class Turn:
    text: str
    reasoning: str
    usage: dict[str, int] | None
    stop_reason: str | None


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _effective_timeout(timeout: Any) -> float:
    env = os.environ.get("HERMES_AMP_TIMEOUT", "").strip()
    if env:
        return _env_float("HERMES_AMP_TIMEOUT", DEFAULT_TIMEOUT)
    for attr in ("read", "total"):  # httpx.Timeout
        value = getattr(timeout, attr, None)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    if isinstance(timeout, (int, float)) and timeout > 0:
        return float(timeout)
    return DEFAULT_TIMEOUT


def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")


# =============================================================================================== process
class AmpProcess:
    """One ``amp`` child speaking stream-json both ways. Not thread-safe: callers serialise on the session."""

    def __init__(self, argv: list[str], cwd: Path, env: dict[str, str]):
        self.argv, self.cwd, self.env = argv, cwd, env
        self.events: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self.thread_id: str | None = None
        self._stderr: collections.deque[str] = collections.deque(maxlen=40)
        self.proc: subprocess.Popen[str] | None = None

    def start(self) -> None:
        try:
            self.proc = subprocess.Popen(
                self.argv, cwd=self.cwd, env=self.env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1, start_new_session=True,
            )
        except OSError as exc:
            raise AmpError(
                f"Could not start the Amp CLI {self.argv[0]!r}: {exc}. Install it from https://ampcode.com, run "
                "`amp login`, or set HERMES_AMP_COMMAND."
            ) from exc
        threading.Thread(target=self._pump_stdout, daemon=True, name="amp-stdout").start()
        threading.Thread(target=self._pump_stderr, daemon=True, name="amp-stderr").start()

    def _pump_stdout(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self.events.put(json.loads(line))
            except ValueError:
                logger.debug("amp: non-JSON stdout line: %.200s", line)
        self.events.put(None)  # EOF

    def _pump_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        for line in self.proc.stderr:
            self._stderr.append(line.rstrip())

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stderr_tail(self) -> str:
        return "\n".join(self._stderr)[-600:]

    def send(self, text: str) -> None:
        if not self.alive or not self.proc or not self.proc.stdin:
            raise AmpError("The Amp process is not running", stale=True)
        msg = {"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": text}]}}
        try:
            self.proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise AmpError(f"The Amp process closed its input: {exc}", stale=True) from exc

    def read_turn(self, timeout: float) -> Turn:
        """Collect assistant output until the model stops (``end_turn`` and friends)."""
        deadline = time.monotonic() + timeout
        texts: list[str] = []
        thoughts: list[str] = []
        usage: dict[str, int] = {}
        stop: str | None = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AmpError(f"Amp did not answer within {timeout:.0f}s")
            try:
                ev = self.events.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                continue
            if ev is None:
                raise AmpError(
                    "Amp exited before answering" + (f": {self.stderr_tail()}" if self.stderr_tail() else ""),
                    stale=True,
                )
            kind = ev.get("type")
            if kind == "system" and ev.get("subtype") == "init":
                self.thread_id = ev.get("session_id") or self.thread_id
            elif kind == "assistant":
                msg = ev.get("message") or {}
                for block in msg.get("content") or []:
                    if block.get("type") == "text":
                        texts.append(block.get("text", ""))
                    elif block.get("type") == "thinking":
                        thoughts.append(block.get("thinking", ""))
                for key, value in (msg.get("usage") or {}).items():
                    if isinstance(value, int) and key in (
                        "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens"
                    ):
                        usage[key] = usage.get(key, 0) + value
                stop = msg.get("stop_reason") or stop
                if msg.get("stop_reason") in _STOP_DONE:
                    return Turn("".join(texts), "\n".join(t for t in thoughts if t), usage or None, stop)
            elif kind == "result" and ev.get("is_error"):
                raise AmpError(str(ev.get("error") or "Amp reported an error"))
            elif kind == "result":  # success result before any end_turn: take what we have
                return Turn("".join(texts), "\n".join(t for t in thoughts if t), usage or None, stop)
            elif kind == "system" and ev.get("error"):
                raise AmpError(str(ev["error"]))

    def close(self, grace: float = 3.0) -> None:
        proc = self.proc
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    pass


# =============================================================================================== session
class AmpSession:
    """A warm Amp thread bound to one (model, effort, system prompt + tools) configuration."""

    def __init__(self, client: "AmpClient", cfg_key: str, spec: amp_agent.AgentSpec, *, ephemeral: bool = False):
        self.client, self.cfg_key, self.spec, self.ephemeral = client, cfg_key, spec, ephemeral
        self.lock = threading.Lock()
        self.canon: list[list[Any]] = []  # canonical form of every Hermes message the thread already holds
        self.last_used = time.monotonic()
        self.dir: Path | None = None
        self.proc: AmpProcess | None = None

    def start(self) -> None:
        self.dir = amp_agent.write_session_dir(self.client.sessions_dir, self.spec)
        argv = [
            self.client.command, "-x", "--stream-json", "--stream-json-input", "--stream-json-thinking",
            "-m", self.spec.key, "--settings-file", str(self.dir / "settings.json"),
            "--no-ide", "--no-notifications", "--no-color", "--visibility", "private",
            "--plugin-ready-timeout", "30",
        ]
        env = dict(os.environ, NO_COLOR="1")
        env.pop("AMP_SETTINGS_FILE", None)  # the private settings file must win
        self.proc = AmpProcess(argv, self.dir, env)
        self.proc.start()

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.alive

    def ask(self, text: str, timeout: float) -> Turn:
        assert self.proc
        self.proc.send(text)
        try:
            turn = self.proc.read_turn(timeout)
        except AmpError as exc:
            level = amp_agent.nearest_supported_effort(str(exc))
            if level and level != self.spec.effort:
                raise _EffortRetry(level) from exc
            raise
        self.last_used = time.monotonic()
        return turn

    def close(self) -> None:
        proc, self.proc = self.proc, None
        thread_id = proc.thread_id if proc else None
        if proc:
            proc.close()
        if self.dir:
            shutil.rmtree(self.dir, ignore_errors=True)
            self.dir = None
        self.client._retire_thread(thread_id)


# ================================================================================================ client
class AmpClient:
    """Drop-in for the OpenAI client Hermes builds for ``external_process`` providers."""

    HERMES_SKIP_TRANSPORT_WRAP = True  # a complete client already: never re-dispatch through a wire adapter
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None,
                 default_headers: dict[str, str] | None = None, command: str | None = None,
                 args: list[str] | None = None, **_: Any):
        self.api_key = api_key or "amp"
        self.base_url = base_url or MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self.command = (command or os.environ.get("HERMES_AMP_COMMAND") or os.environ.get("AMP_CLI_PATH")
                        or shutil.which("amp") or str(Path.home() / ".local" / "bin" / "amp"))
        self.sessions_dir = _hermes_home() / "cache" / "amp-sessions"
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create_chat_completion))
        self.is_closed = False
        self.max_sessions = int(_env_float("HERMES_AMP_MAX_SESSIONS", 4))
        self.idle_ttl = _env_float("HERMES_AMP_IDLE_SECONDS", 600.0)
        self.threads_policy = (os.environ.get("HERMES_AMP_THREADS") or "archive").strip().lower()
        self._effort_fix: dict[tuple[str, str], str] = {}  # (model, asked effort) -> level Amp accepted
        self._sessions: "collections.OrderedDict[str, AmpSession]" = collections.OrderedDict()
        self._mutex = threading.RLock()
        self._reaper: threading.Thread | None = None
        self._prune_stale_dirs()

    # ----------------------------------------------------------------------------------- housekeeping
    def _prune_stale_dirs(self) -> None:
        """Leftovers of a crashed run: session dirs untouched for a day."""
        try:
            for d in self.sessions_dir.iterdir():
                if d.is_dir() and time.time() - d.stat().st_mtime > 86400:
                    shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass

    def _ensure_reaper(self) -> None:
        if self._reaper and self._reaper.is_alive():
            return
        self._reaper = threading.Thread(target=self._reap_loop, daemon=True, name="amp-reaper")
        self._reaper.start()

    def _reap_loop(self) -> None:
        while not self.is_closed:
            time.sleep(30)
            now = time.monotonic()
            with self._mutex:
                for key, sess in list(self._sessions.items()):
                    idle = now - sess.last_used > self.idle_ttl
                    if (idle or not sess.alive) and not sess.lock.locked():
                        self._sessions.pop(key, None)
                        sess.close()

    def _retire_thread(self, thread_id: str | None) -> None:
        """Hide (default) or delete the Amp thread this client created, so Hermes does not litter the account."""
        if not thread_id or self.threads_policy not in ("archive", "delete"):
            return
        try:
            subprocess.Popen(
                [self.command, "threads", self.threads_policy, thread_id], stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
            )
        except OSError:
            pass

    def close(self) -> None:
        self.is_closed = True
        with self._mutex:
            sessions, self._sessions = list(self._sessions.values()), collections.OrderedDict()
        for sess in sessions:
            sess.close()

    # --------------------------------------------------------------------------------------- sessions
    def _acquire(self, cfg_key: str, spec_factory, canon: list[list[Any]]) -> tuple[AmpSession, int, bool]:
        """``(session, already_conveyed, ephemeral)`` with the session's lock HELD (release it when done).

        Reuses the warm thread when the history extends what it holds; otherwise opens a new one. If the warm
        thread is busy with another request, a private one-shot session is returned instead.
        """
        with self._mutex:
            self._ensure_reaper()
            sess = self._sessions.get(cfg_key)
            if sess is not None:
                n = len(sess.canon)
                if sess.lock.locked():  # another request is using this thread: take a private one
                    new = AmpSession(self, cfg_key, spec_factory(), ephemeral=True)
                    new.lock.acquire()
                    return new, 0, True
                if sess.alive and 0 < n < len(canon) and canon[:n] == sess.canon:
                    sess.lock.acquire()
                    self._sessions.move_to_end(cfg_key)
                    return sess, n, False
                self._sessions.pop(cfg_key, None)  # dead, or the history diverged: start over
                sess.close()
            while len(self._sessions) >= max(1, self.max_sessions):
                victim = next(iter(self._sessions.values()))
                if victim.lock.locked():
                    break
                self._sessions.popitem(last=False)
                victim.close()
            new = AmpSession(self, cfg_key, spec_factory())
            new.lock.acquire()
            self._sessions[cfg_key] = new
            return new, 0, False

    # ---------------------------------------------------------------------------------------- request
    def _effort(self, kwargs: dict[str, Any]) -> str | None:
        value = kwargs.get("reasoning_effort")
        extra = kwargs.get("extra_body") or {}
        if not value and isinstance(extra, dict):
            reasoning = extra.get("reasoning") or {}
            value = reasoning.get("effort") if isinstance(reasoning, dict) else None
        value = value or os.environ.get("HERMES_AMP_EFFORT") or None
        return str(value).lower() if value and str(value).lower() in amp_agent.EFFORT_ORDER else None

    def _create_chat_completion(self, *, model: str | None = None, messages: list[dict[str, Any]] | None = None,
                                timeout: Any = None, tools: list[dict[str, Any]] | None = None,
                                tool_choice: Any = None, stream: bool = False, **kwargs: Any) -> Any:
        system_text, convo = amp_bridge.split_system(list(messages or []))
        if not convo:
            raise AmpError("Nothing to send: the conversation has no user message")
        try:
            target = amp_models.resolve(model, amp_models.catalog(self.command))
        except amp_models.UnknownModel as exc:
            raise AmpError(str(exc)) from exc
        instructions = amp_bridge.build_instructions(system_text, tools, tool_choice)
        effort = self._effort(kwargs)
        if effort:
            effort = self._effort_fix.get((target.extends, effort), effort)  # learned from an earlier rejection
        logger.debug("amp request: model=%s effort=%s tools=%d messages=%d stream=%s",
                     target.extends, effort, len(tools or []), len(messages or []), stream)
        canon = [amp_bridge.canonical(m) for m in convo]
        used_ids = amp_bridge.used_tool_ids(convo)
        timeout_s = _effective_timeout(timeout)

        turn: Turn | None = None
        for attempt in range(3):  # 1: warm thread; 2: cold retry after a stale one; 3: effort downgrade
            cfg_key = hashlib.sha256(json.dumps([target.extends, effort, instructions]).encode()).hexdigest()
            sess, done, ephemeral = self._acquire(
                cfg_key,
                lambda: amp_agent.AgentSpec(amp_agent.mode_key(uuid.uuid4().hex), target.extends, instructions, effort),
                canon,
            )
            try:
                if not sess.alive:
                    sess.start()
                text = amp_bridge.render_turn(convo[done:], fresh_history=(done == 0 and len(convo) > 1))
                turn = sess.ask(text, timeout_s)
                break
            except _EffortRetry as retry:
                logger.info("amp: %s does not support effort %r, using %r", target.extends, effort, retry.level)
                if effort:
                    self._effort_fix[(target.extends, effort)] = retry.level
                effort = retry.level
                self._drop(sess)
                continue
            except AmpError as exc:
                self._drop(sess)
                if exc.stale and attempt == 0 and done:
                    logger.info("amp: warm thread went away (%s), retrying on a fresh one", exc)
                    continue
                raise
            except BaseException:
                self._drop(sess)
                raise
            finally:
                sess.lock.release()
        else:
            raise AmpError("Amp rejected the reasoning effort repeatedly")
        assert turn is not None

        calls, cleaned = amp_bridge.extract_tool_calls(turn.text, used_ids)
        reply = {"role": "assistant", "content": cleaned, "tool_calls": calls}
        if ephemeral:
            self._drop(sess)
        else:
            sess.canon = canon + [amp_bridge.canonical(reply)]
        return self._completion(target, turn, cleaned, calls, stream)

    def _drop(self, sess: AmpSession) -> None:
        with self._mutex:
            if self._sessions.get(sess.cfg_key) is sess:
                self._sessions.pop(sess.cfg_key, None)
        sess.close()

    @staticmethod
    def _completion(target: amp_models.Target, turn: Turn, cleaned: str, calls: list[Any], stream: bool) -> Any:
        usage = None
        if turn.usage:
            cached = turn.usage.get("cache_read_input_tokens", 0)
            prompt = turn.usage.get("input_tokens", 0) + turn.usage.get("cache_creation_input_tokens", 0) + cached
            out = turn.usage.get("output_tokens", 0)
            usage = SimpleNamespace(prompt_tokens=prompt, completion_tokens=out, total_tokens=prompt + out,
                                    prompt_tokens_details=SimpleNamespace(cached_tokens=cached))
        message = SimpleNamespace(content=cleaned, tool_calls=calls, reasoning=turn.reasoning or None,
                                  reasoning_content=turn.reasoning or None, reasoning_details=None)
        finish = "tool_calls" if calls else ("length" if turn.stop_reason == "max_tokens" else "stop")
        completion = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish)],
                                     usage=usage, model=target.display)
        if stream:
            try:
                from agent.acp_openai_bridge import completion_to_stream_chunks

                return completion_to_stream_chunks(completion)
            except ImportError:  # pragma: no cover
                return completion
        return completion

    # ------------------------------------------------------------------------------------------ models
    def list_models(self, *, timeout_seconds: float = 30.0) -> list[str]:
        return amp_models.picker_ids(amp_models.catalog(self.command))
