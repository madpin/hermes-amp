"""Which models Amp can run, and how a Hermes model string maps onto one.

``amp -m`` takes a built-in mode or a ``provider/model`` id. ``amp --help`` only names some modes, but the
server validates the value and, on an unknown one, answers with an error that lists every accepted value.
Triggering that error is free (the thread is never created, no model runs), so it is the live catalog.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

# Snapshot of the server's list on 2026-11-12, used when the live probe fails. "Accepted" does not mean
# "enabled for your workspace": workspace admins can switch models (and modes) off.
PINNED = (
    "anthropic/claude-opus-5-5", "anthropic/claude-opus-5", "anthropic/claude-sonnet-5-5",
    "anthropic/claude-sonnet-5", "anthropic/claude-haiku-5-5", "anthropic/claude-fable-5-2",
    "anthropic/claude-fable-5-1", "anthropic/claude-mythos-5-2",
    "openai/gpt-6.1-sol", "openai/gpt-6.1-terra", "openai/gpt-6-sol", "openai/gpt-6-astra",
    "openai/gpt-6-luna", "openai/gpt-5.6-sol", "openai/gpt-5.6-terra", "openai/gpt-5.6-luna",
    "google-vertex/gemini-4-carbon", "google-vertex/gemini-4-argon", "google-vertex/gemini-3.8-flash",
    "xai/grok-4.7", "xai/grok-4.6", "xai/grok-4.5",
    "moonshotai/kimi-k3", "moonshotai/kimi-k3.1-pro",
    "zhipuai/glm-5.3", "zhipuai/glm-5.3-flash",
    "alibaba/qwen3.8-max", "deepseek/deepseek-v4.1-flash", "minimax/MiniMax-M3.1-Flash",
    "meta/muse-spark-1.3", "meta/muse-wonder-1.0", "nvidia/nemotron-3-ultra-550b-a55b",
)

# Built-in modes a plugin agent can extend. The other server-side aliases (puck, nostromo, rush, smart,
# deep, large, project-puck) are internal or deprecated and cannot be extended.
MODES = ("low", "medium", "high", "ultra")

DEFAULT_MODEL = "anthropic/claude-sonnet-5-5"
# Placeholders Hermes may pass when the user just picked the provider.
_PLACEHOLDERS = {"", "amp", "amp-cli", "ampcode", "default"}

# Curated, cheap-to-run picks shown when the live catalog cannot be fetched.
FALLBACK_BARE = (
    "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "gpt-6.1-sol", "gpt-6.1-terra",
    "gemini-4-argon", "grok-4.7", "kimi-k3", "glm-5.3-flash", "high", "medium", "low",
)


@dataclass(frozen=True)
class Target:
    """What ``-m`` should run: ``extends`` goes into the plugin agent."""

    extends: str
    display: str


class UnknownModel(ValueError):
    pass


def bare(model_id: str) -> str:
    """``anthropic/claude-opus-5-5`` -> ``claude-opus-5-5``."""
    return model_id.split("/", 1)[-1]


def _cache_file() -> Path:
    home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(home) / "cache" / "amp-models.json"


def parse_accepted_values(output: str) -> list[str]:
    """Pull the accepted ``agentMode`` values out of Amp's validation error (JSON inside JSON)."""
    text = output.replace("\\", "")  # un-escape the nested JSON so the quotes are plain
    m = re.search(r'"values":\s*\[(.*?)\]', text, re.S)
    return re.findall(r'"([^"]+)"', m.group(1)) if m else []


def probe_catalog(command: str, timeout: float = 60.0) -> list[str]:
    """Live ``provider/model`` ids from the server. Raises RuntimeError when the probe fails."""
    cwd = tempfile.mkdtemp(prefix="amp-models-", dir=os.environ.get("TMPDIR"))
    try:
        proc = subprocess.run(
            [command, "-x", "x", "-m", "nonexistent-model-xyz", "--no-ide", "--no-notifications"],
            cwd=cwd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"could not run {command!r}: {exc}") from exc
    finally:
        try:
            os.rmdir(cwd)
        except OSError:
            pass
    values = parse_accepted_values(proc.stdout + proc.stderr)
    models = [v for v in values if "/" in v]
    if not models:
        raise RuntimeError("Amp did not list its models (format changed, or not logged in: run `amp login`)")
    return models


def catalog(command: str | None = None, *, max_age: float = 24 * 3600, refresh: bool = False) -> list[str]:
    """Models Amp accepts: on-disk cache (24h), else the live probe, else the pinned snapshot."""
    path = _cache_file()
    if not refresh:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if time.time() - float(data["fetched_at"]) < max_age and data["models"]:
                return list(data["models"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
    if command:
        try:
            models = probe_catalog(command)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"fetched_at": time.time(), "models": models}), encoding="utf-8")
            except OSError:
                pass
            return models
        except RuntimeError:
            pass
    return list(PINNED)


def resolve(requested: str | None, models: list[str] | None = None) -> Target:
    """Map a Hermes model string to a Target.

    Accepts a full id (``anthropic/claude-opus-5-5``), a bare name (``claude-opus-5-5``, if unambiguous),
    one of the built-in modes, or a placeholder (-> DEFAULT_MODEL). Anything else raises UnknownModel
    instead of silently running something the user did not pick.
    """
    models = models or list(PINNED)
    want = (requested or "").strip()
    if want.lower() in _PLACEHOLDERS:
        return Target(DEFAULT_MODEL, bare(DEFAULT_MODEL))
    if want.lower() in MODES:
        return Target(want.lower(), f"amp-{want.lower()}")
    if want in models:
        return Target(want, bare(want))
    hits = [m for m in models if bare(m) == want or bare(m).lower() == want.lower()]
    if len(hits) == 1:
        return Target(hits[0], bare(hits[0]))
    if len(hits) > 1:
        raise UnknownModel(f"{want!r} is ambiguous: {', '.join(hits)}. Use the full provider/model id.")
    close = [bare(m) for m in models if want.lower() in m.lower()][:5]
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    raise UnknownModel(f"Amp does not list model {requested!r}.{hint} Run `python amp_models.py` to see the list.")


def picker_ids(models: list[str]) -> list[str]:
    """Names shown in Hermes' picker: bare ids (full id only where the bare name collides) plus the modes."""
    counts: dict[str, int] = {}
    for m in models:
        counts[bare(m)] = counts.get(bare(m), 0) + 1
    out = [bare(m) if counts[bare(m)] == 1 else m for m in models]
    return out + [m for m in ("high", "medium", "low") if m not in out]


if __name__ == "__main__":  # python amp_models.py  -> print the live catalog
    cmd = os.environ.get("HERMES_AMP_COMMAND") or os.environ.get("AMP_CLI_PATH") or "amp"
    for item in catalog(cmd, refresh=True):
        print(item)
