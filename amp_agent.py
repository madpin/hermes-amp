"""Generate the throw-away Amp plugin agent that turns Amp into a bare model.

Amp's CLI always wraps a model in its own system prompt, ~26 tools and every MCP server the user has
configured (about 31k tokens before the first word). A *plugin agent* can opt out of all of it:

* ``omitDefaultSystemPrompt: true``  -> ``instructions`` is the whole system prompt (Hermes' own),
* ``tools: {include: [], exclude: [code_exec, tool_search]}`` -> zero tools; ``code_exec`` and
  ``tool_search`` are added to every agent unless excluded by name, and they reach every MCP server,
* a private ``--settings-file`` -> none of the user's MCP servers or ``dangerouslyAllowAll``.

Measured on glm-5.3-flash: 30,930 first-turn tokens stock Amp vs 89 with this agent.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

# Tools every plugin agent gets unless excluded by name. Both can execute code or reach MCP servers.
ALWAYS_EXCLUDED = ("code_exec", "tool_search")

SETTINGS = {"amp.updates.mode": "disabled"}  # private settings file: nothing of the user's leaks in

# Amp's own vocabulary; ``none`` < ... < ``max``. Used to pick the nearest supported level.
EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

_UNSUPPORTED_EFFORT = re.compile(
    r"Reasoning effort \"(?P<asked>[^\"]+)\" is not supported by model \"[^\"]+\"\. Available efforts: (?P<avail>[^\n]+)"
)


def mode_key(session_id: str) -> str:
    """Mode keys must be unique, non-empty and at most 24 chars."""
    return ("hermes-" + re.sub(r"[^a-z0-9]", "", session_id.lower()))[:24]


def _js(value: str) -> str:
    """A JS/TS string literal. ``ensure_ascii`` escapes U+2028/2029 too, so any prompt is safe."""
    return json.dumps(value, ensure_ascii=True)


@dataclass(frozen=True)
class AgentSpec:
    key: str
    extends: str  # ``provider/model`` from Amp's picker, or one of low|medium|high|ultra
    instructions: str
    effort: str | None = None


def render_plugin(spec: AgentSpec) -> str:
    """TypeScript source of the plugin. Pure function: same spec, same bytes."""
    label = spec.key  # <= 24 chars, shows up in Amp's UI as the thread's agent
    lines = [
        f"// @amp-agent-mode {json.dumps({'key': spec.key, 'label': label})}",
        "import type { PluginAPI } from '@ampcode/plugin'",
        "",
        "export default function (amp: PluginAPI) {",
        "\tconst agent = amp.createAgent({",
        f"\t\textends: {_js(spec.extends)},",
        f"\t\tinstructions: {_js(spec.instructions)},",
        "\t\tomitDefaultSystemPrompt: true,",
        f"\t\ttools: {{ include: [], exclude: [{', '.join(_js(t) for t in ALWAYS_EXCLUDED)}] }},",
    ]
    if spec.effort:
        lines.append(f"\t\treasoningEffort: {_js(spec.effort)},")
    lines += [
        "\t})",
        "\tamp.registerAgentMode({",
        f"\t\tkey: {_js(spec.key)},",
        f"\t\tlabel: {_js(label)},",
        "\t\tdescription: 'Bare model for Hermes: no Amp system prompt, no tools, no MCP.',",
        "\t\tagent: agent.definition,",
        "\t})",
        "}",
        "",
    ]
    return "\n".join(lines)


def write_session_dir(base: Path, spec: AgentSpec) -> Path:
    """Create ``<base>/<key>/`` holding the plugin and the private settings; returns the directory.

    The directory is the Amp process' cwd: project plugins load from ``<cwd>/.amp/plugins``, and the
    empty cwd also means Amp finds no AGENTS.md or repo to read.
    """
    base = Path(base)
    root = base / spec.key
    plugins = root / ".amp" / "plugins"
    plugins.mkdir(parents=True, exist_ok=True)
    # The plugin holds Hermes' whole system prompt (memory, profile, skills): keep it private to this user.
    for directory in (base, root, root / ".amp", plugins):
        os.chmod(directory, 0o700)
    for name, body in (("agent.ts", render_plugin(spec)), ("settings.json", json.dumps(SETTINGS))):
        path = (plugins if name == "agent.ts" else root) / name
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(body)
    return root


def nearest_supported_effort(error_text: str) -> str | None:
    """Parse Amp's "Reasoning effort X is not supported ... Available efforts: a, b" error.

    Returns the supported level closest to the one asked for, or None when the text is another error.
    """
    m = _UNSUPPORTED_EFFORT.search(error_text or "")
    if not m:
        return None
    avail = [a.strip() for a in m.group("avail").split(",") if a.strip()]
    asked = m.group("asked").strip()
    if not avail:
        return None
    if asked not in EFFORT_ORDER:
        return avail[0]
    pos = EFFORT_ORDER.index(asked)

    def distance(level: str) -> int:
        return abs(EFFORT_ORDER.index(level) - pos) if level in EFFORT_ORDER else 99

    return min(avail, key=distance)
