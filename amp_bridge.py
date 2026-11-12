"""Hermes <-> Amp text bridge: tool schemas travel in the system prompt, tool calls come back as text.

Amp runs the model with no tools (see amp_agent.py), so Hermes' OpenAI-style ``tools`` / ``tool_calls`` have no
wire channel. Same contract as Hermes' ACP clients: schemas go in as text, the model answers with
``<tool_call>{...}</tool_call>`` blocks, and we parse those back into OpenAI tool-call objects.
"""

from __future__ import annotations

import hashlib
import json
import re
from types import SimpleNamespace
from typing import Any, Iterable

HARNESS = (
    "You are the language model behind Hermes, an agent that runs on the user's machine. "
    "You have no tools of your own in this session; Hermes owns every tool.\n"
    "To act, reply with one or more blocks of the form "
    '<tool_call>{"id": "call_1", "type": "function", "function": {"name": "<tool>", "arguments": "<JSON string>"}}'
    "</tool_call> as plain text, then stop and wait: Hermes runs them and sends the results in the next user "
    'message, each labelled "Tool result". Use only the tools listed below. If no tool is needed, answer directly.'
)

_ROLE_LABELS = {"system": "System", "developer": "System", "user": "User", "assistant": "Assistant", "tool": "Tool result"}


# ----------------------------------------------------------------------------------------------- content
def text_of(content: Any) -> str:
    """Flatten OpenAI message content (str | list of parts | None) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if part.get("text"):
                    parts.append(str(part["text"]))
                elif part.get("type") in ("image_url", "input_image", "image"):
                    parts.append("[image omitted: the Amp bridge forwards text only]")
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict) and "text" in content:
        return str(content["text"])
    return str(content)


def _tool_calls_of(msg: dict[str, Any]) -> list[tuple[str, str, str]]:
    """``[(id, name, arguments_json)]`` from an assistant message's ``tool_calls`` (dicts or objects)."""
    out = []
    for call in msg.get("tool_calls") or []:
        fn = (call.get("function") if isinstance(call, dict) else getattr(call, "function", None)) or {}
        get = (lambda o, k: o.get(k)) if isinstance(fn, dict) else (lambda o, k: getattr(o, k, None))
        cid = (call.get("id") if isinstance(call, dict) else getattr(call, "id", None)) or "call_1"
        args = get(fn, "arguments")
        if not isinstance(args, str):
            args = json.dumps(args if args is not None else {}, ensure_ascii=False)
        out.append((str(cid), str(get(fn, "name") or ""), args))
    return out


def _tool_call_block(cid: str, name: str, args: str) -> str:
    body = {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}
    return "<tool_call>" + json.dumps(body, ensure_ascii=False) + "</tool_call>"


# --------------------------------------------------------------------------------------------- rendering
def render_message(msg: dict[str, Any]) -> str:
    """One transcript message as text, with the assistant's earlier tool calls spelled out so the model
    sees its own history in the same format it must produce."""
    role = str(msg.get("role") or "user").lower()
    parts = [text_of(msg.get("content")).strip()]
    for cid, name, args in _tool_calls_of(msg):
        parts.append(_tool_call_block(cid, name, args))
    label = _ROLE_LABELS.get(role, "Context")
    if role == "tool" and msg.get("name"):
        label = f"Tool result ({msg['name']})"
    body = "\n".join(p for p in parts if p)
    return f"{label}:\n{body}" if body else ""


def render_turn(messages: list[dict[str, Any]], *, fresh_history: bool = False) -> str:
    """Text for the next Amp user message.

    * a lone user message is sent as-is (no label), which is what a normal first turn looks like;
    * anything else is labelled; with ``fresh_history`` (a new Amp thread opened mid-conversation) the
      whole transcript is introduced as such so the model can pick up where it was.
    """
    if len(messages) == 1 and str(messages[0].get("role")).lower() == "user":
        return text_of(messages[0].get("content")).strip()
    rendered = [r for r in (render_message(m) for m in messages) if r]
    if fresh_history:
        return "Conversation so far:\n\n" + "\n\n".join(rendered) + "\n\nNow continue as the Assistant."
    return "\n\n".join(rendered)


def split_system(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """``(system text, rest)``: leading system/developer messages are the system prompt."""
    system, i = [], 0
    while i < len(messages) and str(messages[i].get("role")).lower() in ("system", "developer"):
        system.append(text_of(messages[i].get("content")).strip())
        i += 1
    return "\n\n".join(s for s in system if s), list(messages[i:])


def tool_sections(tools: list[dict[str, Any]] | None, tool_choice: Any) -> list[str]:
    """Prompt sections carrying Hermes' tool schemas. Uses Hermes' own renderer when importable."""
    try:
        from agent.acp_openai_bridge import render_tool_bridge_sections

        return render_tool_bridge_sections(tools, tool_choice)
    except Exception:  # standalone use (tests, scripts): same contract, rendered here
        specs = []
        for t in tools or []:
            fn = t.get("function") if isinstance(t, dict) else None
            if isinstance(fn, dict) and str(fn.get("name") or "").strip():
                specs.append({"name": fn["name"].strip(), "description": fn.get("description", ""),
                              "parameters": fn.get("parameters", {})})
        out = []
        if specs:
            out.append("Available tools (OpenAI function schema). When using a tool, emit ONLY "
                       "<tool_call>{...}</tool_call> with one JSON object containing id/type/function"
                       "{name,arguments}. arguments must be a JSON string.\n" + json.dumps(specs, ensure_ascii=False))
        if tool_choice is not None:
            out.append(f"Tool choice hint: {json.dumps(tool_choice, ensure_ascii=False)}")
        return out


def build_instructions(system_text: str, tools: list[dict[str, Any]] | None, tool_choice: Any) -> str:
    """The Amp agent's whole system prompt: bridge contract + Hermes' system prompt + tool schemas."""
    sections = [HARNESS]
    if system_text.strip():
        sections.append("Hermes' instructions follow.\n\n" + system_text.strip())
    sections += tool_sections(tools, tool_choice)
    return "\n\n".join(sections)


# --------------------------------------------------------------------------------------- tool-call parse
def build_tool_call(call_id: str, name: str, arguments: str) -> Any:
    try:
        from agent.acp_openai_bridge import build_openai_tool_call

        return build_openai_tool_call(call_id=call_id, name=name, arguments=arguments)
    except Exception:
        return SimpleNamespace(id=call_id, call_id=call_id, type="function",
                               function=SimpleNamespace(name=name, arguments=arguments))


def extract_tool_calls(text: str, used_ids: Iterable[str] = ()) -> tuple[list[Any], str]:
    """Pull every well-formed ``<tool_call>{json}</tool_call>`` out of ``text`` -> ``(calls, cleaned text)``.

    Each tag is decoded on its own, so a model that mentions the tag in prose before the real block does not
    lose the call, and both the ``{"function": {...}}`` and the flat ``{"name", "arguments"}`` shapes work.
    Ids are made unique across the conversation (models love ``call_1``).
    """
    if not isinstance(text, str) or "<tool_call>" not in text:
        return [], (text or "").strip()
    used = set(used_ids)
    decoder = json.JSONDecoder()
    calls: list[Any] = []
    spans: list[tuple[int, int]] = []
    for tag in re.finditer(r"<tool_call>", text):
        pos = tag.end()
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text) or text[pos] != "{":
            continue
        try:
            obj, end = decoder.raw_decode(text, pos)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        fn = obj.get("function") if isinstance(obj.get("function"), dict) else obj
        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        args = fn.get("arguments", {})
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        cid = obj.get("id") if isinstance(obj.get("id"), str) and obj.get("id").strip() else f"amp_call_{len(calls) + 1}"
        base, n = cid, 1
        while cid in used:
            n += 1
            cid = f"{base}_{n}"
        used.add(cid)
        close = re.match(r"\s*</tool_call>", text[end:])
        spans.append((tag.start(), end + (close.end() if close else 0)))
        calls.append(build_tool_call(cid, name.strip(), args))
    parts, cursor = [], 0
    for start, end in spans:
        parts.append(text[cursor:start])
        cursor = end
    parts.append(text[cursor:])
    return calls, "".join(parts).strip()


def used_tool_ids(messages: list[dict[str, Any]]) -> set[str]:
    return {cid for m in messages if isinstance(m, dict) for cid, _, _ in _tool_calls_of(m)}


# ------------------------------------------------------------------------------------ prefix identity
def _norm(s: str) -> str:
    return " ".join(str(s).split())


def canonical(msg: dict[str, Any]) -> list[Any]:
    """Whitespace-insensitive identity of a message, ignoring ids and reasoning fields, so that the assistant
    message Hermes stores matches the one we returned even if it trimmed or re-wrapped it."""
    role = str(msg.get("role") or "user").lower()
    if role == "assistant":
        calls = []
        for _, name, args in _tool_calls_of(msg):
            try:
                args = json.dumps(json.loads(args), sort_keys=True, ensure_ascii=False)
            except ValueError:
                args = _norm(args)
            calls.append([name, args])
        return ["assistant", _norm(text_of(msg.get("content"))), calls]
    if role == "tool":
        return ["tool", _norm(msg.get("name") or ""), _norm(text_of(msg.get("content")))]
    return [role if role != "developer" else "system", _norm(text_of(msg.get("content")))]


def digest(messages: Iterable[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps([canonical(m) for m in messages], ensure_ascii=False).encode()).hexdigest()
