# hermes-amp

Run [Hermes Agent](https://github.com/NousResearch/hermes-agent) on the models of your [Amp](https://ampcode.com) account.

Amp sells its models through an agent harness, not a chat-completions API. This plugin drives the Amp CLI as a
*bare model*: Hermes keeps its own agent loop, tools, memory, skills and approvals, and Amp only supplies the
model. About 35 models are available (Claude, GPT, Gemini, Grok, Kimi, GLM, Qwen, DeepSeek, ...), whichever your
Amp workspace allows.

## Requirements

- Hermes Agent 0.21 or newer, macOS or Linux
- The [Amp CLI](https://ampcode.com/manual#get-started), installed and logged in (`amp login`)

## Install

```bash
hermes plugins install madpin/hermes-amp      # or: git clone into ~/.hermes/plugins/amp
```

## Use

```bash
hermes chat --provider amp -m claude-sonnet-5-5
```

or pick **Amp (ampcode.com)** in `hermes model` / `/model`, or set it in `~/.hermes/config.yaml`:

```yaml
model:
  provider: amp
  default: claude-sonnet-5-5
```

Model names are Amp's, bare (`claude-opus-5-5`) or full (`anthropic/claude-opus-5-5`). The built-in modes
`low`, `medium`, `high` and `ultra` also work (Amp picks the model; your admins may have disabled some).
`python amp_models.py` prints the live list. A model your workspace has turned off fails with Amp's own message
(for example "...is disabled for your workspace"); the plugin never substitutes another model.

## How it works

Amp's CLI always wraps a model in its own system prompt, about 26 tools and every MCP server you configured:
31,000 tokens before the first word (measured, same model). A *plugin agent* can opt out of all of it, so for
each conversation the plugin writes a throw-away Amp plugin and runs the CLI against it:

| | stock `amp -x` | what hermes-amp runs |
|---|---|---|
| system prompt | Amp's | **Hermes' own** (`omitDefaultSystemPrompt`) |
| tools | 26 | **none** (`code_exec` and `tool_search` excluded by name) |
| MCP servers, `dangerouslyAllowAll` | yours | **none** (private `--settings-file`) |
| first-turn tokens | 30,930 | **89** (a one-word prompt) |

- **Tool calls** use the same text bridge as Hermes' ACP providers: Hermes' tool schemas are part of the system
  prompt, the model answers with `<tool_call>{...}</tool_call>` blocks, and the plugin turns them into OpenAI tool
  calls. Hermes executes them, with its usual approvals.
- **One warm process per conversation** (`amp -x --stream-json --stream-json-input`) keeps the Amp thread and the
  provider's prompt cache. Only the first request pays Amp's start-up; after that only what is new is sent (the
  user message, or the tool results). Measured through Hermes: first call 8.3 s, next call 3.5 s, 98 % of the
  prompt cached.
- If the history stops extending the thread (compaction, edits, a different system prompt or model), a fresh
  thread opens and the transcript is replayed. If a warm process died, the request is retried on a fresh one.
- Reasoning effort follows Hermes' setting. If a model does not support the level, the plugin uses the nearest
  one Amp accepts and remembers it.

Verified live with Hermes and with `scripts/live_models.py` (tool call, tool result, answer): claude-haiku-5-5,
claude-sonnet-5-5, gpt-6.1-terra, kimi-k3, glm-5.3-flash, grok-4.7, gemini-4-argon.

## Settings (environment)

| Variable | Default | Purpose |
|---|---|---|
| `HERMES_AMP_COMMAND` / `AMP_CLI_PATH` | `amp` on PATH, then `~/.local/bin/amp` | Path to the Amp CLI |
| `HERMES_AMP_THREADS` | `archive` | What happens to the Amp thread when a conversation ends: `archive`, `delete` or `keep` |
| `HERMES_AMP_EFFORT` | unset | Reasoning effort when Hermes sends none: `low`, `medium`, `high`, `xhigh`, `max` |
| `HERMES_AMP_IDLE_SECONDS` | `600` | Close a warm process after this long without a request |
| `HERMES_AMP_MAX_SESSIONS` | `4` | Warm processes kept at once (aux calls such as titles use their own) |
| `HERMES_AMP_TIMEOUT` | Hermes' timeout, else `900` | Seconds to wait for an answer |
| `HERMES_AMP_CONTEXT` | `200000` | Context window reported to Hermes (Amp reports none) |

## Privacy and cost

- Every message, tool result and your Hermes system prompt (memory, profile, skills) goes to Amp and the model
  provider behind it, as with any hosted model. Threads are created `--visibility private` and archived (or
  deleted with `HERMES_AMP_THREADS=delete`) when the conversation ends.
- Usage is billed to your Amp account like any Amp thread. Amp reports real token counts per call, which Hermes
  shows. A workspace may price Amp usage differently from the provider's list price; check `amp usage`.
- While a conversation is open, the generated plugin (it contains Hermes' system prompt) sits under
  `$HERMES_HOME/cache/amp-sessions/` with mode 0600 in a 0700 directory, and is removed when the conversation
  closes.
- Amp runs in an empty private directory with no tools, so a prompt injection cannot make Amp touch your files.
  What Hermes does with its own tools is governed by Hermes' approvals as usual.

## Limits

- Text only: image parts are replaced by a note.
- Tool calling goes through text, not Amp's native tool channel. It worked on every model tried, but a model that
  ignores the format will simply answer in prose.
- No token streaming: Amp's stream-json delivers whole messages.
- The first request of a conversation takes about 6 to 12 s (Amp start-up plus the model).
- The model list comes from the server and may include models your workspace has disabled.
- macOS and Linux only (process groups).

## Development

```bash
PYTHONPATH=~/.hermes/hermes-agent uv run --no-project --python 3.14 --with openai --with pytest \
  --with ruamel.yaml python -m pytest tests -q -o addopts=
AMP_LIVE=1 ... -k live            # one real round trip (spends a few Amp credits)
python scripts/live_models.py claude-haiku-5-5 kimi-k3
```

The unit tests drive `tests/fake_amp.py`, a scripted stand-in for the CLI.

Inspired by [hermes-cursor-acp](https://github.com/neerazz/hermes-cursor-acp) and Nous Research's
`claude-subscription-directsdk` provider.

## License

MIT
