"""Amp (ampcode.com) as a Hermes model provider.

Hermes keeps its own agent loop, tools, memory and skills. Amp only supplies the model: every turn runs a
throw-away Amp plugin agent with Hermes' system prompt, no tools and no MCP servers, over one warm
``amp -x --stream-json --stream-json-input`` process per conversation. See README.md.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

try:  # a package when Hermes loads it, flat modules when tests or scripts do
    from . import amp_client, amp_models
except ImportError:  # pragma: no cover
    import amp_client
    import amp_models

logger = logging.getLogger(__name__)

AMP_MARKER_BASE_URL = amp_client.MARKER_BASE_URL
# Amp reports no context window; stay conservative so Hermes compresses early rather than late.
DEFAULT_CONTEXT = 200_000


class AmpProfile(ProviderProfile):
    """Amp CLI as an external process: no REST endpoint, Hermes talks to a child over stdio."""

    def create_client(self, **client_kwargs: Any) -> Any:
        return amp_client.AmpClient(**client_kwargs)

    def fetch_models(self, *, api_key: str | None = None, base_url: str | None = None,
                     timeout: float = 30.0) -> list[str] | None:
        try:
            from hermes_cli.auth import resolve_external_process_provider_credentials

            command = resolve_external_process_provider_credentials(self.name).get("command")
        except Exception:
            command = None
        try:
            return amp_models.picker_ids(amp_models.catalog(command or "amp"))
        except Exception as exc:  # never break the picker
            logger.debug("amp fetch_models failed: %s", exc)
            return list(amp_models.FALLBACK_BARE)

    def get_model_context_length(self, model: str) -> int | None:
        try:
            return int(os.environ.get("HERMES_AMP_CONTEXT", "") or DEFAULT_CONTEXT)
        except ValueError:
            return DEFAULT_CONTEXT


amp = AmpProfile(
    name="amp",
    aliases=("ampcode", "amp-cli"),
    display_name="Amp (ampcode.com)",
    description="Run Hermes on the models of your Amp account through the Amp CLI",
    api_mode="chat_completions",
    env_vars=(),
    base_url=AMP_MARKER_BASE_URL,
    auth_type="external_process",
    process_command="amp",
    process_args=(),
    process_command_env_vars=("HERMES_AMP_COMMAND", "AMP_CLI_PATH"),
    default_aux_model="glm-5.3-flash",
    fallback_models=tuple(amp_models.FALLBACK_BARE),
    supports_streaming=False,
)

register_provider(amp)
