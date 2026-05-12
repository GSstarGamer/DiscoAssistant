from __future__ import annotations

from discoassistant.llm.tool_registry import BotServices, ToolRegistry
from discoassistant.llm.tools import (
    lookup_user,
    read_channel_messages,
    read_user_memory,
    remember,
    replace_in_memory,
    send_dm,
    server_log,
)


# Order is load-bearing: some models are sensitive to the order in which
# tool schemas are presented. Keep this list aligned with the historical order
# defined in `_tool_schemas_for_agent` prior to extraction.
_DEFAULT_TOOL_MODULES = (
    read_channel_messages,
    lookup_user,
    remember,
    replace_in_memory,
    read_user_memory,
    server_log,
    send_dm,
)


def build_default_registry(bot_services: BotServices) -> ToolRegistry:
    """Build a `ToolRegistry` populated with every built-in tool spec.

    `bot_services` is accepted for API symmetry and to allow future per-tool
    factories that need to read configuration. The current built-in handlers
    receive their services through the per-call `ToolContext`, so we only need
    to register their `SPEC` objects here.
    """
    del bot_services
    registry = ToolRegistry()
    for module in _DEFAULT_TOOL_MODULES:
        registry.register(module.SPEC)
    return registry


__all__ = ["build_default_registry"]
