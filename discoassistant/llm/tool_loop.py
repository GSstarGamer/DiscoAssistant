from __future__ import annotations

import json
import logging
from time import monotonic
from typing import Any

from discoassistant.llm.openrouter import OpenRouterClient
from discoassistant.llm.tool_registry import ToolContext, ToolRegistry


LOGGER = logging.getLogger("discoassistant")


class ToolLoopRunner:
    def __init__(
        self,
        *,
        openrouter_client: OpenRouterClient,
        registry: ToolRegistry,
        max_calls_per_turn: int,
    ) -> None:
        self._openrouter = openrouter_client
        self._registry = registry
        self._max_calls_per_turn = max_calls_per_turn

    async def run(
        self,
        *,
        model: str | None,
        messages: list[dict[str, Any]],
        extra_payload: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        current_payload = dict(extra_payload)
        response = await self._openrouter.chat(
            model=model,
            messages=messages,
            extra_payload=current_payload,
        )
        max_calls = self._max_calls_per_turn
        tool_calls_used = 0
        unresolved_tool_failure: dict[str, Any] | None = None
        retried_failure_names: set[str] = set()

        while tool_calls_used < max_calls:
            assistant_message = response.get("choices", [{}])[0].get("message", {})
            tool_calls = assistant_message.get("tool_calls", [])
            if not tool_calls:
                if unresolved_tool_failure is not None:
                    failure_name = unresolved_tool_failure["name"]
                    if failure_name in retried_failure_names:
                        LOGGER.info(
                            "Tool failure already retried name=%s; surfacing failure",
                            failure_name,
                        )
                        return response
                    retried_failure_names.add(failure_name)
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Previous tool call failed.\n"
                                f"Tool: {failure_name}\n"
                                f"Error: {unresolved_tool_failure['error_message']}\n"
                                "Retry with corrected arguments. Only answer the user after the tool succeeds or is provably impossible."
                            ),
                        }
                    )
                    current_payload["tool_choice"] = "required"
                    response = await self._openrouter.chat(
                        model=model,
                        messages=messages,
                        extra_payload=current_payload,
                    )
                    continue
                return response

            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message.get("content") or "",
                    "tool_calls": tool_calls,
                }
            )

            current_payload["tool_choice"] = "auto"
            for tool_call in tool_calls:
                tool_calls_used += 1
                try:
                    tool_result = await self._execute_tool_call(tool_call, tool_context)
                except Exception as exc:
                    LOGGER.exception(
                        "Tool call failed name=%s id=%s",
                        tool_call.get("function", {}).get("name", ""),
                        tool_call.get("id", ""),
                    )
                    tool_result = {
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                if tool_result.get("ok") is False:
                    unresolved_tool_failure = {
                        "name": tool_call["function"]["name"],
                        "error_message": str(tool_result.get("error_message", "Tool failed.")),
                    }
                elif (
                    unresolved_tool_failure is not None
                    and unresolved_tool_failure["name"] == tool_call["function"]["name"]
                ):
                    unresolved_tool_failure = None
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": tool_call["function"]["name"],
                        "content": json.dumps(tool_result),
                    }
                )
                if tool_calls_used >= max_calls:
                    break

            response = await self._openrouter.chat(
                model=model,
                messages=messages,
                extra_payload=current_payload,
            )

        if unresolved_tool_failure is not None:
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "Couldn't complete tool action after repeated retries: "
                                f"{unresolved_tool_failure['error_message']}"
                            )
                        }
                    }
                ]
            }

        current_payload["tool_choice"] = "none"
        return await self._openrouter.chat(
            model=model,
            messages=messages,
            extra_payload=current_payload,
        )

    async def _execute_tool_call(
        self,
        tool_call: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        function_payload = tool_call.get("function", {})
        function_name = function_payload.get("name", "")
        raw_arguments = function_payload.get("arguments", "{}")
        arguments = json.loads(raw_arguments)
        started_at = monotonic()
        LOGGER.info(
            "Tool call start name=%s arguments=%s",
            function_name,
            raw_arguments,
        )

        result = await self._registry.dispatch(function_name, arguments, tool_context)

        LOGGER.info(
            "Tool call success name=%s duration=%.2fs ok=%s",
            function_name,
            monotonic() - started_at,
            result.get("ok"),
        )
        return result
