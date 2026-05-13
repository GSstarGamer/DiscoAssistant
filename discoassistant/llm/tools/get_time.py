from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from discoassistant.llm.tool_registry import ToolContext, ToolSpec


SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": (
            "Get the exact current time in a specific timezone. "
            "Always call this when the user asks for the time anywhere - never guess. "
            "Use IANA timezone names like 'America/New_York', 'America/Los_Angeles', "
            "'Europe/London', 'Asia/Tokyo', or 'UTC'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone name. Defaults to UTC if omitted.",
                }
            },
            "required": [],
            "additionalProperties": False,
        },
    },
}


async def handler(arguments: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    del ctx
    tz_name = str(arguments.get("timezone", "UTC")).strip() or "UTC"
    try:
        tz = ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        return {
            "ok": False,
            "error_type": "bad_timezone",
            "error_message": (
                f"Unknown timezone {tz_name!r}. Use an IANA name like "
                "'America/New_York', 'Europe/London', or 'Asia/Tokyo'."
            ),
        }

    now = datetime.now(tz)
    return {
        "ok": True,
        "timezone": tz_name,
        "iso": now.isoformat(),
        "human": now.strftime("%A, %B %d, %Y at %I:%M %p"),
        "utc_offset": now.strftime("%z"),
        "tz_abbreviation": now.strftime("%Z"),
    }


SPEC = ToolSpec(name="get_time", schema=SCHEMA, handler=handler)
