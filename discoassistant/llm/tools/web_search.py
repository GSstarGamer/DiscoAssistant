from __future__ import annotations

import logging
from typing import Any

import aiohttp

from discoassistant.llm.tool_registry import ToolContext, ToolSpec


LOGGER = logging.getLogger("discoassistant")

TAVILY_SEARCH_URL = "https://api.tavily.com/search"

SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the live web for current information. "
            "Use this for recent news, product or company updates, pricing, schedules, documentation changes, "
            "or any fact that could have changed after training."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to run.",
                },
                "topic": {
                    "type": "string",
                    "enum": ["general", "news"],
                    "description": "Use news for current events; otherwise general.",
                },
                "search_depth": {
                    "type": "string",
                    "enum": ["basic", "advanced"],
                    "description": "basic is cheaper and faster; advanced is slower but usually more precise.",
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Maximum number of results to return.",
                },
                "time_range": {
                    "type": "string",
                    "enum": ["day", "week", "month", "year"],
                    "description": "Optional freshness filter for recent content.",
                },
                "include_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional allowlist of domains to include.",
                },
                "exclude_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional blocklist of domains to exclude.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


async def handler(arguments: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    api_key = ctx.services.tavily_api_key
    if not api_key:
        return {
            "ok": False,
            "error_type": "MissingApiKey",
            "error_message": "TAVILY_API_KEY is not configured. Add a free Tavily API key in the environment before using web_search.",
        }

    query = str(arguments.get("query", "")).strip()
    if not query:
        raise ValueError("web_search.query is required.")

    payload: dict[str, Any] = {
        "query": query,
        "topic": str(arguments.get("topic", "general")),
        "search_depth": str(arguments.get("search_depth", "basic")),
        "max_results": int(arguments.get("max_results", 5)),
        "include_answer": False,
        "include_raw_content": False,
        "include_favicon": False,
    }

    time_range = arguments.get("time_range")
    if time_range:
        payload["time_range"] = str(time_range)

    include_domains = arguments.get("include_domains")
    if include_domains:
        payload["include_domains"] = [str(domain) for domain in include_domains if str(domain).strip()]

    exclude_domains = arguments.get("exclude_domains")
    if exclude_domains:
        payload["exclude_domains"] = [str(domain) for domain in exclude_domains if str(domain).strip()]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
        async with session.post(TAVILY_SEARCH_URL, json=payload) as response:
            if response.status >= 400:
                error_text = (await response.text()).strip()
                LOGGER.warning(
                    "Tavily search failed status=%s query=%r body=%s",
                    response.status,
                    query,
                    error_text[:500],
                )
                return {
                    "ok": False,
                    "error_type": "UpstreamError",
                    "error_message": f"Tavily search failed with status {response.status}: {error_text[:300]}",
                }
            raw = await response.json()

    results: list[dict[str, Any]] = []
    for item in raw.get("results", [])[: payload["max_results"]]:
        results.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "content": item.get("content"),
                "score": item.get("score"),
                "published_date": item.get("published_date"),
            }
        )

    return {
        "ok": True,
        "provider": "tavily",
        "query": raw.get("query", query),
        "topic": raw.get("auto_parameters", {}).get("topic", payload["topic"]),
        "search_depth": raw.get("auto_parameters", {}).get("search_depth", payload["search_depth"]),
        "response_time": raw.get("response_time"),
        "results": results,
    }


SPEC = ToolSpec(name="web_search", schema=SCHEMA, handler=handler)
