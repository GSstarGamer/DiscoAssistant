from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable
from time import monotonic
from typing import Any

import aiohttp
from bs4 import BeautifulSoup

from discoassistant.config import WebSearchConfig


LOGGER = logging.getLogger("discoassistant.web_search")

OpenRouterChat = Callable[..., Awaitable[dict[str, Any]]]

_URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)

_TAVILY_ENDPOINT = "https://api.tavily.com/search"


async def run_web_search(
    *,
    question: str,
    openrouter_chat: OpenRouterChat,
    http_session: aiohttp.ClientSession,
    config: WebSearchConfig,
    tavily_api_key: str | None,
) -> dict[str, Any]:
    started = monotonic()
    LOGGER.info("Web search subagent start question=%r", question)
    if not tavily_api_key:
        return _error("no_api_key", "TAVILY_API_KEY not configured")
    try:
        queries = await _generate_queries(question, openrouter_chat, config)
        if not queries:
            return _error("no_queries", "subagent did not generate any queries")
        LOGGER.info("Web search queries generated count=%d queries=%s", len(queries), queries)

        raw_results = await _tavily_search_all(queries, http_session, tavily_api_key, config)
        if not raw_results:
            return _error("no_results", "Tavily returned no results")
        LOGGER.info("Web search Tavily results count=%d", len(raw_results))

        picked = await _pick_urls(question, raw_results, openrouter_chat, config)
        if len(picked) < config.min_sites:
            return _error(
                "too_few_urls",
                f"subagent picked {len(picked)} URLs, need at least {config.min_sites}",
            )
        LOGGER.info("Web search URLs picked count=%d urls=%s", len(picked), picked)

        summaries = await _summarize_in_parallel(
            urls=picked,
            question=question,
            http_session=http_session,
            openrouter_chat=openrouter_chat,
            config=config,
        )
        good = [s for s in summaries if s["ok"]]
        if not good:
            return _error("all_sites_failed", "every fetch/summary failed")

        final_answer = await _merge_summaries(question, good, openrouter_chat, config)
        LOGGER.info(
            "Web search merge complete duration=%.2fs sources=%d",
            monotonic() - started,
            len(good),
        )
        return {
            "ok": True,
            "answer": final_answer,
            "sources": [s["url"] for s in good],
            "queries": queries,
            "picked_urls": picked,
            "failed_urls": [s["url"] for s in summaries if not s["ok"]],
        }
    except Exception as exc:
        LOGGER.exception("Web search subagent failed question=%r", question)
        return _error("exception", f"{type(exc).__name__}: {exc}")


def _error(error_type: str, message: str) -> dict[str, Any]:
    return {"ok": False, "error_type": error_type, "error_message": message}


async def _call_subagent_llm(
    *,
    openrouter_chat: OpenRouterChat,
    config: WebSearchConfig,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    json_mode: bool = False,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    extra_payload: dict[str, Any] = {
        "temperature": config.temperature,
        "max_tokens": max_tokens,
        "service_tier": config.service_tier,
    }
    if json_mode:
        extra_payload["response_format"] = {"type": "json_object"}
    response = await openrouter_chat(
        model=config.model,
        messages=messages,
        extra_payload=extra_payload,
    )
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return (message.get("content") or "").strip()


async def _generate_queries(
    question: str,
    openrouter_chat: OpenRouterChat,
    config: WebSearchConfig,
) -> list[str]:
    system_prompt = config.system_prompts.query_gen.format(
        max_queries=config.max_queries,
        min_sites=config.min_sites,
        max_sites=config.max_sites,
    )
    user_prompt = f"Question: {question}"
    content = await _call_subagent_llm(
        openrouter_chat=openrouter_chat,
        config=config,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=300,
        json_mode=True,
    )
    queries = _parse_json_list(content, "queries")
    if not queries:
        queries = [question]
    return [q for q in queries[: config.max_queries] if isinstance(q, str) and q.strip()]


async def _tavily_search_all(
    queries: list[str],
    http_session: aiohttp.ClientSession,
    api_key: str,
    config: WebSearchConfig,
) -> list[dict[str, str]]:
    tasks = [
        _tavily_search_one(q, http_session, api_key, config.ddg_results_per_query)
        for q in queries
    ]
    nested = await asyncio.gather(*tasks, return_exceptions=True)
    seen: set[str] = set()
    merged: list[dict[str, str]] = []
    for batch in nested:
        if isinstance(batch, BaseException):
            LOGGER.warning("Web search Tavily query failed error=%s", batch)
            continue
        for item in batch:
            url = item.get("url", "")
            if not url or url in seen:
                continue
            seen.add(url)
            merged.append(item)
    return merged


async def _tavily_search_one(
    query: str,
    http_session: aiohttp.ClientSession,
    api_key: str,
    max_results: int,
) -> list[dict[str, str]]:
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
    }
    timeout = aiohttp.ClientTimeout(total=15)
    async with http_session.post(_TAVILY_ENDPOINT, json=payload, timeout=timeout) as response:
        response.raise_for_status()
        data = await response.json()
    out: list[dict[str, str]] = []
    for h in data.get("results", []) or []:
        out.append(
            {
                "title": h.get("title", "") or "",
                "url": h.get("url", "") or "",
                "text": h.get("content", "") or "",
            }
        )
    return out


async def _pick_urls(
    question: str,
    results: list[dict[str, str]],
    openrouter_chat: OpenRouterChat,
    config: WebSearchConfig,
) -> list[str]:
    system_prompt = config.system_prompts.url_pick.format(
        max_queries=config.max_queries,
        min_sites=config.min_sites,
        max_sites=config.max_sites,
    )
    catalog = "\n".join(
        f"[{i}] {r['title']} | {r['url']}\n    {r['text']}"
        for i, r in enumerate(results)
    )
    user_prompt = f"Question: {question}\n\nResults:\n{catalog}"
    content = await _call_subagent_llm(
        openrouter_chat=openrouter_chat,
        config=config,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=400,
        json_mode=True,
    )
    urls = _parse_json_list(content, "urls")
    valid_urls = {r["url"] for r in results}
    picked = [u for u in urls if isinstance(u, str) and u in valid_urls]
    if len(picked) < config.min_sites:
        picked = _fallback_pick_urls(content, valid_urls, config.min_sites)
    return picked[: config.max_sites]


def _fallback_pick_urls(content: str, valid: set[str], min_sites: int) -> list[str]:
    matched = [u for u in _URL_RE.findall(content) if u in valid]
    if len(matched) >= min_sites:
        return matched
    return list(valid)[:min_sites]


async def _summarize_in_parallel(
    *,
    urls: list[str],
    question: str,
    http_session: aiohttp.ClientSession,
    openrouter_chat: OpenRouterChat,
    config: WebSearchConfig,
) -> list[dict[str, Any]]:
    tasks = [
        _fetch_and_summarize(
            url=url,
            question=question,
            http_session=http_session,
            openrouter_chat=openrouter_chat,
            config=config,
        )
        for url in urls
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    out: list[dict[str, Any]] = []
    for url, r in zip(urls, results):
        if isinstance(r, BaseException):
            LOGGER.warning("Web search site task crashed url=%s error=%s", url, r)
            out.append({"ok": False, "url": url, "summary": "", "error": str(r)})
        else:
            out.append(r)
    return out


async def _fetch_and_summarize(
    *,
    url: str,
    question: str,
    http_session: aiohttp.ClientSession,
    openrouter_chat: OpenRouterChat,
    config: WebSearchConfig,
) -> dict[str, Any]:
    started = monotonic()
    try:
        html = await _fetch_html(url, http_session, config.fetch_timeout_seconds)
    except Exception as exc:
        LOGGER.warning(
            "Web search site fetch url=%s duration=%.2fs ok=False error=%s",
            url,
            monotonic() - started,
            exc,
        )
        return {"ok": False, "url": url, "summary": "", "error": f"fetch: {exc}"}

    text = await asyncio.to_thread(_strip_html, html, config.max_html_chars)
    if not text:
        return {"ok": False, "url": url, "summary": "", "error": "empty page"}

    user_prompt = (
        f"Original question: {question}\n\n"
        f"Source URL: {url}\n\n"
        f"Page content:\n{text}"
    )
    try:
        summary = await _call_subagent_llm(
            openrouter_chat=openrouter_chat,
            config=config,
            system_prompt=config.system_prompts.summarize,
            user_prompt=user_prompt,
            max_tokens=config.summary_max_tokens,
        )
    except Exception as exc:
        LOGGER.warning(
            "Web search site summarize url=%s duration=%.2fs ok=False error=%s",
            url,
            monotonic() - started,
            exc,
        )
        return {"ok": False, "url": url, "summary": "", "error": f"summarize: {exc}"}

    LOGGER.info(
        "Web search site fetch url=%s duration=%.2fs ok=True chars=%d",
        url,
        monotonic() - started,
        len(text),
    )
    return {"ok": True, "url": url, "summary": summary}


async def _fetch_html(
    url: str,
    http_session: aiohttp.ClientSession,
    timeout_seconds: int,
) -> str:
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml",
    }
    async with http_session.get(url, timeout=timeout, headers=headers) as response:
        response.raise_for_status()
        return await response.text(errors="ignore")


def _strip_html(raw: str, max_chars: int) -> str:
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "form"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


async def _merge_summaries(
    question: str,
    summaries: list[dict[str, Any]],
    openrouter_chat: OpenRouterChat,
    config: WebSearchConfig,
) -> str:
    body = "\n\n".join(
        f"### Source: {s['url']}\n{s['summary']}" for s in summaries
    )
    user_prompt = f"Original question: {question}\n\nPer-site summaries:\n{body}"
    return await _call_subagent_llm(
        openrouter_chat=openrouter_chat,
        config=config,
        system_prompt=config.system_prompts.merge,
        user_prompt=user_prompt,
        max_tokens=config.merge_max_tokens,
    )


def _parse_json_list(content: str, key: str) -> list[Any]:
    if not content:
        return []
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return []
        try:
            data = json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            return []
    if isinstance(data, dict):
        value = data.get(key)
        if isinstance(value, list):
            return value
    if isinstance(data, list):
        return data
    return []
