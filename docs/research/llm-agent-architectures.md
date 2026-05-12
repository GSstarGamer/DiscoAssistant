# LLM Chat Agent Architectures — Reference Doc

Prior-art survey for building DiscoAssistant (Discord LLM bot). Covers production memory, system prompts, tool calling, sub-agents, and session handling, with extra focus on Discord-specific patterns. Sources cited inline.

## Memory Architectures

### ChatGPT (saved memory + chat history reference, 2024–2025)
ChatGPT runs a two-layer system. **Saved memories** are explicit user-confirmed facts injected into the system prompt as a "Model Set Context" block (timestamped, editable, visible in settings). **Chat history reference** (rolled out April 10, 2025) is a separate, hidden, AI-generated set of dense per-conversation summaries (date + title + bullets) prepended every turn. Notably, OpenAI does not run a heavy retrieval engine — they just stuff the compressed dossier into every request and rely on the model to filter ([OpenAI memory FAQ](https://help.openai.com/en/articles/8590148-memory-faq), [Embrace the Red deep dive](https://embracethered.com/blog/posts/2025/chatgpt-how-does-chat-history-memory-preferences-work/), [Simon Willison on the new dossier](https://simonwillison.net/2025/May/21/chatgpt-new-memory/), [shloked.com — bitter lesson take](https://www.shloked.com/writing/chatgpt-memory-bitter-lesson)).

### Claude / Projects / Claude Code memory
Claude Code uses a layered file-based memory: `CLAUDE.md` (user/project/org-scoped instructions, loaded at session start) plus **auto memory** (model writes notes about corrections, build commands, style preferences) and a `memory.md` pointer index that fans out to structured files. Claude.ai "Projects" provide isolated memory banks per project. Anthropic's "Dreaming" feature (Managed Agents, May 2026) consolidates persistent memory between sessions by deduping and pruning ([Claude Code memory docs](https://code.claude.com/docs/en/memory), [MindStudio on three-layer architecture](https://www.mindstudio.ai/blog/claude-code-source-leak-memory-architecture), [rajiv.com on CLAUDE.md](https://rajiv.com/blog/2025/12/12/how-claude-memory-actually-works-and-why-claude-md-matters/)).

### Gemini context caching
Gemini relies on its huge context window plus **implicit + explicit caching**. Implicit caching is on by default for Gemini 2.5+; cached tokens cost ~10% of fresh input. Explicit caching lets you pin large static prefixes. No formal "saved memory" facts feature equivalent to ChatGPT; they roadmap "context fusion" for cross-app recall ([Gemini context caching docs](https://ai.google.dev/gemini-api/docs/caching), [Vertex AI overview](https://cloud.google.com/vertex-ai/generative-ai/docs/context-cache/context-cache-overview), [Datastudios on Gemini memory 2025](https://www.datastudios.org/post/google-gemini-context-window-token-limits-and-memory-in-2025)).

### Character.AI / Replika
Character.AI famously has near-zero persistent memory — context resets around turn ~21 because the window is tight. Replika tracks emotional patterns and conversation tone over time but factual recall is shaky; users frequently report regressions when memory architecture changes. Lesson: companion bots that lean on "vibe continuity" instead of fact recall consistently disappoint users on factual recall ([roborhythms — companion memory comparison](https://www.roborhythms.com/best-ai-companion-long-term-memory/), [Replika 2.0 explained](https://www.roborhythms.com/replika-2-0-explained/), [DHC on companion forgetting](https://digitalhumancorp.com/en/research/why-ai-companion-forgets-you)).

### Open-source memory frameworks
- **Mem0** — vector storage + optional graph, three-tier scope (user/session/agent), auto-extracts and compresses memories. "Add memory in 3 lines" ergonomics ([Mem0 state of agent memory 2026](https://mem0.ai/blog/state-of-ai-agent-memory-2026)).
- **Letta (ex-MemGPT)** — agent owns its memory as editable state. Three tiers explicitly modeled on OS memory hierarchy: **core memory** (in-context, like RAM), **recall memory** (searchable chat archive, like disk cache), **archival memory** (cold store, queried via tools). Agent self-manages via tool calls ([HydraDB comparison](https://hydradb.com/blog/mem0-vs-zep-vs-letta), [Letta forum thread](https://forum.letta.com/t/agent-memory-letta-vs-mem0-vs-zep-vs-cognee/88)).
- **Zep** — temporal knowledge graph; facts are timestamped and superseded rather than overwritten, so the agent can reason about what was true when ([Asymptotic Spaghetti comparison](https://medium.com/asymptotic-spaghetti-integration/from-beta-to-battle-tested-picking-between-letta-mem0-zep-for-ai-memory-6850ca8703d1)).
- Picking heuristic: Mem0/Zep for shared/team memory, Letta when you want the model itself to drive memory ops via tools.

### Vector RAG vs structured profile vs summarization
Vector recall is great for fuzzy semantic match over big corpora but bad at multi-hop reasoning and explainability. Graph/structured memory wins on traceable joins ("manager who approved budget X") but adds ontology/maintenance burden. Most prod systems hybridize: structured profile (name, prefs, relationships) + vector recall over raw chat + rolling summary. For chatbots, "you probably don't need a vector DB yet" is a real take — start with sqlite + in-context summary ([MachineLearningMastery: vectors vs graph RAG](https://machinelearningmastery.com/vector-databases-vs-graph-rag-for-agent-memory-when-to-use-which/), [Towards Data Science — you don't need a vector DB yet](https://towardsdatascience.com/you-probably-dont-need-a-vector-database-for-your-rag-yet/)).

### Hierarchical memory (working / episodic / semantic / procedural)
Most agent-memory frameworks borrow cognitive-science labels:
- **Working** = current chat context window
- **Episodic** = past conversations, timestamped events
- **Semantic** = stable facts about user/world
- **Procedural** = "how to do X" / tool-use playbooks
Letta maps these to core/recall/archival; Mem0 to user/session/agent scopes ([Atlan agent memory frameworks 2026](https://atlan.com/know/best-ai-agent-memory-frameworks-2026/)).

### Write triggers and retrieval scoring
Three common write paths: **explicit user save** ("remember that..."), **auto-extraction** (LLM scans each turn for fact-worthy snippets), **periodic compaction** (background job summarizes old turns). The Generative Agents paper (Park et al., 2023) defines the canonical retrieval score for episodic memory:

```
score = α_recency · recency + α_importance · importance + α_relevance · relevance
```

`recency` is exponential decay since last access, `importance` is an LLM-rated 1–10 score per memory at write time, `relevance` is cosine similarity between query embedding and memory embedding. All αs default to 1; top-k that fits the budget gets injected ([Generative Agents — ar5iv mirror](https://ar5iv.labs.arxiv.org/html/2304.03442), [ACM survey on LLM-agent memory](https://dl.acm.org/doi/10.1145/3748302)).

## System Prompts

### Structure
Production system prompts (ChatGPT, Claude.ai, Claude Code, custom GPTs) follow a recognizable order: **identity/role → capabilities → tool list and contracts → safety rules → output formatting → environment/context block** (date, user info, env vars). Claude Code's prompt is ~15k tokens, modular: 24 builtin tool descriptions, sub-agent prompts (Plan/Explore/Task), utility prompts (compact, statusline, security review, agent creation) ([asgeirtj system prompt leaks](https://github.com/asgeirtj/system_prompts_leaks/blob/main/Anthropic/claude-code.md), [Piebald-AI Claude Code prompts archive](https://github.com/Piebald-AI/claude-code-system-prompts), [Simon Willison on Claude 4 prompt](https://simonwillison.net/2025/May/25/claude-4-system-prompt/)).

### Static vs dynamic sections + caching
Place static content (role, tool defs, examples, long context docs) at the prefix, dynamic content (date, user ID, current channel state) at the suffix. Anthropic's prompt cache walks **tools → system → messages** in that order; one `cache_control` breakpoint at the end of static content lets the system find the longest matching prefix automatically. Min cache block ~1024 tokens (Sonnet 3.7). Cache hits are ~10% the cost and ~85% lower latency, with a 5-minute default TTL (1-hour available) ([Anthropic prompt caching docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching), [Spring AI caching strategies](https://spring.io/blog/2025/10/27/spring-ai-anthropic-prompt-caching-blog/)). Implication: keep DiscoAssistant's system prompt + tool defs + per-user persistent memory as one cacheable prefix; inject the volatile (current message, channel scrollback, timestamp) after the breakpoint.

### Per-conversation injection
Standard pattern: append a small env block — `<env>` with cwd / timestamp / user email / platform — and a `<user_context>` block with whatever user-scoped facts you've decided to surface. Claude Code and ChatGPT both do exactly this.

## Tool Calling

### Formats
- **OpenAI**: `tools` array with `type: "function"`; model returns a `tool_calls` list on the assistant message; you reply with one `role: "tool"` message per call ([OpenAI function calling guide](https://developers.openai.com/api/docs/guides/function-calling)).
- **Anthropic**: content-block model. Assistant emits `tool_use` blocks alongside text blocks; you reply with a user message containing `tool_result` blocks ([ofox guide 2026](https://ofox.ai/blog/function-calling-tool-use-complete-guide-2026/)).
- **MCP**: not a calling format itself; a transport layer that exposes tools/resources/prompts uniformly so the same server works across Claude, ChatGPT, Gemini, Copilot, etc.

### Parallel tool calls
Default-on for GPT-4o/Claude 3.5+/Gemini. Model emits N tool blocks in one assistant turn; you must execute and return all N before the next assistant turn. Order is not guaranteed — don't rely on it for dependent calls; use sequential turns instead ([OpenRouter tool calling docs](https://openrouter.ai/docs/guides/features/tool-calling), [TokenMix function calling guide](https://tokenmix.ai/blog/function-calling-guide)).

### Result handling and error loops
Best practice: return errors as normal `tool_result` content with `is_error: true` (Anthropic) or just include the error string (OpenAI). The model self-corrects — don't crash the loop. Cap retry attempts per tool to avoid infinite loops. Long tool outputs (file reads, web fetches) should be truncated/summarized before re-injection.

### Authorization patterns
Three common gates: (1) **per-tool permission tiers** (read-only vs write vs destructive — Claude Code's bash auto-approval list is the canonical example), (2) **per-user RBAC** (DiscoAssistant: owner can call admin tools, members cannot), (3) **interactive approval prompts** for destructive ops. MCP servers commonly mark tools with `readOnlyHint`/`destructiveHint` annotations the client uses to decide.

### MCP — what changed
[Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) (Anthropic, Nov 2024; donated to Linux Foundation's Agentic AI Foundation Dec 2025) standardizes tool/resource/prompt exposure over JSON-RPC 2.0, modeled on LSP. By 2025 it's natively supported by Claude, ChatGPT, Gemini, Copilot. The win: write a tool server once, all clients use it. The pain: tool descriptions get dumped into every model context, costing tokens — Anthropic now recommends "code execution with MCP" where the model calls MCP servers via generated TypeScript instead of native tool blocks for token efficiency ([MCP spec 2025-11](https://modelcontextprotocol.io/specification/2025-11-25), [Anthropic on code execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp)).

## Agents and Sub-agents

### Orchestrator-worker
The dominant pattern. A coordinator spawns specialized workers (each with their own system prompt, tool subset, and isolated context) and aggregates results. Claude Code's `Task` tool is the reference impl: workers run in fresh contexts; only their final summary returns to the orchestrator. This is the main lever against context rot in long sessions ([Claude Code subagents docs](https://code.claude.com/docs/en/sub-agents), [MindStudio on subagents](https://www.mindstudio.ai/blog/sub-agents-claude-code-context-management), [clouatre.ca on subagent architecture](https://clouatre.ca/posts/orchestrating-ai-agents-subagent-architecture/)).

### Framework comparison
- **LangGraph** — graph/state machine, durable, reducer-merged concurrent updates, v1.0 late 2025, default LangChain runtime. Production: LinkedIn, Uber, 400+ companies.
- **CrewAI** — role-based "team of agents" abstraction, easy mental model.
- **AutoGen** (Microsoft) — multi-agent conversation as the primitive.
- **OpenAI Swarm** — minimal handoff library, explicitly "not production-ready," good for learning.
- **Letta** — stateful agents with built-in self-managed memory, agent-as-a-service.
([DataCamp comparison](https://www.datacamp.com/tutorial/crewai-vs-langgraph-vs-autogen), [Maxim AI top frameworks](https://www.getmaxim.ai/articles/top-5-ai-agent-frameworks-in-2025-a-practical-guide-for-ai-builders/)).

### Handoff vs delegation vs fan-out
- **Handoff**: parent stops, child takes over the conversation (Swarm style).
- **Delegation**: child runs to completion, returns result, parent continues with summary only (Claude Code style).
- **Fan-out**: parent spawns N children in parallel for independent subtasks, joins results.

Cost/context tradeoff: each subagent is a fresh context (cheaper per call, but pays full system-prompt + tool-def overhead again, often kills cache). Worth it when subtask output is much smaller than its working context (research, search, file scanning); not worth it for tiny tasks.

## Chat / Session Handling

### Threading
ChatGPT uses linear threads with branching on edits; "Projects" group threads with shared instructions and files. Claude.ai "Projects" similarly bundle a shared knowledge base + system instructions across many chats. Both expose memory at the **account** scope, not the **thread** scope.

### Context window management
Three production strategies, usually combined:
1. **Sliding window** — keep last N turns verbatim, drop older ones.
2. **Hierarchical summarization** — recent turns verbatim, older turns collapsed to running summary; LangChain's `ConversationSummaryBufferMemory` is the canonical impl.
3. **Compaction** — Anthropic's automatic compaction API (now on Claude API, Bedrock, Vertex, Foundry); Google ADK's context compaction does the same. Anchored iterative summarization (merging new summary into persistent state) outperforms full reconstruction on accuracy/continuity ([Microsoft Agent Framework compaction](https://learn.microsoft.com/en-us/agent-framework/agents/conversations/compaction), [Google ADK compaction](https://google.github.io/adk-docs/context/compaction/), [Maxim AI on context window strategies](https://www.getmaxim.ai/articles/context-window-management-strategies-for-long-context-ai-agents-and-chatbots/)).

### Multi-user / multi-channel chat (Discord/Slack)
The scoping decision is the single most consequential design call for a Discord bot. Common keying:
- **Per-user** — personal facts, preferences, persistent identity. Survives across guilds/DMs.
- **Per-channel** — current conversation context, recent scrollback, channel topic. Dies/rotates on inactivity.
- **Per-guild** — server-wide rules, owner-set persona overrides, admin-only facts.
- **DM vs guild** — DMs typically get fuller per-user memory; guild messages should not leak DM-private memory back into a channel where other users can read it.

llmcord (popular reference Discord LLM bot) and the LlamaIndex Discord chatbot tutorial both store messages with `{author, timestamp, channel_id, guild_id}` metadata and filter retrieval by `guild_id` to prevent cross-server bleed ([jakobdylanc/llmcord](https://github.com/jakobdylanc/llmcord), [ClusteredBytes LlamaIndex Discord tutorial](https://clusteredbytes.pages.dev/posts/2024/create-a-discord-chatbot-using-llamaindex-for-your-server/)).

### Streaming, presence, typing
Discord doesn't support true token streaming, so bots simulate it by sending an initial message then editing it as tokens arrive. Throttle edits to ~1.2s to stay under rate limits. `sendTyping()` lasts 10s — re-fire on a 10s loop while generating. Cap total typing duration to avoid stuck indicators ([agentscope-ai issue on streaming pattern](https://github.com/agentscope-ai/QwenPaw/issues/1296)).

### Rate limiting and batching
Always parse `X-RateLimit-*` and `Retry-After` headers; never hardcode limits. Use ephemeral interaction responses where possible (don't count against channel rate limits). Per-user cooldowns prevent abuse. Bulk endpoints (e.g. `bulk_delete`) over loops. Xenon's blog post is the canonical at-scale account ([Discord rate limits docs](https://docs.discord.com/developers/topics/rate-limits), [Xenon on rate limits at scale](https://blog.xenon.bot/handling-rate-limits-at-scale-fb7b453cb235)).

## Discord-Bot-Specific Prior Art

### Clyde (Discord's own, retired)
Clyde routed each conversation into a fresh channel-shaped construct (`conversationID` ~= channel name) so context stayed scoped per chat; capped at ~450 concurrent conversations per server (near the 500-channel limit). Permissions were standard Discord role-based. Initially OpenAI-backed, later migrated to xAI Grok in 2024 ([dotesports — how to use Clyde](https://dotesports.com/general/news/how-to-use-clyde-ai-on-discord), [mrrfv/Discord-Clyde-API](https://github.com/mrrfv/Discord-Clyde-API)).

### MEE6 / Carl-bot (non-LLM, but instructive at scale)
MEE6 handles 21M+ guilds and ~700k events/sec. Architecture is guild-scoped configuration storage (not per-user state-heavy) — server admins configure, bot reacts. Carl-bot similarly is guild-config-first with logging modules. Takeaway: at scale, **guild-scoped config + per-user lightweight counters** is far cheaper than full per-user memory; reserve per-user memory for users who actually invoke the bot ([MEE6 Scaleway case study](https://www.scaleway.com/en/customer-testimonials/mee6/), [Carl-bot docs](https://docs.carl.gg/)).

### llmcord and discord.py LLM examples
[llmcord](https://github.com/jakobdylanc/llmcord) is the de facto reference: any OpenAI-compatible API, reply-chain message threading (the bot reconstructs context by walking Discord's reply links rather than storing chat history itself — clever, leverages Discord as the DB), per-user/per-channel/per-server allow/deny lists, vision support. Other examples (DocShotgun/LLM-discordchatbot, alpaca-discord) lean on simple in-memory dicts keyed by user ID and reset on restart — fine for prototypes, lossy for production.

### Permissions: owner / admin / member
Standard pattern: bot reads Discord roles + guild ownership at command time; owner-only and admin-only tools live behind a permission check before the tool is even exposed to the model. Critical: scope memory writes from owner-issued instructions (e.g. "remember that the rule is X") to a separate **owner rules** layer, not member-visible memory — otherwise members can prompt-inject the bot to leak or override owner rules. (DiscoAssistant's recent commit `Keep owner rules out of member memory` is exactly this pattern.)

---

## Sources cited

Inline above. Key landing pages worth bookmarking:
- [Anthropic MCP intro](https://www.anthropic.com/news/model-context-protocol) and [MCP spec](https://modelcontextprotocol.io/specification/2025-11-25)
- [Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- [Claude Code memory](https://code.claude.com/docs/en/memory) and [subagents](https://code.claude.com/docs/en/sub-agents)
- [Generative Agents paper (ar5iv)](https://ar5iv.labs.arxiv.org/html/2304.03442)
- [Mem0 state of agent memory 2026](https://mem0.ai/blog/state-of-ai-agent-memory-2026)
- [Letta vs Mem0 vs Zep](https://hydradb.com/blog/mem0-vs-zep-vs-letta)
- [llmcord reference Discord LLM bot](https://github.com/jakobdylanc/llmcord)
- [Discord rate limits](https://docs.discord.com/developers/topics/rate-limits)
- [System prompt leaks archive](https://github.com/asgeirtj/system_prompts_leaks)
