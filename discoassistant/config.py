from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ENV_PATH = BASE_DIR / ".env"
DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "app_config.json"


class ToolDefinition(BaseModel):
    name: str
    description: str
    enabled: bool = True


class ToolPolicy(BaseModel):
    allow_model_tool_calls: bool = False
    max_calls_per_turn: int = 4
    timeout_seconds: int = 30
    registry: list[ToolDefinition] = Field(default_factory=list)


class AgentDefinition(BaseModel):
    name: str
    description: str
    system_prompt: str
    model: str | None = None
    temperature: float = 0.7
    max_output_tokens: int = 1024
    tools: list[str] = Field(default_factory=list)
    handoff_targets: list[str] = Field(default_factory=list)


class DiscordConfig(BaseModel):
    respond_to_bots: bool = False
    max_history_messages: int = 20
    conversation_window_seconds: int = 10
    reply_debounce_seconds: float = 2.5
    presence: "DiscordPresenceConfig" = Field(default_factory=lambda: DiscordPresenceConfig())


class DiscordPresenceConfig(BaseModel):
    enabled: bool = False
    type: str = "streaming"
    name: str = "Live on Twitch"
    url: str | None = None
    state: str | None = None
    details: str | None = None
    status: str = "online"
    token_usage_enabled: bool = False
    token_usage_name_template: str = "{total_tokens:,} tokens used"
    token_usage_state_template: str | None = None
    token_usage_details_template: str | None = None


class MemoryConfig(BaseModel):
    enabled: bool = True
    user_directory: str = "memories/users"
    guild_directory: str = "memories/guilds"
    dm_history_db_path: str = "memories/dm_history.sqlite3"
    max_user_chars_in_prompt: int = 6000
    max_guild_chars_in_prompt: int = 6000


class OpenRouterConfig(BaseModel):
    base_url: str = "https://openrouter.ai/api/v1"
    default_model: str = "google/gemma-4-31b-it"
    app_name: str = "DiscoAssistant"
    site_url: str | None = None


class WebToolsConfig(BaseModel):
    enabled: bool = False
    fetch_model: str = "openai/gpt-5-nano"
    service_tier: str = "flex"
    temperature: float = 0.3
    tavily_results_per_query: int = 10
    fetch_timeout_seconds: int = 12
    max_html_chars: int = 200_000
    summary_max_tokens: int = 600
    fetch_system_prompt: str = (
        "You read one web page and answer a specific question about it. "
        "Be concise and factual. Quote sparingly (under 125 chars per quote). "
        "If the page does not contain the answer, say so plainly."
    )


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    discord: DiscordConfig = Field(default_factory=DiscordConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    openrouter: OpenRouterConfig = Field(default_factory=OpenRouterConfig)
    web_tools: WebToolsConfig = Field(default_factory=WebToolsConfig)
    tool_policy: ToolPolicy = Field(default_factory=ToolPolicy)
    prompts: dict[str, str] = Field(default_factory=dict)
    agents: dict[str, AgentDefinition] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_agent_references(self) -> "RuntimeConfig":
        known_tools = {tool.name for tool in self.tool_policy.registry}
        known_agents = set(self.agents)

        for agent_key, agent in self.agents.items():
            unknown_tools = sorted(set(agent.tools) - known_tools)
            if unknown_tools:
                raise ValueError(
                    f"Agent '{agent_key}' references undefined tools: {', '.join(unknown_tools)}"
                )

            unknown_handoffs = sorted(set(agent.handoff_targets) - known_agents)
            if unknown_handoffs:
                raise ValueError(
                    f"Agent '{agent_key}' references undefined handoff targets: {', '.join(unknown_handoffs)}"
                )

        return self


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=DEFAULT_ENV_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    discord_token: str
    openrouter_api_key: str
    owner_user_id: int
    tavily_api_key: str | None = None
    app_config_path: Path = DEFAULT_CONFIG_PATH


class AppConfig(BaseModel):
    settings: Settings
    runtime: RuntimeConfig


def load_settings() -> Settings:
    return Settings()


def load_runtime_config(config_path: Path) -> RuntimeConfig:
    if not config_path.exists():
        raise RuntimeError(f"Config file not found: {config_path}")

    return RuntimeConfig.model_validate_json(config_path.read_text(encoding="utf-8"))


def load_app_config() -> AppConfig:
    settings = load_settings()
    runtime = load_runtime_config(settings.app_config_path)
    return AppConfig(settings=settings, runtime=runtime)
