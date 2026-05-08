from functools import lru_cache
from typing import Literal
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class RiskConfig(BaseModel):
    starting_balance_usd: float = 1000.0
    max_position_pct: float = 0.10
    max_concurrent_positions: int = 5
    max_leverage: int = 5
    daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.15
    dead_man_timeout_s: int = 60
    heartbeat_interval_s: int = 5


class FeeConfig(BaseModel):
    taker_rate: float = 0.0004       # 0.04%
    maker_rate: float = 0.0002       # 0.02%
    slippage_rate: float = 0.0005    # 0.05%
    maintenance_margin_rate: float = 0.004  # 0.4%


class Settings(BaseSettings):
    valkey_url: str = "redis://localhost:6379"
    timescale_dsn: str = "postgresql://postgres:postgres@localhost:5432/cryptoswarm_ts"
    postgres_dsn: str = "postgresql://postgres:postgres@localhost:5433/cryptoswarm"

    binance_api_key: str = ""
    binance_api_secret: str = ""
    binance_testnet: bool = True

    paper_trading: bool = True

    symbols: str = (
        "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT,"
        "BNBUSDT,ADAUSDT,AVAXUSDT,LINKUSDT,SUIUSDT"
    )

    risk: RiskConfig = RiskConfig()
    fees: FeeConfig = FeeConfig()

    # ------------------------------------------------------------------ #
    # LLM provider selection
    # Set LLM_PROVIDER to one of: anthropic | openai | ollama | gemini
    # ------------------------------------------------------------------ #
    llm_provider: Literal["anthropic", "openai", "ollama", "gemini"] = "anthropic"

    # Anthropic (Claude)
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-sonnet-20241022"

    # OpenAI (GPT-4o, GPT-4-turbo, GPT-3.5-turbo, …)
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # Ollama  (local; requires `ollama serve` + model pulled)
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"   # must support function calling

    # Google Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-pro"

    # ------------------------------------------------------------------ #
    # Per-agent LLM overrides  (format: "provider:model")
    # Leave blank to inherit the global llm_provider + model above.
    #
    # Examples:
    #   QUANT_LLM=openai:gpt-4o
    #   RISK_LLM=anthropic:claude-3-haiku-20240307
    #   PORTFOLIO_LLM=ollama:llama3.1
    #   DIRECTOR_LLM=anthropic:claude-3-5-sonnet-20241022
    # ------------------------------------------------------------------ #
    quant_llm: str = ""        # Quant Agent  — technical indicators + regime
    risk_llm: str = ""         # Risk Agent   — Kelly criterion + sizing
    portfolio_llm: str = ""    # Portfolio Agent — correlation / concentration
    director_llm: str = ""     # Director Agent — final synthesis (use best model)

    # ------------------------------------------------------------------ #
    # Scraper settings
    # ------------------------------------------------------------------ #
    scraper_interval_s: int = 1800          # 30 minutes
    scraper_ollama_model: str = "qwen2.5:7b"
    scraper_ollama_url: str = "http://localhost:11434"
    scraper_min_relevance: float = 0.3

    # ------------------------------------------------------------------ #
    # ML / RL settings
    # ------------------------------------------------------------------ #
    ml_retrain_interval_s: int = 21600      # 6 hours
    ml_min_samples: int = 500
    ml_model_dir: str = "models"
    ml_feature_lookback: int = 30

    # Composite reward weights
    reward_w1: float = 0.40
    reward_w2: float = 0.20
    reward_w3: float = 0.20
    reward_w4: float = 0.15
    reward_w5: float = 0.05

    # ------------------------------------------------------------------ #
    # Prompt evolution settings
    # ------------------------------------------------------------------ #
    prompt_evolution_interval_s: int = 25200   # 7 hours
    prompt_evolution_lookback: int = 50

    # Director / agent timing
    director_interval_s: int = 60    # how often Director cycles all symbols
    agent_timeout_s: int = 30        # max wait for sub-agent responses

    @property
    def symbol_list(self) -> list[str]:
        return [s.strip() for s in self.symbols.split(",") if s.strip()]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
