from functools import lru_cache
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

    # Agent / LLM configuration
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    ollama_url: str = "http://localhost:11434"
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
