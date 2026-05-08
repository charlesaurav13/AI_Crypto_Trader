from cryptoswarm.config.settings import get_settings, Settings


def test_settings_defaults():
    s = Settings()
    assert s.symbol_list == [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
        "BNBUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT",
    ]
    assert s.risk.max_leverage == 5
    assert s.risk.daily_loss_pct == 0.03
    assert s.paper_trading is True


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("RISK__MAX_LEVERAGE", "3")
    monkeypatch.setenv("PAPER_TRADING", "false")
    s = Settings()
    assert s.risk.max_leverage == 3
    assert s.paper_trading is False


def test_get_settings_cached():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_phase3_scraper_defaults():
    from cryptoswarm.config.settings import Settings
    s = Settings()
    assert s.scraper_interval_s == 1800
    assert s.scraper_ollama_model == "qwen2.5:7b"
    assert s.scraper_ollama_url == "http://localhost:11434"
    assert s.scraper_min_relevance == 0.3


def test_phase3_ml_defaults():
    from cryptoswarm.config.settings import Settings
    s = Settings()
    assert s.ml_retrain_interval_s == 21600
    assert s.ml_min_samples == 500
    assert s.ml_model_dir == "models"
    assert s.ml_feature_lookback == 30


def test_phase3_reward_weights_sum_to_one():
    from cryptoswarm.config.settings import Settings
    s = Settings()
    total = s.reward_w1 + s.reward_w2 + s.reward_w3 + s.reward_w4 + s.reward_w5
    assert abs(total - 1.0) < 1e-9


def test_phase3_prompt_evolution_defaults():
    from cryptoswarm.config.settings import Settings
    s = Settings()
    assert s.prompt_evolution_interval_s == 25200
    assert s.prompt_evolution_lookback == 50
