"""Tests for MLTrainer — DB and model calls mocked."""
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from cryptoswarm.ml.trainer import MLTrainer
from cryptoswarm.ml.features import FEATURE_SIZE


def _make_trainer():
    mock_ts = MagicMock()
    mock_pg = MagicMock()
    mock_pg.insert_training_run = AsyncMock(return_value=1)
    mock_pg.update_training_run = AsyncMock()
    mock_pg.get_recent_closed_trades = AsyncMock(return_value=[])
    mock_xgb = MagicMock()
    mock_xgb.fit = MagicMock()
    mock_xgb.version = "2026-05-09T06:00:00"
    mock_lstm = MagicMock()
    mock_lstm.fit = MagicMock()
    mock_lstm.version = "2026-05-09T06:00:00"
    mock_store = MagicMock()
    mock_store.save = MagicMock()
    mock_features = MagicMock()
    mock_features.build = AsyncMock(
        return_value=np.zeros(FEATURE_SIZE, dtype=np.float32)
    )
    return MLTrainer(
        ts=mock_ts, pg=mock_pg,
        xgb_model=mock_xgb, lstm_model=mock_lstm,
        model_store=mock_store, features=mock_features,
        min_samples=5,
        seq_len=2,   # small seq_len so test data produces sequences
    )


async def test_run_once_skips_when_no_data():
    trainer = _make_trainer()
    trainer._pg.get_recent_closed_trades = AsyncMock(return_value=[])
    await trainer.run_once()
    trainer._xgb.fit.assert_not_called()
    trainer._lstm.fit.assert_not_called()


async def test_run_once_trains_when_enough_data():
    trainer = _make_trainer()
    fake_trades = [
        {
            "correlation_id": f"cid-{i}",
            "symbol": "BTCUSDT",
            "side": "LONG",
            "realized_pnl": float(i % 3 - 1),
            "state": f'{{"f0": {float(i)/10}}}',
            "action": '{"side": "LONG"}',
            "reward": float(i % 3 - 1) * 0.1,
        }
        for i in range(10)
    ]
    trainer._pg.get_recent_closed_trades = AsyncMock(return_value=fake_trades)
    await trainer.run_once()
    trainer._xgb.fit.assert_called_once()
    trainer._lstm.fit.assert_called_once()
    trainer._store.save.assert_called()


async def test_run_once_records_training_run():
    trainer = _make_trainer()
    fake_trades = [
        {
            "correlation_id": f"cid-{i}", "symbol": "BTCUSDT",
            "side": "LONG", "realized_pnl": 1.0,
            "state": f'{{"f0": {float(i)/10}}}',
            "action": '{"side": "LONG"}', "reward": 0.5,
        }
        for i in range(10)
    ]
    trainer._pg.get_recent_closed_trades = AsyncMock(return_value=fake_trades)
    await trainer.run_once()
    trainer._pg.insert_training_run.assert_called()
    trainer._pg.update_training_run.assert_called()


async def test_run_once_tolerates_malformed_state_rows():
    """Malformed JSON in state column should be skipped, not crash training."""
    trainer = _make_trainer()
    bad_trades = [
        {"correlation_id": "bad", "symbol": "BTCUSDT", "side": "LONG",
         "realized_pnl": 1.0, "state": "not-valid-json",
         "action": "{}", "reward": 0.5},
    ]
    # Mix of bad rows + enough good rows to hit min_samples=5
    good_trades = [
        {"correlation_id": f"g{i}", "symbol": "BTCUSDT", "side": "LONG",
         "realized_pnl": 1.0, "state": f'{{"f0": {float(i)/10}}}',
         "action": "{}", "reward": 0.5}
        for i in range(8)
    ]
    trainer._pg.get_recent_closed_trades = AsyncMock(return_value=bad_trades + good_trades)
    await trainer.run_once()  # must not raise
    # Good rows were enough to train
    trainer._xgb.fit.assert_called_once()


async def test_run_once_records_error_when_fit_raises():
    """If model.fit() raises, the error is recorded in metrics passed to update_training_run."""
    trainer = _make_trainer()
    trainer._xgb.fit = MagicMock(side_effect=RuntimeError("GPU OOM"))
    good_trades = [
        {"correlation_id": f"g{i}", "symbol": "BTCUSDT", "side": "LONG",
         "realized_pnl": 1.0, "state": f'{{"f0": {float(i)/10}}}',
         "action": "{}", "reward": 0.5}
        for i in range(8)
    ]
    trainer._pg.get_recent_closed_trades = AsyncMock(return_value=good_trades)
    await trainer.run_once()  # must not raise
    call_kwargs = trainer._pg.update_training_run.call_args[1]
    assert "xgb_error" in call_kwargs["metrics"]
