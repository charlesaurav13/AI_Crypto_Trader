"""MLAgent — 5th signal agent. Uses XGBoost + LSTM + PPO to produce MLSignal."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from cryptoswarm.bus.client import BusClient
from cryptoswarm.bus.messages import AnalyzeRequest, MLSignal

if TYPE_CHECKING:
    from cryptoswarm.ml.features import FeatureEngine
    from cryptoswarm.ml.lstm_model import LSTMModel
    from cryptoswarm.ml.ppo_policy import PPOPolicy
    from cryptoswarm.ml.xgboost_model import XGBoostModel
    from cryptoswarm.storage.postgres import PostgresWriter

logger = logging.getLogger(__name__)


class MLAgent:
    def __init__(
        self,
        bus: BusClient,
        features: "FeatureEngine",
        xgb: "XGBoostModel",
        lstm: "LSTMModel",
        ppo: "PPOPolicy",
        pg: "PostgresWriter",
        seq_len: int = 30,
    ) -> None:
        self._bus = bus
        self._features = features
        self._xgb = xgb
        self._lstm = lstm
        self._ppo = ppo
        self._pg = pg
        self._seq_len = seq_len

    async def run(self) -> None:
        async for _, data in self._bus.psubscribe("agent.analyze.*"):
            req = AnalyzeRequest.model_validate_json(data)
            try:
                await self._handle(req)
            except Exception as exc:
                logger.error("MLAgent error for %s: %s", req.symbol, exc)

    async def _handle(self, req: AnalyzeRequest) -> None:
        try:
            feat_vec = await self._features.build(req.symbol)
            feat_seq = await self._features.build_sequence(req.symbol, self._seq_len)
        except Exception as exc:
            logger.warning("MLAgent: feature build failed for %s: %s", req.symbol, exc)
            await self._publish_neutral(req)
            return

        try:
            regime, direction, xgb_conf = self._xgb.predict(feat_vec)
            short_dir, lstm_conf = self._lstm.predict(feat_seq)
            size_adj, ppo_conf = self._ppo.predict(feat_vec)
        except Exception as exc:
            logger.warning("MLAgent: model inference failed for %s: %s", req.symbol, exc)
            await self._publish_neutral(req, reason=f"model inference failed — neutral fallback")
            return

        # Overall confidence: average of models that have been trained (conf > 0)
        confidences = [c for c in [xgb_conf, lstm_conf, ppo_conf] if c > 0.0]
        confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

        reasoning = (
            f"XGBoost: regime={regime} dir={direction} conf={xgb_conf:.2f} | "
            f"LSTM: short_dir={short_dir} conf={lstm_conf:.2f} | "
            f"PPO: size_adj={size_adj}"
        )

        msg = MLSignal(
            symbol=req.symbol,
            correlation_id=req.correlation_id,
            regime_pred=regime,
            direction_pred=direction,
            short_direction=short_dir,
            size_adjustment=size_adj,
            confidence=confidence,
            reasoning=reasoning,
        )
        await self._bus.publish(f"agent.result.ml.{req.symbol}", msg)

        model_version = self._xgb.version or "untrained"
        await self._pg.insert_ml_signal(
            symbol=req.symbol,
            regime_pred=regime,
            direction_pred=direction,
            short_direction=short_dir,
            confidence=confidence,
            size_adjustment=size_adj,
            model_version=model_version,
        )
        logger.info(
            "MLAgent: %s regime=%s dir=%s short=%s adj=%s conf=%.2f",
            req.symbol, regime, direction, short_dir, size_adj, confidence,
        )

    async def _publish_neutral(self, req: AnalyzeRequest, reason: str = "feature build failed — neutral fallback") -> None:
        msg = MLSignal(
            symbol=req.symbol,
            correlation_id=req.correlation_id,
            regime_pred="ranging",
            direction_pred="up",
            short_direction="up",
            size_adjustment="hold",
            confidence=0.0,
            reasoning=reason,
        )
        await self._bus.publish(f"agent.result.ml.{req.symbol}", msg)
        try:
            await self._pg.insert_ml_signal(
                symbol=req.symbol,
                regime_pred="ranging",
                direction_pred="up",
                short_direction="up",
                confidence=0.0,
                size_adjustment="hold",
                model_version="neutral_fallback",
            )
        except Exception as exc:
            logger.warning("MLAgent: failed to persist neutral signal for %s: %s", req.symbol, exc)
