"""RewardComputer — subscribes to position.update(is_closed=True), computes reward,
updates rl_tuples, and triggers PPO online update."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from cryptoswarm.bus.client import BusClient
from cryptoswarm.bus.messages import PositionUpdate
from cryptoswarm.ml.features import FEATURE_SIZE
from cryptoswarm.ml.reward import RewardConfig, compute_reward

if TYPE_CHECKING:
    from cryptoswarm.ml.features import FeatureEngine
    from cryptoswarm.ml.ppo_policy import PPOPolicy
    from cryptoswarm.storage.postgres import PostgresWriter

logger = logging.getLogger(__name__)


class RewardComputer:
    def __init__(
        self,
        bus: BusClient,
        pg: "PostgresWriter",
        ppo: "PPOPolicy",
        features: "FeatureEngine",
        reward_config: RewardConfig | None = None,
    ) -> None:
        self._bus = bus
        self._pg = pg
        self._ppo = ppo
        self._features = features
        self._cfg = reward_config or RewardConfig()
        # Track per-symbol entry state for reward computation
        self._entry_info: dict[str, dict] = {}  # symbol → {entry_price, entry_state, sl_pct, tp_pct, side, open_ts}

    async def run(self) -> None:
        async for _, data in self._bus.subscribe("position.update"):
            update = PositionUpdate.model_validate_json(data)
            if update.is_closed:
                await self._handle_close(update)

    async def register_open(
        self,
        symbol: str,
        entry_price: float,
        sl_pct: float,
        tp_pct: float,
        side: str,
        entry_state: dict,
        open_ts: float,
    ) -> None:
        """Called by main.py when a trade opens. Keyed by symbol."""
        self._entry_info[symbol] = {
            "entry_price": entry_price,
            "entry_state": entry_state,
            "sl_pct": sl_pct,
            "tp_pct": tp_pct,
            "side": side,
            "open_ts": open_ts,
        }

    async def _handle_close(self, update: PositionUpdate) -> None:
        info = self._entry_info.pop(update.symbol, None)

        # Compute P&L from mark price vs entry
        entry = info["entry_price"] if info else update.entry_price
        if update.side == "LONG":
            realized_pnl = (update.mark_price - entry) * update.qty
        else:
            realized_pnl = (entry - update.mark_price) * update.qty

        sl_pct = info["sl_pct"] if info else 0.02
        tp_pct = info["tp_pct"] if info else 0.04
        open_ts = info["open_ts"] if info else time.time()
        total_seconds = max(time.time() - open_ts, 1.0)
        # Approximate seconds in loss — use total if loss, 0 if profit
        seconds_in_loss = total_seconds if realized_pnl < 0 else 0.0

        position_size = update.qty * entry
        reward = compute_reward(
            realized_pnl=realized_pnl,
            position_size_usd=max(position_size, 1.0),
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            seconds_in_loss=seconds_in_loss,
            total_seconds=total_seconds,
            cfg=self._cfg,
        )

        # Build next_state feature vector
        next_state = await self._features.build(update.symbol)
        next_state_dict = {f"f{i}": float(v) for i, v in enumerate(next_state)}

        # Update rl_tuple in DB
        await self._pg.update_rl_tuple_reward(
            correlation_id=update.correlation_id,
            reward=reward,
            next_state=next_state_dict,
        )

        # PPO online update
        if info and "entry_state" in info:
            prev_state_dict = info["entry_state"]
            prev_state = [float(prev_state_dict.get(f"f{i}", 0.0)) for i in range(FEATURE_SIZE)]
            self._ppo.update(
                state=np.array(prev_state, dtype=np.float32),
                action="scale_up",   # approximate — Director chose to trade
                reward=reward,
                next_state=next_state,
                done=True,
            )
            await asyncio.to_thread(self._ppo.maybe_train)

        logger.info(
            "RewardComputer: %s %s pnl=%.4f reward=%.4f",
            update.symbol, update.side, realized_pnl, reward,
        )
