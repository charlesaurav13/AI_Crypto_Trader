"""
Paper trade engine.
Subscribes to: signal.execute, market.mark.*
Publishes to: trade.executed, position.update, risk.veto, circuit.tripped
"""
from __future__ import annotations
import asyncio
import json
import logging

from cryptoswarm.bus.client import BusClient
from cryptoswarm.bus.messages import (
    Signal, TradeExecuted, PositionUpdate, RiskVeto,
    MarkPrice, CircuitTripped,
)
from cryptoswarm.config.settings import Settings
from cryptoswarm.papertrade.account import Account, OpenPosition
from cryptoswarm.papertrade.math import (
    calc_qty, calc_liq_price, calc_isolated_margin,
    calc_entry_fee, calc_exit_fee,
)
from cryptoswarm.risk.guards import SignalGuard
from cryptoswarm.risk.breakers import DailyLossBreaker, MaxDrawdownBreaker

logger = logging.getLogger(__name__)


class PaperTradeEngine:
    def __init__(self, bus: BusClient, settings: Settings) -> None:
        self._bus = bus
        self._cfg = settings
        self._fees = settings.fees
        self._account = Account(settings.risk.starting_balance_usd)
        self._daily_loss = DailyLossBreaker(
            settings.risk.starting_balance_usd, settings.risk.daily_loss_pct
        )
        self._max_dd = MaxDrawdownBreaker(settings.risk.max_drawdown_pct)

    async def run(self) -> None:
        """Main loop: listens on signal.execute and market.mark.* simultaneously."""
        await asyncio.gather(
            self._handle_signals(),
            self._handle_mark_prices(),
        )

    async def _handle_signals(self) -> None:
        async for _, data in self._bus.subscribe("signal.execute"):
            signal = Signal.model_validate_json(data)
            await self._process_signal(signal)

    async def _handle_mark_prices(self) -> None:
        async for _, data in self._bus.psubscribe("market.mark.*"):
            mark = MarkPrice.model_validate_json(data)
            await self._process_mark(mark)

    async def _process_signal(self, signal: Signal) -> None:
        # Check circuit breakers first
        if self._daily_loss.is_tripped() or self._max_dd.is_tripped():
            veto = RiskVeto(
                original_correlation_id=signal.correlation_id,
                reason="circuit breaker tripped",
                breaker_name="daily_loss_or_drawdown",
            )
            await self._bus.publish("risk.veto", veto)
            return

        # Per-signal guards
        guard = SignalGuard(
            self._cfg,
            len(self._account.open_positions),
            self._account.equity,
        )
        result = guard.check(signal)
        if not result.allowed:
            veto = RiskVeto(
                original_correlation_id=signal.correlation_id,
                reason=result.reason,
                breaker_name=result.breaker_name,
            )
            await self._bus.publish("risk.veto", veto)
            return

        # Determine paper fill price from reasoning JSON, or midpoint fallback
        try:
            meta = json.loads(signal.reasoning) if signal.reasoning.startswith("{") else {}
            entry_price = float(meta["entry"])
        except Exception:
            entry_price = (signal.sl + signal.tp) / 2  # midpoint fallback

        qty = calc_qty(signal.size_usd, entry_price)
        margin = calc_isolated_margin(signal.size_usd, signal.leverage)
        liq = calc_liq_price(
            entry_price, signal.side, signal.leverage,  # type: ignore[arg-type]
            self._fees.maintenance_margin_rate,
        )
        entry_fee = calc_entry_fee(signal.size_usd, self._fees.taker_rate)

        pos = OpenPosition(
            symbol=signal.symbol, side=signal.side, qty=qty,
            entry_price=entry_price, leverage=signal.leverage,
            sl=signal.sl, tp=signal.tp,
            isolated_margin=margin, liq_price=liq, fees=entry_fee,
        )
        self._account.open(pos)

        executed = TradeExecuted(
            original_correlation_id=signal.correlation_id,
            symbol=signal.symbol, side=signal.side,
            qty=qty, entry_price=entry_price, leverage=signal.leverage,
            sl=signal.sl, tp=signal.tp, fees=entry_fee,
        )
        await self._bus.publish("trade.executed", executed)
        await self._publish_position_update(signal.symbol)
        logger.info("Paper fill: %s %s @ %.2f qty=%.6f", signal.side, signal.symbol, entry_price, qty)

    async def _process_mark(self, mark: MarkPrice) -> None:
        symbol = mark.symbol
        if symbol not in self._account.open_positions:
            return

        self._account.update_mark(symbol, mark.mark_price)
        pos = self._account.open_positions[symbol]

        sl_hit = (pos.side == "LONG" and mark.mark_price <= pos.sl) or \
                 (pos.side == "SHORT" and mark.mark_price >= pos.sl)
        tp_hit = (pos.side == "LONG" and mark.mark_price >= pos.tp) or \
                 (pos.side == "SHORT" and mark.mark_price <= pos.tp)
        liq_hit = (pos.side == "LONG" and mark.mark_price <= pos.liq_price) or \
                  (pos.side == "SHORT" and mark.mark_price >= pos.liq_price)

        if liq_hit:
            await self._close_position(symbol, pos.liq_price, "liq")
        elif sl_hit:
            await self._close_position(symbol, pos.sl, "sl")
        elif tp_hit:
            await self._close_position(symbol, pos.tp, "tp")
        else:
            await self._publish_position_update(symbol)

    async def _close_position(self, symbol: str, exit_price: float, reason: str) -> None:
        pos = self._account.open_positions[symbol]
        exit_fee = calc_exit_fee(pos.qty, exit_price, self._fees.taker_rate)
        net_pnl = self._account.close(symbol, exit_price, reason, exit_fee)

        # Update circuit breakers
        self._daily_loss.update_pnl(net_pnl)
        self._max_dd.update_equity(self._account.equity)

        if self._daily_loss.is_tripped():
            await self._bus.publish(
                "circuit.tripped",
                CircuitTripped(
                    breaker_name="daily_loss",
                    value=self._daily_loss.last_value,
                    threshold=self._cfg.risk.starting_balance_usd * self._cfg.risk.daily_loss_pct,
                ),
            )

        closed_update = PositionUpdate(
            symbol=symbol, side=pos.side, qty=pos.qty,
            entry_price=pos.entry_price, mark_price=exit_price,
            unrealized_pnl=0.0, isolated_margin=0.0,
            liq_price=pos.liq_price, is_closed=True, close_reason=reason,  # type: ignore[arg-type]
        )
        await self._bus.publish("position.update", closed_update)
        logger.info("Closed %s %s @ %.2f reason=%s net_pnl=%.4f", pos.side, symbol, exit_price, reason, net_pnl)

    async def _publish_position_update(self, symbol: str) -> None:
        if symbol not in self._account.open_positions:
            return
        pos = self._account.open_positions[symbol]
        upd = PositionUpdate(
            symbol=symbol, side=pos.side, qty=pos.qty,
            entry_price=pos.entry_price, mark_price=pos.mark_price,
            unrealized_pnl=pos.unrealized_pnl,
            isolated_margin=pos.isolated_margin,
            liq_price=pos.liq_price,
        )
        await self._bus.publish("position.update", upd)
