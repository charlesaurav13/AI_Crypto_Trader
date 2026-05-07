from dataclasses import dataclass
from cryptoswarm.bus.messages import Signal
from cryptoswarm.config.settings import Settings


@dataclass
class GuardResult:
    allowed: bool
    reason: str = ""
    breaker_name: str = ""


class SignalGuard:
    """Stateless per-signal checks. Called by risk module before passing to paper engine."""

    def __init__(self, settings: Settings, open_positions: int, current_equity: float) -> None:
        self._s = settings
        self._open_positions = open_positions
        self._current_equity = current_equity

    def check(self, signal: Signal) -> GuardResult:
        cfg = self._s.risk

        if signal.symbol not in self._s.symbol_list:
            return GuardResult(False, f"symbol {signal.symbol} not in configured list", "symbol_guard")

        if self._open_positions >= cfg.max_concurrent_positions:
            return GuardResult(
                False,
                f"max concurrent positions reached ({cfg.max_concurrent_positions})",
                "concurrent_positions_guard",
            )

        max_size = self._current_equity * cfg.max_position_pct
        if signal.size_usd > max_size:
            return GuardResult(
                False,
                f"size_usd {signal.size_usd:.2f} exceeds {max_size:.2f} (10% of equity)",
                "position_size_guard",
            )

        if signal.leverage > cfg.max_leverage:
            return GuardResult(
                False,
                f"leverage {signal.leverage}x exceeds max {cfg.max_leverage}x",
                "leverage_guard",
            )

        if signal.sl <= 0 or signal.tp <= 0:
            return GuardResult(False, "sl and tp must be > 0", "sl_tp_guard")

        return GuardResult(True)
