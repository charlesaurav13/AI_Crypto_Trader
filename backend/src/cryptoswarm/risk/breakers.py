from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class CircuitBreakerState:
    name: str
    tripped: bool = False
    tripped_at: datetime | None = None
    last_value: float = 0.0

    def trip(self, value: float) -> None:
        self.tripped = True
        self.last_value = value
        self.tripped_at = datetime.now(timezone.utc)

    def reset(self) -> None:
        self.tripped = False
        self.tripped_at = None
        self.last_value = 0.0

    def is_tripped(self) -> bool:
        return self.tripped


class DailyLossBreaker:
    """Trips when cumulative daily PnL drops below threshold_pct of starting_balance."""

    def __init__(self, starting_balance: float, threshold_pct: float) -> None:
        self._starting_balance = starting_balance
        self._threshold = starting_balance * threshold_pct
        self._cumulative_pnl: float = 0.0
        self._state = CircuitBreakerState(name="daily_loss")

    def update_pnl(self, delta: float) -> None:
        self._cumulative_pnl += delta
        if self._cumulative_pnl <= -abs(self._threshold):
            self._state.trip(self._cumulative_pnl)

    def is_tripped(self) -> bool:
        return self._state.is_tripped()

    @property
    def last_value(self) -> float:
        return self._state.last_value

    def reset(self) -> None:
        self._cumulative_pnl = 0.0
        self._state.reset()


class MaxDrawdownBreaker:
    """Trips when equity drops more than threshold_pct from its peak."""

    def __init__(self, threshold_pct: float) -> None:
        self._threshold_pct = threshold_pct
        self._peak: float = 0.0
        self._state = CircuitBreakerState(name="max_drawdown")

    def update_equity(self, equity: float) -> None:
        if equity > self._peak:
            self._peak = equity
        if self._peak > 0:
            drawdown = (self._peak - equity) / self._peak
            if drawdown >= self._threshold_pct:
                self._state.trip(drawdown)

    def is_tripped(self) -> bool:
        return self._state.is_tripped()

    def reset(self) -> None:
        self._peak = 0.0
        self._state.reset()
