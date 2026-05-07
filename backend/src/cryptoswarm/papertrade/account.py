from dataclasses import dataclass, field
from .math import calc_unrealized_pnl, calc_realized_pnl


@dataclass
class OpenPosition:
    symbol: str
    side: str            # "LONG" | "SHORT"
    qty: float
    entry_price: float
    leverage: int
    sl: float
    tp: float
    isolated_margin: float
    liq_price: float
    fees: float          # entry fees
    mark_price: float = 0.0
    funding_paid: float = 0.0

    @property
    def unrealized_pnl(self) -> float:
        if self.mark_price == 0:
            return 0.0
        return calc_unrealized_pnl(self.qty, self.entry_price, self.mark_price, self.side)  # type: ignore[arg-type]


class Account:
    def __init__(self, starting_balance: float) -> None:
        self._balance: float = starting_balance
        self.open_positions: dict[str, OpenPosition] = {}

    @property
    def balance(self) -> float:
        return self._balance

    @property
    def equity(self) -> float:
        return self._balance + sum(p.unrealized_pnl for p in self.open_positions.values())

    def open(self, pos: OpenPosition) -> None:
        self.open_positions[pos.symbol] = pos
        self._balance -= pos.isolated_margin + pos.fees

    def close(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str,
        exit_fees: float,
    ) -> float:
        """Returns realized PnL net of all fees. Releases isolated margin back to balance."""
        pos = self.open_positions.pop(symbol)
        pnl = calc_realized_pnl(pos.qty, pos.entry_price, exit_price, pos.side)  # type: ignore[arg-type]
        # return margin + gross pnl - exit fees; entry fees already deducted on open
        self._balance += pos.isolated_margin + pnl - exit_fees + pos.funding_paid
        return pnl - pos.fees - exit_fees

    def apply_funding(self, symbol: str, funding_delta: float) -> None:
        if symbol in self.open_positions:
            self.open_positions[symbol].funding_paid += funding_delta
            self._balance += funding_delta

    def update_mark(self, symbol: str, mark: float) -> None:
        if symbol in self.open_positions:
            self.open_positions[symbol].mark_price = mark
