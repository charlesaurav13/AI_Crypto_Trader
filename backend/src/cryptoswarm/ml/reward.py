"""Composite reward function for RL training.

reward = w1 * norm_pnl
       + w2 * win_contribution
       + w3 * rr_contribution
       - w4 * drawdown_penalty
       - w5 * time_in_loss_penalty
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardConfig:
    w1: float = 0.40   # normalized P&L
    w2: float = 0.20   # win/loss contribution
    w3: float = 0.20   # R:R ratio
    w4: float = 0.15   # drawdown penalty
    w5: float = 0.05   # time-in-loss penalty
    drawdown_scale: float = 3.0  # amplifier for drawdown penalty component


def compute_reward(
    realized_pnl: float,
    position_size_usd: float,
    tp_pct: float,
    sl_pct: float,
    seconds_in_loss: float,
    total_seconds: float,
    cfg: RewardConfig | None = None,
) -> float:
    """Compute composite reward. Returned value is clamped to [-5.0, +5.0]; typical positive range is [0.0, +0.80]."""
    if cfg is None:
        cfg = RewardConfig()

    ps = max(position_size_usd, 1.0)
    ts = max(total_seconds, 1.0)

    # Component 1: normalised P&L (-1 to +1 typical)
    norm_pnl = max(-1.0, min(1.0, realized_pnl / ps))

    # Component 2: win/loss (+1 profitable, 0.0 break-even, -0.5 loss)
    if realized_pnl > 0:
        win_contribution = 1.0
    elif realized_pnl == 0.0:
        win_contribution = 0.0
    else:
        win_contribution = -0.5

    # Component 3: R:R ratio (0–1 normalised, capped at 5x)
    rr = abs(tp_pct) / max(abs(sl_pct), 1e-6)
    rr_contribution = min(rr / 5.0, 1.0)

    # Component 4: drawdown penalty (extra weight on losses)
    drawdown = max(0.0, -realized_pnl / ps)
    drawdown_penalty = drawdown * cfg.drawdown_scale

    # Component 5: fraction of time in negative P&L
    time_in_loss = min(seconds_in_loss / ts, 1.0)

    reward = (
        cfg.w1 * norm_pnl
        + cfg.w2 * win_contribution
        + cfg.w3 * rr_contribution
        - cfg.w4 * drawdown_penalty
        - cfg.w5 * time_in_loss
    )
    return round(max(-5.0, min(5.0, reward)), 6)
