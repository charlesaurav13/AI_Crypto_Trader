"""LSTMModel — short-horizon (15min) direction prediction from bar sequences."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal

import numpy as np

from cryptoswarm.ml.features import FEATURE_SIZE

logger = logging.getLogger(__name__)

_MIN_SAMPLES = 50


class LSTMModel:
    def __init__(
        self,
        seq_len: int = 30,
        hidden_size: int = 64,
        num_layers: int = 2,
        epochs: int = 10,
        lr: float = 1e-3,
    ) -> None:
        self._seq_len = seq_len
        self._hidden = hidden_size
        self._layers = num_layers
        self._epochs = epochs
        self._lr = lr
        self._net = None
        self.version: str | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """X: (N, seq_len, FEATURE_SIZE), y: (N,) int 0=up 1=down."""
        if len(X) < _MIN_SAMPLES:
            raise ValueError(f"LSTMModel minimum {_MIN_SAMPLES} samples, got {len(X)}")
        import torch
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self, input_size, hidden, layers):
                super().__init__()
                # dropout only valid when num_layers > 1
                dropout = 0.2 if layers > 1 else 0.0
                self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True, dropout=dropout)
                self.fc = nn.Linear(hidden, 2)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        net = _Net(FEATURE_SIZE, self._hidden, self._layers)
        opt = torch.optim.Adam(net.parameters(), lr=self._lr)
        criterion = nn.CrossEntropyLoss()

        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)

        net.train()
        for _ in range(self._epochs):
            opt.zero_grad()
            loss = criterion(net(Xt), yt)
            loss.backward()
            opt.step()

        self._net = net
        self.version = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        logger.info("LSTMModel: trained on %d samples version=%s", len(X), self.version)

    def predict(self, seq: np.ndarray) -> tuple[Literal["up", "down"], float]:
        """seq: (seq_len, FEATURE_SIZE). Returns (direction, confidence)."""
        if self._net is None:
            return "up", 0.0
        import torch
        self._net.eval()
        with torch.no_grad():
            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
            logits = self._net(x)
            proba = torch.softmax(logits, dim=-1)[0].numpy()
        direction: Literal["up", "down"] = "up" if proba[0] >= 0.5 else "down"
        return direction, round(float(max(proba)), 4)
