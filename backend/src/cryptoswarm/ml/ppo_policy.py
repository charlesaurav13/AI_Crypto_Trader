"""PPOPolicy — lightweight PPO via Stable-Baselines3.

Action space: 3 discrete actions
  0 = hold        (no size adjustment)
  1 = scale_up    (increase size_pct by 20%)
  2 = scale_down  (decrease size_pct by 20%)

The policy is trained online: experiences are buffered and
maybe_train() is called after each trade close.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

_ACTIONS = ["hold", "scale_up", "scale_down"]
_ACTION_IDX = {a: i for i, a in enumerate(_ACTIONS)}


class PPOPolicy:
    def __init__(
        self,
        state_size: int,
        min_train_samples: int = 256,
        max_buffer_size: int = 10_000,
    ) -> None:
        self._state_size = state_size
        self._min_train = min_train_samples
        self._model = None
        self._env = None
        self.is_trained: bool = False
        self.experience_count: int = 0
        # Buffer: deque of (state, action_idx, reward, next_state, done)
        self._buffer: deque = deque(maxlen=max_buffer_size)

    def predict(
        self, state: np.ndarray
    ) -> tuple[Literal["hold", "scale_up", "scale_down"], float]:
        """Predict action. Returns ('hold', 0.0) if not trained."""
        if not self.is_trained or self._model is None:
            return "hold", 0.0
        action_idx, _states = self._model.predict(state, deterministic=False)
        action = _ACTIONS[int(action_idx)]
        return action, 0.6  # SB3 doesn't expose per-action confidence easily

    def update(
        self,
        state: np.ndarray,
        action: Literal["hold", "scale_up", "scale_down"],
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Buffer one experience tuple."""
        self._buffer.append((
            state.copy(), _ACTION_IDX[action], reward, next_state.copy(), done
        ))
        self.experience_count += 1

    def maybe_train(self) -> None:
        """Train PPO if enough experience buffered. No-op if insufficient."""
        if len(self._buffer) < self._min_train:
            return
        try:
            self._train()
        except Exception as exc:
            logger.warning("PPOPolicy.maybe_train error: %s", exc, exc_info=True)

    def _train(self) -> None:
        import gymnasium as gym
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env

        class _ReplayEnv(gym.Env):
            """Minimal gym env that replays buffered experiences."""
            def __init__(self_, buffer, state_size):
                super().__init__()
                self_._buffer = buffer
                self_._idx = 0
                self_.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32
                )
                self_.action_space = gym.spaces.Discrete(3)

            def reset(self_, **kwargs):
                self_._idx = 0
                return self_._buffer[0][0], {}

            def step(self_, action):
                exp = self_._buffer[self_._idx % len(self_._buffer)]
                self_._idx += 1
                obs = exp[3]  # next_state
                reward = exp[2]
                done = exp[4] or self_._idx >= len(self_._buffer)
                return obs, reward, done, False, {}

        buffer_copy = list(self._buffer)[-self._min_train:]
        make_env = lambda: _ReplayEnv(buffer_copy, self._state_size)
        vec_env = make_vec_env(make_env)
        if self._model is None:
            self._model = PPO("MlpPolicy", vec_env, verbose=0, n_steps=64, batch_size=32)
        else:
            self._model.set_env(vec_env)
        self._model.learn(total_timesteps=len(buffer_copy) * 2, reset_num_timesteps=False)
        self.is_trained = True
        logger.info("PPOPolicy: trained on %d experiences", len(buffer_copy))
