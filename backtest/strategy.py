"""
Backtesting strategy that wraps a trained PPO model.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from backtesting import Strategy
from stable_baselines3 import PPO

from environment.features.feature_aggregator import FeatureAggregator


class PPOTradingStrategy(Strategy):
    """
    Strategy adapter for Stable-Baselines3 PPO.

    Class variables are used so Backtest can pass parameters.
    """

    model_path: Optional[str] = None
    feature_config: Optional[dict] = None
    position_size_pct: float = 0.15
    stop_loss_pct: float = 0.015
    deterministic: bool = True

    def init(self) -> None:
        if not self.model_path:
            raise ValueError("model_path is required for PPOTradingStrategy")

        self._model = PPO.load(self.model_path, device="cpu")
        self._feature_aggregator = FeatureAggregator(config=self.feature_config)

        data_df = self.data.df.copy()
        data_df.columns = [col.lower() for col in data_df.columns]

        if not isinstance(data_df.index, pd.DatetimeIndex):
            if "timestamp" in data_df.columns:
                data_df["timestamp"] = pd.to_datetime(data_df["timestamp"], errors="coerce")
                data_df = data_df.set_index("timestamp")
            else:
                data_df.index = pd.to_datetime(data_df.index, errors="coerce")

        self._feature_df = data_df

    def next(self) -> None:
        current_idx = len(self.data.Close) - 1
        if current_idx < 0:
            return

        state = self._feature_aggregator.get_state_vector(self._feature_df, current_idx)
        action, _ = self._model.predict(state.reshape(1, -1), deterministic=self.deterministic)
        action = int(action[0]) if isinstance(action, (np.ndarray, list)) else int(action)

        price = float(self.data.Close[-1])

        if action == 0:
            if self.position:
                self.position.close()
            return

        if action == 1:
            if self.position.is_short:
                self.position.close()
            if not self.position:
                sl_price = price * (1 - self.stop_loss_pct)
                self.buy(size=self.position_size_pct, sl=sl_price, tag="long")
            return

        if action == 2:
            if self.position.is_long:
                self.position.close()
            if not self.position:
                sl_price = price * (1 + self.stop_loss_pct)
                self.sell(size=self.position_size_pct, sl=sl_price, tag="short")
            return

        # action == 3: hold
        return
