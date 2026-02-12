"""
Backtesting strategy that wraps a trained PPO model.

優化：使用預計算特徵緩存，將 O(n²) 降低到 O(n)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from backtesting import Strategy
from stable_baselines3 import PPO


class PPOTradingStrategy(Strategy):
    """
    Strategy adapter for Stable-Baselines3 PPO.

    Class variables are used so Backtest can pass parameters.

    優化：在 init() 中一次性預計算所有特徵，next() 中 O(1) 查詢
    """

    model_path: Optional[str] = None
    feature_config: Optional[dict] = None
    position_size_pct: float = 0.15
    stop_loss_pct: float = 0.015
    deterministic: bool = True
    # 可選：外部傳入預計算特徵（避免重複計算）
    precomputed_features: Optional[np.ndarray] = None

    def init(self) -> None:
        if not self.model_path:
            raise ValueError("model_path is required for PPOTradingStrategy")

        self._model = PPO.load(self.model_path, device="cpu")

        data_df = self.data.df.copy()
        data_df.columns = [col.lower() for col in data_df.columns]

        if not isinstance(data_df.index, pd.DatetimeIndex):
            if "timestamp" in data_df.columns:
                data_df["timestamp"] = pd.to_datetime(data_df["timestamp"], errors="coerce")
                data_df = data_df.set_index("timestamp")
            else:
                data_df.index = pd.to_datetime(data_df.index, errors="coerce")

        self._feature_df = data_df

        # === 優化：預計算所有特徵（帶硬碟緩存）===
        if self.precomputed_features is not None:
            # 使用外部傳入的預計算特徵
            self._feature_cache = self.precomputed_features
            print(f"[Strategy] Using precomputed features: {self._feature_cache.shape}")
        else:
            # 使用硬碟緩存系統
            from utils.feature_cache import precompute_features_with_cache

            # 需要原始 DataFrame（帶所有欄位）給緩存系統
            original_df = self.data.df.copy()
            original_df.columns = [col.lower() for col in original_df.columns]

            print("[Strategy] Precomputing features with disk cache...")
            self._feature_cache = precompute_features_with_cache(
                df=original_df,
                config=self.feature_config or {},
                cache_dir="data/cache",
                verbose=True
            )
            print(f"[Strategy] Feature cache ready: {self._feature_cache.shape}")

    def next(self) -> None:
        current_idx = len(self.data.Close) - 1
        if current_idx < 0:
            return

        # === 優化：O(1) 直接索引預計算特徵 ===
        state = self._feature_cache[current_idx]
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
