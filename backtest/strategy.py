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

        # === v8.0：持倉狀態追蹤（用於生成 25 維觀察空間）===
        self._entry_price = 0.0       # 開倉價格
        self._holding_steps = 0       # 持倉步數
        self._initial_equity = None   # 初始權益（延遲初始化）
        self._last_close_step = -999  # 上次平倉步數
        self._current_step = 0       # 當前步數計數器

    def _get_position_features(self, price: float) -> np.ndarray:
        """
        計算 5 維持倉狀態特徵，與 trading_env._get_observation() 對齊

        Returns:
            np.ndarray: [position_state, floating_pnl_pct, holding_time_norm,
                         distance_to_stop_loss, equity_change_pct]
        """
        # (1) 持倉方向: -1=空倉, 0=無倉位, 1=多倉
        if self.position.is_long:
            position_state = 1.0
        elif self.position.is_short:
            position_state = -1.0
        else:
            position_state = 0.0

        # (2) 浮動盈虧百分比
        if self.position and self._entry_price > 0:
            if self.position.is_long:
                floating_pnl_pct = (price - self._entry_price) / self._entry_price
            else:
                floating_pnl_pct = (self._entry_price - price) / self._entry_price
            floating_pnl_pct = np.clip(floating_pnl_pct, -1.0, 1.0)
        else:
            floating_pnl_pct = 0.0

        # (3) 持倉時間正規化 (0~1, 120 步飽和)
        if self.position:
            holding_time_norm = min(self._holding_steps / 120.0, 1.0)
        else:
            holding_time_norm = 0.0

        # (4) 距止損距離百分比 (0~1)
        if self.position and self._entry_price > 0:
            if self.position.is_long:
                sl_price = self._entry_price * (1 - self.stop_loss_pct)
                dist_to_sl = (price - sl_price) / (self._entry_price - sl_price + 1e-10)
            else:
                sl_price = self._entry_price * (1 + self.stop_loss_pct)
                dist_to_sl = (sl_price - price) / (sl_price - self._entry_price + 1e-10)
            dist_to_sl = np.clip(dist_to_sl, 0.0, 2.0) / 2.0
        else:
            dist_to_sl = 0.0

        # (5) 權益變化百分比
        if self._initial_equity is not None and self._initial_equity > 0:
            current_equity = self.equity
            equity_change_pct = (current_equity - self._initial_equity) / self._initial_equity
            equity_change_pct = np.clip(equity_change_pct, -1.0, 1.0)
        else:
            equity_change_pct = 0.0

        return np.array([
            position_state,
            floating_pnl_pct,
            holding_time_norm,
            dist_to_sl,
            equity_change_pct
        ], dtype=np.float32)

    def next(self) -> None:
        current_idx = len(self.data.Close) - 1
        if current_idx < 0:
            return

        # 延遲初始化初始權益
        if self._initial_equity is None:
            self._initial_equity = self.equity

        price = float(self.data.Close[-1])

        # === 組合 25 維觀察空間：20 維市場特徵 + 5 維持倉狀態 ===
        market_features = self._feature_cache[current_idx]
        position_features = self._get_position_features(price)
        state = np.concatenate([market_features, position_features])

        action, _ = self._model.predict(state.reshape(1, -1), deterministic=self.deterministic)
        action = int(action[0]) if isinstance(action, (np.ndarray, list)) else int(action)

        # 更新持倉時間
        if self.position:
            self._holding_steps += 1

        if action == 0:
            if self.position:
                self.position.close()
                self._entry_price = 0.0
                self._holding_steps = 0
                self._last_close_step = self._current_step
            self._current_step += 1
            return

        if action == 1:
            if self.position.is_short:
                self.position.close()
                self._entry_price = 0.0
                self._holding_steps = 0
                self._last_close_step = self._current_step
            if not self.position:
                sl_price = price * (1 - self.stop_loss_pct)
                self.buy(size=self.position_size_pct, sl=sl_price, tag="long")
                self._entry_price = price
                self._holding_steps = 0
            self._current_step += 1
            return

        if action == 2:
            if self.position.is_long:
                self.position.close()
                self._entry_price = 0.0
                self._holding_steps = 0
                self._last_close_step = self._current_step
            if not self.position:
                sl_price = price * (1 + self.stop_loss_pct)
                self.sell(size=self.position_size_pct, sl=sl_price, tag="short")
                self._entry_price = price
                self._holding_steps = 0
            self._current_step += 1
            return

        # action == 3: hold
        self._current_step += 1
        return
