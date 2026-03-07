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
    atr_stop_multiplier: float = 2.0
    trailing_stop: bool = True
    deterministic: bool = True
    # 可選：外部傳入預計算特徵（避免重複計算）
    precomputed_features: Optional[np.ndarray] = None
    # LSTM 模式
    use_lstm: bool = False
    # 與訓練環境對齊的參數
    episode_length: int = 480    # rolling equity_change_pct 窗口大小（對應訓練 episode 長度）
    max_holding_steps: int = 9999  # 最大持倉步數（與 trading_env 一致）

    def init(self) -> None:
        if not self.model_path:
            raise ValueError("model_path is required for PPOTradingStrategy")

        if self.use_lstm:
            from sb3_contrib import RecurrentPPO
            self._model = RecurrentPPO.load(self.model_path, device="cpu")
        else:
            self._model = PPO.load(self.model_path, device="cpu")

        # LSTM 狀態追蹤
        self._lstm_states = None
        self._episode_starts = np.array([True])

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

        # === 預計算 ATR（供動態止損使用）===
        closes = data_df['close'].to_numpy(dtype=np.float64)
        highs = data_df['high'].to_numpy(dtype=np.float64)
        lows = data_df['low'].to_numpy(dtype=np.float64)
        prev_close = np.roll(closes, 1)
        prev_close[0] = closes[0]
        true_range = np.maximum(
            highs - lows,
            np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close))
        )
        self._atr_values = pd.Series(true_range).rolling(14, min_periods=1).mean().to_numpy()

        # === 持倉狀態追蹤（用於生成 28 維觀察空間）===
        self._entry_price = 0.0       # 開倉價格
        self._holding_steps = 0       # 持倉步數
        self._initial_equity = None   # 初始權益（延遲初始化）
        self._last_close_step = -999  # 上次平倉步數
        self._current_step = 0       # 當前步數計數器
        self._current_sl = 0.0       # 當前止損價格（含追蹤）
        self._equity_history: list = []  # 用於滾動 equity_change_pct 窗口

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

        # (4) 距止損距離百分比 (0~1)，使用追蹤止損價
        if self.position and self._entry_price > 0 and self._current_sl > 0:
            if self.position.is_long:
                dist_to_sl = (price - self._current_sl) / (self._entry_price - self._current_sl + 1e-10)
            else:
                dist_to_sl = (self._current_sl - price) / (self._current_sl - self._entry_price + 1e-10)
            dist_to_sl = np.clip(dist_to_sl, 0.0, 2.0) / 2.0
        else:
            dist_to_sl = 0.0

        # (5) 權益變化百分比：滾動 episode_length 窗口，對應訓練環境每 episode 重置的語意
        #     避免回測全期累積導致特徵超出訓練分布（OOD）
        if self._initial_equity is not None and self._initial_equity > 0:
            if len(self._equity_history) >= self.episode_length:
                baseline = self._equity_history[-self.episode_length]
            else:
                baseline = self._initial_equity
            equity_change_pct = (self.equity - baseline) / (baseline + 1e-10)
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

        # 偵測 backtesting.py 自動平倉（止損觸發）→ 同步內部追蹤狀態
        # backtesting.py 執行止損後不會呼叫我們的程式碼，需主動偵測
        if not self.position and self._entry_price != 0.0:
            self._entry_price = 0.0
            self._holding_steps = 0
            self._current_sl = 0.0
            self._last_close_step = self._current_step

        price = float(self.data.Close[-1])

        # === 追蹤止損更新（trailing_stop=true 時才執行）===
        # 更新 self._current_sl 後同步寫入 trade.sl，
        # backtesting.py 下一根 K 線會用更新後的止損做盤中觸發判斷
        if self.position and self.trailing_stop and self._current_sl > 0:
            current_idx_for_atr = len(self.data.Close) - 1
            atr = self._atr_values[current_idx_for_atr] if current_idx_for_atr < len(self._atr_values) else 0
            if atr > 0:
                if self.position.is_long:
                    new_sl = price - self.atr_stop_multiplier * atr
                    if new_sl > self._current_sl:
                        self._current_sl = new_sl
                        for trade in self.trades:   # self.trades = 開倉中的 Trade 物件
                            trade.sl = self._current_sl
                elif self.position.is_short:
                    new_sl = price + self.atr_stop_multiplier * atr
                    if new_sl < self._current_sl:
                        self._current_sl = new_sl
                        for trade in self.trades:
                            trade.sl = self._current_sl

        # === 組合觀察空間：N 維市場特徵 + 5 維持倉狀態 ===
        market_features = self._feature_cache[current_idx]
        position_features = self._get_position_features(price)
        state = np.concatenate([market_features, position_features])

        if self.use_lstm:
            action, self._lstm_states = self._model.predict(
                state.reshape(1, -1),
                state=self._lstm_states,
                episode_start=self._episode_starts,
                deterministic=self.deterministic
            )
            self._episode_starts = np.array([False])
        else:
            action, _ = self._model.predict(state.reshape(1, -1), deterministic=self.deterministic)
        action = int(action[0]) if isinstance(action, (np.ndarray, list)) else int(action)

        # 記錄 equity 歷史（用於滾動 equity_change_pct）
        self._equity_history.append(self.equity)

        # 更新持倉時間
        if self.position:
            self._holding_steps += 1

        # 最大持倉時間強制平倉（與 trading_env 一致）
        if self.position and self._holding_steps >= self.max_holding_steps:
            self.position.close()
            self._entry_price = 0.0
            self._holding_steps = 0
            self._current_sl = 0.0
            self._last_close_step = self._current_step
            self._current_step += 1
            return

        if action == 0:
            if self.position:
                self.position.close()
                self._entry_price = 0.0
                self._holding_steps = 0
                self._current_sl = 0.0
                self._last_close_step = self._current_step
            self._current_step += 1
            return

        if action == 1:
            if self.position.is_short:
                self.position.close()
                self._entry_price = 0.0
                self._holding_steps = 0
                self._current_sl = 0.0
                self._last_close_step = self._current_step
            if not self.position:
                atr = self._atr_values[current_idx] if current_idx < len(self._atr_values) else 0
                if atr > 0:
                    sl_price = price - self.atr_stop_multiplier * atr
                else:
                    sl_price = price * (1 - self.stop_loss_pct)
                self.buy(size=self.position_size_pct, sl=sl_price, tag="long")
                self._entry_price = price
                self._holding_steps = 0
                self._current_sl = sl_price
            self._current_step += 1
            return

        if action == 2:
            if self.position.is_long:
                self.position.close()
                self._entry_price = 0.0
                self._holding_steps = 0
                self._current_sl = 0.0
                self._last_close_step = self._current_step
            if not self.position:
                atr = self._atr_values[current_idx] if current_idx < len(self._atr_values) else 0
                if atr > 0:
                    sl_price = price + self.atr_stop_multiplier * atr
                else:
                    sl_price = price * (1 + self.stop_loss_pct)
                self.sell(size=self.position_size_pct, sl=sl_price, tag="short")
                self._entry_price = price
                self._holding_steps = 0
                self._current_sl = sl_price
            self._current_step += 1
            return

        # action == 3: hold
        self._current_step += 1
        return
