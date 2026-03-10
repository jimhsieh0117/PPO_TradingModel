"""
PPO Trading Environment - Gymnasium 實現
基於 ICT 策略的加密貨幣交易環境

作為交易員的核心原則：
1. 風險第一，利潤第二
2. 止損是保護本金的防線
3. 不過度交易（手續費會吃掉利潤）
4. 讓利潤奔跑，快速止損
"""

import math
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any
from collections import deque

from environment.features.feature_aggregator import FeatureAggregator
from utils.data_pipeline import FEATURE_COLUMNS

# 從 FEATURE_COLUMNS 取得特徵索引，避免寫死 magic number
_IDX_ADX_NORMALIZED = FEATURE_COLUMNS.index('adx_normalized')
_IDX_VOLATILITY_REGIME = FEATURE_COLUMNS.index('volatility_regime')


class TradingEnv(gym.Env):
    """
    加密貨幣交易環境（機構交易員視角）

    動作空間:
        0: 平倉 (Close Position)
        1: 做多 (Long)
        2: 做空 (Short)
        3: 持有 (Hold)

    觀察空間:
        N 維 = 市場特徵 + 5 持倉狀態特徵（市場特徵數量由 FeatureAggregator 定義）

    止損設計:
        - ATR 動態止損（2x ATR 倍數，適應市場波動）
        - 追蹤止損（止損價只朝有利方向移動，鎖住利潤）
        - 固定百分比作為 ATR 不可用時的 fallback
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000.0,
        leverage: int = 10,
        position_size_pct: float = 0.15,
        stop_loss_pct: float = 0.015,
        max_daily_drawdown: float = 0.10,
        trading_fee: float = 0.0004,
        slippage: float = 0.0,
        episode_length: int = 1440,
        feature_config: Optional[Dict] = None,
        reward_config: Optional[Dict] = None,
        precomputed_features: Optional[np.ndarray] = None,
        atr_stop_multiplier: float = 2.0,
        trailing_stop: bool = True,
    ):
        """
        初始化交易環境

        Args:
            df: OHLCV 數據（必須包含 timestamp, open, high, low, close, volume）
            initial_balance: 初始資金（USDT）
            leverage: 槓桿倍數
            position_size_pct: 每次開倉使用的資金比例
            stop_loss_pct: 止損百分比（ATR 不可用時的 fallback）
            max_daily_drawdown: 單日最大回撤限制
            trading_fee: 交易手續費（taker fee）
            episode_length: 每個 episode 的步數（1440 = 24小時）
            feature_config: 特徵檢測器配置
            reward_config: 獎勵函數配置
            atr_stop_multiplier: ATR 止損倍數（預設 2.0）
            trailing_stop: 是否啟用追蹤止損
        """
        super().__init__()

        # === 市場數據 ===
        # 確保數據有正確的 datetime index（MultiTimeframeAnalyzer 需要）
        if 'timestamp' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df_copy = df.copy()
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
            df_copy = df_copy.set_index('timestamp')
            self.df = df_copy
        else:
            self.df = df.copy()

        self.total_steps = len(self.df)
        self.episode_length = episode_length

        # === 交易參數（交易員視角）===
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.position_size_pct = position_size_pct  # 15%
        self.actual_exposure_pct = position_size_pct * leverage  # 150%
        self.stop_loss_pct = stop_loss_pct  # 1.5%（ATR fallback）
        self.max_daily_drawdown = max_daily_drawdown  # 10%
        self.trading_fee = trading_fee  # 0.04%
        self.slippage = slippage  # 滑點（0.05% = 0.0005）

        # === ATR 動態止損 + 追蹤止損 ===
        self.atr_period = 14
        self.atr_stop_multiplier = atr_stop_multiplier
        self.trailing_stop_enabled = trailing_stop

        # === 動態 ATR 止損倍數（根據 volatility_regime 線性插值）===
        self.dynamic_atr_stop = bool((reward_config or {}).get('dynamic_atr_stop', True))
        self.atr_stop_high_vol = float((reward_config or {}).get('atr_stop_high_vol', 1.5))
        self.atr_stop_low_vol = float((reward_config or {}).get('atr_stop_low_vol', 2.5))

        # === 最大持倉時間 ===
        self.max_holding_steps = int((reward_config or {}).get('max_holding_steps', 360))

        # === 特徵提取器 ===
        self.feature_aggregator = FeatureAggregator(
            config=feature_config
        )

        # === 優化：預提取價格數據為 NumPy 數組（避免 DataFrame 訪問開銷）===
        self._close_prices = self.df['close'].to_numpy(dtype=np.float64)
        self._high_prices = self.df['high'].to_numpy(dtype=np.float64)
        self._low_prices = self.df['low'].to_numpy(dtype=np.float64)
        self._open_prices = self.df['open'].to_numpy(dtype=np.float64)

        # === 預計算 ATR 數組（供動態止損使用）===
        self._atr_values = self._precompute_atr()

        # === 優化：預計算所有特徵 ===
        if precomputed_features is not None:
            # 使用外部傳入的預計算特徵（避免多環境重複計算）
            self.feature_aggregator._feature_cache = precomputed_features
            self.feature_aggregator._cache_valid = True
        else:
            # 單環境模式：自行預計算
            print("[TradingEnv] Precomputing features...")
            self.feature_aggregator.precompute_all_features(self.df, verbose=True)

        # === 預計算 volatility_regime / adx_normalized 數組 ===
        self._adx_values = self.feature_aggregator._feature_cache[:, _IDX_ADX_NORMALIZED]
        self._vol_regime_values = self.feature_aggregator._feature_cache[:, _IDX_VOLATILITY_REGIME]

        # === 獎勵參數 (v8.0：盈虧信號 + 品質獎勵 + 頻率懲罰) ===
        self.reward_config = reward_config or {}

        # v8.0：已實現盈虧 + 浮動信號 + 品質獎勵
        self.pnl_reward_scale = float(self.reward_config.get('pnl_reward_scale', 500))
        self.floating_reward_scale = float(self.reward_config.get('floating_reward_scale', 80))  # v8.0: 降至 80
        self.stop_loss_extra_penalty = float(self.reward_config.get('stop_loss_extra_penalty', 3.0))
        self.take_profit_multiplier = float(self.reward_config.get('take_profit_multiplier', 1.7))  # 止盈獎勵倍數（相對止損 1.0x）
        self.holding_bonus_max = float(self.reward_config.get('holding_bonus_max', 1.5))  # v8.0: 盈利持倉獎勵
        self.holding_bonus_steps = float(self.reward_config.get('holding_bonus_steps', 30.0))  # 達到最大獎勵的步數
        self.rapid_reentry_penalty = float(self.reward_config.get('rapid_reentry_penalty', 0.5))  # v8.0: 頻繁交易懲罰
        self.rapid_reentry_threshold = int(self.reward_config.get('rapid_reentry_threshold', 3))  # 快速重開倉閾值（步）
        self.episode_profit_bonus = float(self.reward_config.get('episode_profit_bonus', 100))  # Episode 結算獎勵縮放

        # === 空倉機會成本 ===
        self.idle_penalty_enabled = bool(self.reward_config.get('idle_penalty_enabled', False))
        self.idle_penalty_atr_threshold = float(self.reward_config.get('idle_penalty_atr_threshold', 0.5))
        self.idle_penalty_scale = float(self.reward_config.get('idle_penalty_scale', 0.3))
        self.idle_penalty_cooldown = int(self.reward_config.get('idle_penalty_cooldown', 5))

        # === 低波動持倉獎勵 ===
        self.low_vol_hold_bonus = float(self.reward_config.get('low_vol_hold_bonus', 0.0))
        self.low_vol_threshold = float(self.reward_config.get('low_vol_threshold', 0.3))

        # === Regime-Conditional Reward ===
        self.regime_reward_enabled = bool(self.reward_config.get('regime_reward_enabled', True))
        self.regime_low_vol_threshold = float(self.reward_config.get('regime_low_vol_threshold', 0.3))
        self.regime_low_adx_threshold = float(self.reward_config.get('regime_low_adx_threshold', 0.2))
        self.regime_pnl_bonus = float(self.reward_config.get('regime_pnl_bonus', 1.5))

        # 夏普比率相關（用於統計追蹤，不用於獎勵）
        self.sharpe_window = int(self.reward_config.get('sharpe_window', 60))

        # === Gymnasium 空間定義 ===
        # 動作空間: 0=平倉, 1=做多, 2=做空, 3=持有
        self.action_space = spaces.Discrete(4)

        # 觀察空間: N 維 = 市場特徵 + 5 持倉狀態特徵（動態計算）
        # 市場特徵由 FeatureAggregator 定義
        # 持倉: position_state, floating_pnl_pct, holding_time_norm,
        #        distance_to_stop_loss, equity_change_pct = 5
        n_market_features = self.feature_aggregator.get_state_dimension()
        n_position_features = 5
        obs_dim = n_market_features + n_position_features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # === 交易狀態 ===
        self.balance = initial_balance
        self.equity = initial_balance
        self.position = 0  # 0=無倉位, 1=多倉, -1=空倉
        self.position_size = 0.0  # 當前持倉數量（BTC）
        self.entry_price = 0.0  # 開倉價格
        self.stop_loss_price = 0.0  # 止損價格

        # === Episode 追蹤 ===
        self.current_step = 0
        self.episode_start_step = 0
        self.daily_start_balance = initial_balance
        self.recent_returns = deque(maxlen=self.sharpe_window)  # 用於計算夏普比率
        self.previous_sharpe = 0.0
        self.last_realized_pnl = 0.0
        self.realized_this_step = False
        self.no_position_steps = 0

        # === 交易統計（交易員最關心的指標）===
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.stop_loss_count = 0
        self.holding_time = 0  # 當前持倉時間（K線數）

        # v3 新增：追蹤上次開倉時間（用於檢測快速換倉）
        self.last_open_step = -999  # 上次開倉的 step（初始化為遠古時間）

        # v4 新增：追蹤連續 Hold 次數（用於檢測過度保守）
        self.consecutive_hold_steps = 0  # 連續 Hold 的步數

        # v5 新增：追蹤上次平倉時間（用於檢測過度平倉）
        self.last_close_step = -999  # 上次平倉的 step

        # === 權益曲線（用於計算回撤）===
        self.equity_curve = [initial_balance]
        self.peak_equity = initial_balance

        # === Reward Clipping (v10: 關閉 EMA normalization，改用固定 clip) ===
        # EMA normalization 會因策略變化導致 reward 分布漂移，造成 value loss 上升
        # 改用簡單 clip：穩定、無狀態、不受策略變化影響
        self.normalize_reward = False
        self._reward_clip = 10.0         # Clip raw rewards to [-10, 10]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        重置環境到初始狀態

        Returns:
            observation: 初始觀察（20維特徵）
            info: 額外信息
        """
        super().reset(seed=seed)

        # 隨機選擇一個起始點（確保有足夠的歷史數據和未來數據）
        max_start = self.total_steps - self.episode_length - 100
        self.episode_start_step = self.np_random.integers(100, max_start)
        self.current_step = self.episode_start_step

        # 重置交易狀態
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.stop_loss_price = 0.0

        # 重置 episode 追蹤
        self.daily_start_balance = self.initial_balance
        self.recent_returns.clear()
        self.previous_sharpe = 0.0
        self.last_realized_pnl = 0.0
        self.realized_this_step = False
        self.no_position_steps = 0

        # 重置交易統計
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.stop_loss_count = 0
        self.holding_time = 0

        # 重置動作追蹤（v3/v4/v5）
        self.last_open_step = -999
        self.consecutive_hold_steps = 0
        self.last_close_step = -999

        # 重置權益曲線
        self.equity_curve = [self.initial_balance]
        self.peak_equity = self.initial_balance

        # 注意：reward normalization EMA 統計不重置，保持跨 episode 連續性
        # 只重置 step count 以確保 warmup 正確（可選，這裡保持累積）

        # 獲取初始觀察
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        執行一步交易動作

        Args:
            action: 0=平倉, 1=做多, 2=做空, 3=持有

        Returns:
            observation: 下一步觀察
            reward: 獎勵
            terminated: 是否終止（觸發停止條件）
            truncated: 是否截斷（episode 結束）
            info: 額外信息
        """
        # 獲取當前價格（使用 NumPy 數組，O(1) 操作）
        current_price = self._close_prices[self.current_step]
        self.realized_this_step = False
        self.last_realized_pnl = 0.0

        # === 1. 先檢查止損（用盤中高低點），觸發後不再更新追蹤止損 ===
        stop_loss_triggered = self._check_stop_loss()
        if not stop_loss_triggered:
            self._update_trailing_stop(current_price)

        # === 1.5 最大持倉時間檢查（避免被動式長時間曝險）===
        if (not stop_loss_triggered
                and self.position != 0
                and self.holding_time >= self.max_holding_steps):
            self._close_position(current_price, reason="max_holding_time")

        # === 2. 執行交易動作 ===
        trade_executed = False
        if not stop_loss_triggered:
            trade_executed = self._execute_action(action, current_price)

        if self.position == 0:
            self.no_position_steps += 1
        else:
            self.no_position_steps = 0

        # === 3. 更新權益 ===
        self._update_equity(current_price)

        # === 4. 計算獎勵 ===
        will_truncate = (self.current_step + 1 >= self.episode_start_step + self.episode_length)
        reward = self._calculate_reward(
            action=action,
            trade_executed=trade_executed,
            stop_loss_triggered=stop_loss_triggered,
            episode_done=will_truncate
        )

        # === 5. 檢查終止條件 ===
        terminated = self._check_termination()

        # === 6. 前進到下一步 ===
        self.current_step += 1
        if self.position != 0:
            self.holding_time += 1

        # === 7. 檢查 episode 是否結束 ===
        truncated = bool(self.current_step >= self.episode_start_step + self.episode_length)

        # 獲取下一步觀察
        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        獲取當前觀察（28維 = 23 ICT 特徵 + 5 持倉狀態）

        交易員視角：我需要知道市場結構、關鍵區域、流動性、波動率、時段，
        以及自己的持倉狀態（方向、盈虧、風險距離）
        """
        # 23 維市場特徵（20 ICT + atr_normalized + hour_sin + hour_cos）
        market_features = self.feature_aggregator.get_state_vector(self.df, self.current_step)

        # 5 維持倉狀態特徵
        current_price = self._close_prices[self.current_step]

        # (21) 持倉方向: -1=空倉, 0=無倉位, 1=多倉
        position_state = float(self.position)

        # (22) 浮動盈虧百分比 (clipped to [-1, 1])
        if self.position != 0 and self.entry_price > 0:
            if self.position == 1:
                floating_pnl_pct = (current_price - self.entry_price) / self.entry_price
            else:
                floating_pnl_pct = (self.entry_price - current_price) / self.entry_price
            floating_pnl_pct = np.clip(floating_pnl_pct, -1.0, 1.0)
        else:
            floating_pnl_pct = 0.0

        # (23) 持倉時間正規化 (0~1, 以 120 步 = 2 小時為上限)
        holding_time_norm = min(self.holding_time / 120.0, 1.0) if self.position != 0 else 0.0

        # (24) 距止損距離百分比 (0~1, 0=已觸及止損, 1=遠離止損)
        if self.position != 0 and self.entry_price > 0:
            if self.position == 1:  # 多倉: 價格越高離止損越遠
                dist_to_sl = (current_price - self.stop_loss_price) / (self.entry_price - self.stop_loss_price + 1e-10)
            else:  # 空倉: 價格越低離止損越遠
                dist_to_sl = (self.stop_loss_price - current_price) / (self.stop_loss_price - self.entry_price + 1e-10)
            dist_to_sl = np.clip(dist_to_sl, 0.0, 2.0) / 2.0  # 正規化到 0~1
        else:
            dist_to_sl = 0.0

        # (25) Episode 至今的權益變化百分比 (clipped to [-1, 1])
        equity_change_pct = (self.equity - self.initial_balance) / self.initial_balance
        equity_change_pct = np.clip(equity_change_pct, -1.0, 1.0)

        # 組合 28 維觀察向量（23 市場 + 5 持倉）
        position_features = np.array([
            position_state,
            floating_pnl_pct,
            holding_time_norm,
            dist_to_sl,
            equity_change_pct
        ], dtype=np.float32)

        return np.concatenate([market_features, position_features])

    def _precompute_atr(self) -> np.ndarray:
        """預計算整個數據集的 ATR 數組（供動態止損使用）"""
        highs = self._high_prices
        lows = self._low_prices
        closes = self._close_prices
        prev_close = np.roll(closes, 1)
        prev_close[0] = closes[0]
        true_range = np.maximum(
            highs - lows,
            np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close))
        )
        import pandas as _pd
        atr = _pd.Series(true_range).rolling(self.atr_period, min_periods=1).mean().to_numpy()
        return atr

    def _update_trailing_stop(self, current_price: float):
        """
        追蹤止損：止損價只能朝有利方向移動，鎖住已有利潤。
        """
        if self.position == 0 or not self.trailing_stop_enabled:
            return

        atr = self._atr_values[self.current_step]

        if self.position == 1:  # 多倉：止損只能上移
            new_sl = current_price - self.atr_stop_multiplier * atr
            if new_sl > self.stop_loss_price:
                self.stop_loss_price = new_sl
        else:  # 空倉：止損只能下移
            new_sl = current_price + self.atr_stop_multiplier * atr
            if new_sl < self.stop_loss_price:
                self.stop_loss_price = new_sl

    def _check_stop_loss(self) -> bool:
        """
        檢查是否觸發止損（使用 K 線盤中最高/最低點，與真實合約和 backtesting.py 一致）

        觸發邏輯：
          - 多倉：K 線最低點 <= stop_loss_price → 以 stop_loss_price 成交（加滑點）
          - 空倉：K 線最高點 >= stop_loss_price → 以 stop_loss_price 成交（加滑點）
        """
        if self.position == 0:
            return False

        low = self._low_prices[self.current_step]
        high = self._high_prices[self.current_step]

        # 多倉止損：K 線最低點觸及止損價
        if self.position == 1 and low <= self.stop_loss_price:
            self._close_position(self.stop_loss_price, reason="stop_loss")
            self.stop_loss_count += 1
            return True

        # 空倉止損：K 線最高點觸及止損價
        if self.position == -1 and high >= self.stop_loss_price:
            self._close_position(self.stop_loss_price, reason="stop_loss")
            self.stop_loss_count += 1
            return True

        return False

    def _execute_action(self, action: int, current_price: float) -> bool:
        """
        執行交易動作

        Args:
            action: 0=平倉, 1=做多, 2=做空, 3=持有
            current_price: 當前價格

        Returns:
            trade_executed: 是否執行了交易
        """
        # 動作 3: 持有（不做任何操作）
        if action == 3:
            return False

        # 動作 0: 平倉
        if action == 0 and self.position != 0:
            self._close_position(current_price, reason="manual_close")
            return True

        # 動作 1: 做多
        if action == 1:
            # 如果已有空倉，先平倉
            if self.position == -1:
                self._close_position(current_price, reason="reverse")
            # 如果無倉位，開多倉
            if self.position == 0:
                self._open_position(direction=1, current_price=current_price)
                return True

        # 動作 2: 做空
        if action == 2:
            # 如果已有多倉，先平倉
            if self.position == 1:
                self._close_position(current_price, reason="reverse")
            # 如果無倉位，開空倉
            if self.position == 0:
                self._open_position(direction=-1, current_price=current_price)
                return True

        return False

    def _open_position(self, direction: int, current_price: float):
        """
        開倉（交易員視角：嚴格風險控制）

        Args:
            direction: 1=做多, -1=做空
            current_price: 開倉價格
        """
        # 計算倉位大小（15% 資金）
        position_value = self.balance * self.position_size_pct

        # 計算實際購買數量（考慮槓桿）
        self.position_size = (position_value * self.leverage) / current_price

        # 記錄開倉信息（含滑點：買入成交價偏高，賣出成交價偏低）
        self.position = direction
        if direction == 1:  # 做多（買入）：成交價偏高
            self.entry_price = current_price * (1 + self.slippage)
        else:  # 做空（賣出）：成交價偏低
            self.entry_price = current_price * (1 - self.slippage)
        self.holding_time = 0

        # 設置止損價格（ATR 動態止損，fallback 到固定百分比）
        atr = self._atr_values[self.current_step]
        if atr > 0:
            # 動態 ATR 倍數：根據 volatility_regime 線性插值
            # vol_regime=0 → low_vol(2.5x), vol_regime=1 → high_vol(1.5x)
            if self.dynamic_atr_stop:
                vol_regime = self._vol_regime_values[self.current_step]
                effective_multiplier = (self.atr_stop_low_vol
                                        + (self.atr_stop_high_vol - self.atr_stop_low_vol)
                                        * vol_regime)
            else:
                effective_multiplier = self.atr_stop_multiplier

            if direction == 1:  # 多倉止損在下方
                self.stop_loss_price = current_price - effective_multiplier * atr
            else:  # 空倉止損在上方
                self.stop_loss_price = current_price + effective_multiplier * atr
        else:
            # ATR 不可用（warmup 期），使用固定百分比
            if direction == 1:
                self.stop_loss_price = current_price * (1 - self.stop_loss_pct)
            else:
                self.stop_loss_price = current_price * (1 + self.stop_loss_pct)

        # 扣除手續費
        fee = position_value * self.leverage * self.trading_fee
        self.balance -= fee

        self.total_trades += 1

    def _close_position(self, exit_price_raw: float, reason: str):
        """
        平倉（使用開倉時鎖定的持幣數量計算盈虧，與真實合約一致）

        Args:
            exit_price_raw: 平倉基準價（止損價或收盤價），滑點在此基礎上加減
            reason: 平倉原因（manual_close, stop_loss, reverse, max_holding_time）

        修正：使用開倉鎖定的 self.position_size（持幣數），而非以當前 balance 重算倉位
              確保帳戶虧損後，倉位大小不會縮水，PnL 計算符合真實合約邏輯
        """
        if self.position == 0:
            return

        # 計算含滑點的實際成交價
        # 平多（賣出）：成交價偏低；平空（買入）：成交價偏高
        if self.position == 1:  # 平多倉（賣出）
            exit_price = exit_price_raw * (1 - self.slippage)
            pnl = self.position_size * (exit_price - self.entry_price)
        else:  # 平空倉（買入）
            exit_price = exit_price_raw * (1 + self.slippage)
            pnl = self.position_size * (self.entry_price - exit_price)

        # 手續費 = 平倉名義金額（基準價）× fee rate
        fee = self.position_size * exit_price_raw * self.trading_fee
        pnl -= fee

        # 更新餘額
        self.balance += pnl

        # 統計交易結果
        if pnl > 0:
            self.winning_trades += 1
            self.total_profit += pnl
        else:
            self.losing_trades += 1
            self.total_loss += abs(pnl)

        # 記錄回報率（用於計算夏普比率）
        self.last_realized_pnl = pnl
        self.realized_this_step = True

        # 重置倉位
        self.position = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.stop_loss_price = 0.0
        self.holding_time = 0

    def _update_equity(self, current_price: float):
        """
        更新權益（使用開倉時鎖定的持幣數量計算浮動盈虧，與真實合約一致）
        """
        floating_pnl = 0.0
        if self.position != 0 and self.position_size > 0:
            if self.position == 1:  # 多倉
                floating_pnl = self.position_size * (current_price - self.entry_price)
            else:  # 空倉
                floating_pnl = self.position_size * (self.entry_price - current_price)

        self.equity = self.balance + floating_pnl
        self.equity_curve.append(self.equity)

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

    def _calculate_reward(
        self,
        action: int,
        trade_executed: bool,
        stop_loss_triggered: bool,
        episode_done: bool = False
    ) -> float:
        """
        【v8.0】盈虧信號 + 交易品質獎勵 + 頻率懲罰

        目的：讓 Value Network 能學到「什麼狀態能帶來利潤」

        設計原則：
        1. 主要獎勵：已實現盈虧（平倉時）
        2. 輔助信號：浮動盈虧（每步，權重降低以避免壓過主信號）
        3. 品質獎勵：盈利交易持倉越久獎勵越高（鼓勵讓利潤奔跑）
        4. 頻率懲罰：快速重新開倉給予懲罰（減少無效交易）
        5. 止損額外懲罰
        6. 獎勵範圍：約 [-10, +10]

        v8.0 改進：
        - 浮動信號權重由 config 控制（建議 50~80）
        - 新增盈利交易品質獎勵
        - 新增頻繁交易懲罰
        """
        reward = 0.0

        # 追蹤 equity 變化（用於統計）
        equity_change = self.equity - self.equity_curve[-2] if len(self.equity_curve) > 1 else 0.0
        step_return = equity_change / self.initial_balance
        self.recent_returns.append(step_return)

        # === 核心：已實現盈虧（平倉時才有，主要獎勵）===
        if self.realized_this_step:
            realized_return = self.last_realized_pnl / self.initial_balance
            # 不對稱獎勵：止盈 × take_profit_multiplier（1.7x），止損 × 1.0x
            if self.last_realized_pnl > 0:
                reward = realized_return * self.pnl_reward_scale * self.take_profit_multiplier
            else:
                reward = realized_return * self.pnl_reward_scale

            # === Regime-Conditional 獎勵調節 ===
            # 在橫盤低波動期：盈利交易額外獎勵（精準出手），虧損交易加重懲罰（抑制假突破）
            if self.regime_reward_enabled:
                vol_regime = self._vol_regime_values[self.current_step]
                adx = self._adx_values[self.current_step]
                is_choppy = (vol_regime < self.regime_low_vol_threshold
                             and adx < self.regime_low_adx_threshold)
                if is_choppy:
                    if self.last_realized_pnl > 0:
                        reward *= self.regime_pnl_bonus   # 精準出手獎勵 (×1.5)
                    else:
                        reward *= 1.3                     # 假突破懲罰加重 (×1.3)

            # === v8.0 新增：盈利交易品質獎勵（鼓勵讓利潤奔跑）===
            if self.last_realized_pnl > 0:
                # 持倉越久的盈利交易，給越多 bonus（非線性飽和）
                # holding_time=30 步時達到最大獎勵
                holding_ratio = min(self.holding_time / self.holding_bonus_steps, 1.0)
                holding_bonus = holding_ratio * self.holding_bonus_max
                reward += holding_bonus

        # === 浮動盈虧信號（每步，輔助學習）===
        if self.position != 0 and self.entry_price > 0:
            current_price = self._close_prices[self.current_step]
            if self.position == 1:  # 多倉
                floating_pct = (current_price - self.entry_price) / self.entry_price
            else:  # 空倉
                floating_pct = (self.entry_price - current_price) / self.entry_price

            # 浮動信號：權重由 config 控制（v8.0 建議 50~80）
            reward += floating_pct * self.floating_reward_scale

        # === 空倉機會成本（只在無持倉時觸發）===
        if self.idle_penalty_enabled and self.position == 0:
            steps_since_close = self.current_step - self.last_close_step
            if steps_since_close > self.idle_penalty_cooldown:
                current_price = self._close_prices[self.current_step]
                if self.current_step > 0:
                    prev_price = self._close_prices[self.current_step - 1]
                    price_move = abs(current_price - prev_price)
                else:
                    price_move = 0.0

                atr_val = self._atr_values[self.current_step]
                if atr_val > 0:
                    move_ratio = price_move / atr_val
                    if move_ratio > self.idle_penalty_atr_threshold:
                        penalty = (move_ratio - self.idle_penalty_atr_threshold) * self.idle_penalty_scale
                        reward -= min(penalty, 1.0)

        # === 低波動持倉獎勵（鼓勵在低波動期耐心等待）===
        if self.position == 0 and self.low_vol_hold_bonus > 0:
            vol_regime = self._vol_regime_values[self.current_step]
            if vol_regime < self.low_vol_threshold:
                reward += self.low_vol_hold_bonus

        # === v8.0 新增：頻繁交易懲罰（抑制無效交易）===
        if trade_executed and action in [1, 2]:  # 開倉動作
            steps_since_last_close = self.current_step - self.last_close_step
            if steps_since_last_close < self.rapid_reentry_threshold:
                reward -= self.rapid_reentry_penalty

        # === 止損額外懲罰（風險管理信號）===
        if stop_loss_triggered:
            reward -= self.stop_loss_extra_penalty

        # === Episode 結算獎勵（整體表現回饋）===
        if episode_done:
            episode_return_pct = (self.equity - self.initial_balance) / self.initial_balance
            reward += episode_return_pct * self.episode_profit_bonus

        # === 更新追蹤狀態 ===
        if action in [1, 2]:
            self.last_open_step = self.current_step
        if action == 0 and self.realized_this_step:
            self.last_close_step = self.current_step
        if action == 3:
            self.consecutive_hold_steps += 1
        else:
            self.consecutive_hold_steps = 0

        # 清理標記
        self.realized_this_step = False
        self.last_realized_pnl = 0.0

        # Soft clip：tanh 壓縮，保留量級資訊（大虧 vs 小虧可區分）
        reward = self._reward_clip * math.tanh(reward / self._reward_clip)

        return reward

    def _check_termination(self) -> bool:
        """
        檢查是否觸發終止條件

        交易員視角：保護本金是首要任務
        """
        # 檢查單日回撤是否超過限制（10%）
        daily_drawdown = (self.daily_start_balance - self.equity) / self.daily_start_balance
        if daily_drawdown > self.max_daily_drawdown:
            return True

        # 檢查賬戶是否爆倉（餘額 < 初始資金的 20%）
        if self.equity < self.initial_balance * 0.2:
            return True

        return False

    def _get_info(self) -> Dict[str, Any]:
        """
        獲取額外信息（用於監控和調試）
        """
        # 計算當前最大回撤
        current_drawdown = (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0

        # 計算勝率
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0

        # 計算盈虧比
        profit_factor = self.total_profit / self.total_loss if self.total_loss > 0 else 0

        return {
            "step": self.current_step,
            "balance": self.balance,
            "equity": self.equity,
            "position": self.position,
            "position_size": self.position_size,
            "entry_price": self.entry_price,
            "stop_loss_price": self.stop_loss_price,
            "holding_time": self.holding_time,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "stop_loss_count": self.stop_loss_count,
            "max_drawdown": current_drawdown,
            "total_return_pct": (self.equity - self.initial_balance) / self.initial_balance * 100
        }

    def render(self):
        """渲染環境（可選）"""
        if len(self.equity_curve) > 0:
            print(f"Step: {self.current_step} | "
                  f"Equity: ${self.equity:.2f} | "
                  f"Position: {self.position} | "
                  f"Trades: {self.total_trades} | "
                  f"Win Rate: {self.winning_trades / max(self.total_trades, 1) * 100:.1f}%")

    def close(self):
        """關閉環境"""
        pass

