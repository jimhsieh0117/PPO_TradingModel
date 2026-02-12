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


class TradingEnv(gym.Env):
    """
    加密貨幣交易環境（機構交易員視角）

    動作空間:
        0: 平倉 (Close Position)
        1: 做多 (Long)
        2: 做空 (Short)
        3: 持有 (Hold)

    觀察空間:
        20 維 ICT 特徵向量

    獎勵設計:
        - 即時損益（正規化）
        - 夏普比率改善
        - 高風險倉位懲罰
        - 止損嚴重懲罰
        - 交易成本
        - 持倉時間獎勵
        - 單日回撤限制懲罰
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
        episode_length: int = 1440,
        feature_config: Optional[Dict] = None,
        reward_config: Optional[Dict] = None,
        precomputed_features: Optional[np.ndarray] = None
    ):
        """
        初始化交易環境

        Args:
            df: OHLCV 數據（必須包含 timestamp, open, high, low, close, volume）
            initial_balance: 初始資金（USDT）
            leverage: 槓桿倍數
            position_size_pct: 每次開倉使用的資金比例
            stop_loss_pct: 止損百分比（價格波動）
            max_daily_drawdown: 單日最大回撤限制
            trading_fee: 交易手續費（taker fee）
            episode_length: 每個 episode 的步數（1440 = 24小時）
            feature_config: 特徵檢測器配置
            reward_config: 獎勵函數配置
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
        self.stop_loss_pct = stop_loss_pct  # 1.5%
        self.max_daily_drawdown = max_daily_drawdown  # 10%
        self.trading_fee = trading_fee  # 0.04%

        # === 特徵提取器 ===
        self.feature_aggregator = FeatureAggregator(
            config=feature_config
        )

        # === 優化：預提取價格數據為 NumPy 數組（避免 DataFrame 訪問開銷）===
        self._close_prices = self.df['close'].to_numpy(dtype=np.float64)
        self._high_prices = self.df['high'].to_numpy(dtype=np.float64)
        self._low_prices = self.df['low'].to_numpy(dtype=np.float64)
        self._open_prices = self.df['open'].to_numpy(dtype=np.float64)

        # === 優化：預計算所有特徵 ===
        if precomputed_features is not None:
            # 使用外部傳入的預計算特徵（避免多環境重複計算）
            self.feature_aggregator._feature_cache = precomputed_features
            self.feature_aggregator._cache_valid = True
        else:
            # 單環境模式：自行預計算
            print("[TradingEnv] Precomputing features...")
            self.feature_aggregator.precompute_all_features(self.df, verbose=True)

        # === 獎勵參數 ===
        self.reward_config = reward_config or {}

        # 新版：基礎盈利獎勵
        self.profit_multiplier = float(self.reward_config.get('profit_multiplier', 500))
        self.loss_multiplier = float(self.reward_config.get('loss_multiplier', 100))

        # 舊版權重（保留兼容性）
        self.floating_weight = float(
            self.reward_config.get('floating_weight', self.reward_config.get('pnl_weight', 1.0))
        )
        self.realized_weight = float(
            self.reward_config.get('realized_weight', self.reward_config.get('pnl_weight', 1.0))
        )

        # 開倉激勵（關鍵！）
        self.open_position_reward = float(self.reward_config.get('open_position_reward', 20.0))
        self.profitable_open_bonus = float(self.reward_config.get('profitable_open_bonus', 50.0))

        # 交易成本調整
        self.trading_fee_multiplier = float(self.reward_config.get('trading_fee_multiplier', 0.3))

        # 空倉懲罰
        self.no_position_window = int(self.reward_config.get('no_position_window', 20))
        self.no_position_penalty = float(self.reward_config.get('no_position_penalty', -1.0))

        # 其他參數
        self.sharpe_weight = float(self.reward_config.get('sharpe_weight', 0.3))
        self.sharpe_window = int(self.reward_config.get('sharpe_window', 60))
        self.high_risk_penalty = float(self.reward_config.get('high_risk_penalty', -15))
        self.stop_loss_penalty = float(self.reward_config.get('stop_loss_penalty', -200))
        self.daily_drawdown_penalty = float(self.reward_config.get('daily_drawdown_penalty', -400))
        self.holding_time_reward = float(self.reward_config.get('holding_time_reward', 2.0))
        self.max_holding_reward_bars = int(self.reward_config.get('max_holding_reward_bars', 120))

        # 平倉獎勵（v3 新增）
        self.close_position_reward = float(self.reward_config.get('close_position_reward', 20.0))

        # 平倉訊號獎勵（v3 調整）
        self.take_profit_reward = float(self.reward_config.get('take_profit_reward', 50.0))
        self.take_profit_threshold = float(self.reward_config.get('take_profit_threshold', 0.001))
        self.cut_loss_early_reward = float(self.reward_config.get('cut_loss_early_reward', 30.0))
        self.cut_loss_threshold_min = float(self.reward_config.get('cut_loss_threshold_min', -0.015))
        self.cut_loss_threshold_max = float(self.reward_config.get('cut_loss_threshold_max', -0.005))

        # 快速換倉懲罰（v4 調整）
        self.rapid_switch_penalty = float(self.reward_config.get('rapid_switch_penalty', -15.0))
        self.rapid_switch_window = int(self.reward_config.get('rapid_switch_window', 5))

        # 持有獎勵/懲罰（v4 調整）
        self.holding_profit_reward = float(self.reward_config.get('holding_profit_reward', 2.0))
        self.holding_loss_penalty = float(self.reward_config.get('holding_loss_penalty', -2.0))
        self.holding_loss_threshold = float(self.reward_config.get('holding_loss_threshold', 0.015))

        # 過度 Hold 懲罰（v4 新增）
        self.excessive_hold_penalty = float(self.reward_config.get('excessive_hold_penalty', -1.0))
        self.excessive_hold_window = int(self.reward_config.get('excessive_hold_window', 20))

        # 持倉時間限制（v4 調整）
        self.max_holding_bars = int(self.reward_config.get('max_holding_bars', 120))
        self.min_holding_bars = int(self.reward_config.get('min_holding_bars', 3))
        self.overtime_holding_penalty = float(self.reward_config.get('overtime_holding_penalty', -15.0))
        self.undertime_holding_penalty = float(self.reward_config.get('undertime_holding_penalty', -5.0))

        # === Gymnasium 空間定義 ===
        # 動作空間: 0=平倉, 1=做多, 2=做空, 3=持有
        self.action_space = spaces.Discrete(4)

        # 觀察空間: 20 維 ICT 特徵
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(20,),
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

        # === 權益曲線（用於計算回撤）===
        self.equity_curve = [initial_balance]
        self.peak_equity = initial_balance
        # === ??????Welford?===
        self.normalize_reward = True
        self._reward_count = 0
        self._reward_mean = 0.0
        self._reward_m2 = 0.0
        self._reward_clip = 10.0

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
        self.no_position_steps = 0

        # 重置交易統計
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.stop_loss_count = 0
        self.holding_time = 0

        # 重置權益曲線
        self.equity_curve = [self.initial_balance]
        self.peak_equity = self.initial_balance

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

        # === 1. 檢查止損（交易員的生命線）===
        stop_loss_triggered = self._check_stop_loss(current_price)

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
        reward = self._calculate_reward(
            action=action,
            trade_executed=trade_executed,
            stop_loss_triggered=stop_loss_triggered
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
        獲取當前觀察（20維 ICT 特徵）

        交易員視角：我需要知道市場結構、關鍵區域、流動性等信息
        """
        features = self.feature_aggregator.get_state_vector(self.df, self.current_step)
        return features

    def _check_stop_loss(self, current_price: float) -> bool:
        """
        檢查是否觸發止損

        交易員原則：止損是保護本金的防線，絕不能違反！
        """
        if self.position == 0:
            return False

        # 多倉止損：價格跌破止損價
        if self.position == 1 and current_price <= self.stop_loss_price:
            self._close_position(current_price, reason="stop_loss")
            self.stop_loss_count += 1
            return True

        # 空倉止損：價格突破止損價
        if self.position == -1 and current_price >= self.stop_loss_price:
            self._close_position(current_price, reason="stop_loss")
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

        # 記錄開倉信息
        self.position = direction
        self.entry_price = current_price
        self.holding_time = 0

        # 設置止損價格（1.5% 價格波動）
        if direction == 1:  # 多倉止損在下方
            self.stop_loss_price = current_price * (1 - self.stop_loss_pct)
        else:  # 空倉止損在上方
            self.stop_loss_price = current_price * (1 + self.stop_loss_pct)

        # 扣除手續費
        fee = position_value * self.leverage * self.trading_fee
        self.balance -= fee

        self.total_trades += 1

    def _close_position(self, current_price: float, reason: str):
        """
        平倉（計算盈虧）

        Args:
            current_price: 平倉價格
            reason: 平倉原因（manual_close, stop_loss, reverse）
        """
        if self.position == 0:
            return

        # 計算盈虧（考慮槓桿）
        if self.position == 1:  # 平多倉
            price_change_pct = (current_price - self.entry_price) / self.entry_price
        else:  # 平空倉
            price_change_pct = (self.entry_price - current_price) / self.entry_price

        # 實際盈虧 = 倉位價值 × 槓桿 × 價格變化百分比
        position_value = self.balance * self.position_size_pct
        pnl = position_value * self.leverage * price_change_pct

        # 扣除手續費
        fee = position_value * self.leverage * self.trading_fee
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
        更新權益（用於計算回撤）

        交易員視角：權益曲線是評估表現的關鍵
        """
        # 如果有持倉，計算浮動盈虧
        floating_pnl = 0.0
        if self.position != 0:
            if self.position == 1:  # 多倉
                price_change_pct = (current_price - self.entry_price) / self.entry_price
            else:  # 空倉
                price_change_pct = (self.entry_price - current_price) / self.entry_price

            position_value = self.balance * self.position_size_pct
            floating_pnl = position_value * self.leverage * price_change_pct

        # 更新權益
        self.equity = self.balance + floating_pnl
        self.equity_curve.append(self.equity)

        # 更新峰值權益（用於計算回撤）
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

    def _calculate_reward(
        self,
        action: int,
        trade_executed: bool,
        stop_loss_triggered: bool
    ) -> float:
        """
        【v4】平衡版獎勵函數 - 修正 v3 的過度懲罰

        核心改變（相對 v3）：
        1. 提高開倉獎勵（3 → 10，修正過度保守）
        2. 降低平倉獎勵（20 → 15，平衡開倉/平倉）
        3. 【核心】大幅減輕快速換倉懲罰（-50 → -15，減少 70%）
        4. 【核心】縮短快速換倉窗口（10分鐘 → 5分鐘）
        5. 【核心】新增過度 Hold 懲罰（防止過於保守）
        6. 提高獲利門檻（0.1% → 0.2%，更現實）
        7. 略提高虧損懲罰（-1 → -2，鼓勵及時平倉）
        8. 放寬最短持倉時間（5分鐘 → 3分鐘）

        預期行為：
        - 完整循環（開倉→持有10分鐘→獲利平倉）= +85 獎勵
        - 快速換倉（5分鐘內）= +5 獎勵（仍可行但次優）
        - 過度 Hold（20分鐘不動作）= -20 懲罰（防止過於保守）
        """
        reward = 0.0

        equity_change = self.equity - self.equity_curve[-2] if len(self.equity_curve) > 1 else 0.0
        step_return = equity_change / self.initial_balance
        self.recent_returns.append(step_return)

        # === 1. 基礎盈利獎勵（大幅提高！）===
        if equity_change > 0:
            # 盈利時：使用高倍數獎勵
            reward += step_return * self.profit_multiplier
        else:
            # 虧損時：使用較低倍數懲罰（鼓勵冒險）
            reward += step_return * self.loss_multiplier

        # === 2. 開倉獎勵（v3 大幅降低）===
        if action in [1, 2]:  # Long or Short
            # 開倉行為本身給小獎勵（從 20 → 3）
            reward += self.open_position_reward

            # 【v3 新增】檢測快速換倉
            steps_since_last_open = self.current_step - self.last_open_step
            if steps_since_last_open < self.rapid_switch_window:
                # 快速換倉懲罰
                reward += self.rapid_switch_penalty

            # 更新上次開倉時間
            self.last_open_step = self.current_step

            # 如果開倉後有浮盈，額外獎勵（減少）
            if self.position != 0 and equity_change > 0:
                reward += self.profitable_open_bonus

        # === 3. 【v3 核心改進】平倉基礎獎勵 ===
        if action == 0 and self.realized_this_step:
            # 任何平倉都給基礎獎勵（鼓勵主動平倉）
            reward += self.close_position_reward

            # 計算已實現盈虧
            realized_return = self.last_realized_pnl / self.initial_balance
            if realized_return > 0:
                reward += realized_return * self.profit_multiplier * 0.5  # 平倉盈利
            else:
                reward += realized_return * self.loss_multiplier * 0.5  # 平倉虧損

            # 計算持倉時間（用於檢測過短平倉）
            if self.holding_time < self.min_holding_bars:
                # 過短持倉懲罰
                reward += self.undertime_holding_penalty

        # === 3.1 【v3 調整】獲利平倉額外獎勵（門檻大幅降低）===
        if action == 0 and self.realized_this_step and self.last_realized_pnl > 0:
            if self.entry_price > 0:
                position_value = self.balance * self.position_size_pct
                trade_return_pct = self.last_realized_pnl / position_value

                # 超過獲利門檻時給予額外獎勵（0.1% vs 舊版 0.7%）
                if trade_return_pct >= self.take_profit_threshold:
                    reward += self.take_profit_reward

        # === 3.2 【v3 調整】小虧主動平倉獎勵 ===
        if action == 0 and self.realized_this_step and self.last_realized_pnl < 0:
            if self.entry_price > 0:
                position_value = self.balance * self.position_size_pct
                trade_return_pct = self.last_realized_pnl / position_value

                # 在小虧範圍內主動平倉給予獎勵（鼓勵及時止損）
                if self.cut_loss_threshold_min <= trade_return_pct <= self.cut_loss_threshold_max:
                    reward += self.cut_loss_early_reward

        # === 3.3 【v3 新增】持有獲利倉位獎勵 + 持有虧損懲罰 ===
        if self.position != 0 and self.entry_price > 0:
            # 計算當前浮盈/浮虧百分比（使用 NumPy 數組，O(1) 操作）
            current_price_for_pnl = self._close_prices[self.current_step]
            if self.position == 1:  # 做多
                unrealized_pnl_pct = (current_price_for_pnl - self.entry_price) / self.entry_price
            else:  # 做空
                unrealized_pnl_pct = (self.entry_price - current_price_for_pnl) / self.entry_price

            # 持有獲利倉位獎勵（鼓勵持倉）
            if unrealized_pnl_pct > 0:
                reward += self.holding_profit_reward

            # 浮虧超過門檻時給予持續懲罰（但懲罰減輕）
            elif abs(unrealized_pnl_pct) >= self.holding_loss_threshold:
                reward += self.holding_loss_penalty

        # === 4. 夏普比率改善（保留）===
        if self.position != 0 and len(self.recent_returns) >= 2:
            returns_array = np.array(self.recent_returns, dtype=np.float64)
            std = returns_array.std()
            if std > 0.0:
                current_sharpe = (returns_array.mean() / (std + 1e-8)) * np.sqrt(252)
                sharpe_improvement = current_sharpe - self.previous_sharpe
                reward += sharpe_improvement * self.sharpe_weight
                self.previous_sharpe = current_sharpe

        # === 5. 交易成本懲罰（大幅減少！）===
        if trade_executed:
            position_value = self.balance * self.position_size_pct * self.leverage
            fee = position_value * self.trading_fee
            # 使用 trading_fee_multiplier 減少懲罰（0.3 = 減少 70%）
            reward -= (fee / self.initial_balance) * 100 * self.trading_fee_multiplier

        # === 6. 空倉懲罰（關鍵！）===
        if self.no_position_window > 0 and self.no_position_penalty != 0.0:
            if self.position == 0 and self.no_position_steps >= self.no_position_window:
                # 持續懲罰空倉
                reward += self.no_position_penalty

        # === 6.1 【v4 新增】過度 Hold 懲罰（防止過於保守）===
        if action == 3:  # Hold 動作
            self.consecutive_hold_steps += 1
            # 連續 Hold 超過門檻時懲罰
            if self.consecutive_hold_steps >= self.excessive_hold_window:
                reward += self.excessive_hold_penalty
        else:
            # 非 Hold 動作，重置計數器
            self.consecutive_hold_steps = 0

        # === 7. 高風險倉位懲罰（保留）===
        if self.position != 0:
            position_value = self.balance * self.position_size_pct * self.leverage
            exposure_ratio = position_value / self.equity if self.equity > 0 else 0.0
            if exposure_ratio > 0.5:
                reward += (exposure_ratio - 0.5) * self.high_risk_penalty

        # === 8. 止損重懲罰（保留嚴格）===
        if stop_loss_triggered:
            reward += self.stop_loss_penalty

        # === 9. 【v3 移除舊版持倉時間獎勵】===
        # 舊版機制已被 holding_profit_reward 取代

        # === 9.1 【v3 調整】超時持倉懲罰（收緊至 2 小時）===
        if self.position != 0 and self.holding_time > self.max_holding_bars:
            # 超過最大持倉時間時給予持續懲罰（120 根 K 線 = 2 小時）
            reward += self.overtime_holding_penalty

        # === 10. 單日回撤限制懲罰（保留嚴格）===
        daily_drawdown = (self.daily_start_balance - self.equity) / self.daily_start_balance
        if daily_drawdown > self.max_daily_drawdown:
            reward += self.daily_drawdown_penalty

        # 清理標記
        self.realized_this_step = False
        self.last_realized_pnl = 0.0

        # 正規化（如果啟用）
        if self.normalize_reward:
            reward = self._normalize_reward(reward)

        return reward

    def _normalize_reward(self, reward: float) -> float:
        """
        以Welfor線上統計做標準化，比免獎勵尺度過大
        """
        self._reward_count += 1
        delta = reward - self._reward_mean
        self._reward_mean += delta / self._reward_count
        delta2 = reward - self._reward_mean
        self._reward_m2 += delta * delta2

        if self._reward_count < 2:
            return 0.0

        variance = self._reward_m2 / (self._reward_count - 1)
        std = math.sqrt(variance) if variance > 0.0 else 0.0
        if std == 0.0:
            return 0.0

        normalized = (reward - self._reward_mean) / (std + 1e-8)
        if self._reward_clip is not None:
            normalized = max(min(normalized, self._reward_clip), -self._reward_clip)
        return float(normalized)

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

