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
        reward_config: Optional[Dict] = None
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

        # === 獎勵參數 ===
        self.reward_config = reward_config or {}
        self.floating_weight = float(
            self.reward_config.get('floating_weight', self.reward_config.get('pnl_weight', 1.0))
        )
        self.realized_weight = float(
            self.reward_config.get('realized_weight', self.reward_config.get('pnl_weight', 1.0))
        )
        self.sharpe_weight = float(self.reward_config.get('sharpe_weight', 0.3))
        self.sharpe_window = int(self.reward_config.get('sharpe_window', 60))
        self.high_risk_penalty = float(self.reward_config.get('high_risk_penalty', -20))
        self.stop_loss_penalty = float(self.reward_config.get('stop_loss_penalty', -50))
        self.daily_drawdown_penalty = float(self.reward_config.get('daily_drawdown_penalty', -100))
        self.holding_time_reward = float(self.reward_config.get('holding_time_reward', 0.1))
        self.max_holding_reward_bars = int(self.reward_config.get('max_holding_reward_bars', 10))

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

        # === 交易統計（交易員最關心的指標）===
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.stop_loss_count = 0
        self.holding_time = 0  # 當前持倉時間（K線數）

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
        # 獲取當前價格（使用 iloc 因為 current_step 是整數索引）
        current_price = self.df.iloc[self.current_step]['close']
        self.realized_this_step = False
        self.last_realized_pnl = 0.0

        # === 1. 檢查止損（交易員的生命線）===
        stop_loss_triggered = self._check_stop_loss(current_price)

        # === 2. 執行交易動作 ===
        trade_executed = False
        if not stop_loss_triggered:
            trade_executed = self._execute_action(action, current_price)

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
        Calculate reward for the current step.
        """
        reward = 0.0

        equity_change = self.equity - self.equity_curve[-2] if len(self.equity_curve) > 1 else 0.0
        step_return = equity_change / self.initial_balance
        self.recent_returns.append(step_return)

        # Floating reward (only while holding and not on close step)
        if self.position != 0 and not self.realized_this_step:
            reward += self.floating_weight * step_return * 100

        # Realized reward (only on close step)
        if self.realized_this_step:
            realized_return = self.last_realized_pnl / self.initial_balance
            reward += self.realized_weight * realized_return * 100

        # Sharpe improvement (rolling window)
        if len(self.recent_returns) >= 2:
            returns_array = np.array(self.recent_returns, dtype=np.float64)
            std = returns_array.std()
            if std > 0.0:
                current_sharpe = (returns_array.mean() / (std + 1e-8)) * np.sqrt(252)
                sharpe_improvement = current_sharpe - self.previous_sharpe
                reward += sharpe_improvement * self.sharpe_weight
                self.previous_sharpe = current_sharpe

        # High risk penalty
        if self.position != 0:
            position_value = self.balance * self.position_size_pct * self.leverage
            exposure_ratio = position_value / self.equity if self.equity > 0 else 0.0
            if exposure_ratio > 0.5:
                reward += (exposure_ratio - 0.5) * self.high_risk_penalty

        # Stop loss penalty
        if stop_loss_triggered:
            reward += self.stop_loss_penalty

        # Trading cost penalty
        if trade_executed:
            position_value = self.balance * self.position_size_pct * self.leverage
            fee = position_value * self.trading_fee
            reward -= (fee / self.initial_balance) * 100

        # Holding time reward
        if self.position != 0 and equity_change > 0:
            reward += min(self.holding_time, self.max_holding_reward_bars) * self.holding_time_reward

        # Daily drawdown penalty
        daily_drawdown = (self.daily_start_balance - self.equity) / self.daily_start_balance
        if daily_drawdown > self.max_daily_drawdown:
            reward += self.daily_drawdown_penalty

        self.realized_this_step = False
        self.last_realized_pnl = 0.0

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

