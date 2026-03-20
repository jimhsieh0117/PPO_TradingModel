"""
持倉狀態管理 — Single Source of Truth

職責：
- 維護持倉狀態（方向、開倉價、止損、持倉步數等）
- 組合 33 維觀察向量（28 市場特徵 + 5 持倉狀態）
- 追蹤風控統計（daily_pnl、consecutive_losses 等）
- 與交易所狀態同步

設計原則：
- equity_change_pct 使用滾動窗口（對齊 PPOTradingStrategy，非 TradingEnv）
- deque(maxlen=episode_length) 自動淘汰最舊值
- 所有模組從這裡讀取持倉狀態，不自行維護
"""

import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger("live_trading.state")


class TradingState:
    """
    持倉狀態管理器

    Usage:
        state = TradingState(initial_balance=200.0, episode_length=720)
        obs = state.build_observation(market_features)  # → [33]
        state.open_position(side=1, entry_price=2450.0, quantity=0.04, sl_price=2401.0, sl_order_id="123")
        state.close_position(exit_price=2478.0, pnl=1.12)
        state.step()  # 每根 K 線呼叫一次
    """

    def __init__(self, initial_balance: float, episode_length: int = 720,
                 max_holding_steps: int = 120,
                 on_exchange_close=None):
        """
        初始化狀態

        Args:
            initial_balance: 啟動時的帳戶餘額
            episode_length: 滾動窗口大小（對齊模型訓練的 episode_length）
            max_holding_steps: 最大持倉步數（超過強制平倉）
            on_exchange_close: 交易所平倉回調（sync 偵測到 DESYNC 時通知 bot 層）
                               signature: (entry_price, quantity, side, estimated_pnl, reason) -> None
        """
        self._on_exchange_close = on_exchange_close
        # === 帳戶 ===
        self.balance: float = initial_balance
        self.equity: float = initial_balance
        self.initial_balance: float = initial_balance

        # === 持倉 ===
        self.position: int = 0          # -1=空, 0=無, 1=多
        self.entry_price: float = 0.0
        self.entry_time: Optional[datetime] = None
        self.quantity: float = 0.0
        self.holding_steps: int = 0
        self.current_sl: float = 0.0    # 當前止損價
        self.sl_order_id: str = ""      # 止損單 ID

        # === 滾動 equity 歷史（對齊 PPOTradingStrategy） ===
        self.episode_length: int = episode_length
        self.max_holding_steps: int = max_holding_steps
        self.equity_history: deque = deque(maxlen=episode_length)

        # === 統計 ===
        self.daily_pnl: float = 0.0
        self.total_pnl: float = 0.0
        self.consecutive_losses: int = 0
        self.trade_count: int = 0
        self.daily_reset_date: str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # === 交易歷史（供 Telegram 指令查詢） ===
        self.trade_history: deque = deque(maxlen=100)
        self.daily_win_count: int = 0
        self.daily_loss_count: int = 0
        self.daily_max_pnl: float = 0.0
        self.daily_min_pnl: float = 0.0

        # === 步數計數 ===
        self._step_count: int = 0
        self._last_close_step: int = -999

        logger.info(
            f"TradingState initialized | "
            f"balance={initial_balance:.2f} | "
            f"episode_length={episode_length} | "
            f"max_holding_steps={max_holding_steps}"
        )

    # ================================================================
    # 觀察向量
    # ================================================================

    def build_observation(self, market_features: np.ndarray,
                          current_price: float) -> np.ndarray:
        """
        組合 33 維觀察向量（28 市場特徵 + 5 持倉狀態）

        與 PPOTradingStrategy._get_position_features() + next() 完全對齊

        Args:
            market_features: 28 維市場特徵 (from FeatureEngine)
            current_price: 當前收盤價（計算浮動盈虧用）

        Returns:
            np.ndarray shape [33], dtype=float32
        """
        position_features = self._get_position_features(current_price)
        obs = np.concatenate([market_features, position_features])
        return obs.astype(np.float32)

    def _get_position_features(self, price: float) -> np.ndarray:
        """
        計算 5 維持倉狀態特徵

        與 backtest/strategy.py PPOTradingStrategy._get_position_features() 邏輯一致

        Returns:
            np.ndarray: [position_state, floating_pnl_pct, holding_time_norm,
                         distance_to_stop_loss, equity_change_pct]
        """
        # (1) 持倉方向
        position_state = float(self.position)

        # (2) 浮動盈虧百分比
        if self.position != 0 and self.entry_price > 0:
            if self.position == 1:  # 多倉
                floating_pnl_pct = (price - self.entry_price) / self.entry_price
            else:  # 空倉
                floating_pnl_pct = (self.entry_price - price) / self.entry_price
            floating_pnl_pct = float(np.clip(floating_pnl_pct, -1.0, 1.0))
        else:
            floating_pnl_pct = 0.0

        # (3) 持倉時間正規化 (0~1, max_holding_steps 步飽和)
        if self.position != 0:
            holding_time_norm = min(self.holding_steps / float(self.max_holding_steps), 1.0)
        else:
            holding_time_norm = 0.0

        # (4) 距止損距離百分比 (0~1)
        if self.position != 0 and self.entry_price > 0 and self.current_sl > 0:
            if self.position == 1:  # 多倉
                dist_to_sl = (price - self.current_sl) / (self.entry_price - self.current_sl + 1e-10)
            else:  # 空倉
                dist_to_sl = (self.current_sl - price) / (self.current_sl - self.entry_price + 1e-10)
            dist_to_sl = float(np.clip(dist_to_sl, 0.0, 2.0) / 2.0)
        else:
            dist_to_sl = 0.0

        # (5) 權益變化百分比：滾動窗口
        #     與 PPOTradingStrategy._get_position_features() 完全對齊
        equity_change_pct = self._equity_change_pct()

        return np.array([
            position_state,
            floating_pnl_pct,
            holding_time_norm,
            dist_to_sl,
            equity_change_pct,
        ], dtype=np.float32)

    def _equity_change_pct(self) -> float:
        """
        滾動窗口 equity 變化百分比

        對齊 PPOTradingStrategy（非 TradingEnv）：
        - TradingEnv: (equity - initial_balance) / initial_balance（episode 累積）
        - PPOTradingStrategy: 滾動 episode_length 窗口的 baseline
        - 實盤無 episode reset → 必須用滾動窗口
        """
        if self.equity_history:
            if len(self.equity_history) >= self.episode_length:
                baseline = self.equity_history[0]  # deque(maxlen) → [0] 即為窗口起點
            else:
                baseline = self.initial_balance
        else:
            baseline = self.initial_balance

        return float(np.clip(
            (self.equity - baseline) / (baseline + 1e-10),
            -1.0, 1.0
        ))

    # ================================================================
    # 狀態更新
    # ================================================================

    def step(self, current_equity: Optional[float] = None) -> None:
        """
        每根 K 線結束時呼叫

        Args:
            current_equity: 當前帳戶權益（含浮動盈虧）
        """
        self._step_count += 1

        if self.position != 0:
            self.holding_steps += 1

        if current_equity is not None:
            self.equity = current_equity

        # 記錄 equity 到滾動窗口
        self.equity_history.append(self.equity)

        # 每日重置檢查（UTC 0:00）
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self.daily_reset_date:
            logger.info(
                f"Daily reset: {self.daily_reset_date} → {today} | "
                f"yesterday_pnl={self.daily_pnl:.2f}"
            )
            self.daily_pnl = 0.0
            self.trade_count = 0
            self.daily_win_count = 0
            self.daily_loss_count = 0
            self.daily_max_pnl = 0.0
            self.daily_min_pnl = 0.0
            self.daily_reset_date = today

    def open_position(self, side: int, entry_price: float,
                      quantity: float, sl_price: float,
                      sl_order_id: str = "") -> None:
        """
        記錄開倉

        Args:
            side: 1=多, -1=空
            entry_price: 實際成交均價
            quantity: 成交數量
            sl_price: 止損價
            sl_order_id: 止損單 ID
        """
        self.position = side
        self.entry_price = entry_price
        self.entry_time = datetime.now(timezone.utc)
        self.quantity = quantity
        self.holding_steps = 0
        self.current_sl = sl_price
        self.sl_order_id = sl_order_id

        logger.info(
            f"Position opened | side={'LONG' if side == 1 else 'SHORT'} "
            f"entry={entry_price:.2f} qty={quantity} sl={sl_price:.2f}"
        )

    def close_position(self, exit_price: float, pnl: float,
                       fee: float = 0.0, reason: str = "model") -> Dict[str, Any]:
        """
        記錄平倉

        Args:
            exit_price: 平倉成交均價
            pnl: 已實現盈虧（扣除手續費後）
            fee: 手續費
            reason: 平倉原因

        Returns:
            平倉記錄 dict
        """
        record = {
            "side": "LONG" if self.position == 1 else "SHORT",
            "entry_price": self.entry_price,
            "exit_price": exit_price,
            "quantity": self.quantity,
            "pnl": pnl,
            "pnl_pct": pnl / (self.entry_price * self.quantity + 1e-10) * 100 if self.quantity > 0 else 0.0,
            "fee": fee,
            "holding_steps": self.holding_steps,
            "reason": reason,
        }

        logger.info(
            f"Position closed | {record['side']} "
            f"entry={record['entry_price']:.2f} exit={exit_price:.2f} "
            f"pnl={pnl:+.2f} ({record['pnl_pct']:+.2f}%) "
            f"held={self.holding_steps} steps | reason={reason}"
        )

        # H2 fix: 先清除持倉，再更新統計
        # 避免其他線程看到「有倉位 + 已結算 balance」的不一致狀態
        self.position = 0
        self.entry_price = 0.0
        self.entry_time = None
        self.quantity = 0.0
        self.holding_steps = 0
        self.current_sl = 0.0
        self.sl_order_id = ""

        # 更新統計
        self.balance += pnl
        self.equity = self.balance
        self.daily_pnl += pnl
        self.total_pnl += pnl
        self.trade_count += 1

        if pnl < 0:
            self.consecutive_losses += 1
            self.daily_loss_count += 1
        else:
            self.consecutive_losses = 0
            self.daily_win_count += 1
        self.daily_max_pnl = max(self.daily_max_pnl, pnl)
        self.daily_min_pnl = min(self.daily_min_pnl, pnl)

        # 記錄到交易歷史（deque maxlen=100 自動淘汰）
        record["close_time"] = datetime.now(timezone.utc).isoformat()
        self.trade_history.append(record)

        self._last_close_step = self._step_count

        return record

    def update_stop_loss(self, new_sl: float, new_sl_order_id: str = "") -> None:
        """更新止損價（追蹤止損用）"""
        old_sl = self.current_sl
        self.current_sl = new_sl
        if new_sl_order_id:
            self.sl_order_id = new_sl_order_id
        logger.debug(f"Stop loss updated: {old_sl:.2f} → {new_sl:.2f}")

    # ================================================================
    # 與交易所同步
    # ================================================================

    def sync_from_exchange(self, position_data: Dict, balance: float) -> None:
        """
        從交易所 API 同步持倉狀態

        Args:
            position_data: positionRisk API 回傳
            balance: USDT 錢包餘額
        """
        # Fix 2: 先保存舊 balance，再覆蓋（修正 estimated_pnl 永遠 = 0 的 bug）
        old_balance = self.balance
        self.balance = balance

        pos_amt = float(position_data.get("positionAmt", "0"))
        entry_price = float(position_data.get("entryPrice", "0"))
        unrealized_pnl = float(position_data.get("unRealizedProfit", "0"))

        if pos_amt > 0:
            exchange_side = 1
        elif pos_amt < 0:
            exchange_side = -1
        else:
            exchange_side = 0

        # 檢查本地 vs 交易所一致性
        if exchange_side != self.position:
            logger.warning(
                f"STATE DESYNC! local={self.position}, "
                f"exchange={exchange_side} (amt={pos_amt}). "
                f"Syncing to exchange state."
            )

            if exchange_side != 0:
                # 交易所有倉但本地沒有 → 同步倉位
                self.position = exchange_side
                self.entry_price = entry_price
                self.quantity = abs(pos_amt)
                # holding_steps 無法從交易所恢復，保持現有值
            else:
                # 本地有倉但交易所無倉 → 止損被觸發，需結算
                if self.position != 0 and self.entry_price > 0:
                    estimated_pnl = balance - old_balance
                    closed_side = self.position
                    closed_entry = self.entry_price
                    closed_qty = self.quantity
                    logger.warning(
                        f"SL likely triggered on exchange | "
                        f"estimated_pnl={estimated_pnl:+.4f} "
                        f"(balance: {old_balance:.2f} → {balance:.2f})"
                    )

                    # Fix 1: 透過回調通知 bot 層（記錄 trades.jsonl + 發 Telegram）
                    # M7: 回調回傳精確的 exit_price/fee/pnl
                    close_info = None
                    if self._on_exchange_close:
                        close_info = self._on_exchange_close(
                            entry_price=closed_entry,
                            quantity=closed_qty,
                            side=closed_side,
                            estimated_pnl=estimated_pnl,
                            reason="exchange_sl_triggered",
                        )

                    if isinstance(close_info, dict):
                        real_exit = close_info.get("exit_price", 0.0)
                        real_fee = close_info.get("fee", 0.0)
                        real_pnl = close_info.get("pnl", estimated_pnl)
                    else:
                        real_exit = 0.0
                        real_fee = 0.0
                        real_pnl = estimated_pnl

                    self.close_position(
                        exit_price=real_exit,
                        pnl=real_pnl,
                        fee=real_fee,
                        reason="exchange_sl_triggered",
                    )
                    # close_position 已更新 balance，但交易所 balance 才是真值
                    self.balance = balance
                else:
                    self.position = 0
                    self.entry_price = 0.0
                    self.quantity = 0.0
                    self.holding_steps = 0
                    self.current_sl = 0.0
                    self.sl_order_id = ""

        self.equity = balance + unrealized_pnl

    # ================================================================
    # 快照（供 state_snapshot 使用）
    # ================================================================

    def to_snapshot(self) -> Dict[str, Any]:
        """匯出可持久化的狀態快照"""
        return {
            "consecutive_losses": self.consecutive_losses,
            "daily_pnl": self.daily_pnl,
            "total_pnl": self.total_pnl,
            "trade_count": self.trade_count,
            "daily_reset_date": self.daily_reset_date,
            "balance": self.balance,
            "step_count": self._step_count,
            # M1 + H5: 持倉資料（crash 後恢復 holding_steps 和 sl_order_id）
            "position": self.position,
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "holding_steps": self.holding_steps,
            "current_sl": self.current_sl,
            "sl_order_id": self.sl_order_id,
        }

    def restore_from_snapshot(self, data: Dict[str, Any]) -> None:
        """從快照恢復風控統計與持倉資料"""
        self.consecutive_losses = data.get("consecutive_losses", 0)
        self.daily_pnl = data.get("daily_pnl", 0.0)
        self.total_pnl = data.get("total_pnl", 0.0)
        self.trade_count = data.get("trade_count", 0)
        self.daily_reset_date = data.get("daily_reset_date", self.daily_reset_date)
        # M1 + H5: 恢復持倉資料
        # sync_from_exchange 會覆蓋 position/entry_price/quantity，
        # 但 holding_steps 和 sl_order_id 無法從交易所取得
        self.holding_steps = data.get("holding_steps", 0)
        self.sl_order_id = data.get("sl_order_id", "")
        self.current_sl = data.get("current_sl", 0.0)
        logger.info(
            f"State restored from snapshot | "
            f"consecutive_losses={self.consecutive_losses} "
            f"daily_pnl={self.daily_pnl:.2f} "
            f"total_pnl={self.total_pnl:.2f} "
            f"holding_steps={self.holding_steps} "
            f"sl_order_id={self.sl_order_id or 'none'}"
        )

    # ================================================================
    # 查詢
    # ================================================================

    @property
    def has_position(self) -> bool:
        return self.position != 0

    @property
    def is_max_holding(self) -> bool:
        """是否已達最大持倉步數"""
        return self.position != 0 and self.holding_steps >= self.max_holding_steps

    def get_stats(self) -> Dict:
        return {
            "position": self.position,
            "entry_price": self.entry_price,
            "holding_steps": self.holding_steps,
            "balance": round(self.balance, 2),
            "equity": round(self.equity, 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "total_pnl": round(self.total_pnl, 2),
            "consecutive_losses": self.consecutive_losses,
            "trade_count": self.trade_count,
            "step_count": self._step_count,
        }
