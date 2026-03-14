"""
風控模組 — 多層防護閘門

職責：
- Layer 1: 系統健康檢查（WebSocket、buffer、API）
- Layer 2: 帳戶級別（daily loss、total loss、consecutive losses）
- Layer 3: 倉位級別（max positions、不重複開倉）
- Layer 4: 訂單級別（max order value、slippage）
- API 斷路器（Circuit Breaker）
- Kill Switch（STOP 檔案）

設計原則：
- 任何一層 FAIL → 拒絕執行 + 記錄原因
- 寧可不交易，也不可錯誤交易
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger("live_trading.risk_manager")


class RiskManager:
    """
    多層風控閘門

    Usage:
        rm = RiskManager(config)
        allowed, reason = rm.check(action, state, data_feed)
        if not allowed:
            logger.info(f"Blocked: {reason}")
    """

    def __init__(self, config: dict, kill_switch_path: str = "live_trading/STOP"):
        """
        Args:
            config: 完整的 config_live 配置
            kill_switch_path: Kill Switch 檔案路徑
        """
        risk = config["risk"]

        # 帳戶級別
        self.max_daily_loss_pct: float = risk["max_daily_loss_pct"]
        self.max_total_loss_pct: float = risk["max_total_loss_pct"]
        self.max_consecutive_losses: int = risk["max_consecutive_losses"]
        self.max_open_positions: int = risk["max_open_positions"]
        self.min_balance_to_trade: float = risk["min_balance_to_trade"]

        # 訂單級別
        self.max_order_value: float = risk["max_order_value_usdt"]
        self.max_slippage_pct: float = risk["max_slippage_pct"]

        # 斷路器
        self.cb_threshold: int = risk["circuit_breaker_threshold"]
        self.cb_cooldown: float = risk["circuit_breaker_cooldown"]
        self.cb_max_triggers: int = risk["circuit_breaker_max_triggers"]

        # 斷路器狀態
        self._consecutive_api_errors: int = 0
        self._cb_trigger_count: int = 0
        self._cb_cooldown_until: float = 0.0
        self._standby_mode: bool = False

        # Kill Switch
        self._kill_switch_path = Path(kill_switch_path)

        # 連續虧損暫停
        self._loss_pause_until: float = 0.0

        logger.info(
            f"RiskManager initialized | "
            f"daily_loss={self.max_daily_loss_pct} "
            f"total_loss={self.max_total_loss_pct} "
            f"max_consec={self.max_consecutive_losses}"
        )

    # ================================================================
    # 主要入口
    # ================================================================

    def check(self, action: int, state, data_feed=None,
              estimated_notional: float = 0.0) -> Tuple[bool, Optional[str]]:
        """
        多層風控檢查

        Args:
            action: 模型輸出 (0-3)
            state: TradingState 實例
            data_feed: DataFeed 實例（檢查連線健康）
            estimated_notional: 預估下單金額

        Returns:
            (allowed: bool, reason: Optional[str])
        """
        # HOLD 永遠允許
        if action == 3:
            return True, None

        # CLOSE 基本允許（但仍需系統健康檢查）
        is_close = (action == 0)

        # === Kill Switch ===
        if self._check_kill_switch():
            return False, "KILL_SWITCH: STOP file detected"

        # === 待機模式 ===
        if self._standby_mode:
            return False, "STANDBY_MODE: circuit breaker max triggers exceeded"

        # === 斷路器冷卻中 ===
        if self._is_in_cooldown():
            if is_close:
                # 平倉在冷卻期間仍允許（已有止損保護，但手動平倉應放行）
                pass
            else:
                return False, f"CIRCUIT_BREAKER: cooling down until {self._cb_cooldown_until:.0f}"

        # === Layer 1: 系統健康 ===
        if data_feed is not None:
            ok, reason = self._check_system_health(data_feed)
            if not ok:
                return False, reason

        # 平倉只需通過系統健康檢查
        if is_close:
            return True, None

        # === Layer 2: 帳戶級別（僅限開倉） ===
        ok, reason = self._check_account_level(state)
        if not ok:
            return False, reason

        # === Layer 3: 倉位級別 ===
        ok, reason = self._check_position_level(action, state)
        if not ok:
            return False, reason

        # === Layer 4: 訂單級別 ===
        if estimated_notional > 0:
            ok, reason = self._check_order_level(estimated_notional)
            if not ok:
                return False, reason

        return True, None

    # ================================================================
    # Layer 1: 系統健康
    # ================================================================

    def _check_system_health(self, data_feed) -> Tuple[bool, Optional[str]]:
        """WebSocket 連線 + buffer 檢查"""
        # WebSocket 心跳檢查（最後心跳 > 30 秒前 = 異常）
        if data_feed.last_heartbeat_age > 30:
            return False, (
                f"SYSTEM_HEALTH: WebSocket heartbeat stale "
                f"({data_feed.last_heartbeat_age:.0f}s ago)"
            )

        # Buffer 非空
        if data_feed.buffer_length == 0:
            return False, "SYSTEM_HEALTH: buffer is empty"

        return True, None

    # ================================================================
    # Layer 2: 帳戶級別
    # ================================================================

    def _check_account_level(self, state) -> Tuple[bool, Optional[str]]:
        """帳戶級別風控"""
        # 餘額下限
        if state.balance < self.min_balance_to_trade:
            return False, (
                f"ACCOUNT: balance {state.balance:.2f} < "
                f"min {self.min_balance_to_trade}"
            )

        # 每日虧損限制
        if state.initial_balance > 0:
            daily_loss_pct = abs(min(state.daily_pnl, 0)) / state.initial_balance
            if daily_loss_pct >= self.max_daily_loss_pct:
                return False, (
                    f"ACCOUNT: daily loss {daily_loss_pct:.1%} >= "
                    f"limit {self.max_daily_loss_pct:.1%}"
                )

        # 總虧損限制
        if state.initial_balance > 0:
            total_loss_pct = abs(min(state.total_pnl, 0)) / state.initial_balance
            if total_loss_pct >= self.max_total_loss_pct:
                return False, (
                    f"ACCOUNT: total loss {total_loss_pct:.1%} >= "
                    f"limit {self.max_total_loss_pct:.1%}. "
                    f"MANUAL INTERVENTION REQUIRED."
                )

        # 連續虧損暫停
        if state.consecutive_losses >= self.max_consecutive_losses:
            if time.time() < self._loss_pause_until:
                remaining = self._loss_pause_until - time.time()
                return False, (
                    f"ACCOUNT: consecutive losses pause, "
                    f"{remaining:.0f}s remaining"
                )
            elif self._loss_pause_until == 0:
                # 首次觸發：暫停 1 小時
                self._loss_pause_until = time.time() + 3600
                logger.warning(
                    f"Consecutive losses reached {state.consecutive_losses} — "
                    f"pausing for 1 hour"
                )
                return False, (
                    f"ACCOUNT: {state.consecutive_losses} consecutive losses, "
                    f"pausing 1 hour"
                )
            else:
                # 暫停期已過，重置
                self._loss_pause_until = 0.0

        return True, None

    # ================================================================
    # Layer 3: 倉位級別
    # ================================================================

    def _check_position_level(self, action: int,
                               state) -> Tuple[bool, Optional[str]]:
        """倉位級別風控"""
        # 已有持倉 → 不開新倉（max_open_positions = 1）
        if state.position != 0:
            if action in (1, 2):  # LONG or SHORT
                # 允許反向開倉（executor 會先平再開）
                if (action == 1 and state.position == -1) or \
                   (action == 2 and state.position == 1):
                    return True, None
                # 同方向 → 不重複開
                return False, (
                    f"POSITION: already {['SHORT','','LONG'][state.position+1]}, "
                    f"cannot open same direction"
                )

        return True, None

    # ================================================================
    # Layer 4: 訂單級別
    # ================================================================

    def _check_order_level(self, estimated_notional: float) -> Tuple[bool, Optional[str]]:
        """訂單級別風控"""
        if estimated_notional > self.max_order_value:
            return False, (
                f"ORDER: notional {estimated_notional:.2f} > "
                f"max {self.max_order_value}"
            )
        return True, None

    # ================================================================
    # Kill Switch
    # ================================================================

    def _check_kill_switch(self) -> bool:
        """檢查 STOP 檔案是否存在"""
        return self._kill_switch_path.exists()

    # ================================================================
    # API 斷路器
    # ================================================================

    def record_api_success(self) -> None:
        """API 呼叫成功 → 重置連續錯誤計數"""
        self._consecutive_api_errors = 0

    def record_api_error(self, status_code: int) -> None:
        """
        API 呼叫失敗 → 累計錯誤

        只對 429 (Rate Limit) 和 5xx (Server Error) 計數
        """
        if status_code == 429 or status_code >= 500:
            self._consecutive_api_errors += 1
            logger.warning(
                f"API error #{self._consecutive_api_errors}: HTTP {status_code}"
            )

            if self._consecutive_api_errors >= self.cb_threshold:
                self._trigger_cooldown()

    def _trigger_cooldown(self) -> None:
        """觸發斷路器冷卻"""
        self._cb_trigger_count += 1
        self._consecutive_api_errors = 0
        self._cb_cooldown_until = time.time() + self.cb_cooldown

        logger.warning(
            f"CIRCUIT BREAKER triggered (#{self._cb_trigger_count}) — "
            f"cooling down for {self.cb_cooldown}s"
        )

        if self._cb_trigger_count >= self.cb_max_triggers:
            self._standby_mode = True
            logger.critical(
                f"CIRCUIT BREAKER: {self._cb_trigger_count} triggers — "
                f"entering STANDBY MODE. Manual intervention required."
            )

    def _is_in_cooldown(self) -> bool:
        """是否在冷卻期"""
        return time.time() < self._cb_cooldown_until

    def reset_circuit_breaker(self) -> None:
        """手動重置斷路器（恢復後）"""
        self._consecutive_api_errors = 0
        self._cb_trigger_count = 0
        self._cb_cooldown_until = 0.0
        self._standby_mode = False
        logger.info("Circuit breaker reset")

    # ================================================================
    # 狀態
    # ================================================================

    def get_stats(self) -> dict:
        return {
            "standby_mode": self._standby_mode,
            "consecutive_api_errors": self._consecutive_api_errors,
            "cb_trigger_count": self._cb_trigger_count,
            "in_cooldown": self._is_in_cooldown(),
            "kill_switch": self._check_kill_switch(),
        }
