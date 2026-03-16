"""
通知推送模組 — Telegram Bot / log_only

職責：
- 交易通知（開倉/平倉）
- 風控警告
- 系統異常
- 心跳（每小時）

設計原則：
- 通知失敗不影響交易邏輯（fire-and-forget）
- 支援 Telegram / log_only 兩種模式
- Rate limiting：防止短時間大量通知
- 所有通知同時寫入日誌（即使推送失敗也有記錄）
"""

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger("live_trading.notifier")

# 通知類型前綴
PREFIX_TRADE = "[TRADE]"
PREFIX_RISK = "[RISK]"
PREFIX_ERROR = "[ERROR]"
PREFIX_HEARTBEAT = "[HEARTBEAT]"
PREFIX_SYSTEM = "[SYSTEM]"


class Notifier:
    """
    通知推送管理器

    Usage:
        notifier = Notifier(config["notification"])
        notifier.send_trade("ETHUSDT 做多 @ 2450.30 | 數量 0.04")
        notifier.send_heartbeat(balance=98.5, daily_pnl=2.1)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化通知模組

        Args:
            config: notification 配置區塊
                - enabled: bool
                - method: "telegram" / "log_only"
                - telegram_bot_token_env: str
                - telegram_chat_id_env: str
                - notify_on_trade: bool
                - notify_on_error: bool
                - notify_heartbeat_minutes: int
        """
        self.enabled = config.get("enabled", True)
        self.method = config.get("method", "log_only")
        self.notify_on_trade = config.get("notify_on_trade", True)
        self.notify_on_error = config.get("notify_on_error", True)
        self.heartbeat_interval = config.get("notify_heartbeat_minutes", 60) * 60

        # 交易通知模式：1=靜音, 2=僅平倉, 3=開倉+平倉
        self.trade_notify_mode = config.get("trade_notify_mode", 3)

        # Rate limiting
        self._last_send_time = 0.0
        self._min_interval = 1.0  # 最少間隔 1 秒
        self._last_heartbeat_time = 0.0

        # Telegram 設定
        self._telegram_token: Optional[str] = None
        self._telegram_chat_id: Optional[str] = None

        if self.method == "telegram" and self.enabled:
            self._init_telegram(config)

        mode_label = self.method if self.enabled else "disabled"
        logger.info(f"Notifier initialized | mode={mode_label}")

    def _init_telegram(self, config: Dict) -> None:
        """初始化 Telegram Bot"""
        token_env = config.get("telegram_bot_token_env", "TELEGRAM_BOT_TOKEN")
        chat_id_env = config.get("telegram_chat_id_env", "TELEGRAM_CHAT_ID")

        self._telegram_token = os.environ.get(token_env)
        self._telegram_chat_id = os.environ.get(chat_id_env)

        if not self._telegram_token or not self._telegram_chat_id:
            logger.warning(
                f"Telegram credentials not found "
                f"(env: {token_env}, {chat_id_env}). "
                f"Falling back to log_only mode."
            )
            self.method = "log_only"
        else:
            logger.info("Telegram Bot configured successfully")

    # ================================================================
    # 公開方法
    # ================================================================

    def send_trade(self, message: str) -> None:
        """推送交易通知"""
        if not self.notify_on_trade:
            return
        self._send(f"{PREFIX_TRADE} {message}")

    def send_trade_open(self, symbol: str, side: str, price: float,
                        quantity: float, sl_price: float) -> None:
        """推送開倉通知（trade_notify_mode >= 3 時發送）"""
        if not self.notify_on_trade or self.trade_notify_mode < 3:
            return
        direction = "做多" if side == "BUY" else "做空"
        notional = price * quantity
        msg = (
            f"{PREFIX_TRADE} {symbol} {direction} @ {price:.2f}\n"
            f"  數量: {quantity}\n"
            f"  倉位價值: {notional:.2f}U\n"
            f"  止損: {sl_price:.2f}"
        )
        self._send(msg)

    def send_trade_close(self, symbol: str, price: float,
                         pnl: float, pnl_pct: float,
                         holding_minutes: int,
                         reason: str = "model") -> None:
        """推送平倉通知（trade_notify_mode >= 2 時發送）"""
        if not self.notify_on_trade or self.trade_notify_mode < 2:
            return
        result = "賺" if pnl >= 0 else "賠"
        msg = (
            f"{PREFIX_TRADE} {symbol} 平倉 @ {price:.2f}\n"
            f"  {result} {abs(pnl):.4f}U ({pnl_pct:+.2f}%)\n"
            f"  持倉: {holding_minutes} 分鐘\n"
            f"  原因: {reason}"
        )
        self._send(msg)

    def send_emergency_close(self, symbol: str, reason: str) -> None:
        """推送緊急平倉通知（無視通知模式，永遠發送）"""
        self._send(
            f"{PREFIX_ERROR} {symbol} 緊急平倉!\n"
            f"  原因: {reason}"
        )

    def send_risk_warning(self, message: str) -> None:
        """推送風控警告"""
        self._send(f"{PREFIX_RISK} {message}")

    def send_error(self, message: str) -> None:
        """推送系統錯誤"""
        if not self.notify_on_error:
            return
        self._send(f"{PREFIX_ERROR} {message}")

    def send_system(self, message: str) -> None:
        """推送系統事件（啟動/關閉等）"""
        self._send(f"{PREFIX_SYSTEM} {message}")

    def send_heartbeat(self, balance: float, daily_pnl: float,
                       position: int, total_trades: int) -> None:
        """
        推送心跳通知

        根據 heartbeat_interval 控制頻率，不會每分鐘都發
        """
        now = time.time()
        if now - self._last_heartbeat_time < self.heartbeat_interval:
            return

        position_str = {-1: "空單", 0: "無持倉", 1: "多單"}.get(position, "未知")
        pnl_sign = "+" if daily_pnl >= 0 else ""
        msg = (
            f"{PREFIX_HEARTBEAT} 系統運行中\n"
            f"  餘額: {balance:.2f}U\n"
            f"  今日 PnL: {pnl_sign}{daily_pnl:.2f}U\n"
            f"  持倉: {position_str}\n"
            f"  今日交易: {total_trades} 筆"
        )
        self._send(msg)
        self._last_heartbeat_time = now

    def check_heartbeat_due(self) -> bool:
        """檢查是否該發送心跳"""
        return time.time() - self._last_heartbeat_time >= self.heartbeat_interval

    # ================================================================
    # 內部方法
    # ================================================================

    def _send(self, message: str) -> None:
        """統一發送入口"""
        if not self.enabled:
            return

        # Rate limiting
        now = time.time()
        if now - self._last_send_time < self._min_interval:
            time.sleep(self._min_interval - (now - self._last_send_time))

        # 永遠寫入日誌
        logger.info(f"NOTIFY | {message}")

        # 根據 method 推送
        if self.method == "telegram":
            self._send_telegram(message)
        # log_only 模式：只寫日誌（上面已完成）

        self._last_send_time = time.time()

    def _send_telegram(self, message: str) -> None:
        """透過 Telegram Bot API 發送訊息"""
        if not self._telegram_token or not self._telegram_chat_id:
            return

        url = f"https://api.telegram.org/bot{self._telegram_token}/sendMessage"
        payload = {
            "chat_id": self._telegram_chat_id,
            "text": message,
            "parse_mode": "HTML",
        }

        try:
            response = requests.post(url, json=payload, timeout=5)
            if response.status_code != 200:
                logger.warning(
                    f"Telegram send failed: HTTP {response.status_code} "
                    f"{response.text[:200]}"
                )
        except requests.RequestException as e:
            # 通知失敗不影響交易邏輯
            logger.warning(f"Telegram send error: {e}")


class LogOnlyNotifier(Notifier):
    """純日誌模式通知器（測試用）"""

    def __init__(self):
        self.enabled = True
        self.method = "log_only"
        self.notify_on_trade = True
        self.notify_on_error = True
        self.heartbeat_interval = 3600
        self._last_send_time = 0.0
        self._min_interval = 0.0
        self._last_heartbeat_time = 0.0
        self._telegram_token = None
        self._telegram_chat_id = None
        logger.info("LogOnlyNotifier initialized (no external push)")


def create_notifier(config: Dict) -> Notifier:
    """
    工廠函數：根據配置建立 Notifier

    Args:
        config: notification 配置區塊

    Returns:
        Notifier 或 LogOnlyNotifier
    """
    if not config.get("enabled", True):
        return LogOnlyNotifier()

    method = config.get("method", "log_only")
    if method == "log_only":
        return LogOnlyNotifier()

    return Notifier(config)
