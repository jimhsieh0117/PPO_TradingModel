"""
Telegram 指令處理模組 — 接收並執行遠端指令

職責：
- Polling Telegram getUpdates API（非阻塞）
- 授權驗證（僅接受指定 chat_id）
- 指令分發與執行
- 倉位操作加鎖（防止與 on_bar_close 競態）

設計原則：
- 唯讀指令不需要鎖
- 倉位操作指令（force_close, stop）需要 position_lock
- polling 失敗不影響交易邏輯（fire-and-forget）
"""

import logging
import time
import threading
from datetime import datetime, timezone
from typing import Dict, Optional

import requests

logger = logging.getLogger("live_trading.command_handler")


class TelegramCommandHandler:

    def __init__(
        self,
        bot_token: str,
        authorized_chat_id: str,
        bot_ref,
        position_lock: threading.Lock,
    ):
        self._token = bot_token
        self._authorized_chat_id = str(authorized_chat_id)
        self._bot = bot_ref
        self._position_lock = position_lock

        self._last_update_id: int = 0
        self._paused: bool = False
        self._force_close_pending: Dict[str, float] = {}

        # 回覆頻率限制
        self._last_reply_time: float = 0.0
        self._min_reply_interval: float = 1.0

        # 指令分發表
        self._commands = {
            "/help": self._cmd_help,
            "/status": self._cmd_status,
            "/position": self._cmd_position,
            "/today": self._cmd_today,
            "/stop": self._cmd_stop,
            "/pause": self._cmd_pause,
            "/resume": self._cmd_resume,
            "/trades": self._cmd_trades,
            "/risk": self._cmd_risk,
            "/config": self._cmd_config,
            "/force_close": self._cmd_force_close,
        }

        # 啟動時清除舊訊息（避免處理離線期間累積的指令）
        self._flush_old_updates()

        logger.info("TelegramCommandHandler initialized")

    @property
    def is_paused(self) -> bool:
        return self._paused

    # ================================================================
    # Polling
    # ================================================================

    def poll(self) -> None:
        """
        非阻塞 polling — 從 Telegram 取得新訊息並處理

        在 _main_loop 中每 3 秒呼叫一次
        """
        try:
            url = f"https://api.telegram.org/bot{self._token}/getUpdates"
            params = {
                "offset": self._last_update_id + 1,
                "timeout": 0,  # 非阻塞
                "allowed_updates": '["message"]',
            }
            resp = requests.get(url, params=params, timeout=5)
            if resp.status_code != 200:
                return

            data = resp.json()
            if not data.get("ok"):
                return

            for update in data.get("result", []):
                self._last_update_id = update["update_id"]
                message = update.get("message", {})
                text = message.get("text", "").strip()
                chat_id = str(message.get("chat", {}).get("id", ""))

                if not text or not chat_id:
                    continue

                # 授權檢查
                if chat_id != self._authorized_chat_id:
                    logger.warning(
                        f"Unauthorized command attempt | "
                        f"chat_id={chat_id} text={text[:50]}"
                    )
                    continue

                self._handle_command(text, chat_id)

        except Exception as e:
            logger.debug(f"Telegram polling error: {e}")

    def _flush_old_updates(self) -> None:
        """啟動時清除所有待處理的舊訊息"""
        try:
            url = f"https://api.telegram.org/bot{self._token}/getUpdates"
            params = {"offset": -1, "timeout": 0}
            resp = requests.get(url, params=params, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("result", [])
                if results:
                    self._last_update_id = results[-1]["update_id"]
                    logger.info(
                        f"Flushed {len(results)} old Telegram update(s)"
                    )
        except Exception as e:
            logger.debug(f"Flush old updates failed: {e}")

    # ================================================================
    # 指令分發
    # ================================================================

    def _handle_command(self, text: str, chat_id: str) -> None:
        parts = text.split()
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        # 移除 @bot_username（群組中的指令格式）
        if "@" in cmd:
            cmd = cmd.split("@")[0]

        handler = self._commands.get(cmd)
        if handler is None:
            self._reply(chat_id, f"Unknown command: {cmd}\nSend /help for available commands.")
            return

        logger.info(f"Telegram command received: {text}")

        try:
            if cmd in ("/trades", "/force_close"):
                handler(chat_id, args)
            else:
                handler(chat_id)
        except Exception as e:
            logger.error(f"Command handler error: {e}", exc_info=True)
            self._reply(chat_id, f"Command error: {e}")

    # ================================================================
    # 指令實作
    # ================================================================

    def _cmd_help(self, chat_id: str) -> None:
        msg = (
            "<b>Available Commands</b>\n\n"
            "/help — Show this message. 顯示所有可用指令及說明\n"
            "/status — Balance, position, daily PnL, uptime. 餘額、權益、持倉方向、日/總 PnL、今日交易數、運行時間、步數\n"
            "/position — Current position details. 當前持倉詳情：方向、入場價、數量、止損價、當前價、浮動盈虧、持倉時間\n"
            "/today — Today's trade summary. 今日統計：交易數、勝/敗次數、勝率、日0PnL、最佳/最差單筆\n"
            "/trades [N] — Recent N trades (default 5). 最近 N 筆交易紀錄（預設 5，最多 20），顯示方向、PnL、持倉時間、平倉原因\n"
            "/risk — Risk manager status. 風控狀態：日 PnL、連敗次數、斷路器狀態、API 錯誤數、待機/kill switch\n"
            "/config — Key config parameters. 關鍵配置：交易對、槓桿、倉位比例、ATR 止損、最大持倉、模型名稱\n"
            "/pause — Pause trading (bot keeps running). 暫停交易（bot 持續運行，現有倉位保留，但不開新倉）\n"
            "/resume — Resume trading. 恢復交易\n"
            "/stop — Graceful shutdown. 優雅關閉 bot（依 shutdown_action 決定是平倉還是保留止損）\n"
            "/force_close — Force close position (requires confirm). 強制平倉，需 30 秒內再發 /force_close confirm 確認才執行"
        )
        self._reply(chat_id, msg)

    def _cmd_status(self, chat_id: str) -> None:
        state = self._bot.state
        uptime = datetime.now(timezone.utc) - self._bot._start_time
        hours, remainder = divmod(int(uptime.total_seconds()), 3600)
        minutes = remainder // 60

        pos_str = {-1: "SHORT", 0: "None", 1: "LONG"}.get(state.position, "?")
        pnl_sign = "+" if state.daily_pnl >= 0 else ""
        paused_str = " [PAUSED]" if self._paused else ""

        msg = (
            f"<b>Status{paused_str}</b>\n"
            f"Balance: {state.balance:.2f} U\n"
            f"Equity: {state.equity:.2f} U\n"
            f"Position: {pos_str}\n"
            f"Daily PnL: {pnl_sign}{state.daily_pnl:.2f} U\n"
            f"Total PnL: {state.total_pnl:+.2f} U\n"
            f"Today Trades: {state.trade_count}\n"
            f"Uptime: {hours}h {minutes}m\n"
            f"Steps: {state._step_count}"
        )
        self._reply(chat_id, msg)

    def _cmd_position(self, chat_id: str) -> None:
        state = self._bot.state

        if state.position == 0:
            self._reply(chat_id, "No open position.")
            return

        side = "LONG" if state.position == 1 else "SHORT"

        # 取當前價格
        current_price = self._get_current_price()
        if current_price and state.entry_price > 0:
            if state.position == 1:
                floating_pnl = (current_price - state.entry_price) * state.quantity
            else:
                floating_pnl = (state.entry_price - current_price) * state.quantity
            price_str = f"\nCurrent Price: {current_price:.2f}"
            pnl_str = f"\nFloating PnL: {floating_pnl:+.4f} U"
        else:
            price_str = ""
            pnl_str = ""

        holding_min = state.holding_steps  # 1 step = 1 min

        msg = (
            f"<b>Position: {side}</b>\n"
            f"Entry: {state.entry_price:.2f}\n"
            f"Quantity: {state.quantity}\n"
            f"Stop Loss: {state.current_sl:.2f}"
            f"{price_str}"
            f"{pnl_str}\n"
            f"Holding: {holding_min} min\n"
            f"SL Order ID: {state.sl_order_id or 'N/A'}"
        )
        self._reply(chat_id, msg)

    def _cmd_today(self, chat_id: str) -> None:
        state = self._bot.state
        total = state.daily_win_count + state.daily_loss_count
        win_rate = (state.daily_win_count / total * 100) if total > 0 else 0.0

        msg = (
            f"<b>Today's Summary</b>\n"
            f"Trades: {state.trade_count}\n"
            f"Wins: {state.daily_win_count} | Losses: {state.daily_loss_count}\n"
            f"Win Rate: {win_rate:.1f}%\n"
            f"Daily PnL: {state.daily_pnl:+.4f} U\n"
            f"Best Trade: {state.daily_max_pnl:+.4f} U\n"
            f"Worst Trade: {state.daily_min_pnl:+.4f} U"
        )
        self._reply(chat_id, msg)

    def _cmd_trades(self, chat_id: str, args: list) -> None:
        n = 5
        if args:
            try:
                n = min(int(args[0]), 20)
            except ValueError:
                pass

        history = self._bot.state.trade_history
        if not history:
            self._reply(chat_id, "No trade history.")
            return

        recent = history[-n:]
        lines = [f"<b>Last {len(recent)} Trade(s)</b>\n"]
        for i, t in enumerate(reversed(recent), 1):
            pnl = t.get("pnl", 0)
            pnl_pct = t.get("pnl_pct", 0)
            lines.append(
                f"{i}. {t.get('side', '?')} | "
                f"PnL: {pnl:+.4f} ({pnl_pct:+.2f}%) | "
                f"{t.get('holding_steps', 0)}min | "
                f"{t.get('reason', '')}"
            )
        self._reply(chat_id, "\n".join(lines))

    def _cmd_risk(self, chat_id: str) -> None:
        state = self._bot.state
        rm = self._bot.risk_manager
        rm_stats = rm.get_stats()

        msg = (
            f"<b>Risk Status</b>\n"
            f"Daily PnL: {state.daily_pnl:+.4f} U\n"
            f"Consecutive Losses: {state.consecutive_losses}\n"
            f"Max Consec Losses: {rm.max_consecutive_losses}\n"
            f"Daily Loss Limit: {rm.max_daily_loss_pct:.0%}\n"
            f"Total Loss Limit: {rm.max_total_loss_pct:.0%}\n"
            f"---\n"
            f"Circuit Breaker: {'ACTIVE' if rm_stats['in_cooldown'] else 'OK'}\n"
            f"CB Triggers: {rm_stats['cb_trigger_count']}\n"
            f"API Errors: {rm_stats['consecutive_api_errors']}\n"
            f"Standby Mode: {'YES' if rm_stats['standby_mode'] else 'No'}\n"
            f"Kill Switch: {'ACTIVE' if rm_stats['kill_switch'] else 'No'}"
        )
        self._reply(chat_id, msg)

    def _cmd_config(self, chat_id: str) -> None:
        cfg = self._bot.config
        trading = cfg.get("trading", {})
        risk = cfg.get("risk", {})
        model = cfg.get("model", {})
        from pathlib import Path

        msg = (
            f"<b>Config</b>\n"
            f"Symbol: {trading.get('symbol', '?')}\n"
            f"Leverage: {trading.get('leverage', '?')}x\n"
            f"Position Size: {trading.get('position_size_pct', 0) * 100:.0f}%\n"
            f"ATR Stop: {risk.get('atr_stop_multiplier', '?')}x\n"
            f"Stop Loss: {risk.get('stop_loss_pct', 0) * 100:.1f}%\n"
            f"Trailing Stop: {risk.get('trailing_stop', False)}\n"
            f"Max Holding: {trading.get('max_holding_steps', '?')} steps\n"
            f"Max Order: {risk.get('max_order_value_usdt', '?')} U\n"
            f"Model: {Path(model.get('path', '?')).name}\n"
            f"Shutdown: {cfg.get('system', {}).get('shutdown_action', '?')}"
        )
        self._reply(chat_id, msg)

    def _cmd_pause(self, chat_id: str) -> None:
        if self._paused:
            self._reply(chat_id, "Already paused.")
            return
        self._paused = True
        logger.info("Trading PAUSED via Telegram command")
        self._reply(chat_id, "Trading paused. Existing positions kept. Send /resume to continue.")

    def _cmd_resume(self, chat_id: str) -> None:
        if not self._paused:
            self._reply(chat_id, "Not paused.")
            return
        self._paused = False
        logger.info("Trading RESUMED via Telegram command")
        self._reply(chat_id, "Trading resumed.")

    def _cmd_stop(self, chat_id: str) -> None:
        self._reply(chat_id, "Shutting down bot...")
        logger.info("Shutdown requested via Telegram command")
        with self._position_lock:
            self._bot._shutdown_requested = True

    def _cmd_force_close(self, chat_id: str, args: list) -> None:
        state = self._bot.state

        if state.position == 0:
            self._reply(chat_id, "No open position to close.")
            return

        if not args or args[0].lower() != "confirm":
            self._force_close_pending[chat_id] = time.time()
            side = "LONG" if state.position == 1 else "SHORT"
            self._reply(
                chat_id,
                f"Force close {side} position?\n"
                f"Send <code>/force_close confirm</code> within 30s to execute."
            )
            return

        # 檢查確認是否在 30 秒內
        pending_time = self._force_close_pending.get(chat_id, 0)
        if time.time() - pending_time > 30:
            self._reply(chat_id, "Confirmation expired. Send /force_close again.")
            self._force_close_pending.pop(chat_id, None)
            return

        self._force_close_pending.pop(chat_id, None)

        # 取得鎖並平倉
        with self._position_lock:
            if state.position == 0:
                self._reply(chat_id, "Position already closed.")
                return

            result = self._bot.executor.force_close(
                state, reason="telegram_force_close"
            )
            if result:
                record = state.close_position(
                    exit_price=result.get("exit_price", 0),
                    pnl=result.get("pnl", 0),
                    fee=result.get("fee", 0),
                    reason="telegram_force_close",
                )
                self._bot.tlogger.log_trade(result)
                self._reply(
                    chat_id,
                    f"Position force closed.\n"
                    f"PnL: {result.get('pnl', 0):+.4f} U"
                )
                logger.info(f"Force close executed via Telegram | pnl={result.get('pnl', 0):+.4f}")
            else:
                self._reply(chat_id, "Force close failed. Check logs.")

    # ================================================================
    # 回覆
    # ================================================================

    def _reply(self, chat_id: str, text: str) -> None:
        """回覆 Telegram 訊息"""
        # 頻率限制
        now = time.time()
        elapsed = now - self._last_reply_time
        if elapsed < self._min_reply_interval:
            time.sleep(self._min_reply_interval - elapsed)

        url = f"https://api.telegram.org/bot{self._token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
        }
        try:
            resp = requests.post(url, json=payload, timeout=5)
            if resp.status_code != 200:
                logger.warning(f"Telegram reply failed: HTTP {resp.status_code}")
        except requests.RequestException as e:
            logger.warning(f"Telegram reply error: {e}")

        self._last_reply_time = time.time()

    # ================================================================
    # 輔助
    # ================================================================

    def _get_current_price(self) -> Optional[float]:
        """從 buffer 取最新收盤價"""
        try:
            buffer = self._bot.data_feed.get_buffer()
            if buffer is not None and not buffer.empty:
                return float(buffer.iloc[-1]["close"])
        except Exception:
            pass
        return None
