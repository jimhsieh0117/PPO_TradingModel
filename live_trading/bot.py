"""
主控迴圈 — 整合所有模組的入口點

職責：
1. 載入配置 + 驗證環境變數
2. 初始化所有模組（client, model, state, feed, engine, executor, risk, logger, notifier）
3. 同步帳戶狀態
4. 暖機：等待 buffer 填充 warmup_bars
5. 進入事件驅動主迴圈（每根 K 線觸發一次決策）
6. Graceful shutdown

設計原則：
- 單執行緒 + WebSocket 回調（避免競態條件）
- 事件驅動（非輪詢）
- 寧可不交易，也不可錯誤交易
"""

import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

# 確保專案根目錄在 sys.path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from live_trading.command_handler import TelegramCommandHandler
from live_trading.data_feed import DataFeed
from live_trading.executor import Executor
from live_trading.feature_engine import FeatureEngine
from live_trading.inference import InferenceEngine
from live_trading.logger import TradingLogger, _obs_hash
from live_trading.notifier import create_notifier
from live_trading.risk_manager import RiskManager
from live_trading.state import TradingState
from live_trading.state_snapshot import StateSnapshot
from live_trading.user_data_stream import UserDataStream
from live_trading.utils.binance_client import BinanceFuturesClient

logger = logging.getLogger("live_trading.bot")


class TradingBot:
    """
    實盤交易機器人

    Usage:
        bot = TradingBot("live_trading/config_live.yaml")
        bot.run()
    """

    def __init__(self, config_path: str = "live_trading/config_live.yaml"):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self._shutdown_requested = False
        self._config_mtime: float = 0.0  # config 檔案最後修改時間
        self._last_processed_ts = None   # 去重：上一根已處理 K 線的時間戳
        self._start_time: datetime = None  # 啟動時間（供 /status 指令使用）
        self._last_snapshot_date: str = ""  # M2: 每日特徵快照去重

        # 倉位操作鎖（防止 Telegram 指令與 on_bar_close 競態）
        self.position_lock = threading.Lock()

        # 所有模組在 _initialize() 中初始化
        self.client: BinanceFuturesClient = None
        self.inference: InferenceEngine = None
        self.state: TradingState = None
        self.data_feed: DataFeed = None
        self.feature_engine: FeatureEngine = None
        self.executor: Executor = None
        self.risk_manager: RiskManager = None
        self.tlogger: TradingLogger = None
        self.notifier = None
        self.snapshot: StateSnapshot = None
        self.command_handler: TelegramCommandHandler = None
        self.user_data_stream: UserDataStream = None

    # ================================================================
    # 啟動
    # ================================================================

    def run(self) -> None:
        """主入口 — 初始化 → 暖機 → 主迴圈"""
        try:
            self._initialize()
            self._warmup()
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received")
        except Exception as e:
            logger.critical(f"Fatal error: {e}", exc_info=True)
            if self.notifier:
                self.notifier.send_error(f"FATAL: {e}")
        finally:
            self._shutdown()

    # ================================================================
    # 初始化
    # ================================================================

    def _initialize(self) -> None:
        """初始化所有模組"""
        logger.info("=" * 60)
        logger.info("TRADING BOT STARTING")
        logger.info("=" * 60)

        cfg = self.config

        # 1. Logger（最先，讓後續模組可以記錄）
        log_cfg = cfg.get("logging", {})
        self.tlogger = TradingLogger(
            log_dir=log_cfg.get("log_dir", "live_trading/logs"),
            max_size_mb=log_cfg.get("max_log_size_mb", 50),
            backup_count=log_cfg.get("backup_count", 10),
            console_level=log_cfg.get("level", "INFO"),
        )

        # 2. Notifier
        self.notifier = create_notifier(cfg.get("notification", {}))

        # 3. Binance Client
        exc_cfg = cfg["exchange"]
        self.client = BinanceFuturesClient(
            testnet=exc_cfg["testnet"],
            api_key_env=exc_cfg["api_key_env"],
            api_secret_env=exc_cfg["api_secret_env"],
        )

        # 驗證連通性
        if not self.client.ping():
            raise ConnectionError("Cannot connect to Binance API")
        logger.info("Binance API connection verified")

        # 4. 設定帳戶（槓桿 + 保證金模式）
        trading_cfg = cfg["trading"]
        symbol = trading_cfg["symbol"]
        self.client.setup_account(
            symbol=symbol,
            leverage=trading_cfg["leverage"],
            margin_type=trading_cfg.get("margin_type", "CROSSED"),
        )

        # 5. Model
        model_cfg = cfg["model"]
        self.inference = InferenceEngine(
            model_path=model_cfg["path"],
            expected_md5=model_cfg.get("expected_md5", ""),
            use_lstm=model_cfg.get("use_lstm", False),
            deterministic=model_cfg.get("deterministic", True),
        )

        # 6. State
        balance = self.client.get_balance()
        episode_length = cfg.get("_training_reference", {}).get("episode_length", 720)
        max_holding = trading_cfg.get("max_holding_steps", 120)

        self.state = TradingState(
            initial_balance=balance,
            episode_length=episode_length,
            max_holding_steps=max_holding,
            on_exchange_close=self._on_exchange_close,
        )

        # 恢復快照（防崩潰後風控歸零）
        self.snapshot = StateSnapshot()
        snap_data = self.snapshot.get_recoverable_fields()
        self.state.restore_from_snapshot(snap_data)

        # 同步交易所持倉（強制成功，最多重試 3 次）
        pos_data = None
        for attempt in range(3):
            pos_data = self.client.get_position_risk(symbol)
            if pos_data is not None:
                break
            logger.warning(f"Position sync failed, retry {attempt + 1}/3")
            time.sleep(2)

        if pos_data is not None:
            self.state.sync_from_exchange(pos_data, balance)
        else:
            raise RuntimeError(
                "Cannot sync position from exchange after 3 retries — aborting startup"
            )

        # H5: 啟動時檢查交易所是否已有 Algo SL 掛單
        # 防止 snapshot 恢復的 sl_order_id 對應到已失效的訂單，或遺留孤兒 SL
        if self.state.has_position and self.state.sl_order_id:
            try:
                algo_orders = self.client.get_algo_open_orders(symbol)
                active_ids = {str(o.get("algoId", "")) for o in algo_orders}
                if self.state.sl_order_id not in active_ids:
                    logger.warning(
                        f"Restored sl_order_id={self.state.sl_order_id} "
                        f"not found in exchange algo orders — clearing"
                    )
                    self.state.sl_order_id = ""
                else:
                    logger.info(
                        f"Restored sl_order_id={self.state.sl_order_id} "
                        f"confirmed active on exchange"
                    )
            except Exception as e:
                logger.warning(f"Failed to verify SL order on startup: {e}")

        # 7. Feature Engine
        feature_config = self._load_training_feature_config()
        self.feature_engine = FeatureEngine(feature_config)
        self.feature_engine.setup_daily_snapshot(
            log_cfg.get("log_dir", "live_trading/logs")
        )

        # 8. Executor
        self.executor = Executor(self.client, cfg)
        self.executor.load_symbol_filters()

        # 9. Risk Manager
        self.risk_manager = RiskManager(cfg)

        # 10. User Data Stream（接收訂單/帳戶推送，取代輪詢）
        self.user_data_stream = UserDataStream(
            client=self.client,
            on_order_update=self._on_order_update,
            on_account_update=self._on_account_update,
        )
        if not self.user_data_stream.start():
            logger.warning(
                "User Data Stream failed to start — "
                "falling back to polling-based sync"
            )

        # 12. Data Feed（最後，因為啟動後會開始收 K 線）
        ws_url = self.client.get_ws_kline_url(symbol, "1m")
        self.data_feed = DataFeed(
            ws_url=ws_url,
            buffer_size=cfg.get("data", {}).get("buffer_size", 500),
            on_bar_close=self._on_bar_close,
        )

        # 註冊 signal handler
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # 13. Telegram Command Handler（可選）
        self._start_time = datetime.now(timezone.utc)
        notif_cfg = cfg.get("notification", {})
        tg_token = os.environ.get(
            notif_cfg.get("telegram_bot_token_env", ""), ""
        )
        tg_chat_id = os.environ.get(
            notif_cfg.get("telegram_chat_id_env", ""), ""
        )
        if tg_token and tg_chat_id:
            self.command_handler = TelegramCommandHandler(
                bot_token=tg_token,
                authorized_chat_id=tg_chat_id,
                bot_ref=self,
                position_lock=self.position_lock,
            )
        else:
            logger.info("Telegram command handler disabled (no token/chat_id)")

        env_label = "TESTNET" if exc_cfg["testnet"] else "PRODUCTION"
        logger.info(
            f"All modules initialized | "
            f"env={env_label} symbol={symbol} "
            f"balance={balance:.2f}U"
        )
        self.notifier.send_system(
            f"Bot started [{env_label}]\n"
            f"  Symbol: {symbol}\n"
            f"  Balance: {balance:.2f}U\n"
            f"  Model: {Path(model_cfg['path']).name}"
        )

    def _warmup(self) -> None:
        """暖機：用 REST API 填充 buffer + 啟動 WebSocket"""
        cfg = self.config
        symbol = cfg["trading"]["symbol"]
        warmup_bars = cfg.get("data", {}).get("warmup_bars", 200)

        # REST 填充
        loaded = self.data_feed.warmup(self.client, symbol, warmup_bars)
        if loaded < warmup_bars:
            logger.warning(
                f"Warmup incomplete: {loaded}/{warmup_bars} bars loaded"
            )

        # 驗證特徵可計算
        buffer = self.data_feed.get_buffer()
        test_features = self.feature_engine.compute(buffer)
        logger.info(
            f"Feature warmup test passed | shape={test_features.shape} "
            f"no_nan={not any(map(lambda x: x != x, test_features))}"
        )

        # 啟動 WebSocket
        self.data_feed.start()
        logger.info("Warmup complete — entering main loop")

    # ================================================================
    # 主迴圈（事件驅動）
    # ================================================================

    def _on_bar_close(self, feed: DataFeed) -> None:
        """
        每根 K 線收盤觸發一次

        這是整個系統的核心 — 從接收數據到執行交易的完整流程
        """
        try:
            symbol = self.config["trading"]["symbol"]
            buffer = feed.get_buffer()

            if buffer.empty:
                return

            # === 去重：防止同一根 K 線觸發兩次 ===
            last_ts = buffer.index[-1] if hasattr(buffer.index, 'dtype') else None
            if last_ts is not None and last_ts == self._last_processed_ts:
                return
            self._last_processed_ts = last_ts

            current_price = float(buffer.iloc[-1]["close"])

            # === 1. 計算特徵 ===
            market_features = self.feature_engine.compute(buffer)

            # === 2. 組合觀察向量 ===
            obs = self.state.build_observation(market_features, current_price)

            # === 3. 更新 equity（浮動盈虧） ===
            # H2: 在 position_lock 內讀取 balance/position 快照，
            # 防止 _on_account_update 在計算途中修改 balance
            with self.position_lock:
                snap_balance = self.state.balance
                snap_position = self.state.position
                snap_entry = self.state.entry_price
                snap_qty = self.state.quantity

            if snap_position != 0:
                if snap_position == 1:
                    unrealized = (current_price - snap_entry) * snap_qty
                else:
                    unrealized = (snap_entry - current_price) * snap_qty
                current_equity = snap_balance + unrealized
            else:
                current_equity = snap_balance

            self.state.step(current_equity=current_equity)

            # === 3.5 Client-side 止損檢查（Algo SL 的 backup） ===
            if self.state.position != 0 and self.state.current_sl > 0:
                sl_hit = False
                if self.state.position == 1 and current_price <= self.state.current_sl:
                    sl_hit = True
                elif self.state.position == -1 and current_price >= self.state.current_sl:
                    sl_hit = True
                if sl_hit:
                    logger.warning(
                        f"Client-side SL triggered | "
                        f"price={current_price:.2f} sl={self.state.current_sl:.2f}"
                    )
                    with self.position_lock:
                        # Double-close 防護：User Data Stream 可能已處理
                        if not self.state.has_position:
                            logger.info(
                                "SL triggered but position already closed "
                                "(likely by User Data Stream) — skipping"
                            )
                            return

                        # 先查交易所是否還有倉位（Algo SL 可能已觸發）
                        exchange_pos = self.executor._check_exchange_position()
                        if exchange_pos is None:
                            logger.error(
                                "Cannot verify exchange position — "
                                "aborting client-side SL close"
                            )
                            return
                        if exchange_pos == 0:
                            logger.warning(
                                "SL triggered but exchange has no position — "
                                "Algo SL likely already fired, clearing local state"
                            )
                            # C1: 查詢交易所成交記錄取得精確 exit_price
                            real_exit = current_price
                            real_pnl = 0.0
                            real_fee = 0.0
                            try:
                                trades = self.client.get_recent_user_trades(
                                    symbol, limit=5
                                )
                                for t in reversed(trades):
                                    realized = float(t.get("realizedPnl", "0"))
                                    if realized != 0:
                                        real_exit = float(t.get("price", current_price))
                                        real_fee = float(t.get("commission", "0"))
                                        real_pnl = realized - real_fee
                                        break
                            except Exception:
                                pass
                            self.state.close_position(
                                exit_price=real_exit,
                                pnl=real_pnl,
                                fee=real_fee,
                                reason="algo_sl_fired(client_side_sl)",
                            )
                            return

                        result = self.executor.force_close(
                            self.state, reason="client_side_sl"
                        )
                        if result:
                            self._handle_trade_result(
                                result, obs, market_features,
                                current_price, action=0,
                                executed=True, risk_passed=True,
                            )
                    return

            # === 4-7. 倉位操作（加鎖防止與 Telegram 指令競態） ===
            with self.position_lock:
                # === 3.6 暫停檢查（在鎖內，防止 pause 與開倉競態） ===
                if self.command_handler and self.command_handler.is_paused:
                    self.tlogger.log_decision(
                        bar_close=current_price, action=3,  # HOLD
                        executed=False, risk_passed=True,
                        risk_block_reason="paused_via_telegram",
                        position_before=self.state.position,
                        position_after=self.state.position,
                        obs=obs,
                    )
                    return

                # === 4. max_holding_steps 強制平倉 ===
                if self.state.is_max_holding:
                    logger.info(
                        f"Max holding steps reached ({self.state.holding_steps}) — "
                        f"force closing"
                    )
                    result = self.executor.force_close(
                        self.state, reason="max_holding_steps"
                    )
                    if result:
                        self._handle_trade_result(result, obs, market_features,
                                                  current_price, action=0,
                                                  executed=True, risk_passed=True)
                    return

                # === 5. 模型推論 ===
                action = self.inference.predict(obs)

                # === 6. 風控檢查 ===
                estimated_notional = 0.0
                if action in (1, 2):
                    estimated_notional = self.state.balance * self.executor.position_size_pct

                allowed, block_reason = self.risk_manager.check(
                    action, self.state, self.data_feed, estimated_notional
                )

                # === 7. 執行或記錄 ===
                if allowed:
                    atr = self.feature_engine.get_atr(buffer)
                    result = self.executor.execute(
                        action, self.state, atr=atr, current_price=current_price
                    )
                    executed = result is not None
                    if executed:
                        # 反向開倉時 result 是 list[dict]（平倉+開倉）
                        results = result if isinstance(result, list) else [result]
                        for r in results:
                            self._handle_trade_result(r, obs, market_features,
                                                      current_price, action,
                                                      executed=True, risk_passed=True)
                    else:
                        # HOLD 或無操作
                        self.tlogger.log_decision(
                            bar_close=current_price, action=action,
                            executed=False, risk_passed=True,
                            risk_block_reason=None,
                            position_before=self.state.position,
                            position_after=self.state.position,
                            obs=obs,
                        )
                else:
                    self.tlogger.log_decision(
                        bar_close=current_price, action=action,
                        executed=False, risk_passed=False,
                        risk_block_reason=block_reason,
                        position_before=self.state.position,
                        position_after=self.state.position,
                        obs=obs,
                    )
                    if block_reason:
                        logger.info(f"Action {action} blocked: {block_reason}")

            # === 8. 心跳 ===
            if self.notifier.check_heartbeat_due():
                self.notifier.send_heartbeat(
                    balance=self.state.balance,
                    daily_pnl=self.state.daily_pnl,
                    position=self.state.position,
                    total_trades=self.state.trade_count,
                )

            # === 9. 持久化快照 ===
            if self.config.get("system", {}).get("state_snapshot", True):
                self.snapshot.save(self.state.to_snapshot())

            # === 10. 定期健康檢查（每 5 分鐘） ===
            if self.state._step_count % 5 == 0:
                self._health_check()
                self._check_config_reload()

            # === 11. 每日特徵快照（UTC 0:00 的第一根 bar） ===
            now_utc = datetime.now(timezone.utc)
            today_str = now_utc.strftime("%Y-%m-%d")
            if now_utc.hour == 0 and today_str != self._last_snapshot_date:
                self.feature_engine.save_daily_snapshot(buffer)
                self._last_snapshot_date = today_str

        except Exception as e:
            logger.error(f"Error in on_bar_close: {e}", exc_info=True)
            if self.notifier:
                self.notifier.send_error(f"on_bar_close error: {e}")

    def _handle_trade_result(self, result: dict, obs, market_features,
                             current_price: float, action: int,
                             executed: bool, risk_passed: bool) -> None:
        """處理交易結果：更新 state + 記錄 + 通知"""
        # Algo SL 已觸發 — 交易所無倉位，只清本地 state
        if result.get("_algo_sl_fired"):
            logger.info(
                "Algo SL already fired on exchange — "
                "clearing local state without sending order"
            )
            self.state.close_position(
                exit_price=current_price,
                pnl=0.0,  # 真實 PnL 已由 Algo SL 結算
                fee=0.0,
                reason=f"algo_sl_fired({result.get('reason', '')})",
            )
            return

        position_before = self.state.position

        if "entry_price" in result and "exit_price" not in result:
            # 開倉
            if result.get("error") == "stop_order_failed":
                # SL 失敗 + 緊急平倉 — 不更新 state（已在 exchange 平倉）
                self.notifier.send_emergency_close(
                    symbol=result["symbol"],
                    reason="止損單下單失敗，已緊急市價平倉",
                )
                return

            side = 1 if result["side"] == "BUY" else -1
            self.state.open_position(
                side=side,
                entry_price=result["entry_price"],
                quantity=result["quantity"],
                sl_price=result.get("sl_price", 0.0),
                sl_order_id=result.get("sl_order_id", ""),
            )
            self.notifier.send_trade_open(
                symbol=result["symbol"],
                side=result["side"],
                price=result["entry_price"],
                quantity=result["quantity"],
                sl_price=result.get("sl_price", 0.0),
            )
        elif "exit_price" in result:
            # 平倉
            record = self.state.close_position(
                exit_price=result["exit_price"],
                pnl=result.get("pnl", 0.0),
                fee=result.get("fee", 0.0),
                reason=result.get("reason", "model"),
            )
            self.notifier.send_trade_close(
                symbol=result["symbol"],
                price=result["exit_price"],
                pnl=result.get("pnl", 0.0),
                pnl_pct=result.get("pnl_pct", 0.0),
                holding_minutes=record.get("holding_steps", 0),
                reason=result.get("reason", "model"),
            )

        # 記錄交易
        result["model_obs_hash"] = _obs_hash(obs)
        result["balance_after"] = self.state.balance
        self.tlogger.log_trade(result)

        # 記錄決策
        self.tlogger.log_decision(
            bar_close=current_price, action=action,
            executed=executed, risk_passed=risk_passed,
            risk_block_reason=None,
            position_before=position_before,
            position_after=self.state.position,
            features=obs, obs=obs,
        )

    # ================================================================
    # 交易所平倉回調（sync_from_exchange 偵測到 DESYNC 時觸發）
    # ================================================================

    def _on_exchange_close(self, entry_price: float, quantity: float,
                           side: int, estimated_pnl: float,
                           reason: str) -> dict:
        """
        交易所已平倉但本地 state 未更新時的回調

        注意：此回調由 sync_from_exchange 觸發，呼叫者必須持有 position_lock。
        目前的呼叫路徑：
        - _health_check → with position_lock → sync_from_exchange → 此回調
        - _initialize → sync_from_exchange → 此回調（啟動時單執行緒，無競態）

        Returns:
            dict with exit_price, fee, pnl（M7: 供 sync_from_exchange 傳入 close_position）
        """
        symbol = self.config["trading"]["symbol"]
        exit_price = 0.0
        fee = 0.0
        pnl = estimated_pnl

        # 嘗試從交易所成交記錄取得精確數據
        try:
            trades = self.client.get_recent_user_trades(symbol, limit=10)
            if trades:
                # 找最近的 realizedPnl != 0 的成交（即平倉成交）
                for t in reversed(trades):
                    realized = float(t.get("realizedPnl", "0"))
                    if realized != 0:
                        exit_price = float(t.get("price", "0"))
                        fee = float(t.get("commission", "0"))
                        pnl = realized - fee
                        logger.info(
                            f"Found exchange close trade | "
                            f"price={exit_price} pnl={realized:+.4f} "
                            f"fee={fee:.4f}"
                        )
                        break
        except Exception as e:
            logger.warning(f"Failed to query user trades: {e}")

        # 記錄到 trades.jsonl
        side_str = "LONG" if side == 1 else "SHORT"
        close_side = "SELL" if side == 1 else "BUY"
        pnl_pct = pnl / (entry_price * quantity + 1e-10) * 100 if quantity > 0 else 0.0

        trade_record = {
            "action": 0,  # CLOSE
            "symbol": symbol,
            "side": close_side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": quantity,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "fee": fee,
            "reason": reason,
            "balance_after": self.state.balance,
            "_source": "exchange_sync",
        }
        self.tlogger.log_trade(trade_record)

        # 發送 Telegram 通知
        self.notifier.send_trade_close(
            symbol=symbol,
            price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_minutes=self.state.holding_steps,
            reason=reason,
        )

        logger.info(
            f"Exchange close recorded | {side_str} "
            f"entry={entry_price:.2f} exit={exit_price:.2f} "
            f"pnl={pnl:+.4f} ({pnl_pct:+.2f}%) | reason={reason}"
        )

        # M7: 回傳精確資料供 sync_from_exchange 的 close_position 使用
        return {"exit_price": exit_price, "fee": fee, "pnl": pnl}

    # ================================================================
    # User Data Stream 回調
    # ================================================================

    def _on_order_update(self, order: dict) -> None:
        """
        ORDER_TRADE_UPDATE 回調 — 交易所推送訂單狀態變更

        用於即時偵測 Algo SL 觸發（不用等 health_check 輪詢）
        注意：此回調在 WebSocket 執行緒上運行，必須加鎖防競態
        """
        symbol = order.get("s", "")
        if symbol != self.config["trading"]["symbol"]:
            return

        status = order.get("X", "")
        order_type = order.get("o", "")
        realized_pnl = float(order.get("rp", "0"))

        # 偵測 STOP_MARKET 被觸發（Algo SL）
        if order_type == "STOP_MARKET" and status == "FILLED":
            avg_price = float(order.get("ap", "0"))
            filled_qty = float(order.get("z", "0"))
            commission = float(order.get("n", "0"))

            logger.info(
                f"ALGO SL FILLED (via User Data Stream) | "
                f"price={avg_price} qty={filled_qty} "
                f"pnl={realized_pnl:+.4f} fee={commission:.4f}"
            )

            with self.position_lock:
                # Double-close 防護：確認本地還有倉位
                if not self.state.has_position:
                    logger.info(
                        "ALGO SL FILLED but local position already closed — skipping"
                    )
                    return

                # C1: 先保存 close_position 會清除的欄位
                saved_entry_price = self.state.entry_price
                saved_quantity = self.state.quantity
                saved_holding_steps = self.state.holding_steps

                pnl = realized_pnl - commission
                pnl_pct = pnl / (saved_entry_price * saved_quantity + 1e-10) * 100

                # 先更新 state（close_position 會清除倉位欄位）
                self.state.close_position(
                    exit_price=avg_price,
                    pnl=pnl,
                    fee=commission,
                    reason="algo_sl_filled",
                )

                # 記錄 + 通知（使用保存的值和更新後的 balance）
                trade_record = {
                    "action": 0,
                    "symbol": symbol,
                    "side": order.get("S", ""),
                    "entry_price": saved_entry_price,
                    "exit_price": avg_price,
                    "quantity": filled_qty,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "fee": commission,
                    "reason": "algo_sl_filled",
                    "balance_after": self.state.balance,
                    "_source": "user_data_stream",
                }
                self.tlogger.log_trade(trade_record)

                self.notifier.send_trade_close(
                    symbol=symbol,
                    price=avg_price,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    holding_minutes=saved_holding_steps,
                    reason="algo_sl_filled",
                )

    def _on_account_update(self, account: dict) -> None:
        """
        ACCOUNT_UPDATE 回調 — 交易所推送餘額/持倉變動

        即時更新餘額，減少對 health_check 的依賴
        注意：此回調在 WebSocket 執行緒上運行，必須加鎖防競態
        """
        with self.position_lock:
            # 更新餘額
            for b in account.get("B", []):
                if b.get("a") == "USDT":
                    new_balance = float(b.get("wb", "0"))
                    if new_balance > 0:
                        self.state.balance = new_balance
                        logger.debug(f"Balance updated via stream: {new_balance:.2f}")
                    break

            # 更新持倉（同步方向，不覆蓋 state 的精細狀態）
            symbol = self.config["trading"]["symbol"]
            for p in account.get("P", []):
                if p.get("s") != symbol:
                    continue
                pos_amt = float(p.get("pa", "0"))
                if pos_amt == 0 and self.state.has_position:
                    # 交易所倉位已清空但 state 還有 — 可能是 SL 觸發
                    # 不在這裡處理（由 ORDER_TRADE_UPDATE 或 health_check 處理）
                    logger.info(
                        "ACCOUNT_UPDATE: position cleared on exchange, "
                        "waiting for ORDER_TRADE_UPDATE to reconcile"
                    )

    # ================================================================
    # 定期健康檢查
    # ================================================================

    def _health_check(self) -> None:
        """每 5 分鐘：state vs 交易所同步"""
        try:
            symbol = self.config["trading"]["symbol"]
            balance = self.client.get_balance()
            pos_data = self.client.get_position_risk(symbol)

            if pos_data:
                # C3: sync_from_exchange 可能觸發 _on_exchange_close 回調，
                # 該回調會修改 state，必須在 position_lock 下執行
                with self.position_lock:
                    self.state.sync_from_exchange(pos_data, balance)

            self.risk_manager.record_api_success()
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            # 嘗試判斷 HTTP status
            status = getattr(e, "status_code", 0)
            if status:
                self.risk_manager.record_api_error(status)

    # ================================================================
    # 主迴圈等待
    # ================================================================

    def _main_loop(self) -> None:
        """等待 WebSocket 事件（主執行緒保持活躍）"""
        logger.info("Main loop running — waiting for kline events...")

        poll_counter = 0
        while not self._shutdown_requested:
            time.sleep(1)
            poll_counter += 1

            # Telegram 指令 polling（每 3 秒）
            if self.command_handler and poll_counter % 3 == 0:
                self.command_handler.poll()

            # 檢查 WebSocket 連線
            if not self.data_feed.is_connected:
                if self.data_feed.last_heartbeat_age > 120:
                    logger.warning("WebSocket disconnected > 2min, attempting gap fill...")
                    try:
                        filled = self.data_feed.fill_gap(
                            self.client, self.config["trading"]["symbol"]
                        )
                        if filled > 5:
                            self.notifier.send_error(
                                f"WebSocket gap filled: {filled} bars recovered"
                            )
                    except Exception as e:
                        logger.error(f"Gap fill failed: {e}")

    # ================================================================
    # 關閉
    # ================================================================

    def _shutdown(self) -> None:
        """Graceful shutdown"""
        logger.info("Shutting down...")

        shutdown_action = self.config.get("system", {}).get(
            "shutdown_action", "keep_with_sl"
        )

        if self.user_data_stream:
            self.user_data_stream.stop()

        if self.data_feed:
            self.data_feed.stop()

        if self.state and self.state.has_position:
            if shutdown_action == "close_all":
                logger.info("Shutdown action: closing all positions")
                if self.executor:
                    result = self.executor.force_close(
                        self.state, reason="shutdown"
                    )
                    if result:
                        self.state.close_position(
                            exit_price=result.get("exit_price", 0),
                            pnl=result.get("pnl", 0),
                            reason="shutdown",
                        )
            else:
                logger.info(
                    "Shutdown action: keeping position with SL protection "
                    f"(sl={self.state.current_sl})"
                )

        # 最終快照
        if self.snapshot and self.state:
            self.snapshot.save(self.state.to_snapshot())

        if self.notifier:
            self.notifier.send_system(
                f"Bot stopped\n"
                f"  Balance: {self.state.balance:.2f}U\n"
                f"  Position: {self.state.position}\n"
                f"  Action: {shutdown_action}"
            )

        logger.info("Shutdown complete")

    def _signal_handler(self, signum, frame) -> None:
        """Signal handler for graceful shutdown"""
        logger.info(f"Signal {signum} received — requesting shutdown")
        self._shutdown_requested = True

    # ================================================================
    # 配置載入
    # ================================================================

    @staticmethod
    def _load_config(path: str) -> dict:
        """載入 YAML 配置"""
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"Config loaded: {path}")
        return config

    def _load_training_feature_config(self) -> dict:
        """
        從模型的訓練 config 載入特徵配置

        確保實盤特徵計算使用與訓練完全相同的參數
        """
        model_path = Path(self.config["model"]["path"])
        training_config_path = model_path.parent / "config.yaml"

        if training_config_path.exists():
            with open(training_config_path, "r", encoding="utf-8") as f:
                training_config = yaml.safe_load(f)
            feature_config = training_config.get("features", {})
            logger.info(
                f"Loaded training feature config from {training_config_path}"
            )
            return feature_config

        logger.warning(
            f"Training config not found at {training_config_path} — "
            f"using default feature config"
        )
        return {}

    # ================================================================
    # Config 熱重載
    # ================================================================

    def _check_config_reload(self) -> None:
        """
        檢查 config 檔案是否被修改，有的話重新載入風控/訂單參數

        可熱重載的參數（不影響模型行為）：
        - risk: max_daily_loss_pct, max_total_loss_pct, max_consecutive_losses,
                max_order_value_usdt, min_balance_to_trade, max_slippage_pct
        - system: shutdown_action

        不可熱重載（需重啟）：
        - model, trading.symbol, trading.position_size_pct, exchange
        """
        try:
            config_path = Path(self.config_path)
            mtime = config_path.stat().st_mtime

            if self._config_mtime == 0:
                self._config_mtime = mtime
                return

            if mtime <= self._config_mtime:
                return

            # 檔案有更新
            self._config_mtime = mtime
            new_config = self._load_config(self.config_path)

            # 更新風控參數
            new_risk = new_config.get("risk", {})
            old_risk = self.config.get("risk", {})
            changed = []

            for key in ["max_daily_loss_pct", "max_total_loss_pct",
                        "max_consecutive_losses", "max_order_value_usdt",
                        "min_balance_to_trade", "max_slippage_pct"]:
                if new_risk.get(key) != old_risk.get(key):
                    changed.append(f"{key}: {old_risk.get(key)} → {new_risk.get(key)}")

            if changed:
                # M3: 驗證所有必要 key 存在，防止半途更新
                required_risk_keys = [
                    "max_daily_loss_pct", "max_total_loss_pct",
                    "max_consecutive_losses", "max_order_value_usdt",
                    "min_balance_to_trade", "max_slippage_pct",
                ]
                missing = [k for k in required_risk_keys if k not in new_risk]
                if missing:
                    logger.warning(
                        f"Config reload aborted: missing risk keys {missing}"
                    )
                    return

                rm = self.risk_manager
                rm.max_daily_loss_pct = new_risk["max_daily_loss_pct"]
                rm.max_total_loss_pct = new_risk["max_total_loss_pct"]
                rm.max_consecutive_losses = new_risk["max_consecutive_losses"]
                rm.max_order_value = new_risk["max_order_value_usdt"]
                rm.min_balance_to_trade = new_risk["min_balance_to_trade"]
                rm.max_slippage_pct = new_risk["max_slippage_pct"]

                self.executor.max_order_value = new_risk["max_order_value_usdt"]

                self.config = new_config
                msg = "Config hot-reloaded:\n  " + "\n  ".join(changed)
                logger.info(msg)
                self.notifier.send_system(msg)
            else:
                # 其他欄位更新（shutdown_action 等）
                self.config = new_config
                logger.info("Config reloaded (no risk parameter changes)")

        except Exception as e:
            logger.warning(f"Config reload failed: {e}")


PID_FILE = Path("live_trading/bot.pid")
_pid_lock_fd = None  # 持有 flock 的 fd（程式結束前不可關閉）


def _check_single_instance() -> None:
    """確保只有一個 bot 實例在運行（fcntl.flock + PID file）"""
    global _pid_lock_fd

    # 嘗試使用 fcntl.flock（macOS/Linux，原子性保證）
    try:
        import fcntl
        _pid_lock_fd = open(PID_FILE, "a+")
        try:
            fcntl.flock(_pid_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            # 無法取得鎖 → 另一個實例正在運行
            _pid_lock_fd.seek(0)
            old_pid = _pid_lock_fd.read().strip()
            print(f"ERROR: Another bot instance is already running (PID {old_pid})")
            print(f"  If this is a stale lock, delete {PID_FILE}")
            sys.exit(1)
        # 取得鎖成功 → 寫入 PID
        _pid_lock_fd.seek(0)
        _pid_lock_fd.truncate()
        _pid_lock_fd.write(str(os.getpid()))
        _pid_lock_fd.flush()
        return
    except ImportError:
        pass  # Windows — fallback 到 PID 檢查

    # Fallback: PID 存在性檢查（Windows 或 fcntl 不可用）
    if PID_FILE.exists():
        try:
            old_pid = int(PID_FILE.read_text().strip())
            os.kill(old_pid, 0)
            print(f"ERROR: Another bot instance is already running (PID {old_pid})")
            print(f"  If this is a stale lock, delete {PID_FILE}")
            sys.exit(1)
        except (ProcessLookupError, ValueError):
            pass

    PID_FILE.write_text(str(os.getpid()))


def _cleanup_pid() -> None:
    """清理 PID file 和釋放 flock"""
    global _pid_lock_fd
    try:
        if _pid_lock_fd is not None:
            import fcntl
            fcntl.flock(_pid_lock_fd, fcntl.LOCK_UN)
            _pid_lock_fd.close()
            _pid_lock_fd = None
    except (ImportError, OSError):
        pass
    try:
        if PID_FILE.exists():
            PID_FILE.unlink()
    except OSError:
        pass


def main():
    """CLI 入口"""
    import argparse

    parser = argparse.ArgumentParser(description="PPO Trading Bot - Live")
    parser.add_argument(
        "--config", default="live_trading/config_live.yaml",
        help="Path to config_live.yaml",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Initialize all modules but don't start trading",
    )
    args = parser.parse_args()

    _check_single_instance()

    bot = TradingBot(config_path=args.config)

    try:
        if args.dry_run:
            print("=== DRY RUN MODE ===")
            bot._initialize()
            print("\nAll modules initialized successfully.")
            print(f"  Balance: {bot.state.balance:.2f}U")
            print(f"  Position: {bot.state.position}")
            print(f"  Model obs_dim: {bot.inference.obs_dimension}")
            print(f"  Feature dim: {bot.feature_engine.aggregator.get_state_dimension()}")
            print("\nDry run complete — no trading started.")
            bot._shutdown()
        else:
            bot.run()
    finally:
        _cleanup_pid()


if __name__ == "__main__":
    main()
