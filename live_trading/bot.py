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
from live_trading.logger import TradingLogger
from live_trading.notifier import create_notifier
from live_trading.risk_manager import RiskManager
from live_trading.state import TradingState
from live_trading.state_snapshot import StateSnapshot
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
        )

        # 恢復快照（防崩潰後風控歸零）
        self.snapshot = StateSnapshot()
        snap_data = self.snapshot.get_recoverable_fields()
        self.state.restore_from_snapshot(snap_data)

        # 同步交易所持倉
        pos_data = self.client.get_position_risk(symbol)
        if pos_data:
            self.state.sync_from_exchange(pos_data, balance)

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

        # 10. Data Feed（最後，因為啟動後會開始收 K 線）
        ws_url = self.client.get_ws_kline_url(symbol, "1m")
        self.data_feed = DataFeed(
            ws_url=ws_url,
            buffer_size=cfg.get("data", {}).get("buffer_size", 500),
            on_bar_close=self._on_bar_close,
        )

        # 註冊 signal handler
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # 11. Telegram Command Handler（可選）
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
            if self.state.position != 0:
                if self.state.position == 1:
                    unrealized = (current_price - self.state.entry_price) * self.state.quantity
                else:
                    unrealized = (self.state.entry_price - current_price) * self.state.quantity
                current_equity = self.state.balance + unrealized
            else:
                current_equity = self.state.balance

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

            # === 3.6 暫停檢查（Telegram /pause 指令） ===
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

            # === 4-7. 倉位操作（加鎖防止與 Telegram 指令競態） ===
            with self.position_lock:
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

            # === 11. 每日特徵快照（UTC 0:00 附近） ===
            now_utc = datetime.now(timezone.utc)
            if now_utc.hour == 0 and now_utc.minute == 0:
                self.feature_engine.save_daily_snapshot(buffer)

        except Exception as e:
            logger.error(f"Error in on_bar_close: {e}", exc_info=True)
            if self.notifier:
                self.notifier.send_error(f"on_bar_close error: {e}")

    def _handle_trade_result(self, result: dict, obs, market_features,
                             current_price: float, action: int,
                             executed: bool, risk_passed: bool) -> None:
        """處理交易結果：更新 state + 記錄 + 通知"""
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
        from live_trading.logger import _obs_hash
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
    # 定期健康檢查
    # ================================================================

    def _health_check(self) -> None:
        """每 5 分鐘：state vs 交易所同步"""
        try:
            symbol = self.config["trading"]["symbol"]
            balance = self.client.get_balance()
            pos_data = self.client.get_position_risk(symbol)

            if pos_data:
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


def _check_single_instance() -> None:
    """確保只有一個 bot 實例在運行（PID lock file）"""
    if PID_FILE.exists():
        try:
            old_pid = int(PID_FILE.read_text().strip())
            # 檢查舊進程是否還活著
            os.kill(old_pid, 0)
            print(f"ERROR: Another bot instance is already running (PID {old_pid})")
            print(f"  If this is a stale lock, delete {PID_FILE}")
            sys.exit(1)
        except (ProcessLookupError, ValueError):
            # 舊進程已死，清理 PID file
            pass

    PID_FILE.write_text(str(os.getpid()))


def _cleanup_pid() -> None:
    """清理 PID file"""
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
