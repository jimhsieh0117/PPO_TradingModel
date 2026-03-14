"""
即時 K 線數據接收 — WebSocket 事件驅動

職責：
- 連接 Binance Futures WebSocket（kline_1m stream）
- 維護滾動 K 線 buffer（DataFrame, FIFO）
- K 線收盤時回調通知 bot.py
- 斷線自動重連（指數退避）
- 重連後用 REST API 補回缺失 K 線

安全要點：
- 每根 K 線驗證 timestamp 嚴格遞增（防重複/亂序）
- 間距檢查：正常 60s，容許 ±2s 誤差
- buffer 是唯一數據來源，不依賴交易所歷史 API 做即時決策
"""

import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np
import pandas as pd
import websocket

from live_trading.utils.retry import ReconnectManager

logger = logging.getLogger("live_trading.data_feed")

# 期望的 K 線間距（秒）
EXPECTED_INTERVAL_S = 60
# 間距容許誤差（秒）
INTERVAL_TOLERANCE_S = 2

# Buffer DataFrame 欄位（與 data_pipeline 一致）
BUFFER_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "trades"]


class DataFeed:
    """
    即時 K 線數據接收器

    Usage:
        feed = DataFeed(
            ws_url="wss://fstream.binancefuture.com/ws/ethusdt@kline_1m",
            buffer_size=500,
            on_bar_close=my_callback,
        )
        feed.start()          # 啟動 WebSocket（背景執行緒）
        feed.warmup(client)   # 用 REST API 填充初始 buffer
        # ... 等待 K 線事件 ...
        feed.stop()           # 關閉
    """

    def __init__(self, ws_url: str, buffer_size: int = 500,
                 on_bar_close: Optional[Callable[["DataFeed"], None]] = None):
        """
        初始化 DataFeed

        Args:
            ws_url: WebSocket URL（e.g. wss://fstream.binancefuture.com/ws/ethusdt@kline_1m）
            buffer_size: 滾動 buffer 最大長度
            on_bar_close: K 線收盤回調函數，參數為 DataFeed 實例
        """
        self.ws_url = ws_url
        self.buffer_size = buffer_size
        self.on_bar_close = on_bar_close

        # 滾動 buffer（空 DataFrame，暖機後填充）
        self._buffer = pd.DataFrame(columns=BUFFER_COLUMNS)
        self._buffer_lock = threading.Lock()

        # WebSocket
        self._ws: Optional[websocket.WebSocketApp] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._running = False
        self._connected = False
        self._last_heartbeat = 0.0

        # 重連管理器
        self._reconnect = ReconnectManager(
            base_delay=1.0, max_delay=60.0, max_attempts=50
        )

        # 統計
        self._bars_received = 0
        self._bars_dropped = 0
        self._gaps_detected = 0

    # ================================================================
    # 暖機：用 REST API 填充初始 buffer
    # ================================================================

    def warmup(self, client, symbol: str, warmup_bars: int = 200) -> int:
        """
        用 REST API 載入歷史 K 線填充 buffer

        Args:
            client: BinanceFuturesClient 實例
            symbol: 交易對（e.g. "ETHUSDT"）
            warmup_bars: 需要載入的 K 線數量

        Returns:
            實際載入的 K 線數量
        """
        logger.info(f"Warming up buffer: fetching {warmup_bars} historical klines...")

        # Binance REST API 最多回傳 1500 根
        raw_klines = client.get_klines(symbol, interval="1m", limit=warmup_bars)

        rows = []
        for k in raw_klines:
            rows.append({
                "timestamp": pd.Timestamp(k[0], unit="ms", tz="UTC"),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "trades": int(k[8]),
            })

        if not rows:
            logger.error("Warmup failed: no klines received from REST API")
            return 0

        df = pd.DataFrame(rows)
        df = df.set_index("timestamp")

        with self._buffer_lock:
            self._buffer = df.tail(self.buffer_size).copy()
            self._buffer.reset_index(inplace=True)

        loaded = len(self._buffer)
        logger.info(
            f"Warmup complete: {loaded} bars loaded "
            f"(from {self._buffer.iloc[0]['timestamp']} "
            f"to {self._buffer.iloc[-1]['timestamp']})"
        )
        return loaded

    # ================================================================
    # WebSocket 生命週期
    # ================================================================

    def start(self) -> None:
        """啟動 WebSocket 連線（背景執行緒）"""
        if self._running:
            logger.warning("DataFeed already running")
            return

        self._running = True
        self._ws_thread = threading.Thread(
            target=self._run_ws_loop, daemon=True, name="DataFeed-WS"
        )
        self._ws_thread.start()
        logger.info(f"DataFeed started | url={self.ws_url}")

    def stop(self) -> None:
        """停止 WebSocket 連線"""
        self._running = False
        if self._ws:
            self._ws.close()
        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=5)
        logger.info("DataFeed stopped")

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def last_heartbeat_age(self) -> float:
        """距離上次收到 WebSocket 訊息的秒數"""
        if self._last_heartbeat == 0:
            return float("inf")
        return time.time() - self._last_heartbeat

    # ================================================================
    # Buffer 存取
    # ================================================================

    def get_buffer(self) -> pd.DataFrame:
        """
        取得 buffer 的 copy（thread-safe）

        Returns:
            K 線 DataFrame（DatetimeIndex）供 feature_engine 使用
        """
        with self._buffer_lock:
            df = self._buffer.copy()

        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        return df

    @property
    def buffer_length(self) -> int:
        with self._buffer_lock:
            return len(self._buffer)

    # ================================================================
    # 斷線補缺口
    # ================================================================

    def fill_gap(self, client, symbol: str) -> int:
        """
        重連後補回缺失的 K 線

        Args:
            client: BinanceFuturesClient 實例
            symbol: 交易對

        Returns:
            補回的 K 線數量
        """
        with self._buffer_lock:
            if self._buffer.empty:
                return 0
            last_ts = self._buffer.iloc[-1]["timestamp"]

        # 計算缺口
        now = pd.Timestamp.now(tz="UTC")
        gap_minutes = int((now - last_ts).total_seconds() / 60)

        if gap_minutes <= 1:
            return 0

        logger.warning(f"Detected gap: {gap_minutes} minutes since last bar")
        self._gaps_detected += 1

        if gap_minutes > 5:
            logger.error(
                f"Gap > 5 minutes ({gap_minutes}m). "
                f"Notifying user — will NOT auto-trade until buffer stable."
            )

        # 用 REST 補回
        raw = client.get_klines(symbol, interval="1m", limit=min(gap_minutes + 5, 500))
        filled = 0

        for k in raw:
            ts = pd.Timestamp(k[0], unit="ms", tz="UTC")
            if ts <= last_ts:
                continue

            new_row = {
                "timestamp": ts,
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "trades": int(k[8]),
            }
            self._append_bar(new_row, from_rest=True)
            filled += 1

        logger.info(f"Gap filled: {filled} bars recovered")
        return filled

    # ================================================================
    # WebSocket 內部
    # ================================================================

    def _run_ws_loop(self) -> None:
        """WebSocket 主迴圈（含自動重連）"""
        while self._running:
            try:
                self._ws = websocket.WebSocketApp(
                    self.ws_url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_open=self._on_open,
                )
                # run_forever 會阻塞直到斷線
                self._ws.run_forever(
                    ping_interval=20,
                    ping_timeout=10,
                )
            except Exception as e:
                logger.error(f"WebSocket exception: {e}")

            self._connected = False

            if not self._running:
                break

            # 重連退避
            if self._reconnect.should_give_up():
                logger.critical(
                    f"WebSocket reconnect gave up after "
                    f"{self._reconnect.attempt_count} attempts"
                )
                break

            delay = self._reconnect.next_delay()
            logger.info(f"Reconnecting in {delay:.1f}s...")
            time.sleep(delay)

    def _on_open(self, ws) -> None:
        self._connected = True
        self._last_heartbeat = time.time()
        self._reconnect.reset()
        logger.info("WebSocket connected")

    def _on_close(self, ws, close_status_code, close_msg) -> None:
        self._connected = False
        logger.warning(
            f"WebSocket closed: code={close_status_code} msg={close_msg}"
        )

    def _on_error(self, ws, error) -> None:
        logger.error(f"WebSocket error: {error}")

    def _on_message(self, ws, message: str) -> None:
        """處理 WebSocket 訊息"""
        self._last_heartbeat = time.time()

        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from WebSocket: {message[:100]}")
            return

        kline = data.get("k")
        if not kline:
            return

        # 只處理已收盤的 K 線
        if not kline.get("x", False):
            return

        # 解析 K 線
        new_bar = {
            "timestamp": pd.Timestamp(kline["t"], unit="ms", tz="UTC"),
            "open": float(kline["o"]),
            "high": float(kline["h"]),
            "low": float(kline["l"]),
            "close": float(kline["c"]),
            "volume": float(kline["v"]),
            "trades": int(kline["n"]),
        }

        if self._append_bar(new_bar, from_rest=False):
            # 通知主迴圈
            if self.on_bar_close:
                try:
                    self.on_bar_close(self)
                except Exception as e:
                    logger.error(f"on_bar_close callback error: {e}", exc_info=True)

    def _append_bar(self, bar: dict, from_rest: bool = False) -> bool:
        """
        將新 K 線加入 buffer

        Args:
            bar: K 線數據 dict
            from_rest: 是否來自 REST API（補缺口）

        Returns:
            True = 成功加入, False = 被丟棄
        """
        with self._buffer_lock:
            # 驗證 timestamp 嚴格遞增
            if not self._buffer.empty:
                last_ts = self._buffer.iloc[-1]["timestamp"]
                new_ts = bar["timestamp"]

                if new_ts <= last_ts:
                    self._bars_dropped += 1
                    source = "REST" if from_rest else "WS"
                    logger.warning(
                        f"Dropped bar ({source}): ts={new_ts} <= last={last_ts}"
                    )
                    return False

                # 間距檢查
                gap_s = (new_ts - last_ts).total_seconds()
                if abs(gap_s - EXPECTED_INTERVAL_S) > INTERVAL_TOLERANCE_S:
                    if not from_rest:
                        logger.warning(
                            f"Abnormal bar interval: {gap_s:.0f}s "
                            f"(expected {EXPECTED_INTERVAL_S}s)"
                        )

            # 加入 buffer
            new_row = pd.DataFrame([bar])
            self._buffer = pd.concat(
                [self._buffer, new_row], ignore_index=True
            )

            # FIFO：保持 buffer_size
            if len(self._buffer) > self.buffer_size:
                self._buffer = self._buffer.iloc[-self.buffer_size:].reset_index(drop=True)

            self._bars_received += 1

        return True

    # ================================================================
    # 統計
    # ================================================================

    def get_stats(self) -> dict:
        """取得接收統計"""
        return {
            "bars_received": self._bars_received,
            "bars_dropped": self._bars_dropped,
            "gaps_detected": self._gaps_detected,
            "buffer_length": self.buffer_length,
            "connected": self._connected,
            "heartbeat_age_s": round(self.last_heartbeat_age, 1),
        }
