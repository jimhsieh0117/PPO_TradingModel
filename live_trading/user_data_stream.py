"""
Binance User Data Stream — 即時帳戶事件推送

職責：
- 透過 WebSocket 接收交易所推送的帳戶事件
- ORDER_TRADE_UPDATE：訂單成交/取消/過期（取代 _resolve_order_status 輪詢）
- ACCOUNT_UPDATE：倉位和餘額變動（取代 _health_check 定期同步）
- listenKey 自動續期（每 30 分鐘 PUT 一次，60 分鐘過期）
- 斷線自動重連

設計原則：
- 事件回調通知 bot 層，不直接修改 state
- 與現有 health_check 並存（User Data Stream 為主，health_check 為 fallback）
"""

import json
import logging
import threading
import time
from typing import Callable, Dict, Optional

import websocket

logger = logging.getLogger("live_trading.user_data_stream")

# listenKey 續期間隔（秒）— 幣安要求 60 分鐘內續期，30 分鐘較安全
KEEPALIVE_INTERVAL_S = 1800


class UserDataStream:
    """
    Binance User Data Stream 接收器

    Usage:
        stream = UserDataStream(
            client=binance_client,
            on_order_update=my_order_callback,
            on_account_update=my_account_callback,
        )
        stream.start()
        # ... 收到推送時自動回調 ...
        stream.stop()
    """

    def __init__(self, client,
                 on_order_update: Optional[Callable[[Dict], None]] = None,
                 on_account_update: Optional[Callable[[Dict], None]] = None):
        """
        Args:
            client: BinanceFuturesClient 實例
            on_order_update: ORDER_TRADE_UPDATE 回調
            on_account_update: ACCOUNT_UPDATE 回調
        """
        self.client = client
        self.on_order_update = on_order_update
        self.on_account_update = on_account_update

        self._listen_key: Optional[str] = None
        self._ws: Optional[websocket.WebSocketApp] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._keepalive_thread: Optional[threading.Thread] = None
        self._running = False
        self._connected = False

        # L2: 連線監控指標
        self._reconnect_count: int = 0
        self._last_message_time: float = 0.0
        self._backoff: int = 5  # 重連指數退避（秒）

    # ================================================================
    # 生命週期
    # ================================================================

    def start(self) -> bool:
        """啟動 User Data Stream（取得 listenKey + 連接 WebSocket）"""
        try:
            self._listen_key = self._create_listen_key()
            if not self._listen_key:
                logger.error("Failed to create listenKey — User Data Stream disabled")
                return False

            self._running = True

            # WebSocket 連線
            self._ws_thread = threading.Thread(
                target=self._run_ws, daemon=True, name="UserDataStream-WS"
            )
            self._ws_thread.start()

            # listenKey 續期
            self._keepalive_thread = threading.Thread(
                target=self._keepalive_loop, daemon=True, name="UserDataStream-KA"
            )
            self._keepalive_thread.start()

            logger.info("User Data Stream started")
            return True

        except Exception as e:
            logger.error(f"Failed to start User Data Stream: {e}")
            return False

    def stop(self) -> None:
        """停止 User Data Stream"""
        self._running = False
        if self._ws:
            self._ws.close()
        if self._listen_key:
            try:
                self._delete_listen_key()
            except Exception:
                pass
        logger.info("User Data Stream stopped")

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def reconnect_count(self) -> int:
        return self._reconnect_count

    @property
    def last_message_age(self) -> float:
        """距離上次收到消息的秒數（0 = 尚未收到任何消息）"""
        if self._last_message_time == 0.0:
            return 0.0
        return time.time() - self._last_message_time

    # ================================================================
    # listenKey 管理
    # ================================================================

    def _create_listen_key(self) -> Optional[str]:
        """POST /fapi/v1/listenKey — 取得 listenKey"""
        try:
            data = self.client._request(
                "POST", "/fapi/v1/listenKey", params={}, signed=False
            )
            key = data.get("listenKey", "")
            if key:
                logger.info(f"listenKey created: {key[:16]}...")
            return key
        except Exception as e:
            logger.error(f"Create listenKey failed: {e}")
            return None

    def _keepalive_listen_key(self) -> bool:
        """PUT /fapi/v1/listenKey — 續期 listenKey（透過 _request 統一錯誤處理）"""
        try:
            self.client._request(
                "PUT", "/fapi/v1/listenKey", params={}, signed=False, timeout=5.0
            )
            logger.debug("listenKey keepalive sent")
            return True
        except Exception as e:
            logger.warning(f"listenKey keepalive failed: {e}")
            return False

    def _delete_listen_key(self) -> None:
        """DELETE /fapi/v1/listenKey — 關閉 stream（M4: 統一使用 _request）"""
        try:
            self.client._request(
                "DELETE", "/fapi/v1/listenKey", params={}, signed=False, timeout=5.0
            )
            logger.debug("listenKey deleted")
        except Exception as e:
            logger.warning(f"listenKey delete failed: {e}")

    def _keepalive_loop(self) -> None:
        """定期續期 listenKey"""
        while self._running:
            time.sleep(KEEPALIVE_INTERVAL_S)
            if not self._running:
                break
            if not self._keepalive_listen_key():
                # 續期失敗 → 重新建立
                logger.warning("listenKey keepalive failed — recreating")
                new_key = self._create_listen_key()
                if new_key:
                    self._listen_key = new_key
                    # 重連 WebSocket
                    if self._ws:
                        self._ws.close()

    # ================================================================
    # WebSocket
    # ================================================================

    def _run_ws(self) -> None:
        """WebSocket 主迴圈（含自動重連 + 指數退避）"""
        self._backoff = 5  # 初始重連等待秒數（_on_open 重連成功時重設）
        max_backoff = 60  # 最大等待秒數

        while self._running:
            ws_url = f"{self.client.ws_url}/ws/{self._listen_key}"
            try:
                self._ws = websocket.WebSocketApp(
                    ws_url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_open=self._on_open,
                )
                self._ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                logger.error(f"User Data Stream WS exception: {e}")

            self._connected = False

            if not self._running:
                break

            self._reconnect_count += 1
            logger.info(
                f"User Data Stream reconnecting in {self._backoff}s... "
                f"(attempt #{self._reconnect_count})"
            )
            time.sleep(self._backoff)
            self._backoff = min(self._backoff * 2, max_backoff)

            # M5: 重連前刪除舊 listenKey，避免累積到上限
            if self._listen_key:
                self._delete_listen_key()
            new_key = self._create_listen_key()
            if new_key:
                self._listen_key = new_key

    def _on_open(self, ws) -> None:
        self._connected = True
        self._backoff = 5  # 重連成功，重設退避計時器
        logger.info("User Data Stream WebSocket connected")

    def _on_close(self, ws, close_status_code, close_msg) -> None:
        self._connected = False
        logger.warning(
            f"User Data Stream closed: code={close_status_code} msg={close_msg}"
        )

    def _on_error(self, ws, error) -> None:
        logger.error(f"User Data Stream error: {error}")

    def _on_message(self, ws, message: str) -> None:
        """處理 User Data Stream 事件"""
        self._last_message_time = time.time()
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from User Data Stream: {message[:200]}")
            return

        event_type = data.get("e", "")

        if event_type == "ORDER_TRADE_UPDATE":
            self._handle_order_update(data)
        elif event_type == "ACCOUNT_UPDATE":
            self._handle_account_update(data)
        elif event_type == "listenKeyExpired":
            logger.warning("listenKey expired — reconnecting")
            new_key = self._create_listen_key()
            if new_key:
                self._listen_key = new_key
            if self._ws:
                self._ws.close()  # 觸發重連

    def _handle_order_update(self, data: Dict) -> None:
        """
        處理 ORDER_TRADE_UPDATE 事件

        data.o 包含：
        - s: symbol
        - S: side (BUY/SELL)
        - o: order type (MARKET/LIMIT/STOP_MARKET...)
        - X: order status (NEW/FILLED/PARTIALLY_FILLED/CANCELED/EXPIRED)
        - ap: average price (成交均價)
        - q: original quantity
        - z: cumulative filled quantity
        - rp: realized profit (已實現盈虧)
        - n: commission (手續費)
        - i: orderId
        """
        order = data.get("o", {})
        status = order.get("X", "")
        symbol = order.get("s", "")
        order_type = order.get("o", "")

        logger.info(
            f"ORDER_TRADE_UPDATE | {symbol} {order.get('S')} "
            f"{order_type} status={status} "
            f"avgPrice={order.get('ap')} filledQty={order.get('z')} "
            f"realizedPnl={order.get('rp')}"
        )

        if self.on_order_update:
            try:
                self.on_order_update(order)
            except Exception as e:
                logger.error(f"on_order_update callback error: {e}", exc_info=True)

    def _handle_account_update(self, data: Dict) -> None:
        """
        處理 ACCOUNT_UPDATE 事件

        data.a 包含：
        - B: 餘額列表 [{a: asset, wb: wallet_balance, cw: cross_wallet_balance}]
        - P: 持倉列表 [{s: symbol, pa: position_amount, ep: entry_price, up: unrealized_pnl}]
        - m: event reason (ORDER/FUNDING_FEE/WITHDRAW/DEPOSIT...)
        """
        account = data.get("a", {})
        reason = account.get("m", "")

        logger.info(f"ACCOUNT_UPDATE | reason={reason}")

        if self.on_account_update:
            try:
                self.on_account_update(account)
            except Exception as e:
                logger.error(f"on_account_update callback error: {e}", exc_info=True)
