"""
Binance Futures API Client — Testnet / Production 切換

職責：
- 封裝 Binance Futures REST API 連線
- 根據 config 切換 Testnet / Production
- 統一錯誤處理與 API 回應格式
- 提供帳戶查詢、下單、止損單管理等方法

安全要點：
- API Key 從環境變數讀取，永不硬編碼
- 每次 API 呼叫紀錄請求與回應（供審計）
- 非冪等操作（下單）不自動重試
"""

import hashlib
import hmac
import logging
import os
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)

# === Binance API Endpoints ===
BINANCE_FUTURES_PROD = "https://fapi.binance.com"
BINANCE_FUTURES_TESTNET = "https://testnet.binancefuture.com"

BINANCE_WS_PROD = "wss://fstream.binance.com"
BINANCE_WS_TESTNET = "wss://fstream.binancefuture.com"


class BinanceClientError(Exception):
    """Binance API 錯誤"""
    def __init__(self, status_code: int, error_code: int, message: str):
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        super().__init__(f"HTTP {status_code} | Binance Error {error_code}: {message}")


class BinanceFuturesClient:
    """
    Binance Futures REST API Client

    Usage:
        client = BinanceFuturesClient(testnet=True)
        account = client.get_account()
        client.place_market_order("ETHUSDT", "BUY", 0.04)
    """

    def __init__(self, testnet: bool = True,
                 api_key_env: str = "BINANCE_API_KEY",
                 api_secret_env: str = "BINANCE_API_SECRET"):
        """
        初始化 Binance Futures Client

        Args:
            testnet: True = Testnet, False = Production
            api_key_env: API Key 環境變數名稱
            api_secret_env: API Secret 環境變數名稱

        Raises:
            EnvironmentError: 環境變數未設定
        """
        self.testnet = testnet
        self.base_url = BINANCE_FUTURES_TESTNET if testnet else BINANCE_FUTURES_PROD
        self.ws_url = BINANCE_WS_TESTNET if testnet else BINANCE_WS_PROD

        # 從環境變數讀取 API Key
        self.api_key = os.environ.get(api_key_env)
        self.api_secret = os.environ.get(api_secret_env)

        if not self.api_key or not self.api_secret:
            raise EnvironmentError(
                f"Missing API credentials. Set environment variables: "
                f"{api_key_env} and {api_secret_env}"
            )

        self.session = requests.Session()
        self.session.headers.update({
            "X-MBX-APIKEY": self.api_key,
            "Content-Type": "application/x-www-form-urlencoded",
        })

        # 請求計數（供斷路器使用）
        self._request_count = 0
        self._last_request_time = 0.0

        env_label = "TESTNET" if testnet else "PRODUCTION"
        logger.info(f"Binance Futures Client initialized [{env_label}]")
        logger.info(f"  Base URL: {self.base_url}")

    # ================================================================
    # 簽名與請求
    # ================================================================

    def _sign(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """為 params 加入 timestamp 和 HMAC-SHA256 簽名"""
        params["timestamp"] = int(time.time() * 1000)
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = signature
        return params

    def _request(self, method: str, path: str, params: Optional[Dict] = None,
                 signed: bool = True, timeout: float = 5.0) -> Dict:
        """
        發送 API 請求

        Args:
            method: HTTP method (GET/POST/DELETE)
            path: API endpoint path (e.g. "/fapi/v1/order")
            params: 請求參數
            signed: 是否需要簽名（私有 API）
            timeout: 請求超時秒數

        Returns:
            API 回應 JSON

        Raises:
            BinanceClientError: API 回傳錯誤
            requests.Timeout: 請求超時
        """
        params = params or {}
        if signed:
            params = self._sign(params)

        url = f"{self.base_url}{path}"
        self._request_count += 1
        self._last_request_time = time.time()

        logger.debug(f"API {method} {path} | params={_safe_log_params(params)}")

        try:
            if method == "GET":
                response = self.session.get(url, params=params, timeout=timeout)
            elif method == "POST":
                response = self.session.post(url, data=params, timeout=timeout)
            elif method == "DELETE":
                response = self.session.delete(url, params=params, timeout=timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
        except requests.Timeout:
            logger.error(f"API TIMEOUT {method} {path} ({timeout}s)")
            raise
        except requests.ConnectionError as e:
            logger.error(f"API CONNECTION ERROR {method} {path}: {e}")
            raise

        # 解析回應
        data = response.json() if response.text else {}

        if response.status_code != 200:
            error_code = data.get("code", -1)
            error_msg = data.get("msg", "Unknown error")
            logger.error(
                f"API ERROR {method} {path} | "
                f"HTTP {response.status_code} | "
                f"code={error_code} msg={error_msg}"
            )
            raise BinanceClientError(response.status_code, error_code, error_msg)

        logger.debug(f"API OK {method} {path}")
        return data

    # ================================================================
    # 帳戶 & 市場資訊
    # ================================================================

    def ping(self) -> bool:
        """測試 API 連通性"""
        try:
            self._request("GET", "/fapi/v1/ping", signed=False, timeout=3.0)
            return True
        except Exception as e:
            logger.warning(f"Ping failed: {e}")
            return False

    def get_server_time(self) -> int:
        """取得伺服器時間（毫秒）"""
        data = self._request("GET", "/fapi/v1/time", signed=False)
        return data["serverTime"]

    def get_account(self) -> Dict:
        """取得帳戶資訊（餘額、持倉等）"""
        return self._request("GET", "/fapi/v2/account")

    def get_balance(self) -> float:
        """取得 USDT 可用餘額"""
        account = self.get_account()
        for asset in account.get("assets", []):
            if asset["asset"] == "USDT":
                return float(asset["availableBalance"])
        return 0.0

    def get_position_risk(self, symbol: str) -> Optional[Dict]:
        """
        取得指定合約的持倉資訊

        Returns:
            持倉資訊 dict，無持倉時 positionAmt == "0"
        """
        data = self._request("GET", "/fapi/v2/positionRisk", params={"symbol": symbol})
        for pos in data:
            if pos["symbol"] == symbol:
                return pos
        return None

    def get_exchange_info(self, symbol: str) -> Optional[Dict]:
        """取得交易對資訊（精度、最小下單量等）"""
        data = self._request("GET", "/fapi/v1/exchangeInfo", signed=False)
        for s in data.get("symbols", []):
            if s["symbol"] == symbol:
                return s
        return None

    def get_symbol_filters(self, symbol: str) -> Dict[str, Dict]:
        """
        取得交易對的各種 filter（LOT_SIZE, MIN_NOTIONAL, PRICE_FILTER 等）

        Returns:
            {filter_type: filter_dict} 格式
        """
        info = self.get_exchange_info(symbol)
        if not info:
            return {}
        return {f["filterType"]: f for f in info.get("filters", [])}

    def get_mark_price(self, symbol: str) -> float:
        """取得標記價格"""
        data = self._request(
            "GET", "/fapi/v1/premiumIndex",
            params={"symbol": symbol}, signed=False
        )
        return float(data["markPrice"])

    def get_ticker_price(self, symbol: str) -> Dict:
        """取得最新成交價"""
        return self._request(
            "GET", "/fapi/v1/ticker/price",
            params={"symbol": symbol},
            signed=False,
        )

    def get_klines(self, symbol: str, interval: str = "1m",
                   limit: int = 500, **kwargs) -> List[List]:
        """
        取得歷史 K 線（REST API，用於暖機和斷線補缺口）

        Args:
            symbol: 交易對
            interval: K 線間隔（"1m", "5m", etc.）
            limit: 數量上限（最大 1500）
            **kwargs: 額外參數（startTime, endTime 等）

        Returns:
            K 線列表 [[open_time, open, high, low, close, volume, ...], ...]
        """
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        params.update(kwargs)
        return self._request(
            "GET", "/fapi/v1/klines",
            params=params,
            signed=False,
        )

    # ================================================================
    # 下單操作
    # ================================================================

    def set_leverage(self, symbol: str, leverage: int) -> Dict:
        """設定槓桿倍數"""
        return self._request("POST", "/fapi/v1/leverage", params={
            "symbol": symbol,
            "leverage": leverage,
        })

    def set_margin_type(self, symbol: str, margin_type: str) -> Dict:
        """
        設定保證金模式（CROSSED / ISOLATED）

        Note: 如果已經是目標模式，Binance 會回傳 error code -4046，
              這不是真正的錯誤，需要特別處理
        """
        try:
            return self._request("POST", "/fapi/v1/marginType", params={
                "symbol": symbol,
                "marginType": margin_type,
            })
        except BinanceClientError as e:
            if e.error_code == -4046:
                # "No need to change margin type." — 已經是目標模式
                logger.info(f"Margin type already {margin_type} for {symbol}")
                return {"msg": "Already set"}
            raise

    def place_market_order(self, symbol: str, side: str,
                           quantity: float) -> Dict:
        """
        發送市價單

        Args:
            symbol: 交易對 (e.g. "ETHUSDT")
            side: "BUY" / "SELL"
            quantity: 下單數量

        Returns:
            訂單成交結果

        Note:
            市價單為非冪等操作，失敗時不自動重試
        """
        params = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": self._format_quantity(quantity, symbol),
        }

        logger.info(f"MARKET ORDER | {side} {quantity} {symbol}")
        result = self._request("POST", "/fapi/v1/order", params=params)
        logger.info(
            f"ORDER FILLED | orderId={result.get('orderId')} "
            f"avgPrice={result.get('avgPrice')} "
            f"executedQty={result.get('executedQty')} "
            f"status={result.get('status')}"
        )
        return result

    def place_limit_ioc(self, symbol: str, side: str,
                        quantity: float, price: float) -> Dict:
        """
        發送限價 IOC 單（Immediate-Or-Cancel）

        比市價單多一層滑點保護：
        - 能成交就立即成交（跟市價單一樣快）
        - 價格超過限價則自動取消（避免高波動下的不利滑點）

        Args:
            symbol: 交易對
            side: "BUY" / "SELL"
            quantity: 下單數量
            price: 限價（BUY 的上限 / SELL 的下限）

        Returns:
            訂單結果（status 可能為 FILLED / PARTIALLY_FILLED / EXPIRED）
        """
        params = {
            "symbol": symbol,
            "side": side,
            "type": "LIMIT",
            "timeInForce": "IOC",
            "quantity": self._format_quantity(quantity, symbol),
            "price": self._format_price(price, symbol),
        }

        logger.info(
            f"LIMIT IOC ORDER | {side} {quantity} {symbol} @ {price:.2f}"
        )
        result = self._request("POST", "/fapi/v1/order", params=params)
        status = result.get("status", "UNKNOWN")
        logger.info(
            f"ORDER RESULT | orderId={result.get('orderId')} "
            f"avgPrice={result.get('avgPrice')} "
            f"executedQty={result.get('executedQty')} "
            f"status={status}"
        )
        return result

    def place_stop_market(self, symbol: str, side: str,
                          stop_price: float,
                          close_position: bool = True,
                          quantity: Optional[float] = None) -> Dict:
        """
        發送 STOP_MARKET 止損單（reduce-only）

        Args:
            symbol: 交易對
            side: "BUY"（空單止損）/ "SELL"（多單止損）
            stop_price: 觸發價格
            close_position: True = closePosition 模式（平掉全部持倉）
            quantity: close_position=False 時需指定數量

        Returns:
            止損單結果
        """
        params = {
            "symbol": symbol,
            "side": side,
            "type": "STOP_MARKET",
            "stopPrice": self._format_price(stop_price, symbol),
            "workingType": "MARK_PRICE",  # 用標記價格觸發，防止插針
        }

        if close_position:
            params["closePosition"] = "true"
        else:
            if quantity is None:
                raise ValueError("quantity required when close_position=False")
            params["quantity"] = self._format_quantity(quantity, symbol)
            params["reduceOnly"] = "true"

        logger.info(
            f"STOP_MARKET ORDER | {side} {symbol} "
            f"stopPrice={stop_price}"
        )
        result = self._request("POST", "/fapi/v1/order", params=params)
        logger.info(f"STOP ORDER PLACED | orderId={result.get('orderId')}")
        return result

    def cancel_order(self, symbol: str, order_id: int) -> Dict:
        """取消指定訂單"""
        logger.info(f"CANCEL ORDER | {symbol} orderId={order_id}")
        return self._request("DELETE", "/fapi/v1/order", params={
            "symbol": symbol,
            "orderId": order_id,
        })

    def cancel_all_orders(self, symbol: str) -> Dict:
        """取消指定合約的所有掛單"""
        logger.info(f"CANCEL ALL ORDERS | {symbol}")
        return self._request("DELETE", "/fapi/v1/allOpenOrders", params={
            "symbol": symbol,
        })

    def get_open_orders(self, symbol: str) -> List[Dict]:
        """取得指定合約的所有掛單"""
        return self._request("GET", "/fapi/v1/openOrders", params={
            "symbol": symbol,
        })

    def get_order(self, symbol: str, order_id: int) -> Dict:
        """查詢指定訂單狀態"""
        return self._request("GET", "/fapi/v1/order", params={
            "symbol": symbol,
            "orderId": order_id,
        })

    # ================================================================
    # 精度處理
    # ================================================================

    def _format_quantity(self, quantity: float, symbol: str) -> str:
        """
        格式化下單數量至交易所要求的精度

        Note: 首次呼叫時會快取 symbol 的精度資訊
        """
        precision = self._get_quantity_precision(symbol)
        return f"{quantity:.{precision}f}"

    def _format_price(self, price: float, symbol: str) -> str:
        """格式化價格至交易所要求的精度"""
        precision = self._get_price_precision(symbol)
        return f"{price:.{precision}f}"

    def _get_quantity_precision(self, symbol: str) -> int:
        """取得數量精度（小數位數）"""
        if not hasattr(self, "_symbol_precision_cache"):
            self._symbol_precision_cache = {}
        if symbol not in self._symbol_precision_cache:
            self._load_symbol_precision(symbol)
        return self._symbol_precision_cache[symbol]["quantity"]

    def _get_price_precision(self, symbol: str) -> int:
        """取得價格精度（小數位數）"""
        if not hasattr(self, "_symbol_precision_cache"):
            self._symbol_precision_cache = {}
        if symbol not in self._symbol_precision_cache:
            self._load_symbol_precision(symbol)
        return self._symbol_precision_cache[symbol]["price"]

    def _load_symbol_precision(self, symbol: str) -> None:
        """從 exchangeInfo 載入並快取 symbol 精度"""
        info = self.get_exchange_info(symbol)
        if not info:
            raise ValueError(f"Symbol {symbol} not found on exchange")

        self._symbol_precision_cache[symbol] = {
            "quantity": info.get("quantityPrecision", 3),
            "price": info.get("pricePrecision", 2),
        }
        logger.info(
            f"Loaded precision for {symbol}: "
            f"qty={self._symbol_precision_cache[symbol]['quantity']}, "
            f"price={self._symbol_precision_cache[symbol]['price']}"
        )

    # ================================================================
    # WebSocket URL
    # ================================================================

    def get_ws_kline_url(self, symbol: str, interval: str = "1m") -> str:
        """取得 K 線 WebSocket 訂閱 URL"""
        stream = f"{symbol.lower()}@kline_{interval}"
        return f"{self.ws_url}/ws/{stream}"

    # ================================================================
    # 帳戶初始化
    # ================================================================

    def setup_account(self, symbol: str, leverage: int,
                      margin_type: str) -> None:
        """
        初始化帳戶設定（啟動時呼叫一次）

        Args:
            symbol: 交易對
            leverage: 槓桿倍數
            margin_type: 保證金模式
        """
        logger.info(f"Setting up account for {symbol}...")
        self.set_leverage(symbol, leverage)
        self.set_margin_type(symbol, margin_type)
        logger.info(
            f"Account setup complete: "
            f"leverage={leverage}, margin_type={margin_type}"
        )


def _safe_log_params(params: Dict) -> Dict:
    """過濾敏感參數（signature）供日誌記錄"""
    safe = {k: v for k, v in params.items() if k != "signature"}
    if "signature" in params:
        safe["signature"] = "***"
    return safe
