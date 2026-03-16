"""
下單執行模組 — 將 action 轉換為 Binance API 操作

職責：
- Action 0-3 對應 平倉/做多/做空/持有
- 市價單開倉 + STOP_MARKET 止損單
- 止損單驗證（下單後確認存在，否則立即平倉）
- 數量計算（balance * position_size_pct / price）
- minNotional 檢查

安全要點：
- 非冪等操作（下單）不自動重試
- 每次下單前先查詢持倉，與 state 比對
- 止損計算用實際成交均價（avgPrice），非下單前的市價
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from live_trading.utils.binance_client import BinanceFuturesClient, BinanceClientError

logger = logging.getLogger("live_trading.executor")

# Action 對應表（與 TradingEnv 完全一致）
ACTION_CLOSE = 0
ACTION_LONG = 1
ACTION_SHORT = 2
ACTION_HOLD = 3

ACTION_NAMES = {0: "CLOSE", 1: "LONG", 2: "SHORT", 3: "HOLD"}


class Executor:
    """
    下單執行器

    Usage:
        executor = Executor(client, config)
        result = executor.execute(action=1, state=state, atr=25.0)
    """

    def __init__(self, client: BinanceFuturesClient, config: dict):
        """
        Args:
            client: BinanceFuturesClient 實例
            config: 完整的 config_live 配置
        """
        self.client = client
        self.symbol: str = config["trading"]["symbol"]
        self.position_size_pct: float = config["trading"]["position_size_pct"]
        self.atr_stop_multiplier: float = config["risk"]["atr_stop_multiplier"]
        self.stop_loss_pct: float = config["risk"]["stop_loss_pct"]
        self.max_order_value: float = config["risk"]["max_order_value_usdt"]
        self.max_slippage_pct: float = config["risk"].get("max_slippage_pct", 0.003)

        # 快取交易所 filter
        self._min_notional: float = 5.0  # 預設值，啟動時更新
        self._min_qty: float = 0.001
        self._step_size: float = 0.001
        self._loaded_filters = False

        logger.info(
            f"Executor initialized | symbol={self.symbol} "
            f"size_pct={self.position_size_pct} "
            f"atr_mult={self.atr_stop_multiplier}"
        )

    def load_symbol_filters(self) -> None:
        """啟動時從交易所載入 filter（minNotional, LOT_SIZE 等）"""
        try:
            filters = self.client.get_symbol_filters(self.symbol)

            if "MIN_NOTIONAL" in filters:
                self._min_notional = float(filters["MIN_NOTIONAL"].get("notional", 5.0))

            if "LOT_SIZE" in filters:
                lot = filters["LOT_SIZE"]
                self._min_qty = float(lot.get("minQty", 0.001))
                self._step_size = float(lot.get("stepSize", 0.001))

            self._loaded_filters = True
            logger.info(
                f"Symbol filters loaded | minNotional={self._min_notional} "
                f"minQty={self._min_qty} stepSize={self._step_size}"
            )
        except Exception as e:
            logger.error(f"Failed to load symbol filters: {e}")

    # ================================================================
    # 主要入口
    # ================================================================

    def execute(self, action: int, state, atr: float,
                current_price: float) -> Union[None, Dict[str, Any], List[Dict[str, Any]]]:
        """
        執行交易動作

        Args:
            action: 模型輸出 (0=CLOSE, 1=LONG, 2=SHORT, 3=HOLD)
            state: TradingState 實例
            atr: 當前 ATR 值（止損計算用）
            current_price: 當前價格

        Returns:
            - None: HOLD 或無操作
            - dict: 單一交易結果（開倉或平倉）
            - list[dict]: 反向開倉時回傳 [平倉結果, 開倉結果]
        """
        if action == ACTION_HOLD:
            return None

        if action == ACTION_CLOSE:
            if state.position == 0:
                return None
            return self._close_position(state)

        if action == ACTION_LONG:
            if state.position == 1:
                return None
            # 有空倉 → 先平再開，回傳兩個結果
            if state.position == -1:
                close_result = self._close_position(state)
                if close_result is None:
                    return None
                open_result = self._open_position(side=1, state=state, atr=atr,
                                                  current_price=current_price)
                if open_result is None:
                    return close_result  # 平倉成功但開倉失敗，仍回傳平倉
                return [close_result, open_result]
            return self._open_position(side=1, state=state, atr=atr,
                                       current_price=current_price)

        if action == ACTION_SHORT:
            if state.position == -1:
                return None
            if state.position == 1:
                close_result = self._close_position(state)
                if close_result is None:
                    return None
                open_result = self._open_position(side=-1, state=state, atr=atr,
                                                  current_price=current_price)
                if open_result is None:
                    return close_result
                return [close_result, open_result]
            return self._open_position(side=-1, state=state, atr=atr,
                                       current_price=current_price)

        logger.error(f"Unknown action: {action}")
        return None

    def force_close(self, state, reason: str = "forced") -> Optional[Dict[str, Any]]:
        """
        強制平倉（max_holding_steps / kill switch / 風控觸發）

        Args:
            state: TradingState 實例
            reason: 平倉原因

        Returns:
            交易結果 dict
        """
        if state.position == 0:
            return None
        return self._close_position(state, reason=reason)

    # ================================================================
    # 開倉
    # ================================================================

    def _open_position(self, side: int, state, atr: float,
                       current_price: float) -> Optional[Dict[str, Any]]:
        """
        開倉流程：
        1. 計算下單數量
        2. 檢查 minNotional
        3. 發送市價單
        4. 等待成交確認
        5. 用實際成交均價計算止損
        6. 發送 STOP_MARKET 止損單
        7. 驗證止損單存在
        """
        # P1 防護：下單前確認交易所無同方向倉位
        exchange_pos = self._check_exchange_position()
        if exchange_pos == side:
            logger.warning(
                f"Exchange already has {'LONG' if side == 1 else 'SHORT'} "
                f"position — skipping duplicate open"
            )
            return None

        # 計算下單數量
        qty = self._calculate_quantity(state.balance, current_price)
        if qty is None:
            return None

        # 下單方向
        order_side = "BUY" if side == 1 else "SELL"

        try:
            # Step 1: 限價 IOC 單（滑點保護）
            if order_side == "BUY":
                limit_price = current_price * (1 + self.max_slippage_pct)
            else:
                limit_price = current_price * (1 - self.max_slippage_pct)

            order_result = self.client.place_limit_ioc(
                self.symbol, order_side, qty, limit_price
            )

            # 解析成交結果（收到 NEW 時輪詢最終狀態）
            order_result = self._resolve_order_status(order_result)
            status = order_result.get("status", "")

            if status == "EXPIRED":
                # IOC 完全未成交 — 價格超出滑點容忍範圍
                logger.warning(
                    f"LIMIT IOC expired (no fill) | "
                    f"side={order_side} limit={limit_price:.2f} — "
                    f"slippage exceeded {self.max_slippage_pct:.1%}"
                )
                return None
            if status not in ("FILLED", "PARTIALLY_FILLED"):
                logger.error(f"Order unexpected status: {status}")
                return None

            avg_price = float(order_result.get("avgPrice", current_price))
            executed_qty = float(order_result.get("executedQty", qty))
            order_id = order_result.get("orderId", "")

            # 計算手續費（從 API 回傳或估算）
            fee = self._estimate_fee(avg_price, executed_qty)

            # Step 2: 計算止損價（用實際成交均價）
            sl_price = self._calculate_stop_loss(side, avg_price, atr)

            # Step 3: 掛止損單（Algo Order API，獨立 try/except）
            sl_side = "SELL" if side == 1 else "BUY"
            sl_order_id = ""
            sl_ok = False
            try:
                sl_result = self.client.place_algo_stop(
                    self.symbol, sl_side, sl_price,
                    close_position=True,
                )
                sl_order_id = str(sl_result.get("algoId", ""))
                sl_ok = bool(sl_order_id)
            except Exception as e:
                logger.error(f"Algo stop order failed: {e}")

            if not sl_ok:
                # Algo SL 失敗 — 不緊急平倉，改用 client-side SL backup
                logger.warning(
                    "Algo stop order failed — relying on client-side SL"
                )

            logger.info(
                f"Position opened successfully | "
                f"{order_side} {executed_qty} @ {avg_price:.2f} | "
                f"SL={sl_price:.2f} (orderId={sl_order_id})"
            )

            return {
                "action": ACTION_LONG if side == 1 else ACTION_SHORT,
                "symbol": self.symbol,
                "side": order_side,
                "entry_price": avg_price,
                "quantity": executed_qty,
                "sl_price": sl_price,
                "sl_order_id": sl_order_id,
                "fee": fee,
                "order_id": order_id,
                "balance_after": state.balance - fee,
            }

        except BinanceClientError as e:
            logger.error(f"Open position failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error opening position: {e}", exc_info=True)
            return None

    # ================================================================
    # 平倉
    # ================================================================

    def _close_position(self, state,
                        reason: str = "model") -> Optional[Dict[str, Any]]:
        """
        平倉流程：
        1. 發送反方向市價單
        2. 取消止損單
        3. 計算 PnL
        """
        if state.position == 0:
            return None

        close_side = "SELL" if state.position == 1 else "BUY"

        try:
            # Step 1: 限價 IOC 平倉（滑點保護）
            # 平倉方向與持倉相反
            last_price = state.entry_price  # fallback
            try:
                ticker = self.client.get_ticker_price(self.symbol)
                last_price = float(ticker.get("price", last_price))
            except Exception:
                pass

            if close_side == "SELL":
                limit_price = last_price * (1 - self.max_slippage_pct)
            else:
                limit_price = last_price * (1 + self.max_slippage_pct)

            order_result = self.client.place_limit_ioc(
                self.symbol, close_side, state.quantity, limit_price
            )

            # 收到 NEW 時輪詢最終狀態
            order_result = self._resolve_order_status(order_result)
            status = order_result.get("status", "")
            if status in ("EXPIRED", "PARTIALLY_FILLED"):
                # 平倉必須完成 — fallback 到市價單
                remaining = state.quantity
                if status == "PARTIALLY_FILLED":
                    remaining -= float(order_result.get("executedQty", 0))
                if remaining > 0:
                    logger.warning(
                        f"IOC close incomplete ({status}), "
                        f"fallback to market for {remaining}"
                    )
                    order_result = self.client.place_market_order(
                        self.symbol, close_side, remaining
                    )
                    status = order_result.get("status", "")

            if status != "FILLED":
                logger.error(f"Close order not filled: status={status}")
                return None

            exit_price = float(order_result.get("avgPrice", 0))
            order_id = order_result.get("orderId", "")
            fee = self._estimate_fee(exit_price, state.quantity)

            # Step 2: 取消止損單（Algo Order）
            if state.sl_order_id:
                try:
                    self.client.cancel_algo_order(self.symbol, state.sl_order_id)
                except BinanceClientError as e:
                    # 止損單可能已被觸發或過期
                    logger.warning(f"Cancel algo SL failed (may already triggered): {e}")
                    # Fallback：嘗試標準取消（以防舊格式）
                    try:
                        self.client.cancel_order(self.symbol, int(state.sl_order_id))
                    except Exception:
                        pass

            # Step 3: 計算 PnL
            if state.position == 1:
                raw_pnl = (exit_price - state.entry_price) * state.quantity
            else:
                raw_pnl = (state.entry_price - exit_price) * state.quantity

            # 扣除開倉+平倉手續費
            open_fee = self._estimate_fee(state.entry_price, state.quantity)
            total_fee = open_fee + fee
            pnl = raw_pnl - total_fee

            logger.info(
                f"Position closed | {close_side} {state.quantity} @ {exit_price:.2f} | "
                f"PnL={pnl:+.4f} | reason={reason}"
            )

            return {
                "action": ACTION_CLOSE,
                "symbol": self.symbol,
                "side": close_side,
                "entry_price": state.entry_price,
                "exit_price": exit_price,
                "quantity": state.quantity,
                "pnl": pnl,
                "pnl_pct": pnl / (state.entry_price * state.quantity + 1e-10) * 100,
                "fee": total_fee,
                "order_id": order_id,
                "reason": reason,
                "balance_after": state.balance + pnl,
            }

        except BinanceClientError as e:
            logger.error(f"Close position failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error closing position: {e}", exc_info=True)
            return None

    # ================================================================
    # 止損計算
    # ================================================================

    def _calculate_stop_loss(self, side: int, entry_price: float,
                             atr: float) -> float:
        """
        計算止損價

        使用 ATR 動態止損，並以 stop_loss_pct 作為 fallback 上限

        Args:
            side: 1=多, -1=空
            entry_price: 實際成交均價
            atr: 當前 ATR

        Returns:
            止損價
        """
        atr_distance = atr * self.atr_stop_multiplier
        pct_distance = entry_price * self.stop_loss_pct

        # 取較小的止損距離（更保守）
        sl_distance = min(atr_distance, pct_distance)

        if side == 1:  # 多單
            sl_price = entry_price - sl_distance
        else:  # 空單
            sl_price = entry_price + sl_distance

        logger.debug(
            f"SL calculated | entry={entry_price:.2f} "
            f"atr_dist={atr_distance:.2f} pct_dist={pct_distance:.2f} "
            f"→ sl={sl_price:.2f}"
        )
        return sl_price

    # ================================================================
    # 數量計算
    # ================================================================

    def _calculate_quantity(self, balance: float,
                            price: float) -> Optional[float]:
        """
        計算下單數量

        qty = balance * position_size_pct / price
        → 取 stepSize 精度
        → 檢查 minNotional 和 max_order_value

        Returns:
            下單數量，或 None（不滿足條件）
        """
        raw_qty = balance * self.position_size_pct / price
        qty = self._round_to_step(raw_qty)

        # 最小數量檢查
        if qty < self._min_qty:
            logger.warning(
                f"Quantity too small: {qty} < minQty {self._min_qty}"
            )
            return None

        # minNotional 檢查
        notional = qty * price
        if notional < self._min_notional:
            logger.warning(
                f"Notional too small: {notional:.2f} < minNotional {self._min_notional}"
            )
            return None

        # max_order_value 檢查
        if notional > self.max_order_value:
            logger.warning(
                f"Notional exceeds max: {notional:.2f} > {self.max_order_value}"
            )
            # 調降數量至上限
            qty = self._round_to_step(self.max_order_value / price)
            notional = qty * price
            logger.info(f"Adjusted quantity to {qty} (notional={notional:.2f})")

        return qty

    def _round_to_step(self, qty: float) -> float:
        """將數量向下取整到 stepSize"""
        if self._step_size <= 0:
            return qty
        steps = int(qty / self._step_size)
        return steps * self._step_size

    # ================================================================
    # 輔助方法
    # ================================================================

    def _resolve_order_status(self, order_result: Dict,
                              timeout_s: float = 3.0) -> Dict:
        """
        IOC 訂單回傳 status=NEW 時，輪詢查詢最終狀態

        Binance matching engine 可能在低流動性時段有微秒級延遲，
        API 先回傳 NEW 但訂單實際已成交。

        Args:
            order_result: place_limit_ioc 的原始回傳
            timeout_s: 最長等待秒數

        Returns:
            更新後的訂單狀態 dict
        """
        status = order_result.get("status", "")
        if status != "NEW":
            return order_result

        order_id = order_result.get("orderId")
        if not order_id:
            return order_result

        logger.info(
            f"Order status=NEW, polling for final status | "
            f"orderId={order_id}"
        )

        deadline = time.time() + timeout_s
        last_result = order_result
        while time.time() < deadline:
            time.sleep(0.3)
            try:
                queried = self.client.get_order(self.symbol, int(order_id))
                last_result = queried
                q_status = queried.get("status", "")
                if q_status in ("FILLED", "PARTIALLY_FILLED",
                                "CANCELED", "EXPIRED"):
                    logger.info(
                        f"Order resolved | orderId={order_id} "
                        f"status={q_status} "
                        f"avgPrice={queried.get('avgPrice')} "
                        f"executedQty={queried.get('executedQty')}"
                    )
                    return queried
            except Exception as e:
                logger.warning(f"Order query failed: {e}")

        logger.warning(
            f"Order status poll timeout ({timeout_s}s) | "
            f"orderId={order_id} last_status={last_result.get('status')}"
        )
        return last_result

    def _check_exchange_position(self) -> int:
        """
        下單前查詢交易所實際持倉方向

        Returns:
            1=多, -1=空, 0=無倉位
        """
        try:
            pos = self.client.get_position_risk(self.symbol)
            if pos:
                amt = float(pos.get("positionAmt", "0"))
                if amt > 0:
                    return 1
                elif amt < 0:
                    return -1
            return 0
        except Exception as e:
            logger.warning(f"Pre-trade position check failed: {e}")
            return 0  # 查詢失敗時不阻擋交易

    def _estimate_fee(self, price: float, quantity: float) -> float:
        """估算手續費（taker 0.04%）"""
        return price * quantity * 0.0004

    def _verify_stop_order(self, sl_order_id: str,
                           max_retries: int = 2) -> bool:
        """驗證止損單是否存在於交易所"""
        if not sl_order_id:
            return False

        for attempt in range(max_retries + 1):
            try:
                order = self.client.get_order(self.symbol, int(sl_order_id))
                status = order.get("status", "")
                if status in ("NEW", "PARTIALLY_FILLED"):
                    return True
                logger.warning(f"SL order status unexpected: {status}")
                return False
            except BinanceClientError:
                if attempt < max_retries:
                    time.sleep(0.5)
                    continue
                return False
        return False

    def _emergency_close(self, side: int) -> None:
        """緊急平倉（止損單驗證失敗時）"""
        try:
            close_side = "SELL" if side == 1 else "BUY"
            # 取消所有標準掛單
            self.client.cancel_all_orders(self.symbol)
            # 取消所有 Algo 掛單
            try:
                algo_orders = self.client.get_algo_open_orders(self.symbol)
                if isinstance(algo_orders, list):
                    for ao in algo_orders:
                        aid = ao.get("algoId")
                        if aid:
                            self.client.cancel_algo_order(self.symbol, str(aid))
            except Exception:
                pass
            # 查詢實際持倉數量
            pos = self.client.get_position_risk(self.symbol)
            if pos:
                amt = abs(float(pos.get("positionAmt", "0")))
                if amt > 0:
                    self.client.place_market_order(self.symbol, close_side, amt)
                    logger.info("Emergency close executed")
        except Exception as e:
            logger.critical(f"EMERGENCY CLOSE FAILED: {e}")
