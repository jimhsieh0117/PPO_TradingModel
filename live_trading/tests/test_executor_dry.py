"""
下單邏輯乾跑測試 + 風控模組測試

不連接真實交易所，使用 mock 驗證：
1. Executor: 數量計算、止損計算、action 對應
2. RiskManager: 各層風控邏輯、斷路器、Kill Switch
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pytest

from live_trading.executor import Executor, ACTION_CLOSE, ACTION_LONG, ACTION_SHORT, ACTION_HOLD
from live_trading.risk_manager import RiskManager
from live_trading.state import TradingState


def _make_config() -> dict:
    """測試用配置"""
    return {
        "trading": {
            "symbol": "ETHUSDT",
            "leverage": 1,
            "position_size_pct": 0.20,
            "taker_fee": 0.0004,
            "max_holding_steps": 120,
        },
        "risk": {
            "atr_stop_multiplier": 2.0,
            "trailing_stop": False,
            "stop_loss_pct": 0.02,
            "max_daily_loss_pct": 0.10,
            "max_total_loss_pct": 0.30,
            "max_consecutive_losses": 10,
            "max_open_positions": 1,
            "max_order_value_usdt": 50.0,
            "min_balance_to_trade": 10.0,
            "max_slippage_pct": 0.003,
            "circuit_breaker_threshold": 3,
            "circuit_breaker_cooldown": 60,
            "circuit_breaker_max_triggers": 3,
        },
    }


def _make_mock_client():
    """模擬 BinanceFuturesClient"""
    client = MagicMock()
    # 開倉用 LIMIT IOC
    client.place_limit_ioc.return_value = {
        "status": "FILLED",
        "avgPrice": "2450.00",
        "executedQty": "0.016",
        "orderId": "123456",
    }
    # 平倉 IOC 也預設成功
    # 緊急平倉 fallback 用市價單
    client.place_market_order.return_value = {
        "status": "FILLED",
        "avgPrice": "2450.00",
        "executedQty": "0.016",
        "orderId": "123457",
    }
    client.place_stop_market.return_value = {
        "orderId": "789012",
    }
    client.get_order.return_value = {
        "status": "NEW",
    }
    client.get_ticker_price.return_value = {
        "price": "2450.00",
    }
    client.cancel_order.return_value = {}
    return client


# ================================================================
# Executor 測試
# ================================================================

class TestExecutorQuantity:
    """數量計算測試"""

    def test_basic_quantity(self):
        """基本數量計算"""
        config = _make_config()
        client = _make_mock_client()
        executor = Executor(client, config)

        # balance=200, size_pct=0.20, price=2500
        # qty = 200 * 0.20 / 2500 = 0.016
        qty = executor._calculate_quantity(200.0, 2500.0)
        assert qty is not None
        assert qty == pytest.approx(0.016, abs=0.001)

    def test_quantity_respects_max_order_value(self):
        """數量不超過 max_order_value"""
        config = _make_config()
        config["risk"]["max_order_value_usdt"] = 30.0
        client = _make_mock_client()
        executor = Executor(client, config)

        # balance=200, size_pct=0.20, price=2500
        # raw notional = 200*0.20 = 40U > 30U limit
        qty = executor._calculate_quantity(200.0, 2500.0)
        assert qty is not None
        notional = qty * 2500.0
        assert notional <= 30.0 + 0.01

    def test_quantity_too_small_returns_none(self):
        """餘額太小 → 數量低於 minQty → 回傳 None"""
        config = _make_config()
        client = _make_mock_client()
        executor = Executor(client, config)
        executor._min_qty = 0.01

        # balance=5, price=2500 → qty = 5*0.20/2500 = 0.0004 < 0.01
        qty = executor._calculate_quantity(5.0, 2500.0)
        assert qty is None

    def test_min_notional_check(self):
        """notional < minNotional → 回傳 None"""
        config = _make_config()
        client = _make_mock_client()
        executor = Executor(client, config)
        executor._min_notional = 100.0  # 提高門檻

        # balance=200, price=2500 → notional = 200*0.20 = 40 < 100
        qty = executor._calculate_quantity(200.0, 2500.0)
        assert qty is None


class TestExecutorStopLoss:
    """止損計算測試"""

    def test_long_stop_loss(self):
        """多單止損：entry - min(atr*mult, entry*pct)"""
        config = _make_config()
        client = _make_mock_client()
        executor = Executor(client, config)

        # entry=2500, atr=20, mult=2.0 → atr_dist=40
        # pct_dist = 2500 * 0.02 = 50
        # min(40, 50) = 40
        # sl = 2500 - 40 = 2460
        sl = executor._calculate_stop_loss(side=1, entry_price=2500.0, atr=20.0)
        assert sl == pytest.approx(2460.0, abs=0.01)

    def test_short_stop_loss(self):
        """空單止損：entry + min(atr*mult, entry*pct)"""
        config = _make_config()
        client = _make_mock_client()
        executor = Executor(client, config)

        sl = executor._calculate_stop_loss(side=-1, entry_price=2500.0, atr=20.0)
        assert sl == pytest.approx(2540.0, abs=0.01)

    def test_pct_fallback_when_atr_larger(self):
        """ATR 距離 > pct 距離時，使用 pct（更保守）"""
        config = _make_config()
        client = _make_mock_client()
        executor = Executor(client, config)

        # atr=50, mult=2.0 → atr_dist=100
        # pct_dist = 2500 * 0.02 = 50
        # min(100, 50) = 50
        sl = executor._calculate_stop_loss(side=1, entry_price=2500.0, atr=50.0)
        assert sl == pytest.approx(2450.0, abs=0.01)


class TestExecutorActions:
    """Action 對應測試"""

    def test_hold_returns_none(self):
        """HOLD 不操作"""
        config = _make_config()
        client = _make_mock_client()
        executor = Executor(client, config)
        state = TradingState(initial_balance=200.0)

        result = executor.execute(ACTION_HOLD, state, atr=20.0, current_price=2500.0)
        assert result is None

    def test_close_no_position_returns_none(self):
        """無持倉時 CLOSE 不操作"""
        config = _make_config()
        client = _make_mock_client()
        executor = Executor(client, config)
        state = TradingState(initial_balance=200.0)

        result = executor.execute(ACTION_CLOSE, state, atr=20.0, current_price=2500.0)
        assert result is None

    def test_long_when_already_long_returns_none(self):
        """已有多倉時再做多 → 不操作"""
        config = _make_config()
        client = _make_mock_client()
        executor = Executor(client, config)
        state = TradingState(initial_balance=200.0)
        state.open_position(side=1, entry_price=2450.0, quantity=0.04, sl_price=2401.0)

        result = executor.execute(ACTION_LONG, state, atr=20.0, current_price=2500.0)
        assert result is None

    def test_open_long_calls_api(self):
        """開多倉呼叫正確的 API"""
        config = _make_config()
        client = _make_mock_client()
        executor = Executor(client, config)
        state = TradingState(initial_balance=200.0)

        result = executor.execute(ACTION_LONG, state, atr=20.0, current_price=2500.0)

        assert result is not None
        assert result["side"] == "BUY"
        client.place_limit_ioc.assert_called_once()
        client.place_stop_market.assert_called_once()

    def test_open_short_calls_api(self):
        """開空倉呼叫正確的 API"""
        config = _make_config()
        client = _make_mock_client()
        executor = Executor(client, config)
        state = TradingState(initial_balance=200.0)

        result = executor.execute(ACTION_SHORT, state, atr=20.0, current_price=2500.0)

        assert result is not None
        assert result["side"] == "SELL"

    def test_close_position_cancels_sl(self):
        """平倉時取消止損單"""
        config = _make_config()
        client = _make_mock_client()
        executor = Executor(client, config)
        state = TradingState(initial_balance=200.0)
        state.open_position(side=1, entry_price=2450.0, quantity=0.016,
                           sl_price=2401.0, sl_order_id="789")

        result = executor.execute(ACTION_CLOSE, state, atr=20.0, current_price=2470.0)

        assert result is not None
        client.cancel_order.assert_called_once_with("ETHUSDT", 789)

    def test_reverse_position_closes_first(self):
        """反向開倉先平再開，回傳 [平倉結果, 開倉結果]"""
        config = _make_config()
        client = _make_mock_client()
        executor = Executor(client, config)
        state = TradingState(initial_balance=200.0)
        state.open_position(side=1, entry_price=2450.0, quantity=0.016,
                           sl_price=2401.0, sl_order_id="789")

        result = executor.execute(ACTION_SHORT, state, atr=20.0, current_price=2500.0)

        # 回傳 list：[平倉, 開倉]
        assert isinstance(result, list)
        assert len(result) == 2
        assert "exit_price" in result[0]  # 平倉結果
        assert "entry_price" in result[1]  # 開倉結果
        # 平倉用 limit_ioc，開倉也用 limit_ioc
        assert client.place_limit_ioc.call_count == 2

    def test_ioc_expired_returns_none(self):
        """IOC 單 EXPIRED（滑點超限）→ 不開倉"""
        config = _make_config()
        client = _make_mock_client()
        client.place_limit_ioc.return_value = {
            "status": "EXPIRED",
            "avgPrice": "0",
            "executedQty": "0",
            "orderId": "999",
        }
        executor = Executor(client, config)
        state = TradingState(initial_balance=200.0)

        result = executor.execute(ACTION_LONG, state, atr=20.0, current_price=2500.0)
        assert result is None


# ================================================================
# RiskManager 測試
# ================================================================

class TestRiskManagerLayers:
    """風控各層測試"""

    def test_hold_always_allowed(self):
        """HOLD 永遠通過"""
        config = _make_config()
        rm = RiskManager(config)
        state = TradingState(initial_balance=200.0)
        state.balance = 0  # 即使餘額為 0

        allowed, reason = rm.check(ACTION_HOLD, state)
        assert allowed is True

    def test_balance_below_minimum(self):
        """餘額不足 → 拒絕開倉"""
        config = _make_config()
        rm = RiskManager(config)
        state = TradingState(initial_balance=200.0)
        state.balance = 5.0  # < 10U

        allowed, reason = rm.check(ACTION_LONG, state)
        assert allowed is False
        assert "balance" in reason

    def test_daily_loss_exceeded(self):
        """每日虧損超限 → 拒絕開倉"""
        config = _make_config()
        rm = RiskManager(config)
        state = TradingState(initial_balance=200.0)
        state.daily_pnl = -25.0  # -12.5% > 10%

        allowed, reason = rm.check(ACTION_LONG, state)
        assert allowed is False
        assert "daily loss" in reason

    def test_total_loss_exceeded(self):
        """總虧損超限 → 拒絕開倉"""
        config = _make_config()
        rm = RiskManager(config)
        state = TradingState(initial_balance=200.0)
        state.total_pnl = -70.0  # -35% > 30%

        allowed, reason = rm.check(ACTION_LONG, state)
        assert allowed is False
        assert "total loss" in reason

    def test_consecutive_losses_pause(self):
        """連續虧損 → 暫停 1 小時"""
        config = _make_config()
        rm = RiskManager(config)
        state = TradingState(initial_balance=200.0)
        state.consecutive_losses = 10  # = max

        allowed, reason = rm.check(ACTION_LONG, state)
        assert allowed is False
        assert "consecutive" in reason.lower()

    def test_close_allowed_despite_losses(self):
        """即使超過虧損限制，平倉仍允許"""
        config = _make_config()
        rm = RiskManager(config)
        state = TradingState(initial_balance=200.0)
        state.daily_pnl = -25.0
        state.position = 1

        allowed, reason = rm.check(ACTION_CLOSE, state)
        assert allowed is True

    def test_duplicate_direction_blocked(self):
        """已有多倉不能再開多"""
        config = _make_config()
        rm = RiskManager(config)
        state = TradingState(initial_balance=200.0)
        state.position = 1

        allowed, reason = rm.check(ACTION_LONG, state)
        assert allowed is False
        assert "same direction" in reason

    def test_reverse_direction_allowed(self):
        """有多倉可以開空（反向）"""
        config = _make_config()
        rm = RiskManager(config)
        state = TradingState(initial_balance=200.0)
        state.position = 1

        allowed, reason = rm.check(ACTION_SHORT, state)
        assert allowed is True

    def test_order_value_exceeded(self):
        """下單金額超限"""
        config = _make_config()
        rm = RiskManager(config)
        state = TradingState(initial_balance=200.0)

        allowed, reason = rm.check(ACTION_LONG, state, estimated_notional=60.0)
        assert allowed is False
        assert "notional" in reason.lower()


class TestCircuitBreaker:
    """API 斷路器測試"""

    def test_triggers_after_threshold(self):
        """連續 N 次錯誤觸發冷卻"""
        config = _make_config()
        rm = RiskManager(config)

        for _ in range(3):
            rm.record_api_error(429)

        assert rm._is_in_cooldown()

    def test_success_resets_counter(self):
        """成功呼叫重置錯誤計數"""
        config = _make_config()
        rm = RiskManager(config)

        rm.record_api_error(429)
        rm.record_api_error(500)
        rm.record_api_success()

        assert rm._consecutive_api_errors == 0

    def test_standby_after_max_triggers(self):
        """連續觸發 N 次冷卻 → 待機模式"""
        config = _make_config()
        rm = RiskManager(config)
        rm._cb_cooldown = 0.01  # 縮短冷卻時間

        for _ in range(3):
            for _ in range(3):
                rm.record_api_error(429)

        assert rm._standby_mode

    def test_standby_blocks_all(self):
        """待機模式拒絕所有操作"""
        config = _make_config()
        rm = RiskManager(config)
        rm._standby_mode = True
        state = TradingState(initial_balance=200.0)

        allowed, reason = rm.check(ACTION_LONG, state)
        assert allowed is False
        assert "STANDBY" in reason


class TestKillSwitch:
    """Kill Switch 測試"""

    def test_kill_switch_blocks(self, tmp_path):
        """STOP 檔案存在 → 拒絕交易"""
        stop_file = tmp_path / "STOP"
        stop_file.touch()

        config = _make_config()
        rm = RiskManager(config, kill_switch_path=str(stop_file))
        state = TradingState(initial_balance=200.0)

        allowed, reason = rm.check(ACTION_LONG, state)
        assert allowed is False
        assert "KILL_SWITCH" in reason

    def test_no_kill_switch_allows(self, tmp_path):
        """STOP 檔案不存在 → 正常通過"""
        stop_file = tmp_path / "STOP"  # 不建立

        config = _make_config()
        rm = RiskManager(config, kill_switch_path=str(stop_file))
        state = TradingState(initial_balance=200.0)

        allowed, reason = rm.check(ACTION_LONG, state)
        assert allowed is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
