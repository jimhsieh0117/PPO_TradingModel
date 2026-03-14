"""
狀態機轉換測試

驗證 TradingState 的：
1. 持倉狀態轉換（無倉→多/空→平倉）
2. 觀察向量組合（33 維）
3. equity_change_pct 滾動窗口計算
4. 風控統計追蹤
5. 快照保存/恢復
"""

import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pytest

from live_trading.state import TradingState


class TestStateTransitions:
    """持倉狀態轉換測試"""

    def test_initial_state(self):
        """初始狀態為無持倉"""
        state = TradingState(initial_balance=200.0)
        assert state.position == 0
        assert state.entry_price == 0.0
        assert state.holding_steps == 0
        assert not state.has_position
        assert state.balance == 200.0
        assert state.equity == 200.0

    def test_open_long(self):
        """開多倉"""
        state = TradingState(initial_balance=200.0)
        state.open_position(side=1, entry_price=2450.0, quantity=0.04,
                           sl_price=2401.0, sl_order_id="123")
        assert state.position == 1
        assert state.entry_price == 2450.0
        assert state.quantity == 0.04
        assert state.current_sl == 2401.0
        assert state.sl_order_id == "123"
        assert state.has_position
        assert state.holding_steps == 0

    def test_open_short(self):
        """開空倉"""
        state = TradingState(initial_balance=200.0)
        state.open_position(side=-1, entry_price=2450.0, quantity=0.04,
                           sl_price=2499.0)
        assert state.position == -1
        assert state.current_sl == 2499.0

    def test_close_long_profit(self):
        """平多倉（盈利）"""
        state = TradingState(initial_balance=200.0)
        state.open_position(side=1, entry_price=2450.0, quantity=0.04, sl_price=2401.0)

        # 模擬持倉 10 步
        for _ in range(10):
            state.step(current_equity=201.0)
        assert state.holding_steps == 10

        record = state.close_position(exit_price=2478.0, pnl=1.12, reason="model")
        assert state.position == 0
        assert state.balance == pytest.approx(201.12, abs=0.01)
        assert state.consecutive_losses == 0
        assert state.trade_count == 1
        assert record["holding_steps"] == 10
        assert record["pnl"] == 1.12

    def test_close_long_loss(self):
        """平多倉（虧損）→ 連續虧損計數"""
        state = TradingState(initial_balance=200.0)

        # 連續 3 筆虧損
        for i in range(3):
            state.open_position(side=1, entry_price=2450.0, quantity=0.04, sl_price=2401.0)
            state.step()
            state.close_position(exit_price=2440.0, pnl=-0.50, reason="stop_loss")

        assert state.consecutive_losses == 3
        assert state.trade_count == 3
        assert state.daily_pnl == pytest.approx(-1.50, abs=0.01)

    def test_consecutive_losses_reset_on_profit(self):
        """盈利交易重置連續虧損計數"""
        state = TradingState(initial_balance=200.0)

        # 2 筆虧損
        for _ in range(2):
            state.open_position(side=1, entry_price=2450.0, quantity=0.04, sl_price=2401.0)
            state.close_position(exit_price=2440.0, pnl=-0.5)
        assert state.consecutive_losses == 2

        # 1 筆盈利 → 重置
        state.open_position(side=1, entry_price=2450.0, quantity=0.04, sl_price=2401.0)
        state.close_position(exit_price=2460.0, pnl=0.5)
        assert state.consecutive_losses == 0

    def test_holding_steps_increment(self):
        """持倉步數每 step 遞增"""
        state = TradingState(initial_balance=200.0)
        state.open_position(side=1, entry_price=2450.0, quantity=0.04, sl_price=2401.0)

        for i in range(5):
            state.step()
            assert state.holding_steps == i + 1

    def test_max_holding_detection(self):
        """最大持倉步數檢測"""
        state = TradingState(initial_balance=200.0, max_holding_steps=120)
        state.open_position(side=1, entry_price=2450.0, quantity=0.04, sl_price=2401.0)

        for _ in range(119):
            state.step()
        assert not state.is_max_holding

        state.step()  # 第 120 步
        assert state.is_max_holding

    def test_update_stop_loss(self):
        """止損價更新"""
        state = TradingState(initial_balance=200.0)
        state.open_position(side=1, entry_price=2450.0, quantity=0.04,
                           sl_price=2401.0, sl_order_id="old_id")

        state.update_stop_loss(2420.0, "new_id")
        assert state.current_sl == 2420.0
        assert state.sl_order_id == "new_id"


class TestObservationVector:
    """觀察向量測試"""

    def test_observation_shape(self):
        """觀察向量形狀 = [33]"""
        state = TradingState(initial_balance=200.0)
        market_features = np.zeros(28, dtype=np.float32)
        obs = state.build_observation(market_features, current_price=2450.0)
        assert obs.shape == (33,)
        assert obs.dtype == np.float32

    def test_no_position_features(self):
        """無持倉時，5 維持倉特徵全為 0"""
        state = TradingState(initial_balance=200.0)
        market_features = np.ones(28, dtype=np.float32)
        obs = state.build_observation(market_features, current_price=2450.0)

        position_features = obs[28:]
        assert position_features[0] == 0.0  # position_state
        assert position_features[1] == 0.0  # floating_pnl_pct
        assert position_features[2] == 0.0  # holding_time_norm
        assert position_features[3] == 0.0  # dist_to_sl
        assert position_features[4] == 0.0  # equity_change_pct

    def test_long_position_features(self):
        """多倉持倉特徵"""
        state = TradingState(initial_balance=200.0)
        state.open_position(side=1, entry_price=2450.0, quantity=0.04,
                           sl_price=2401.0)

        # 持倉 60 步
        for _ in range(60):
            state.step(current_equity=201.0)

        market_features = np.zeros(28, dtype=np.float32)
        obs = state.build_observation(market_features, current_price=2460.0)

        pf = obs[28:]
        assert pf[0] == 1.0                                    # position_state = LONG
        assert pf[1] == pytest.approx(10.0 / 2450.0, abs=1e-5) # floating_pnl_pct
        assert pf[2] == pytest.approx(60.0 / 120.0, abs=1e-5)  # holding_time_norm = 0.5
        assert pf[3] > 0.0                                     # dist_to_sl > 0
        # equity_change_pct: (201 - 200) / 200 = 0.005
        assert pf[4] == pytest.approx(0.005, abs=1e-3)

    def test_short_position_features(self):
        """空倉持倉特徵"""
        state = TradingState(initial_balance=200.0)
        state.open_position(side=-1, entry_price=2450.0, quantity=0.04,
                           sl_price=2499.0)
        state.step(current_equity=200.0)

        market_features = np.zeros(28, dtype=np.float32)
        obs = state.build_observation(market_features, current_price=2440.0)

        pf = obs[28:]
        assert pf[0] == -1.0  # position_state = SHORT
        # 空倉盈虧: (entry - current) / entry = (2450-2440)/2450
        assert pf[1] == pytest.approx(10.0 / 2450.0, abs=1e-5)

    def test_holding_time_saturates(self):
        """持倉時間在 120 步飽和"""
        state = TradingState(initial_balance=200.0)
        state.open_position(side=1, entry_price=2450.0, quantity=0.04, sl_price=2401.0)

        for _ in range(200):
            state.step()

        pf = state._get_position_features(2450.0)
        assert pf[2] == pytest.approx(1.0)  # 飽和為 1.0


class TestEquityChangeRollingWindow:
    """equity_change_pct 滾動窗口測試"""

    def test_initial_equity_change_is_zero(self):
        """初始時 equity_change_pct = 0"""
        state = TradingState(initial_balance=200.0, episode_length=720)
        assert state._equity_change_pct() == 0.0

    def test_equity_change_uses_initial_balance_before_full(self):
        """窗口未滿時使用 initial_balance 作為 baseline"""
        state = TradingState(initial_balance=200.0, episode_length=720)

        # 前 100 步
        for _ in range(100):
            state.step(current_equity=202.0)

        # baseline = initial_balance = 200
        # change = (202 - 200) / 200 = 0.01
        assert state._equity_change_pct() == pytest.approx(0.01, abs=1e-5)

    def test_equity_change_uses_rolling_window_when_full(self):
        """窗口滿後使用 deque[0] 作為 baseline"""
        episode_length = 10  # 用小窗口方便測試
        state = TradingState(initial_balance=100.0, episode_length=episode_length)

        # 先填滿窗口（equity 從 100 漲到 110）
        for i in range(episode_length):
            state.step(current_equity=100.0 + i)

        # 窗口已滿，deque = [100, 101, ..., 109]
        # baseline = deque[0] = 100
        # equity = 109
        assert state._equity_change_pct() == pytest.approx(
            (109.0 - 100.0) / 100.0, abs=1e-5
        )

        # 繼續步進，deque 自動淘汰最舊值
        state.step(current_equity=115.0)
        # deque = [101, 102, ..., 109, 115]
        # baseline = 101
        # change = (115 - 101) / 101
        assert state._equity_change_pct() == pytest.approx(
            (115.0 - 101.0) / 101.0, abs=1e-5
        )

    def test_equity_change_clipped(self):
        """equity_change_pct 被 clip 到 [-1, 1]"""
        state = TradingState(initial_balance=100.0, episode_length=10)
        state.step(current_equity=300.0)  # +200%
        assert state._equity_change_pct() == 1.0  # clipped

        state2 = TradingState(initial_balance=100.0, episode_length=10)
        state2.step(current_equity=1.0)  # -99%
        assert state2._equity_change_pct() == pytest.approx(-0.99, abs=0.01)


class TestSnapshot:
    """快照保存/恢復測試"""

    def test_snapshot_round_trip(self):
        """快照可正確保存和恢復"""
        state = TradingState(initial_balance=200.0)
        state.daily_pnl = -5.0
        state.total_pnl = -12.0
        state.consecutive_losses = 4
        state.trade_count = 15

        snapshot = state.to_snapshot()

        # 新的 state 恢復
        new_state = TradingState(initial_balance=200.0)
        new_state.restore_from_snapshot(snapshot)

        assert new_state.consecutive_losses == 4
        assert new_state.daily_pnl == -5.0
        assert new_state.total_pnl == -12.0
        assert new_state.trade_count == 15


class TestExchangeSync:
    """交易所同步測試"""

    def test_sync_detects_desync(self):
        """同步時偵測狀態不一致"""
        state = TradingState(initial_balance=200.0)
        # 本地認為無持倉
        assert state.position == 0

        # 交易所實際有多單
        exchange_data = {
            "positionAmt": "0.04",
            "entryPrice": "2450.0",
            "unRealizedProfit": "1.5",
        }
        state.sync_from_exchange(exchange_data, balance=198.5)

        # 應以交易所為準
        assert state.position == 1
        assert state.entry_price == 2450.0
        assert state.quantity == 0.04
        assert state.equity == pytest.approx(200.0, abs=0.01)

    def test_sync_no_position(self):
        """同步空倉狀態"""
        state = TradingState(initial_balance=200.0)
        state.open_position(side=1, entry_price=2450.0, quantity=0.04, sl_price=2401.0)

        exchange_data = {
            "positionAmt": "0",
            "entryPrice": "0",
            "unRealizedProfit": "0",
        }
        state.sync_from_exchange(exchange_data, balance=201.0)

        assert state.position == 0
        assert state.quantity == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
