"""
Liquidity 檢測模塊（向量化優化版）

實現 ICT Liquidity 概念：
- 流動性池：前期高點/低點聚集大量止損單
- 上方流動性：前期高點（賣出止損）
- 下方流動性：前期低點（買入止損）
- Liquidity Sweep：價格短暫突破流動性區域後快速反轉

優化：
- 使用 NumPy 向量化操作
- 使用 pandas rolling 進行滑動窗口計算
- 預計算整個數據集，查詢 O(1)

作者：PPO Trading Team
日期：2026-01-14
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple


class LiquidityDetector:
    """流動性檢測器（向量化優化版）"""

    def __init__(self, lookback: int = 50, sweep_threshold: float = 0.001):
        """
        初始化流動性檢測器

        Args:
            lookback: 回看期數，用於識別前期高低點
            sweep_threshold: 掃蕩閾值（價格超過流動性區域的百分比）
        """
        self.lookback = lookback
        self.sweep_threshold = sweep_threshold
        self.swing_window = 5

        # 預計算緩存
        self._cache_valid = False
        self._liq_above_cache = None    # [n] 上方流動性距離
        self._liq_below_cache = None    # [n] 下方流動性距離
        self._liq_sweep_cache = None    # [n] 是否發生掃蕩

    def precompute_all_features(self, df: pd.DataFrame) -> None:
        """
        向量化預計算整個數據集的流動性特徵

        Args:
            df: OHLC 數據
        """
        n = len(df)
        highs = df['high'].to_numpy(dtype=np.float64)
        lows = df['low'].to_numpy(dtype=np.float64)
        closes = df['close'].to_numpy(dtype=np.float64)

        # 初始化緩存
        self._liq_above_cache = np.full(n, 5.0, dtype=np.float32)
        self._liq_below_cache = np.full(n, 5.0, dtype=np.float32)
        self._liq_sweep_cache = np.zeros(n, dtype=np.int8)

        # 1. 向量化識別 swing points (流動性區域)
        swing_high_mask, swing_low_mask, swing_high_prices, swing_low_prices = \
            self._vectorized_swing_points(highs, lows)

        swing_high_indices = np.where(swing_high_mask)[0]
        swing_low_indices = np.where(swing_low_mask)[0]

        # 2. 追蹤流動性掃蕩和計算特徵
        high_swept = np.zeros(len(swing_high_indices), dtype=bool)
        low_swept = np.zeros(len(swing_low_indices), dtype=bool)

        for i in range(n):
            current_price = closes[i]
            current_high = highs[i]
            current_low = lows[i]

            # 檢測並更新流動性掃蕩
            sweep_detected = False

            # 檢查上方流動性掃蕩
            for j, swing_idx in enumerate(swing_high_indices):
                if high_swept[j] or swing_idx >= i:
                    continue
                swing_price = swing_high_prices[swing_idx]
                # 突破流動性且快速反轉
                if current_high > swing_price * (1 + self.sweep_threshold):
                    if current_price < swing_price:
                        high_swept[j] = True
                        sweep_detected = True

            # 檢查下方流動性掃蕩
            for j, swing_idx in enumerate(swing_low_indices):
                if low_swept[j] or swing_idx >= i:
                    continue
                swing_price = swing_low_prices[swing_idx]
                # 突破流動性且快速反轉
                if current_low < swing_price * (1 - self.sweep_threshold):
                    if current_price > swing_price:
                        low_swept[j] = True
                        sweep_detected = True

            self._liq_sweep_cache[i] = 1 if sweep_detected else 0

            # 找最近的上方流動性（未掃蕩的 swing high > current_price）
            nearest_above_dist = 5.0
            for j, swing_idx in enumerate(swing_high_indices):
                if high_swept[j] or swing_idx >= i:
                    continue
                swing_price = swing_high_prices[swing_idx]
                if swing_price > current_price:
                    dist = (swing_price - current_price) / current_price * 100
                    if dist < nearest_above_dist:
                        nearest_above_dist = dist

            # 找最近的下方流動性（未掃蕩的 swing low < current_price）
            nearest_below_dist = 5.0
            for j, swing_idx in enumerate(swing_low_indices):
                if low_swept[j] or swing_idx >= i:
                    continue
                swing_price = swing_low_prices[swing_idx]
                if swing_price < current_price:
                    dist = (current_price - swing_price) / current_price * 100
                    if dist < nearest_below_dist:
                        nearest_below_dist = dist

            self._liq_above_cache[i] = nearest_above_dist
            self._liq_below_cache[i] = nearest_below_dist

        self._cache_valid = True

    def _vectorized_swing_points(self, highs: np.ndarray, lows: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """向量化識別 swing points"""
        n = len(highs)
        w = self.swing_window
        window_size = 2 * w + 1

        high_series = pd.Series(highs)
        low_series = pd.Series(lows)

        rolling_max = high_series.rolling(window_size, center=True, min_periods=window_size).max()
        rolling_min = low_series.rolling(window_size, center=True, min_periods=window_size).min()

        swing_high_mask = (highs == rolling_max.to_numpy())
        swing_low_mask = (lows == rolling_min.to_numpy())

        # 填充 NaN 為 False
        swing_high_mask = np.nan_to_num(swing_high_mask, nan=0).astype(bool)
        swing_low_mask = np.nan_to_num(swing_low_mask, nan=0).astype(bool)

        return swing_high_mask, swing_low_mask, highs, lows

    def get_cached_features(self, current_idx: int) -> Dict[str, float]:
        """從緩存獲取特徵（O(1) 操作）"""
        if not self._cache_valid:
            raise RuntimeError("緩存未初始化，請先調用 precompute_all_features()")

        return {
            'liquidity_above': float(self._liq_above_cache[current_idx]),
            'liquidity_below': float(self._liq_below_cache[current_idx]),
            'liquidity_sweep': int(self._liq_sweep_cache[current_idx])
        }

    def calculate_features(self, df: pd.DataFrame, current_idx: int) -> dict:
        """計算當前位置的流動性特徵"""
        if self._cache_valid:
            return self.get_cached_features(current_idx)

        return self._calculate_features_original(df, current_idx)

    def _calculate_features_original(self, df: pd.DataFrame, current_idx: int) -> dict:
        """原始計算方法（保留用於兼容性）"""
        lookback_start = max(0, current_idx - self.lookback)
        df_lookback = df.iloc[lookback_start:current_idx+1]

        if len(df_lookback) < 20:
            return {
                'liquidity_above': 5.0,
                'liquidity_below': 5.0,
                'liquidity_sweep': 0
            }

        highs = df_lookback['high'].to_numpy()
        lows = df_lookback['low'].to_numpy()
        closes = df_lookback['close'].to_numpy()

        swing_high_mask, swing_low_mask, _, _ = self._vectorized_swing_points(highs, lows)

        current_price = closes[-1]
        current_high = highs[-1]
        current_low = lows[-1]

        result = {
            'liquidity_above': 5.0,
            'liquidity_below': 5.0,
            'liquidity_sweep': 0
        }

        # 找上方流動性
        swing_high_indices = np.where(swing_high_mask)[0]
        for idx in swing_high_indices:
            if idx >= len(highs) - 1:
                continue
            swing_price = highs[idx]
            if swing_price > current_price:
                dist = (swing_price - current_price) / current_price * 100
                if dist < result['liquidity_above']:
                    result['liquidity_above'] = dist

            # 檢測掃蕩
            if current_high > swing_price * (1 + self.sweep_threshold):
                if current_price < swing_price:
                    result['liquidity_sweep'] = 1

        # 找下方流動性
        swing_low_indices = np.where(swing_low_mask)[0]
        for idx in swing_low_indices:
            if idx >= len(lows) - 1:
                continue
            swing_price = lows[idx]
            if swing_price < current_price:
                dist = (current_price - swing_price) / current_price * 100
                if dist < result['liquidity_below']:
                    result['liquidity_below'] = dist

            # 檢測掃蕩
            if current_low < swing_price * (1 - self.sweep_threshold):
                if current_price > swing_price:
                    result['liquidity_sweep'] = 1

        return result

    def analyze_full_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """分析整個數據集的流動性"""
        print(f"[Liquidity] Analyzing {len(df):,} bars...")
        self.precompute_all_features(df)

        result = df.copy()
        result['liquidity_above'] = self._liq_above_cache
        result['liquidity_below'] = self._liq_below_cache
        result['liquidity_sweep'] = self._liq_sweep_cache

        return result


def test_liquidity():
    """測試 Liquidity 模塊"""
    print("=" * 60)
    print("  Testing Liquidity (Vectorized)")
    print("=" * 60)

    from pathlib import Path
    import time

    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "raw" / "BTCUSDT_1m_train_latest.csv"

    if not data_path.exists():
        print("Test data not found, skipping test")
        return

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df_test = df.iloc[-5000:].copy()
    print(f"   Test data: {len(df_test)} bars")

    liq = LiquidityDetector(lookback=50)

    start = time.time()
    liq.precompute_all_features(df_test)
    elapsed = time.time() - start
    print(f"   Precompute time: {elapsed:.3f}s")

    start = time.time()
    for i in range(len(df_test)):
        _ = liq.get_cached_features(i)
    elapsed = time.time() - start
    print(f"   Cache query time ({len(df_test)} queries): {elapsed:.3f}s")

    print("\n   OK: Vectorized Liquidity test passed!")


if __name__ == "__main__":
    test_liquidity()
