"""
Order Blocks 檢測模塊（向量化優化版）

實現 ICT Order Block 概念：
- 看漲 Order Block: 下降趨勢反轉前的最後一根看跌 K 線
- 看跌 Order Block: 上升趨勢反轉前的最後一根看漲 K 線
- Order Blocks 通常提供支撐/阻力區域

優化：
- 使用 NumPy 向量化操作
- 預計算整個數據集，查詢 O(1)

作者：PPO Trading Team
日期：2026-01-14
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple


class OrderBlockDetector:
    """Order Block 檢測器（向量化優化版）"""

    def __init__(self, lookback: int = 20, min_size_pct: float = 0.002):
        """
        初始化 Order Block 檢測器

        Args:
            lookback: 回看期數
            min_size_pct: OB 最小大小（百分比，默認 0.2%）
        """
        self.lookback = lookback
        self.min_size_pct = min_size_pct
        self.swing_window = 5

        # 預計算緩存
        self._cache_valid = False
        self._dist_bullish_cache = None   # [n] 距離看漲 OB
        self._dist_bearish_cache = None   # [n] 距離看跌 OB
        self._in_bullish_cache = None     # [n] 是否在看漲 OB 內
        self._in_bearish_cache = None     # [n] 是否在看跌 OB 內

    def precompute_all_features(self, df: pd.DataFrame) -> None:
        """
        向量化預計算整個數據集的 Order Block 特徵

        Args:
            df: OHLC 數據
        """
        n = len(df)
        highs = df['high'].to_numpy(dtype=np.float64)
        lows = df['low'].to_numpy(dtype=np.float64)
        opens = df['open'].to_numpy(dtype=np.float64)
        closes = df['close'].to_numpy(dtype=np.float64)

        # 初始化緩存
        self._dist_bullish_cache = np.full(n, 10.0, dtype=np.float32)
        self._dist_bearish_cache = np.full(n, 10.0, dtype=np.float32)
        self._in_bullish_cache = np.zeros(n, dtype=np.int8)
        self._in_bearish_cache = np.zeros(n, dtype=np.int8)

        # 1. 向量化識別 swing points
        swing_high_mask, swing_low_mask = self._vectorized_swing_points(highs, lows)

        # 2. 識別看跌/看漲 K 線
        bearish_candle = closes < opens  # 看跌 K 線
        bullish_candle = closes > opens  # 看漲 K 線

        # 3. 計算 K 線大小
        candle_size = (highs - lows) / np.where(lows > 0, lows, 1)
        valid_size = candle_size >= self.min_size_pct

        # 4. 找到所有 Order Blocks
        # 看漲 OB: swing low 之前的看跌 K 線
        # 看跌 OB: swing high 之前的看漲 K 線
        bullish_ob_highs = []  # [(idx, high, low), ...]
        bearish_ob_highs = []

        swing_low_indices = np.where(swing_low_mask)[0]
        swing_high_indices = np.where(swing_high_mask)[0]

        # 找看漲 OB（在 swing low 之前）
        for swing_idx in swing_low_indices:
            for i in range(swing_idx - 1, max(0, swing_idx - 10), -1):
                if bearish_candle[i] and valid_size[i]:
                    bullish_ob_highs.append((i, highs[i], lows[i]))
                    break

        # 找看跌 OB（在 swing high 之前）
        for swing_idx in swing_high_indices:
            for i in range(swing_idx - 1, max(0, swing_idx - 10), -1):
                if bullish_candle[i] and valid_size[i]:
                    bearish_ob_highs.append((i, highs[i], lows[i]))
                    break

        # 5. 為每個位置計算特徵
        for i in range(n):
            current_price = closes[i]

            # 找最近的有效看漲 OB
            nearest_bullish_dist = 10.0
            in_bullish = 0
            for ob_idx, ob_high, ob_low in bullish_ob_highs:
                if ob_idx >= i:
                    continue
                # 計算距離
                if current_price > ob_high:
                    dist = (current_price - ob_high) / current_price * 100
                elif current_price < ob_low:
                    dist = (ob_low - current_price) / current_price * 100
                else:
                    dist = 0.0
                    in_bullish = 1
                if dist < nearest_bullish_dist:
                    nearest_bullish_dist = dist
                    if ob_low <= current_price <= ob_high:
                        in_bullish = 1

            # 找最近的有效看跌 OB
            nearest_bearish_dist = 10.0
            in_bearish = 0
            for ob_idx, ob_high, ob_low in bearish_ob_highs:
                if ob_idx >= i:
                    continue
                if current_price > ob_high:
                    dist = (current_price - ob_high) / current_price * 100
                elif current_price < ob_low:
                    dist = (ob_low - current_price) / current_price * 100
                else:
                    dist = 0.0
                    in_bearish = 1
                if dist < nearest_bearish_dist:
                    nearest_bearish_dist = dist
                    if ob_low <= current_price <= ob_high:
                        in_bearish = 1

            self._dist_bullish_cache[i] = nearest_bullish_dist
            self._dist_bearish_cache[i] = nearest_bearish_dist
            self._in_bullish_cache[i] = in_bullish
            self._in_bearish_cache[i] = in_bearish

        self._cache_valid = True

    def _vectorized_swing_points(self, highs: np.ndarray, lows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

        return swing_high_mask, swing_low_mask

    def get_cached_features(self, current_idx: int) -> Dict[str, float]:
        """從緩存獲取特徵（O(1) 操作）"""
        if not self._cache_valid:
            raise RuntimeError("緩存未初始化，請先調用 precompute_all_features()")

        return {
            'dist_to_bullish_ob': float(self._dist_bullish_cache[current_idx]),
            'dist_to_bearish_ob': float(self._dist_bearish_cache[current_idx]),
            'in_bullish_ob': int(self._in_bullish_cache[current_idx]),
            'in_bearish_ob': int(self._in_bearish_cache[current_idx])
        }

    def calculate_features(self, df: pd.DataFrame, current_idx: int) -> dict:
        """計算當前位置的 Order Block 特徵"""
        if self._cache_valid:
            return self.get_cached_features(current_idx)

        # 回退到原始計算
        return self._calculate_features_original(df, current_idx)

    def _calculate_features_original(self, df: pd.DataFrame, current_idx: int) -> dict:
        """原始計算方法（保留用於兼容性）"""
        lookback_start = max(0, current_idx - self.lookback)
        df_lookback = df.iloc[lookback_start:current_idx+1]

        if len(df_lookback) < 20:
            return {
                'dist_to_bullish_ob': 10.0,
                'dist_to_bearish_ob': 10.0,
                'in_bullish_ob': 0,
                'in_bearish_ob': 0
            }

        highs = df_lookback['high'].to_numpy()
        lows = df_lookback['low'].to_numpy()
        opens = df_lookback['open'].to_numpy()
        closes = df_lookback['close'].to_numpy()

        # 識別 swing points
        swing_high_mask, swing_low_mask = self._vectorized_swing_points(highs, lows)

        # 找 OB
        bearish_candle = closes < opens
        bullish_candle = closes > opens
        candle_size = (highs - lows) / np.where(lows > 0, lows, 1)
        valid_size = candle_size >= self.min_size_pct

        current_price = closes[-1]
        result = {
            'dist_to_bullish_ob': 10.0,
            'dist_to_bearish_ob': 10.0,
            'in_bullish_ob': 0,
            'in_bearish_ob': 0
        }

        # 找看漲 OB
        swing_low_indices = np.where(swing_low_mask)[0]
        for swing_idx in swing_low_indices:
            for i in range(swing_idx - 1, max(0, swing_idx - 10), -1):
                if bearish_candle[i] and valid_size[i]:
                    ob_high, ob_low = highs[i], lows[i]
                    if ob_low <= current_price <= ob_high:
                        result['in_bullish_ob'] = 1
                        result['dist_to_bullish_ob'] = 0.0
                    elif current_price > ob_high:
                        dist = (current_price - ob_high) / current_price * 100
                        if dist < result['dist_to_bullish_ob']:
                            result['dist_to_bullish_ob'] = dist
                    else:
                        dist = (ob_low - current_price) / current_price * 100
                        if dist < result['dist_to_bullish_ob']:
                            result['dist_to_bullish_ob'] = dist
                    break

        # 找看跌 OB
        swing_high_indices = np.where(swing_high_mask)[0]
        for swing_idx in swing_high_indices:
            for i in range(swing_idx - 1, max(0, swing_idx - 10), -1):
                if bullish_candle[i] and valid_size[i]:
                    ob_high, ob_low = highs[i], lows[i]
                    if ob_low <= current_price <= ob_high:
                        result['in_bearish_ob'] = 1
                        result['dist_to_bearish_ob'] = 0.0
                    elif current_price > ob_high:
                        dist = (current_price - ob_high) / current_price * 100
                        if dist < result['dist_to_bearish_ob']:
                            result['dist_to_bearish_ob'] = dist
                    else:
                        dist = (ob_low - current_price) / current_price * 100
                        if dist < result['dist_to_bearish_ob']:
                            result['dist_to_bearish_ob'] = dist
                    break

        return result

    def analyze_full_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """分析整個數據集的 Order Blocks"""
        print(f"[OrderBlocks] Analyzing {len(df):,} bars...")
        self.precompute_all_features(df)

        result = df.copy()
        result['dist_to_bullish_ob'] = self._dist_bullish_cache
        result['dist_to_bearish_ob'] = self._dist_bearish_cache
        result['in_bullish_ob'] = self._in_bullish_cache
        result['in_bearish_ob'] = self._in_bearish_cache

        return result


def test_order_blocks():
    """測試 Order Blocks 模塊"""
    print("=" * 60)
    print("  Testing Order Blocks (Vectorized)")
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

    ob = OrderBlockDetector(lookback=20)

    start = time.time()
    ob.precompute_all_features(df_test)
    elapsed = time.time() - start
    print(f"   Precompute time: {elapsed:.3f}s")

    start = time.time()
    for i in range(len(df_test)):
        _ = ob.get_cached_features(i)
    elapsed = time.time() - start
    print(f"   Cache query time ({len(df_test)} queries): {elapsed:.3f}s")

    print("\n   OK: Vectorized OrderBlocks test passed!")


if __name__ == "__main__":
    test_order_blocks()
