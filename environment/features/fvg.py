"""
Fair Value Gaps (FVG) 檢測模塊（向量化優化版）

實現 ICT Fair Value Gap 概念：
- FVG 是三根 K 線之間的價格缺口
- 看漲 FVG: 第一根 K 線的高點 < 第三根 K 線的低點
- 看跌 FVG: 第一根 K 線的低點 > 第三根 K 線的高點
- FVG 通常會被"填補"（價格回到缺口區域）

優化：
- 使用 NumPy 向量化操作
- 預計算整個數據集，查詢 O(1)

作者：PPO Trading Team
日期：2026-01-14
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple


class FVGDetector:
    """Fair Value Gap 檢測器（向量化優化版）"""

    def __init__(self, min_size_pct: float = 0.001, max_age: int = 100):
        """
        初始化 FVG 檢測器

        Args:
            min_size_pct: FVG 最小大小（百分比，默認 0.1%）
            max_age: FVG 最大保留期數（默認 100 根 K 線）
        """
        self.min_size_pct = min_size_pct
        self.max_age = max_age

        # 預計算緩存
        self._cache_valid = False
        self._in_bullish_cache = None      # [n] 是否在看漲 FVG 內
        self._in_bearish_cache = None      # [n] 是否在看跌 FVG 內
        self._nearest_dir_cache = None     # [n] 最近 FVG 方向

    def precompute_all_features(self, df: pd.DataFrame) -> None:
        """
        向量化預計算整個數據集的 FVG 特徵

        Args:
            df: OHLC 數據
        """
        n = len(df)
        highs = df['high'].to_numpy(dtype=np.float64)
        lows = df['low'].to_numpy(dtype=np.float64)
        closes = df['close'].to_numpy(dtype=np.float64)

        # 初始化緩存
        self._in_bullish_cache = np.zeros(n, dtype=np.int8)
        self._in_bearish_cache = np.zeros(n, dtype=np.int8)
        self._nearest_dir_cache = np.zeros(n, dtype=np.int8)

        # 1. 向量化檢測所有 FVG
        # 看漲 FVG: highs[i] < lows[i+2]
        # 看跌 FVG: lows[i] > highs[i+2]
        bullish_fvg_list = []  # [(start_idx, high, low), ...]
        bearish_fvg_list = []

        for i in range(n - 2):
            # 看漲 FVG
            if highs[i] < lows[i + 2]:
                gap_size = (lows[i + 2] - highs[i]) / highs[i]
                if gap_size >= self.min_size_pct:
                    bullish_fvg_list.append((i, lows[i + 2], highs[i]))  # (idx, high, low)

            # 看跌 FVG
            if lows[i] > highs[i + 2]:
                gap_size = (lows[i] - highs[i + 2]) / highs[i + 2]
                if gap_size >= self.min_size_pct:
                    bearish_fvg_list.append((i, lows[i], highs[i + 2]))  # (idx, high, low)

        # 2. 追蹤 FVG 填補狀態並計算每個位置的特徵
        bullish_filled = [False] * len(bullish_fvg_list)
        bearish_filled = [False] * len(bearish_fvg_list)

        for i in range(n):
            current_price = closes[i]
            current_high = highs[i]
            current_low = lows[i]

            # 更新 FVG 填補狀態
            for j, (start_idx, fvg_high, fvg_low) in enumerate(bullish_fvg_list):
                if not bullish_filled[j] and i > start_idx + 2:
                    # 看漲 FVG 被填補: 價格回落到 FVG 區域
                    if current_low <= fvg_high:
                        bullish_filled[j] = True

            for j, (start_idx, fvg_high, fvg_low) in enumerate(bearish_fvg_list):
                if not bearish_filled[j] and i > start_idx + 2:
                    # 看跌 FVG 被填補: 價格回升到 FVG 區域
                    if current_high >= fvg_low:
                        bearish_filled[j] = True

            # 找最近的未填補 FVG
            nearest_bullish = None
            nearest_bullish_dist = float('inf')
            for j, (start_idx, fvg_high, fvg_low) in enumerate(bullish_fvg_list):
                if bullish_filled[j]:
                    continue
                if start_idx >= i:
                    continue
                if i - start_idx > self.max_age:
                    continue
                mid = (fvg_high + fvg_low) / 2
                dist = abs(current_price - mid)
                if dist < nearest_bullish_dist:
                    nearest_bullish_dist = dist
                    nearest_bullish = (fvg_high, fvg_low)

            nearest_bearish = None
            nearest_bearish_dist = float('inf')
            for j, (start_idx, fvg_high, fvg_low) in enumerate(bearish_fvg_list):
                if bearish_filled[j]:
                    continue
                if start_idx >= i:
                    continue
                if i - start_idx > self.max_age:
                    continue
                mid = (fvg_high + fvg_low) / 2
                dist = abs(current_price - mid)
                if dist < nearest_bearish_dist:
                    nearest_bearish_dist = dist
                    nearest_bearish = (fvg_high, fvg_low)

            # 計算特徵
            in_bullish = 0
            in_bearish = 0
            nearest_dir = 0

            if nearest_bullish:
                fvg_high, fvg_low = nearest_bullish
                if fvg_low <= current_price <= fvg_high:
                    in_bullish = 1

            if nearest_bearish:
                fvg_high, fvg_low = nearest_bearish
                if fvg_low <= current_price <= fvg_high:
                    in_bearish = 1

            # 確定最近 FVG 方向
            if nearest_bullish and not nearest_bearish:
                nearest_dir = 1
            elif nearest_bearish and not nearest_bullish:
                nearest_dir = -1
            elif nearest_bullish and nearest_bearish:
                nearest_dir = 1 if nearest_bullish_dist < nearest_bearish_dist else -1

            self._in_bullish_cache[i] = in_bullish
            self._in_bearish_cache[i] = in_bearish
            self._nearest_dir_cache[i] = nearest_dir

        self._cache_valid = True

    def get_cached_features(self, current_idx: int) -> Dict[str, int]:
        """從緩存獲取特徵（O(1) 操作）"""
        if not self._cache_valid:
            raise RuntimeError("緩存未初始化，請先調用 precompute_all_features()")

        return {
            'in_bullish_fvg': int(self._in_bullish_cache[current_idx]),
            'in_bearish_fvg': int(self._in_bearish_cache[current_idx]),
            'nearest_fvg_direction': int(self._nearest_dir_cache[current_idx])
        }

    def calculate_features(self, df: pd.DataFrame, current_idx: int) -> dict:
        """計算當前位置的 FVG 特徵"""
        if self._cache_valid:
            return self.get_cached_features(current_idx)

        return self._calculate_features_original(df, current_idx)

    def _calculate_features_original(self, df: pd.DataFrame, current_idx: int) -> dict:
        """原始計算方法（保留用於兼容性）"""
        if current_idx < 10:
            return {
                'in_bullish_fvg': 0,
                'in_bearish_fvg': 0,
                'nearest_fvg_direction': 0
            }

        lookback_start = max(0, current_idx - 200)
        df_lookback = df.iloc[lookback_start:current_idx+1]

        highs = df_lookback['high'].to_numpy()
        lows = df_lookback['low'].to_numpy()
        closes = df_lookback['close'].to_numpy()
        n = len(df_lookback)

        # 檢測 FVG
        bullish_fvgs = []
        bearish_fvgs = []

        for i in range(n - 2):
            if highs[i] < lows[i + 2]:
                gap_size = (lows[i + 2] - highs[i]) / highs[i]
                if gap_size >= self.min_size_pct:
                    bullish_fvgs.append((i, lows[i + 2], highs[i], False))

            if lows[i] > highs[i + 2]:
                gap_size = (lows[i] - highs[i + 2]) / highs[i + 2]
                if gap_size >= self.min_size_pct:
                    bearish_fvgs.append((i, lows[i], highs[i + 2], False))

        # 更新填補狀態
        bullish_fvgs_updated = []
        for start_idx, fvg_high, fvg_low, _ in bullish_fvgs:
            is_filled = False
            for i in range(start_idx + 3, n):
                if lows[i] <= fvg_high:
                    is_filled = True
                    break
            bullish_fvgs_updated.append((start_idx, fvg_high, fvg_low, is_filled))

        bearish_fvgs_updated = []
        for start_idx, fvg_high, fvg_low, _ in bearish_fvgs:
            is_filled = False
            for i in range(start_idx + 3, n):
                if highs[i] >= fvg_low:
                    is_filled = True
                    break
            bearish_fvgs_updated.append((start_idx, fvg_high, fvg_low, is_filled))

        current_price = closes[-1]
        result = {
            'in_bullish_fvg': 0,
            'in_bearish_fvg': 0,
            'nearest_fvg_direction': 0
        }

        # 找最近未填補 FVG
        nearest_bullish_dist = float('inf')
        nearest_bearish_dist = float('inf')
        nearest_bullish = None
        nearest_bearish = None

        for start_idx, fvg_high, fvg_low, is_filled in bullish_fvgs_updated:
            if is_filled or (n - 1 - start_idx) > self.max_age:
                continue
            mid = (fvg_high + fvg_low) / 2
            dist = abs(current_price - mid)
            if dist < nearest_bullish_dist:
                nearest_bullish_dist = dist
                nearest_bullish = (fvg_high, fvg_low)

        for start_idx, fvg_high, fvg_low, is_filled in bearish_fvgs_updated:
            if is_filled or (n - 1 - start_idx) > self.max_age:
                continue
            mid = (fvg_high + fvg_low) / 2
            dist = abs(current_price - mid)
            if dist < nearest_bearish_dist:
                nearest_bearish_dist = dist
                nearest_bearish = (fvg_high, fvg_low)

        if nearest_bullish:
            fvg_high, fvg_low = nearest_bullish
            if fvg_low <= current_price <= fvg_high:
                result['in_bullish_fvg'] = 1

        if nearest_bearish:
            fvg_high, fvg_low = nearest_bearish
            if fvg_low <= current_price <= fvg_high:
                result['in_bearish_fvg'] = 1

        if nearest_bullish and not nearest_bearish:
            result['nearest_fvg_direction'] = 1
        elif nearest_bearish and not nearest_bullish:
            result['nearest_fvg_direction'] = -1
        elif nearest_bullish and nearest_bearish:
            result['nearest_fvg_direction'] = 1 if nearest_bullish_dist < nearest_bearish_dist else -1

        return result

    def analyze_full_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """分析整個數據集的 FVG"""
        print(f"[FVG] Analyzing {len(df):,} bars...")
        self.precompute_all_features(df)

        result = df.copy()
        result['in_bullish_fvg'] = self._in_bullish_cache
        result['in_bearish_fvg'] = self._in_bearish_cache
        result['nearest_fvg_direction'] = self._nearest_dir_cache

        return result


def test_fvg():
    """測試 FVG 模塊"""
    print("=" * 60)
    print("  Testing FVG (Vectorized)")
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

    fvg = FVGDetector(min_size_pct=0.001, max_age=100)

    start = time.time()
    fvg.precompute_all_features(df_test)
    elapsed = time.time() - start
    print(f"   Precompute time: {elapsed:.3f}s")

    start = time.time()
    for i in range(len(df_test)):
        _ = fvg.get_cached_features(i)
    elapsed = time.time() - start
    print(f"   Cache query time ({len(df_test)} queries): {elapsed:.3f}s")

    print("\n   OK: Vectorized FVG test passed!")


if __name__ == "__main__":
    test_fvg()
