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

    def __init__(self, min_size_pct: float = 0.001, max_age: int = 300, min_size_atr: float = 0.1):
        """
        初始化 FVG 檢測器

        Args:
            min_size_pct: FVG 最小大小（百分比，默認 0.1%，無 ATR 時的 fallback）
            max_age: FVG 最大保留期數（默認 100 根 K 線）
            min_size_atr: FVG 最小大小（ATR 倍數，默認 0.3 ATR）
        """
        self.min_size_pct = min_size_pct
        self.min_size_atr = min_size_atr
        self.max_age = max_age

        # 預計算緩存
        self._cache_valid = False
        self._in_bullish_cache = None      # [n] 是否在看漲 FVG 內
        self._in_bearish_cache = None      # [n] 是否在看跌 FVG 內
        self._nearest_dir_cache = None     # [n] 最近 FVG 方向

    def precompute_all_features(self, df: pd.DataFrame, atr_array: np.ndarray = None) -> None:
        """
        預計算整個數據集的 FVG 特徵

        優化版：使用活躍 FVG 清單 + 過期清理，時間複雜度 O(n)

        Args:
            df: OHLC 數據
            atr_array: ATR 陣列（用於動態 FVG 大小閾值，取代固定百分比）
        """
        from collections import deque

        n = len(df)
        highs = df['high'].to_numpy(dtype=np.float64)
        lows = df['low'].to_numpy(dtype=np.float64)
        closes = df['close'].to_numpy(dtype=np.float64)

        # 初始化緩存
        self._in_bullish_cache = np.zeros(n, dtype=np.int8)
        self._in_bearish_cache = np.zeros(n, dtype=np.int8)
        self._nearest_dir_cache = np.zeros(n, dtype=np.int8)

        # 1. 向量化檢測所有 FVG 位置
        # 看漲 FVG: highs[i] < lows[i+2]，且大小 >= min_size_pct
        # 看跌 FVG: lows[i] > highs[i+2]
        bull_fvg_indices = []  # [(start_idx, high=lows[i+2], low=highs[i]), ...]
        bear_fvg_indices = []

        if n > 2:
            h_shifted = highs[:-2]
            l_shifted = lows[2:]
            bull_mask = h_shifted < l_shifted

            l_first = lows[:-2]
            h_third = highs[2:]
            bear_mask = l_first > h_third

            # FVG 大小過濾：使用 ATR 動態閾值（價格無關）
            if atr_array is not None:
                atr_shifted = np.maximum(atr_array[:-2], 1e-10)
                gap_size_bull = (l_shifted - h_shifted) / atr_shifted
                bull_mask &= gap_size_bull >= self.min_size_atr
                gap_size_bear = (l_first - h_third) / atr_shifted
                bear_mask &= gap_size_bear >= self.min_size_atr
            else:
                # 無 ATR 時 fallback 到百分比閾值
                gap_size_bull = (l_shifted - h_shifted) / np.where(h_shifted > 0, h_shifted, 1.0)
                bull_mask &= gap_size_bull >= self.min_size_pct
                gap_size_bear = (l_first - h_third) / np.where(h_third > 0, h_third, 1.0)
                bear_mask &= gap_size_bear >= self.min_size_pct

            for idx in np.where(bull_mask)[0]:
                bull_fvg_indices.append((idx, lows[idx + 2], highs[idx]))
            for idx in np.where(bear_mask)[0]:
                bear_fvg_indices.append((idx, lows[idx], highs[idx + 2]))

        # 2. O(n) 前向掃描：維護活躍（未填補）FVG deque
        bull_ptr = 0
        bear_ptr = 0
        active_bull = deque()  # (start_idx, high, low)
        active_bear = deque()

        for i in range(n):
            current_price = closes[i]
            current_high = highs[i]
            current_low = lows[i]

            # 加入新的 FVG（已形成的，i > start_idx + 2）
            while bull_ptr < len(bull_fvg_indices) and bull_fvg_indices[bull_ptr][0] + 2 <= i:
                active_bull.append(bull_fvg_indices[bull_ptr])
                bull_ptr += 1
            while bear_ptr < len(bear_fvg_indices) and bear_fvg_indices[bear_ptr][0] + 2 <= i:
                active_bear.append(bear_fvg_indices[bear_ptr])
                bear_ptr += 1

            # 清理：移除已過期的 FVG（從左邊移除，deque 前端是最舊的）
            while active_bull and i - active_bull[0][0] > self.max_age:
                active_bull.popleft()
            while active_bear and i - active_bear[0][0] > self.max_age:
                active_bear.popleft()

            # 清理：移除已填補的 FVG（標記後批量清理）
            # 看漲 FVG 被填補: 價格回落到 FVG 區域 (current_low <= fvg_high)
            # 看跌 FVG 被填補: 價格回升到 FVG 區域 (current_high >= fvg_low)
            new_bull = deque()
            for item in active_bull:
                start_idx, fvg_high, fvg_low = item
                if current_low <= fvg_high:  # 填補了
                    continue
                new_bull.append(item)
            active_bull = new_bull

            new_bear = deque()
            for item in active_bear:
                start_idx, fvg_high, fvg_low = item
                if current_high >= fvg_low:  # 填補了
                    continue
                new_bear.append(item)
            active_bear = new_bear

            # 找最近的未填補 FVG
            nearest_bull = None
            nearest_bull_dist = float('inf')
            for start_idx, fvg_high, fvg_low in active_bull:
                mid = (fvg_high + fvg_low) / 2
                dist = abs(current_price - mid)
                if dist < nearest_bull_dist:
                    nearest_bull_dist = dist
                    nearest_bull = (fvg_high, fvg_low)

            nearest_bear = None
            nearest_bear_dist = float('inf')
            for start_idx, fvg_high, fvg_low in active_bear:
                mid = (fvg_high + fvg_low) / 2
                dist = abs(current_price - mid)
                if dist < nearest_bear_dist:
                    nearest_bear_dist = dist
                    nearest_bear = (fvg_high, fvg_low)

            # 計算特徵
            in_bullish = 0
            in_bearish = 0
            nearest_dir = 0

            if nearest_bull:
                fvg_high, fvg_low = nearest_bull
                if fvg_low <= current_price <= fvg_high:
                    in_bullish = 1

            if nearest_bear:
                fvg_high, fvg_low = nearest_bear
                if fvg_low <= current_price <= fvg_high:
                    in_bearish = 1

            if nearest_bull and not nearest_bear:
                nearest_dir = 1
            elif nearest_bear and not nearest_bull:
                nearest_dir = -1
            elif nearest_bull and nearest_bear:
                nearest_dir = 1 if nearest_bull_dist < nearest_bear_dist else -1

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
