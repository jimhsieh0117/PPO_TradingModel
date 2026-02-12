"""
Market Structure 檢測模塊（向量化優化版）

實現 ICT (Inner Circle Trader) 市場結構概念：
1. 趨勢識別（上升/下降/震盪）
2. BOS (Break of Structure) - 趨勢延續
3. ChoCh (Change of Character) - 趨勢轉變

優化：
- 使用 NumPy 向量化操作取代逐點計算
- 使用 pandas rolling 進行滑動窗口計算
- 預計算整個數據集，查詢 O(1)

作者：PPO Trading Team
日期：2026-01-14
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict


class MarketStructure:
    """
    市場結構分析器（向量化優化版）

    根據 ICT 理論檢測市場結構變化：
    - Higher Highs (HH) & Higher Lows (HL) = 上升趨勢
    - Lower Highs (LH) & Lower Lows (LL) = 下降趨勢
    - BOS (Break of Structure) = 趨勢延續信號
    - ChoCh (Change of Character) = 趨勢轉變信號
    """

    def __init__(self, lookback: int = 50):
        """
        初始化市場結構分析器

        Args:
            lookback: 回看期數，用於識別 swing highs/lows（默認 50）
        """
        self.lookback = lookback
        self.swing_window = 5  # 識別 swing point 的窗口大小

        # 預計算緩存
        self._cache_valid = False
        self._trend_cache = None       # [n] array of trend states
        self._signal_cache = None      # [n] array of structure signals
        self._bars_since_cache = None  # [n] array of bars since change

    def precompute_all_features(self, df: pd.DataFrame) -> None:
        """
        向量化預計算整個數據集的市場結構特徵

        時間複雜度: O(n) 總時間（相比原始 O(n × L²)）

        Args:
            df: OHLC 數據
        """
        n = len(df)
        highs = df['high'].to_numpy(dtype=np.float64)
        lows = df['low'].to_numpy(dtype=np.float64)
        closes = df['close'].to_numpy(dtype=np.float64)

        # 1. 向量化識別 swing points（使用 rolling max/min）
        swing_highs, swing_lows = self._vectorized_swing_points(highs, lows)

        # 2. 向量化檢測 BOS/ChoCh
        self._trend_cache, self._signal_cache, self._bars_since_cache = \
            self._vectorized_bos_choch(highs, lows, closes, swing_highs, swing_lows)

        self._cache_valid = True

    def _vectorized_swing_points(self, highs: np.ndarray, lows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        向量化識別 Swing Highs 和 Swing Lows

        使用滑動窗口最大/最小值比較

        Args:
            highs: 高點數組
            lows: 低點數組

        Returns:
            (swing_highs, swing_lows): 兩個數組，非 swing point 為 NaN
        """
        n = len(highs)
        w = self.swing_window
        window_size = 2 * w + 1

        # 使用 pandas rolling（底層是 C 優化的）
        high_series = pd.Series(highs)
        low_series = pd.Series(lows)

        # 計算滑動最大/最小值（center=True 使窗口居中）
        rolling_max = high_series.rolling(window_size, center=True, min_periods=window_size).max()
        rolling_min = low_series.rolling(window_size, center=True, min_periods=window_size).min()

        # Swing High: 當前高點 = 窗口最大值
        swing_highs = np.where(highs == rolling_max.to_numpy(), highs, np.nan)

        # Swing Low: 當前低點 = 窗口最小值
        swing_lows = np.where(lows == rolling_min.to_numpy(), lows, np.nan)

        return swing_highs, swing_lows

    def _vectorized_bos_choch(self, highs: np.ndarray, lows: np.ndarray,
                              closes: np.ndarray, swing_highs: np.ndarray,
                              swing_lows: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        向量化檢測 BOS/ChoCh

        Args:
            highs, lows, closes: 價格數組
            swing_highs, swing_lows: Swing point 數組

        Returns:
            (trend_state, structure_signal, bars_since_change)
        """
        n = len(highs)

        # 初始化結果
        trend_state = np.zeros(n, dtype=np.int8)
        structure_signal = np.zeros(n, dtype=np.int8)
        bars_since_change = np.zeros(n, dtype=np.int32)

        # 追蹤最近的有效 swing points
        last_swing_high = np.nan
        last_swing_low = np.nan
        current_trend = 0
        last_change_idx = 0

        # 遍歷（這部分難以完全向量化，但已優化數據訪問）
        for i in range(n):
            # 更新最近的 swing points
            if not np.isnan(swing_highs[i]):
                last_swing_high = swing_highs[i]
            if not np.isnan(swing_lows[i]):
                last_swing_low = swing_lows[i]

            # 跳過初始無效數據
            if np.isnan(last_swing_high) or np.isnan(last_swing_low):
                trend_state[i] = 0
                structure_signal[i] = 0
                bars_since_change[i] = 0
                continue

            # 檢測結構變化
            signal = 0
            if current_trend >= 0:  # 上升趨勢或震盪
                if highs[i] > last_swing_high:
                    signal = 1  # Bullish BOS
                    current_trend = 1
                    last_change_idx = i
                elif lows[i] < last_swing_low:
                    signal = -1  # Bearish ChoCh
                    current_trend = -1
                    last_change_idx = i

            if current_trend <= 0 and signal == 0:  # 下降趨勢或震盪
                if lows[i] < last_swing_low:
                    signal = -1  # Bearish BOS
                    current_trend = -1
                    last_change_idx = i
                elif highs[i] > last_swing_high:
                    signal = 1  # Bullish ChoCh
                    current_trend = 1
                    last_change_idx = i

            trend_state[i] = current_trend
            structure_signal[i] = signal
            bars_since_change[i] = i - last_change_idx

        return trend_state, structure_signal, bars_since_change

    def get_cached_features(self, current_idx: int) -> Dict[str, int]:
        """
        從緩存獲取特徵（O(1) 操作）

        Args:
            current_idx: 當前索引

        Returns:
            dict: 特徵字典
        """
        if not self._cache_valid:
            raise RuntimeError("緩存未初始化，請先調用 precompute_all_features()")

        return {
            'trend_state': int(self._trend_cache[current_idx]),
            'structure_signal': int(self._signal_cache[current_idx]),
            'bars_since_structure_change': int(self._bars_since_cache[current_idx])
        }

    def calculate_features(self, df: pd.DataFrame, current_idx: int) -> dict:
        """
        計算當前位置的市場結構特徵

        優化版：優先使用緩存，否則回退到原始計算

        Args:
            df: OHLC 數據
            current_idx: 當前 K 線索引

        Returns:
            dict: 包含 3 個特徵的字典
        """
        # 優先使用緩存
        if self._cache_valid:
            return self.get_cached_features(current_idx)

        # 回退到原始計算（用於兼容性）
        return self._calculate_features_original(df, current_idx)

    def _calculate_features_original(self, df: pd.DataFrame, current_idx: int) -> dict:
        """原始計算方法（保留用於兼容性）"""
        lookback_start = max(0, current_idx - self.lookback)
        df_lookback = df.iloc[lookback_start:current_idx+1]

        if len(df_lookback) < self.swing_window * 2:
            return {
                'trend_state': 0,
                'structure_signal': 0,
                'bars_since_structure_change': 0
            }

        highs = df_lookback['high'].to_numpy()
        lows = df_lookback['low'].to_numpy()
        closes = df_lookback['close'].to_numpy()

        swing_highs, swing_lows = self._vectorized_swing_points(highs, lows)
        trend, signal, bars = self._vectorized_bos_choch(highs, lows, closes, swing_highs, swing_lows)

        return {
            'trend_state': int(trend[-1]),
            'structure_signal': int(signal[-1]),
            'bars_since_structure_change': int(bars[-1])
        }

    def identify_swing_points(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """向量化版本的 swing point 識別"""
        highs = df['high'].to_numpy()
        lows = df['low'].to_numpy()
        swing_highs, swing_lows = self._vectorized_swing_points(highs, lows)
        return pd.Series(swing_highs, index=df.index), pd.Series(swing_lows, index=df.index)

    def analyze_full_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """分析整個數據集的市場結構"""
        print(f"[MarketStructure] Analyzing {len(df):,} bars...")

        # 使用向量化預計算
        self.precompute_all_features(df)

        # 統計
        bos_count = np.sum(self._signal_cache == 1)
        choch_count = np.sum(self._signal_cache == -1)
        print(f"   Bullish BOS: {bos_count}, Bearish ChoCh: {choch_count}")

        # 創建結果 DataFrame
        result = df.copy()
        result['trend_state'] = self._trend_cache
        result['structure_signal'] = self._signal_cache
        result['bars_since_structure_change'] = self._bars_since_cache

        return result


def test_market_structure():
    """測試 Market Structure 模塊"""
    print("=" * 60)
    print("  Testing Market Structure (Vectorized)")
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

    ms = MarketStructure(lookback=50)

    # 測試向量化預計算
    start = time.time()
    ms.precompute_all_features(df_test)
    elapsed = time.time() - start
    print(f"   Precompute time: {elapsed:.3f}s")

    # 測試緩存查詢
    start = time.time()
    for i in range(len(df_test)):
        _ = ms.get_cached_features(i)
    elapsed = time.time() - start
    print(f"   Cache query time ({len(df_test)} queries): {elapsed:.3f}s")

    print("\n   OK: Vectorized MarketStructure test passed!")


if __name__ == "__main__":
    test_market_structure()
