"""
Volume & Price 特徵模塊（向量化優化版）

實現成交量與價格行為分析：
- Volume Ratio: 當前成交量 vs 平均成交量
- Price Momentum: 價格變化動量
- VWAP Momentum: 成交量加權價格動量
- Price Position in Range: 當前價格在波段中的位置
- Zone Classification: Premium/Discount/Equilibrium 區域分類

優化：
- 使用 NumPy 向量化操作
- 使用 pandas rolling 進行滑動窗口計算
- 預計算整個數據集，查詢 O(1)

作者：PPO Trading Team
日期：2026-01-14
"""

import numpy as np
import pandas as pd
from typing import Dict


class VolumeAnalyzer:
    """成交量與價格分析器（向量化優化版）"""

    def __init__(self, volume_window: int = 20, swing_window: int = 50):
        """
        初始化分析器

        Args:
            volume_window: 計算平均成交量的窗口
            swing_window: 識別波段高低點的窗口
        """
        self.volume_window = volume_window
        self.swing_window = swing_window

        # 預計算緩存
        self._cache_valid = False
        self._volume_ratio_cache = None
        self._price_momentum_cache = None
        self._vwap_momentum_cache = None
        self._price_position_cache = None
        self._zone_class_cache = None
        self._atr_cache = None              # 原始 ATR（價格單位，供環境止損用）
        self._atr_normalized_cache = None   # ATR / close（正規化，供特徵用）
        self._adx_normalized_cache = None   # ADX(14) / 100, [0, 1]
        self._volatility_regime_cache = None  # ATR 歷史百分位, [0, 1]
        self._trend_strength_cache = None   # (close - EMA200) / ATR, [-1, 1]

    def precompute_all_features(self, df: pd.DataFrame) -> None:
        """
        向量化預計算整個數據集的成交量與價格特徵

        Args:
            df: OHLCV 數據
        """
        n = len(df)
        highs = df['high'].to_numpy(dtype=np.float64)
        lows = df['low'].to_numpy(dtype=np.float64)
        closes = df['close'].to_numpy(dtype=np.float64)
        volumes = df['volume'].to_numpy(dtype=np.float64)

        # 初始化緩存
        self._volume_ratio_cache = np.ones(n, dtype=np.float32)
        self._price_momentum_cache = np.zeros(n, dtype=np.float32)
        self._vwap_momentum_cache = np.zeros(n, dtype=np.float32)
        self._price_position_cache = np.full(n, 50.0, dtype=np.float32)
        self._zone_class_cache = np.zeros(n, dtype=np.int8)

        # 1. 向量化計算 Volume Ratio
        volume_series = pd.Series(volumes)
        rolling_avg_volume = volume_series.rolling(self.volume_window, min_periods=1).mean()
        avg_volume_arr = rolling_avg_volume.to_numpy()
        # 避免除以零
        avg_volume_arr = np.where(avg_volume_arr > 0, avg_volume_arr, 1.0)
        self._volume_ratio_cache = (volumes / avg_volume_arr).astype(np.float32)

        # 2. 向量化計算 Price Momentum (10 bar lookback)
        momentum_lookback = 10
        close_series = pd.Series(closes)
        shifted_closes = close_series.shift(momentum_lookback)
        momentum = (closes - shifted_closes.to_numpy()) / np.where(shifted_closes.to_numpy() > 0, shifted_closes.to_numpy(), 1.0) * 100
        momentum = np.nan_to_num(momentum, nan=0.0)
        self._price_momentum_cache = momentum.astype(np.float32)

        # 3. 向量化計算 VWAP Momentum
        typical_price = (highs + lows + closes) / 3
        tp_volume = typical_price * volumes
        tp_volume_series = pd.Series(tp_volume)
        volume_series = pd.Series(volumes)

        rolling_tp_volume = tp_volume_series.rolling(self.volume_window, min_periods=1).sum()
        rolling_volume = volume_series.rolling(self.volume_window, min_periods=1).sum()

        vwap = rolling_tp_volume.to_numpy() / np.where(rolling_volume.to_numpy() > 0, rolling_volume.to_numpy(), 1.0)
        vwap_momentum = (closes - vwap) / np.where(vwap > 0, vwap, 1.0) * 100
        vwap_momentum = np.nan_to_num(vwap_momentum, nan=0.0)
        self._vwap_momentum_cache = vwap_momentum.astype(np.float32)

        # 4. 向量化計算 Price Position in Range
        high_series = pd.Series(highs)
        low_series = pd.Series(lows)

        rolling_high = high_series.rolling(self.swing_window, min_periods=1).max()
        rolling_low = low_series.rolling(self.swing_window, min_periods=1).min()

        swing_high = rolling_high.to_numpy()
        swing_low = rolling_low.to_numpy()

        range_size = swing_high - swing_low
        # 避免除以零
        range_size = np.where(range_size > 0, range_size, 1.0)
        position = (closes - swing_low) / range_size * 100
        position = np.clip(position, 0, 100)
        self._price_position_cache = position.astype(np.float32)

        # 5. 向量化計算 Zone Classification
        # Premium >= 61.8%, Discount <= 38.2%, Equilibrium 中間
        self._zone_class_cache = np.where(
            position >= 61.8, 1,  # Premium
            np.where(position <= 38.2, -1, 0)  # Discount or Equilibrium
        ).astype(np.int8)

        # 6. 向量化計算 ATR (Average True Range)
        atr_period = 14
        prev_close = np.roll(closes, 1)
        prev_close[0] = closes[0]
        true_range = np.maximum(
            highs - lows,
            np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close))
        )
        atr = pd.Series(true_range).rolling(atr_period, min_periods=1).mean().to_numpy()
        self._atr_cache = atr.astype(np.float64)
        # 正規化：ATR / close（衡量相對波動率）
        self._atr_normalized_cache = (atr / np.maximum(closes, 1e-10)).astype(np.float32)

        # 7. ADX (Average Directional Index, 14 期)
        adx_period = 14
        high_diff = np.diff(highs, prepend=highs[0])
        low_diff = -np.diff(lows, prepend=lows[0])
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)

        alpha = 1.0 / adx_period
        tr_smooth = pd.Series(true_range).ewm(alpha=alpha, adjust=False).mean().to_numpy()
        plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean().to_numpy()
        minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean().to_numpy()

        plus_di = 100.0 * plus_dm_smooth / np.maximum(tr_smooth, 1e-10)
        minus_di = 100.0 * minus_dm_smooth / np.maximum(tr_smooth, 1e-10)

        dx = 100.0 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-10)
        adx = pd.Series(dx).ewm(alpha=alpha, adjust=False).mean().to_numpy()
        self._adx_normalized_cache = np.clip(adx / 100.0, 0.0, 1.0).astype(np.float32)

        # 8. Volatility Regime: ATR 在過去 480 根 K 線中的相對位置 (0~1)
        # 使用 rolling min/max 正規化，O(n) 時間複雜度
        # （原 rolling rank 為 O(n × window)，3.2M 行 × 480 窗口 ≈ 15 億次操作）
        volatility_lookback = 480
        atr_series = pd.Series(atr)
        rolling_min = atr_series.rolling(volatility_lookback, min_periods=1).min().to_numpy()
        rolling_max = atr_series.rolling(volatility_lookback, min_periods=1).max().to_numpy()
        regime_range = rolling_max - rolling_min
        self._volatility_regime_cache = np.where(
            regime_range > 1e-10,
            (atr - rolling_min) / regime_range,
            0.5  # 波動率恆定時返回中間值
        ).astype(np.float32)

        # 9. Trend Strength: (close - EMA200) / ATR, 裁切到 [-1, 1]
        ema200 = pd.Series(closes).ewm(span=200, min_periods=1).mean().to_numpy()
        deviation = (closes - ema200) / np.maximum(atr, 1e-10)
        self._trend_strength_cache = np.clip(deviation / 5.0, -1.0, 1.0).astype(np.float32)

        self._cache_valid = True

    def get_cached_features(self, current_idx: int) -> Dict[str, float]:
        """從緩存獲取特徵（O(1) 操作）"""
        if not self._cache_valid:
            raise RuntimeError("緩存未初始化，請先調用 precompute_all_features()")

        return {
            'volume_ratio': float(self._volume_ratio_cache[current_idx]),
            'price_momentum': float(self._price_momentum_cache[current_idx]),
            'vwap_momentum': float(self._vwap_momentum_cache[current_idx]),
            'price_position_in_range': float(self._price_position_cache[current_idx]),
            'zone_classification': int(self._zone_class_cache[current_idx])
        }

    def calculate_features(self, df: pd.DataFrame, current_idx: int) -> dict:
        """計算當前位置的成交量與價格特徵"""
        if self._cache_valid:
            return self.get_cached_features(current_idx)

        return self._calculate_features_original(df, current_idx)

    def _calculate_features_original(self, df: pd.DataFrame, current_idx: int) -> dict:
        """原始計算方法（保留用於兼容性）"""
        if current_idx < self.volume_window:
            return {
                'volume_ratio': 1.0,
                'price_momentum': 0.0,
                'vwap_momentum': 0.0,
                'price_position_in_range': 50.0,
                'zone_classification': 0
            }

        closes = df['close'].to_numpy()
        highs = df['high'].to_numpy()
        lows = df['low'].to_numpy()
        volumes = df['volume'].to_numpy()

        # Volume Ratio
        current_volume = volumes[current_idx]
        avg_volume = volumes[current_idx - self.volume_window:current_idx].mean()
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

        # Price Momentum
        lookback = 10
        if current_idx >= lookback:
            past_price = closes[current_idx - lookback]
            price_momentum = (closes[current_idx] - past_price) / past_price * 100 if past_price > 0 else 0.0
        else:
            price_momentum = 0.0

        # VWAP Momentum
        start_idx = max(0, current_idx - self.volume_window)
        typical_price = (highs[start_idx:current_idx+1] + lows[start_idx:current_idx+1] + closes[start_idx:current_idx+1]) / 3
        vol_slice = volumes[start_idx:current_idx+1]
        total_vol = vol_slice.sum()
        vwap = (typical_price * vol_slice).sum() / total_vol if total_vol > 0 else closes[current_idx]
        vwap_momentum = (closes[current_idx] - vwap) / vwap * 100 if vwap > 0 else 0.0

        # Price Position
        lookback_start = max(0, current_idx - self.swing_window)
        swing_high = highs[lookback_start:current_idx+1].max()
        swing_low = lows[lookback_start:current_idx+1].min()
        range_size = swing_high - swing_low
        if range_size > 0:
            price_position = (closes[current_idx] - swing_low) / range_size * 100
            price_position = np.clip(price_position, 0, 100)
        else:
            price_position = 50.0

        # Zone Classification
        if price_position >= 61.8:
            zone_class = 1  # Premium
        elif price_position <= 38.2:
            zone_class = -1  # Discount
        else:
            zone_class = 0  # Equilibrium

        return {
            'volume_ratio': volume_ratio,
            'price_momentum': price_momentum,
            'vwap_momentum': vwap_momentum,
            'price_position_in_range': price_position,
            'zone_classification': zone_class
        }

    def analyze_full_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """分析整個數據集的成交量與價格特徵"""
        print(f"[Volume] Analyzing {len(df):,} bars...")
        self.precompute_all_features(df)

        result = df.copy()
        result['volume_ratio'] = self._volume_ratio_cache
        result['price_momentum'] = self._price_momentum_cache
        result['vwap_momentum'] = self._vwap_momentum_cache
        result['price_position_in_range'] = self._price_position_cache
        result['zone_classification'] = self._zone_class_cache

        return result


def test_volume():
    """測試 Volume 模塊"""
    print("=" * 60)
    print("  Testing Volume (Vectorized)")
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

    vol = VolumeAnalyzer(volume_window=20, swing_window=50)

    start = time.time()
    vol.precompute_all_features(df_test)
    elapsed = time.time() - start
    print(f"   Precompute time: {elapsed:.3f}s")

    start = time.time()
    for i in range(len(df_test)):
        _ = vol.get_cached_features(i)
    elapsed = time.time() - start
    print(f"   Cache query time ({len(df_test)} queries): {elapsed:.3f}s")

    print("\n   OK: Vectorized Volume test passed!")


if __name__ == "__main__":
    test_volume()
