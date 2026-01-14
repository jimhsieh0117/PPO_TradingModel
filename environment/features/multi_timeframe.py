"""
Multi-Timeframe 特徵模塊

實現多時間框架趨勢確認：
- 5分K 趨勢方向
- 15分K 趨勢方向

用於過濾交易信號，確保多時間框架一致性

作者：PPO Trading Team
日期：2026-01-14
"""

import numpy as np
import pandas as pd
from typing import Dict


class MultiTimeframeAnalyzer:
    """多時間框架分析器"""

    def __init__(self):
        """初始化分析器"""
        self.timeframes = {
            '5m': 5,   # 5分鐘 = 5 根 1分K
            '15m': 15  # 15分鐘 = 15 根 1分K
        }

    def resample_to_timeframe(self, df: pd.DataFrame, minutes: int) -> pd.DataFrame:
        """
        將 1分K 重採樣到指定時間框架

        Args:
            df: 1分K OHLCV 數據
            minutes: 目標時間框架（分鐘）

        Returns:
            pd.DataFrame: 重採樣後的數據
        """
        # 重採樣規則（使用 'min' 替代已棄用的 'T'）
        resampled = df.resample(f'{minutes}min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        return resampled

    def calculate_trend(self, df: pd.DataFrame, lookback: int = 10) -> int:
        """
        計算趨勢方向

        使用簡單的移動平均交叉方法：
        - 短期均線（5 期）vs 長期均線（10 期）

        Args:
            df: OHLC 數據
            lookback: 用於計算的最小數據量

        Returns:
            int: -1 (下降), 0 (震盪), 1 (上升)
        """
        if len(df) < lookback:
            return 0

        # 計算短期和長期移動平均
        short_ma = df['close'].rolling(window=5, min_periods=1).mean()
        long_ma = df['close'].rolling(window=10, min_periods=1).mean()

        # 最近的均線值
        latest_short_ma = short_ma.iloc[-1]
        latest_long_ma = long_ma.iloc[-1]

        # 判斷趨勢
        if latest_short_ma > latest_long_ma * 1.001:  # 0.1% 緩衝區避免震盪
            return 1  # 上升趨勢
        elif latest_short_ma < latest_long_ma * 0.999:
            return -1  # 下降趨勢
        else:
            return 0  # 震盪

    def calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """
        計算趨勢強度（可選）

        使用 ADX (Average Directional Index) 的簡化版本

        Args:
            df: OHLC 數據

        Returns:
            float: 趨勢強度 (0-100)
        """
        if len(df) < 14:
            return 0.0

        # 計算價格變化
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()

        # 計算方向性移動
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

        # 計算 True Range
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # 平滑計算（14 期）
        atr = tr.rolling(window=14, min_periods=1).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=1).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=1).mean() / atr

        # 計算 DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=14, min_periods=1).mean()

        return adx.iloc[-1] if not adx.empty else 0.0

    def calculate_features(self, df: pd.DataFrame, current_idx: int) -> Dict[str, int]:
        """
        計算多時間框架特徵

        Args:
            df: 1分K OHLCV 數據
            current_idx: 當前索引

        Returns:
            dict: {
                'trend_5m': int,   # -1/0/1
                'trend_15m': int   # -1/0/1
            }
        """
        # 獲取當前時間之前的數據（避免未來洩漏）
        df_before = df.iloc[:current_idx+1].copy()

        if len(df_before) < 20:
            return {
                'trend_5m': 0,
                'trend_15m': 0
            }

        # 重採樣到 5分K
        df_5m = self.resample_to_timeframe(df_before, 5)
        trend_5m = self.calculate_trend(df_5m, lookback=10)

        # 重採樣到 15分K
        df_15m = self.resample_to_timeframe(df_before, 15)
        trend_15m = self.calculate_trend(df_15m, lookback=10)

        return {
            'trend_5m': trend_5m,
            'trend_15m': trend_15m
        }

    def analyze_full_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """分析整個數據集的多時間框架趨勢"""
        print(f"🔍 分析 Multi-Timeframe 趨勢...")
        print(f"   數據範圍: {df.index[0]} 到 {df.index[-1]}")
        print(f"   總 K 線數: {len(df):,}")

        # 創建結果 DataFrame
        result = df.copy()
        result['trend_5m'] = 0
        result['trend_15m'] = 0

        # 計算每個位置的特徵
        for i in range(len(df)):
            features = self.calculate_features(df, i)
            result['trend_5m'].iloc[i] = features['trend_5m']
            result['trend_15m'].iloc[i] = features['trend_15m']

        # 統計 5分K 趨勢分布
        print(f"\n📊 5分K 趨勢分布:")
        trend_5m_counts = result['trend_5m'].value_counts().sort_index()
        for trend, count in trend_5m_counts.items():
            trend_name = {-1: "下降", 0: "震盪", 1: "上升"}[trend]
            print(f"   {trend_name}: {count} ({count/len(result)*100:.1f}%)")

        # 統計 15分K 趨勢分布
        print(f"\n📊 15分K 趨勢分布:")
        trend_15m_counts = result['trend_15m'].value_counts().sort_index()
        for trend, count in trend_15m_counts.items():
            trend_name = {-1: "下降", 0: "震盪", 1: "上升"}[trend]
            print(f"   {trend_name}: {count} ({count/len(result)*100:.1f}%)")

        # 統計多時間框架一致性
        aligned_bullish = ((result['trend_5m'] == 1) & (result['trend_15m'] == 1)).sum()
        aligned_bearish = ((result['trend_5m'] == -1) & (result['trend_15m'] == -1)).sum()
        total_aligned = aligned_bullish + aligned_bearish

        print(f"\n📊 多時間框架一致性:")
        print(f"   看漲一致: {aligned_bullish} ({aligned_bullish/len(result)*100:.1f}%)")
        print(f"   看跌一致: {aligned_bearish} ({aligned_bearish/len(result)*100:.1f}%)")
        print(f"   總一致性: {total_aligned} ({total_aligned/len(result)*100:.1f}%)")

        print(f"✅ Multi-Timeframe 分析完成！\n")

        return result


def test_multi_timeframe():
    """測試 Multi-Timeframe 模塊"""
    print("=" * 60)
    print("  🧪 測試 Multi-Timeframe 特徵模塊")
    print("=" * 60)

    # 載入數據
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent

    print("\n📂 載入測試數據...")
    data_path = project_root / "data" / "raw" / "BTCUSDT_1m_train_latest.csv"
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # 使用最近 1000 根 K 線進行測試
    df_test = df.iloc[-1000:].copy()
    print(f"   測試數據: {len(df_test)} 根 K 線")
    print(f"   時間範圍: {df_test.index[0]} 到 {df_test.index[-1]}\n")

    # 初始化分析器
    mtf_analyzer = MultiTimeframeAnalyzer()

    # 分析整個測試數據集
    result = mtf_analyzer.analyze_full_dataset(df_test)

    print("✅ 測試完成！\n")

    # 測試單點特徵計算
    print("🎯 測試單點特徵計算（最後一根 K 線）:")
    features = mtf_analyzer.calculate_features(df_test, len(df_test) - 1)
    trend_5m_name = {-1: "下降", 0: "震盪", 1: "上升"}[features['trend_5m']]
    trend_15m_name = {-1: "下降", 0: "震盪", 1: "上升"}[features['trend_15m']]
    print(f"   5分K 趨勢: {trend_5m_name} ({features['trend_5m']})")
    print(f"   15分K 趨勢: {trend_15m_name} ({features['trend_15m']})")

    return result


if __name__ == "__main__":
    result = test_multi_timeframe()
