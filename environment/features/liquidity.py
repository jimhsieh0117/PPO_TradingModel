"""
Liquidity 檢測模塊

實現 ICT Liquidity 概念：
- 流動性池：前期高點/低點聚集大量止損單
- 上方流動性：前期高點（賣出止損）
- 下方流動性：前期低點（買入止損）
- Liquidity Sweep：價格短暫突破流動性區域後快速反轉

作者：PPO Trading Team
日期：2026-01-14
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


class LiquidityZone:
    """流動性區域數據結構"""

    def __init__(self, price: float, bar_index: int, zone_type: str,
                 timestamp: pd.Timestamp):
        """
        初始化流動性區域

        Args:
            price: 流動性價格水平
            bar_index: K 線索引
            zone_type: 'high' 或 'low'
            timestamp: 時間戳
        """
        self.price = price
        self.bar_index = bar_index
        self.zone_type = zone_type
        self.timestamp = timestamp
        self.is_swept = False  # 是否已被掃蕩
        self.sweep_idx = None  # 掃蕩位置


class LiquidityDetector:
    """流動性檢測器"""

    def __init__(self, lookback: int = 50, sweep_threshold: float = 0.001):
        """
        初始化流動性檢測器

        Args:
            lookback: 回看期數，用於識別前期高低點
            sweep_threshold: 掃蕩閾值（價格超過流動性區域的百分比）
        """
        self.lookback = lookback
        self.sweep_threshold = sweep_threshold
        self.swing_window = 5  # Swing point 識別窗口

    def identify_liquidity_zones(self, df: pd.DataFrame) -> Tuple[List[LiquidityZone], List[LiquidityZone]]:
        """
        識別流動性區域（前期高點和低點）

        Args:
            df: OHLC 數據

        Returns:
            (highs, lows): 上方和下方流動性區域列表
        """
        highs = df['high'].values
        lows = df['low'].values
        n = len(df)
        w = self.swing_window

        high_liquidity_zones = []
        low_liquidity_zones = []

        # 識別 swing highs 和 swing lows（潛在流動性區域）
        for i in range(w, n - w):
            # Swing High (上方流動性)
            if highs[i] == max(highs[i-w:i+w+1]):
                zone = LiquidityZone(
                    price=highs[i],
                    bar_index=i,
                    zone_type='high',
                    timestamp=df.index[i]
                )
                high_liquidity_zones.append(zone)

            # Swing Low (下方流動性)
            if lows[i] == min(lows[i-w:i+w+1]):
                zone = LiquidityZone(
                    price=lows[i],
                    bar_index=i,
                    zone_type='low',
                    timestamp=df.index[i]
                )
                low_liquidity_zones.append(zone)

        return high_liquidity_zones, low_liquidity_zones

    def detect_liquidity_sweeps(self, df: pd.DataFrame,
                               high_zones: List[LiquidityZone],
                               low_zones: List[LiquidityZone]) -> List[int]:
        """
        檢測流動性掃蕩事件

        Liquidity Sweep: 價格短暫突破流動性區域後快速反轉

        Args:
            df: OHLC 數據
            high_zones: 上方流動性區域
            low_zones: 下方流動性區域

        Returns:
            List[int]: 發生掃蕩的 K 線索引列表
        """
        sweep_indices = []

        # 檢查上方流動性掃蕩
        for zone in high_zones:
            if zone.is_swept:
                continue

            # 檢查後續 K 線是否掃蕩該流動性
            for i in range(zone.bar_index + 1, len(df)):
                current_high = df['high'].iloc[i]
                current_close = df['close'].iloc[i]

                # 檢查是否突破流動性區域
                if current_high > zone.price * (1 + self.sweep_threshold):
                    # 檢查是否快速反轉（收盤價回落）
                    if current_close < zone.price:
                        zone.is_swept = True
                        zone.sweep_idx = i
                        sweep_indices.append(i)
                        break

        # 檢查下方流動性掃蕩
        for zone in low_zones:
            if zone.is_swept:
                continue

            # 檢查後續 K 線是否掃蕩該流動性
            for i in range(zone.bar_index + 1, len(df)):
                current_low = df['low'].iloc[i]
                current_close = df['close'].iloc[i]

                # 檢查是否突破流動性區域
                if current_low < zone.price * (1 - self.sweep_threshold):
                    # 檢查是否快速反轉（收盤價回升）
                    if current_close > zone.price:
                        zone.is_swept = True
                        zone.sweep_idx = i
                        sweep_indices.append(i)
                        break

        return sweep_indices

    def get_nearest_liquidity(self, df: pd.DataFrame, current_idx: int,
                             high_zones: List[LiquidityZone],
                             low_zones: List[LiquidityZone]) -> Tuple[Optional[LiquidityZone], Optional[LiquidityZone]]:
        """
        獲取最近的流動性區域

        Args:
            df: OHLC 數據
            current_idx: 當前索引
            high_zones: 上方流動性區域
            low_zones: 下方流動性區域

        Returns:
            (nearest_high_zone, nearest_low_zone)
        """
        current_price = df['close'].iloc[current_idx]

        # 過濾：只保留當前位置之前的未掃蕩流動性
        valid_high_zones = [
            zone for zone in high_zones
            if zone.bar_index < current_idx
            and not zone.is_swept
            and zone.price > current_price  # 上方流動性
        ]

        valid_low_zones = [
            zone for zone in low_zones
            if zone.bar_index < current_idx
            and not zone.is_swept
            and zone.price < current_price  # 下方流動性
        ]

        # 找到最近的上方流動性
        nearest_high = None
        if valid_high_zones:
            nearest_high = min(valid_high_zones, key=lambda z: abs(z.price - current_price))

        # 找到最近的下方流動性
        nearest_low = None
        if valid_low_zones:
            nearest_low = min(valid_low_zones, key=lambda z: abs(z.price - current_price))

        return nearest_high, nearest_low

    def calculate_features(self, df: pd.DataFrame, current_idx: int) -> dict:
        """
        計算當前位置的流動性特徵

        Args:
            df: OHLC 數據
            current_idx: 當前索引

        Returns:
            dict: 包含 3 個特徵的字典
            {
                'liquidity_above': float,      # 上方流動性距離（百分比）
                'liquidity_below': float,      # 下方流動性距離（百分比）
                'liquidity_sweep': int         # 是否剛發生流動性掃蕩 (0/1)
            }
        """
        # 獲取回看數據
        lookback_start = max(0, current_idx - self.lookback)
        df_lookback = df.iloc[lookback_start:current_idx+1].copy()

        if len(df_lookback) < 20:
            # 數據不足
            return {
                'liquidity_above': 5.0,  # 默認遠離流動性
                'liquidity_below': 5.0,
                'liquidity_sweep': 0
            }

        # 識別流動性區域
        high_zones, low_zones = self.identify_liquidity_zones(df_lookback)

        # 檢測流動性掃蕩
        sweep_indices = self.detect_liquidity_sweeps(df_lookback, high_zones, low_zones)

        # 獲取最近的流動性區域
        nearest_high, nearest_low = self.get_nearest_liquidity(
            df_lookback, len(df_lookback) - 1, high_zones, low_zones
        )

        current_price = df.iloc[current_idx]['close']

        # 計算特徵
        features = {
            'liquidity_above': 5.0,  # 默認值
            'liquidity_below': 5.0,
            'liquidity_sweep': 0
        }

        # 計算到上方流動性的距離
        if nearest_high:
            features['liquidity_above'] = (nearest_high.price - current_price) / current_price * 100

        # 計算到下方流動性的距離
        if nearest_low:
            features['liquidity_below'] = (current_price - nearest_low.price) / current_price * 100

        # 檢查當前位置是否剛發生掃蕩
        if len(df_lookback) - 1 in sweep_indices:
            features['liquidity_sweep'] = 1

        return features

    def analyze_full_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """分析整個數據集的流動性"""
        print(f"🔍 分析 Liquidity Zones...")
        print(f"   數據範圍: {df.index[0]} 到 {df.index[-1]}")
        print(f"   總 K 線數: {len(df):,}")

        # 識別流動性區域
        high_zones, low_zones = self.identify_liquidity_zones(df)

        # 檢測流動性掃蕩
        sweep_indices = self.detect_liquidity_sweeps(df, high_zones, low_zones)

        print(f"   檢測到上方流動性: {len(high_zones)}")
        print(f"   檢測到下方流動性: {len(low_zones)}")
        print(f"   檢測到流動性掃蕩: {len(sweep_indices)}")
        print(f"✅ Liquidity 分析完成！\n")

        # 創建結果 DataFrame
        result = df.copy()
        result['liquidity_above'] = 5.0
        result['liquidity_below'] = 5.0
        result['liquidity_sweep'] = 0

        # 計算每個位置的特徵
        for i in range(len(df)):
            features = self.calculate_features(df, i)
            result['liquidity_above'].iloc[i] = features['liquidity_above']
            result['liquidity_below'].iloc[i] = features['liquidity_below']
            result['liquidity_sweep'].iloc[i] = features['liquidity_sweep']

        return result


def test_liquidity():
    """測試 Liquidity 模塊"""
    print("=" * 60)
    print("  🧪 測試 Liquidity 檢測模塊")
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

    # 初始化檢測器
    liq_detector = LiquidityDetector(lookback=50, sweep_threshold=0.001)

    # 分析整個測試數據集
    result = liq_detector.analyze_full_dataset(df_test)

    # 顯示統計
    print("📊 Liquidity 統計:")
    sweep_count = result['liquidity_sweep'].sum()
    print(f"   流動性掃蕩事件: {sweep_count} 次")

    avg_above = result['liquidity_above'].mean()
    avg_below = result['liquidity_below'].mean()
    print(f"\n   平均上方流動性距離: {avg_above:.2f}%")
    print(f"   平均下方流動性距離: {avg_below:.2f}%")

    print("\n✅ 測試完成！\n")

    # 測試單點特徵計算
    print("🎯 測試單點特徵計算（最後一根 K 線）:")
    features = liq_detector.calculate_features(df_test, len(df_test) - 1)
    print(f"   上方流動性距離: {features['liquidity_above']:.2f}%")
    print(f"   下方流動性距離: {features['liquidity_below']:.2f}%")
    print(f"   流動性掃蕩: {features['liquidity_sweep']}")

    return result


if __name__ == "__main__":
    result = test_liquidity()
