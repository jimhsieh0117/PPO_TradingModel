"""
Order Blocks 檢測模塊

實現 ICT Order Block 概念：
- 看漲 Order Block: 下降趨勢反轉前的最後一根看跌 K 線
- 看跌 Order Block: 上升趨勢反轉前的最後一根看漲 K 線
- Order Blocks 通常提供支撐/阻力區域

作者：PPO Trading Team
日期：2026-01-14
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional


class OrderBlock:
    """Order Block 數據結構"""

    def __init__(self, bar_index: int, high: float, low: float,
                 direction: str, timestamp: pd.Timestamp):
        """
        初始化 Order Block

        Args:
            bar_index: K 線索引
            high: OB 區域高點
            low: OB 區域低點
            direction: 'bullish' 或 'bearish'
            timestamp: 時間戳
        """
        self.bar_index = bar_index
        self.high = high
        self.low = low
        self.direction = direction
        self.timestamp = timestamp
        self.is_mitigated = False  # 是否已被緩解（價格回到OB區域）

    def is_inside(self, price: float) -> bool:
        """檢查價格是否在 OB 區域內"""
        return self.low <= price <= self.high

    def get_distance(self, price: float) -> float:
        """計算價格到 OB 的距離（百分比）"""
        if price > self.high:
            return (price - self.high) / price * 100
        elif price < self.low:
            return (self.low - price) / price * 100
        else:
            return 0.0  # 價格在 OB 內


class OrderBlockDetector:
    """Order Block 檢測器"""

    def __init__(self, lookback: int = 20, min_size_pct: float = 0.002):
        """
        初始化 Order Block 檢測器

        Args:
            lookback: 回看期數
            min_size_pct: OB 最小大小（百分比，默認 0.2%）
        """
        self.lookback = lookback
        self.min_size_pct = min_size_pct
        self.swing_window = 5  # Swing point 識別窗口

    def identify_swing_points(self, df: pd.DataFrame) -> tuple:
        """識別 Swing Highs 和 Swing Lows"""
        highs = df['high'].values
        lows = df['low'].values
        n = len(df)
        w = self.swing_window

        swing_high_indices = []
        swing_low_indices = []

        for i in range(w, n - w):
            # Swing High
            if highs[i] == max(highs[i-w:i+w+1]):
                swing_high_indices.append(i)

            # Swing Low
            if lows[i] == min(lows[i-w:i+w+1]):
                swing_low_indices.append(i)

        return swing_high_indices, swing_low_indices

    def detect_order_blocks(self, df: pd.DataFrame) -> tuple:
        """
        檢測所有 Order Blocks

        邏輯：
        1. 找到所有 swing lows (潛在看漲 OB)
        2. 找到所有 swing highs (潛在看跌 OB)
        3. 在反轉前找到最後一根相反方向的 K 線作為 OB

        Args:
            df: OHLC 數據

        Returns:
            (bullish_obs, bearish_obs): 看漲和看跌 OB 列表
        """
        swing_high_indices, swing_low_indices = self.identify_swing_points(df)

        bullish_obs = []
        bearish_obs = []

        # 檢測看漲 Order Blocks（在 swing low 之前）
        for swing_idx in swing_low_indices:
            # 在 swing low 之前尋找最後一根看跌 K 線
            for i in range(swing_idx - 1, max(0, swing_idx - 10), -1):
                if df['close'].iloc[i] < df['open'].iloc[i]:  # 看跌 K 線
                    ob_high = df['high'].iloc[i]
                    ob_low = df['low'].iloc[i]
                    ob_size = (ob_high - ob_low) / ob_low

                    # 檢查 OB 大小是否符合要求
                    if ob_size >= self.min_size_pct:
                        ob = OrderBlock(
                            bar_index=i,
                            high=ob_high,
                            low=ob_low,
                            direction='bullish',
                            timestamp=df.index[i]
                        )
                        bullish_obs.append(ob)
                        break

        # 檢測看跌 Order Blocks（在 swing high 之前）
        for swing_idx in swing_high_indices:
            # 在 swing high 之前尋找最後一根看漲 K 線
            for i in range(swing_idx - 1, max(0, swing_idx - 10), -1):
                if df['close'].iloc[i] > df['open'].iloc[i]:  # 看漲 K 線
                    ob_high = df['high'].iloc[i]
                    ob_low = df['low'].iloc[i]
                    ob_size = (ob_high - ob_low) / ob_low

                    # 檢查 OB 大小是否符合要求
                    if ob_size >= self.min_size_pct:
                        ob = OrderBlock(
                            bar_index=i,
                            high=ob_high,
                            low=ob_low,
                            direction='bearish',
                            timestamp=df.index[i]
                        )
                        bearish_obs.append(ob)
                        break

        return bullish_obs, bearish_obs

    def get_nearest_obs(self, df: pd.DataFrame, current_idx: int,
                       bullish_obs: List[OrderBlock],
                       bearish_obs: List[OrderBlock]) -> tuple:
        """
        獲取最近的有效 Order Blocks

        Args:
            df: OHLC 數據
            current_idx: 當前索引
            bullish_obs: 看漲 OB 列表
            bearish_obs: 看跌 OB 列表

        Returns:
            (nearest_bullish_ob, nearest_bearish_ob)
        """
        current_price = df['close'].iloc[current_idx]

        # 過濾：只保留當前位置之前的 OB
        valid_bullish = [ob for ob in bullish_obs if ob.bar_index < current_idx]
        valid_bearish = [ob for ob in bearish_obs if ob.bar_index < current_idx]

        # 找到最近的看漲 OB
        nearest_bullish = None
        min_bullish_dist = float('inf')
        for ob in valid_bullish:
            dist = abs(ob.get_distance(current_price))
            if dist < min_bullish_dist:
                min_bullish_dist = dist
                nearest_bullish = ob

        # 找到最近的看跌 OB
        nearest_bearish = None
        min_bearish_dist = float('inf')
        for ob in valid_bearish:
            dist = abs(ob.get_distance(current_price))
            if dist < min_bearish_dist:
                min_bearish_dist = dist
                nearest_bearish = ob

        return nearest_bullish, nearest_bearish

    def calculate_features(self, df: pd.DataFrame, current_idx: int) -> dict:
        """
        計算當前位置的 Order Block 特徵

        Args:
            df: OHLC 數據
            current_idx: 當前索引

        Returns:
            dict: 包含 4 個特徵的字典
            {
                'dist_to_bullish_ob': float,    # 距離看漲 OB 的百分比
                'dist_to_bearish_ob': float,    # 距離看跌 OB 的百分比
                'in_bullish_ob': int,           # 是否在看漲 OB 內 (0/1)
                'in_bearish_ob': int            # 是否在看跌 OB 內 (0/1)
            }
        """
        # 獲取回看數據
        lookback_start = max(0, current_idx - self.lookback)
        df_lookback = df.iloc[lookback_start:current_idx+1].copy()

        if len(df_lookback) < 20:
            # 數據不足
            return {
                'dist_to_bullish_ob': 10.0,  # 默認遠離 OB
                'dist_to_bearish_ob': 10.0,
                'in_bullish_ob': 0,
                'in_bearish_ob': 0
            }

        # 檢測 Order Blocks
        bullish_obs, bearish_obs = self.detect_order_blocks(df_lookback)

        # 獲取最近的 OB
        nearest_bullish, nearest_bearish = self.get_nearest_obs(
            df_lookback, len(df_lookback) - 1, bullish_obs, bearish_obs
        )

        current_price = df.iloc[current_idx]['close']

        # 計算特徵
        features = {
            'dist_to_bullish_ob': 10.0,  # 默認值
            'dist_to_bearish_ob': 10.0,
            'in_bullish_ob': 0,
            'in_bearish_ob': 0
        }

        if nearest_bullish:
            features['dist_to_bullish_ob'] = nearest_bullish.get_distance(current_price)
            features['in_bullish_ob'] = 1 if nearest_bullish.is_inside(current_price) else 0

        if nearest_bearish:
            features['dist_to_bearish_ob'] = nearest_bearish.get_distance(current_price)
            features['in_bearish_ob'] = 1 if nearest_bearish.is_inside(current_price) else 0

        return features

    def analyze_full_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """分析整個數據集的 Order Blocks"""
        print(f"🔍 分析 Order Blocks...")
        print(f"   數據範圍: {df.index[0]} 到 {df.index[-1]}")
        print(f"   總 K 線數: {len(df):,}")

        # 檢測所有 OB
        bullish_obs, bearish_obs = self.detect_order_blocks(df)

        print(f"   檢測到看漲 OB: {len(bullish_obs)}")
        print(f"   檢測到看跌 OB: {len(bearish_obs)}")
        print(f"✅ Order Blocks 分析完成！\n")

        # 創建結果 DataFrame
        result = df.copy()
        result['dist_to_bullish_ob'] = 10.0
        result['dist_to_bearish_ob'] = 10.0
        result['in_bullish_ob'] = 0
        result['in_bearish_ob'] = 0

        # 計算每個位置的特徵
        for i in range(len(df)):
            features = self.calculate_features(df, i)
            result['dist_to_bullish_ob'].iloc[i] = features['dist_to_bullish_ob']
            result['dist_to_bearish_ob'].iloc[i] = features['dist_to_bearish_ob']
            result['in_bullish_ob'].iloc[i] = features['in_bullish_ob']
            result['in_bearish_ob'].iloc[i] = features['in_bearish_ob']

        return result


def test_order_blocks():
    """測試 Order Blocks 模塊"""
    print("=" * 60)
    print("  🧪 測試 Order Blocks 檢測模塊")
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
    ob_detector = OrderBlockDetector(lookback=50, min_size_pct=0.002)

    # 分析整個測試數據集
    result = ob_detector.analyze_full_dataset(df_test)

    # 顯示統計
    print("📊 Order Block 統計:")
    in_bullish_count = result['in_bullish_ob'].sum()
    in_bearish_count = result['in_bearish_ob'].sum()
    print(f"   價格在看漲 OB 內: {in_bullish_count} 次")
    print(f"   價格在看跌 OB 內: {in_bearish_count} 次")

    avg_bullish_dist = result['dist_to_bullish_ob'].mean()
    avg_bearish_dist = result['dist_to_bearish_ob'].mean()
    print(f"\n   平均距離看漲 OB: {avg_bullish_dist:.2f}%")
    print(f"   平均距離看跌 OB: {avg_bearish_dist:.2f}%")

    print("\n✅ 測試完成！\n")

    # 測試單點特徵計算
    print("🎯 測試單點特徵計算（最後一根 K 線）:")
    features = ob_detector.calculate_features(df_test, len(df_test) - 1)
    print(f"   距離看漲 OB: {features['dist_to_bullish_ob']:.2f}%")
    print(f"   距離看跌 OB: {features['dist_to_bearish_ob']:.2f}%")
    print(f"   在看漲 OB 內: {features['in_bullish_ob']}")
    print(f"   在看跌 OB 內: {features['in_bearish_ob']}")

    return result


if __name__ == "__main__":
    result = test_order_blocks()
