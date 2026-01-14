"""
Volume & Price 特徵模塊

實現成交量與價格行為分析：
- Volume Ratio: 當前成交量 vs 平均成交量
- Price Momentum: 價格變化動量
- VWAP Momentum: 成交量加權價格動量
- Price Position in Range: 當前價格在波段中的位置
- Zone Classification: Premium/Discount/Equilibrium 區域分類

作者：PPO Trading Team
日期：2026-01-14
"""

import numpy as np
import pandas as pd
from typing import Tuple


class VolumeAnalyzer:
    """成交量與價格分析器"""

    def __init__(self, volume_window: int = 20, swing_window: int = 50):
        """
        初始化分析器

        Args:
            volume_window: 計算平均成交量的窗口
            swing_window: 識別波段高低點的窗口
        """
        self.volume_window = volume_window
        self.swing_window = swing_window

    def calculate_volume_ratio(self, df: pd.DataFrame, current_idx: int) -> float:
        """
        計算當前成交量比率

        Args:
            df: OHLCV 數據
            current_idx: 當前索引

        Returns:
            float: 當前成交量 / 平均成交量
        """
        if current_idx < self.volume_window:
            return 1.0

        current_volume = df['volume'].iloc[current_idx]
        avg_volume = df['volume'].iloc[current_idx - self.volume_window:current_idx].mean()

        if avg_volume == 0:
            return 1.0

        return current_volume / avg_volume

    def calculate_price_momentum(self, df: pd.DataFrame, current_idx: int,
                                 lookback: int = 10) -> float:
        """
        計算價格動量（正規化）

        Args:
            df: OHLCV 數據
            current_idx: 當前索引
            lookback: 回看期數

        Returns:
            float: 價格變化幅度（百分比）
        """
        if current_idx < lookback:
            return 0.0

        current_price = df['close'].iloc[current_idx]
        past_price = df['close'].iloc[current_idx - lookback]

        if past_price == 0:
            return 0.0

        momentum = (current_price - past_price) / past_price * 100
        return momentum

    def calculate_vwap(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> float:
        """
        計算 VWAP (Volume Weighted Average Price)

        Args:
            df: OHLCV 數據
            start_idx: 起始索引
            end_idx: 結束索引

        Returns:
            float: VWAP 價格
        """
        if start_idx >= end_idx:
            return df['close'].iloc[end_idx]

        df_slice = df.iloc[start_idx:end_idx+1]

        # 使用典型價格 (High + Low + Close) / 3
        typical_price = (df_slice['high'] + df_slice['low'] + df_slice['close']) / 3
        volume = df_slice['volume']

        total_volume = volume.sum()
        if total_volume == 0:
            return typical_price.mean()

        vwap = (typical_price * volume).sum() / total_volume
        return vwap

    def calculate_vwap_momentum(self, df: pd.DataFrame, current_idx: int,
                               vwap_window: int = 20) -> float:
        """
        計算 VWAP 動量

        Args:
            df: OHLCV 數據
            current_idx: 當前索引
            vwap_window: VWAP 計算窗口

        Returns:
            float: (當前價格 - VWAP) / VWAP * 100
        """
        if current_idx < vwap_window:
            return 0.0

        start_idx = max(0, current_idx - vwap_window)
        vwap = self.calculate_vwap(df, start_idx, current_idx)
        current_price = df['close'].iloc[current_idx]

        if vwap == 0:
            return 0.0

        momentum = (current_price - vwap) / vwap * 100
        return momentum

    def find_swing_range(self, df: pd.DataFrame, current_idx: int) -> Tuple[float, float]:
        """
        找到當前的波段高低點

        Args:
            df: OHLCV 數據
            current_idx: 當前索引

        Returns:
            (swing_high, swing_low): 波段高低點
        """
        lookback_start = max(0, current_idx - self.swing_window)
        df_range = df.iloc[lookback_start:current_idx+1]

        swing_high = df_range['high'].max()
        swing_low = df_range['low'].min()

        return swing_high, swing_low

    def calculate_price_position_in_range(self, df: pd.DataFrame,
                                         current_idx: int) -> float:
        """
        計算當前價格在波段中的位置

        Args:
            df: OHLCV 數據
            current_idx: 當前索引

        Returns:
            float: 0-100 的百分比（0 = 波段低點，100 = 波段高點）
        """
        if current_idx < 10:
            return 50.0

        swing_high, swing_low = self.find_swing_range(df, current_idx)
        current_price = df['close'].iloc[current_idx]

        if swing_high == swing_low:
            return 50.0

        position = (current_price - swing_low) / (swing_high - swing_low) * 100
        return np.clip(position, 0, 100)

    def classify_zone(self, price_position: float) -> int:
        """
        分類 Premium/Discount/Equilibrium 區域

        根據 ICT 理論：
        - Premium Zone (>= 61.8%): 價格在波段頂部，可能回調
        - Equilibrium (38.2% - 61.8%): 平衡區域
        - Discount Zone (<= 38.2%): 價格在波段底部，可能反彈

        Args:
            price_position: 價格在波段中的位置 (0-100)

        Returns:
            int: -1 (Discount), 0 (Equilibrium), 1 (Premium)
        """
        if price_position >= 61.8:
            return 1  # Premium
        elif price_position <= 38.2:
            return -1  # Discount
        else:
            return 0  # Equilibrium

    def calculate_features(self, df: pd.DataFrame, current_idx: int) -> dict:
        """
        計算當前位置的成交量與價格特徵

        Args:
            df: OHLCV 數據
            current_idx: 當前索引

        Returns:
            dict: 包含 5 個特徵的字典
            {
                'volume_ratio': float,           # 當前成交量 / 平均成交量
                'price_momentum': float,         # 價格動量（%）
                'vwap_momentum': float,          # VWAP 動量（%）
                'price_position_in_range': float, # 0-100
                'zone_classification': int       # -1/0/1
            }
        """
        if current_idx < self.volume_window:
            return {
                'volume_ratio': 1.0,
                'price_momentum': 0.0,
                'vwap_momentum': 0.0,
                'price_position_in_range': 50.0,
                'zone_classification': 0
            }

        # 計算各項特徵
        volume_ratio = self.calculate_volume_ratio(df, current_idx)
        price_momentum = self.calculate_price_momentum(df, current_idx, lookback=10)
        vwap_momentum = self.calculate_vwap_momentum(df, current_idx, vwap_window=20)
        price_position = self.calculate_price_position_in_range(df, current_idx)
        zone_class = self.classify_zone(price_position)

        return {
            'volume_ratio': volume_ratio,
            'price_momentum': price_momentum,
            'vwap_momentum': vwap_momentum,
            'price_position_in_range': price_position,
            'zone_classification': zone_class
        }

    def analyze_full_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """分析整個數據集的成交量與價格特徵"""
        print(f"🔍 分析 Volume & Price 特徵...")
        print(f"   數據範圍: {df.index[0]} 到 {df.index[-1]}")
        print(f"   總 K 線數: {len(df):,}")

        # 創建結果 DataFrame
        result = df.copy()
        result['volume_ratio'] = 1.0
        result['price_momentum'] = 0.0
        result['vwap_momentum'] = 0.0
        result['price_position_in_range'] = 50.0
        result['zone_classification'] = 0

        # 計算每個位置的特徵
        for i in range(len(df)):
            features = self.calculate_features(df, i)
            result['volume_ratio'].iloc[i] = features['volume_ratio']
            result['price_momentum'].iloc[i] = features['price_momentum']
            result['vwap_momentum'].iloc[i] = features['vwap_momentum']
            result['price_position_in_range'].iloc[i] = features['price_position_in_range']
            result['zone_classification'].iloc[i] = features['zone_classification']

        # 統計
        premium_count = (result['zone_classification'] == 1).sum()
        discount_count = (result['zone_classification'] == -1).sum()
        equilibrium_count = (result['zone_classification'] == 0).sum()

        print(f"   Premium Zone: {premium_count} ({premium_count/len(result)*100:.1f}%)")
        print(f"   Discount Zone: {discount_count} ({discount_count/len(result)*100:.1f}%)")
        print(f"   Equilibrium: {equilibrium_count} ({equilibrium_count/len(result)*100:.1f}%)")

        avg_volume_ratio = result['volume_ratio'].mean()
        print(f"\n   平均成交量比率: {avg_volume_ratio:.2f}")

        print(f"✅ Volume & Price 分析完成！\n")

        return result


def test_volume():
    """測試 Volume & Price 模塊"""
    print("=" * 60)
    print("  🧪 測試 Volume & Price 特徵模塊")
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
    volume_analyzer = VolumeAnalyzer(volume_window=20, swing_window=50)

    # 分析整個測試數據集
    result = volume_analyzer.analyze_full_dataset(df_test)

    # 顯示統計
    print("📊 Volume & Price 統計:")
    print(f"   平均價格動量: {result['price_momentum'].mean():.2f}%")
    print(f"   平均 VWAP 動量: {result['vwap_momentum'].mean():.2f}%")
    print(f"   平均價格位置: {result['price_position_in_range'].mean():.1f}")

    print("\n✅ 測試完成！\n")

    # 測試單點特徵計算
    print("🎯 測試單點特徵計算（最後一根 K 線）:")
    features = volume_analyzer.calculate_features(df_test, len(df_test) - 1)
    print(f"   成交量比率: {features['volume_ratio']:.2f}")
    print(f"   價格動量: {features['price_momentum']:.2f}%")
    print(f"   VWAP 動量: {features['vwap_momentum']:.2f}%")
    print(f"   價格位置: {features['price_position_in_range']:.1f}")
    zone_name = {-1: "Discount", 0: "Equilibrium", 1: "Premium"}[features['zone_classification']]
    print(f"   區域分類: {zone_name}")

    return result


if __name__ == "__main__":
    result = test_volume()
