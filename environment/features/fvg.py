"""
Fair Value Gaps (FVG) 檢測模塊

實現 ICT Fair Value Gap 概念：
- FVG 是三根 K 線之間的價格缺口
- 看漲 FVG: 第一根 K 線的高點 < 第三根 K 線的低點
- 看跌 FVG: 第一根 K 線的低點 > 第三根 K 線的高點
- FVG 通常會被"填補"（價格回到缺口區域）

作者：PPO Trading Team
日期：2026-01-14
"""

import numpy as np
import pandas as pd
from typing import List, Optional


class FairValueGap:
    """Fair Value Gap 數據結構"""

    def __init__(self, start_idx: int, high: float, low: float,
                 direction: str, timestamp: pd.Timestamp):
        """
        初始化 FVG

        Args:
            start_idx: 起始 K 線索引
            high: FVG 區域高點
            low: FVG 區域低點
            direction: 'bullish' 或 'bearish'
            timestamp: 時間戳
        """
        self.start_idx = start_idx
        self.high = high
        self.low = low
        self.direction = direction
        self.timestamp = timestamp
        self.is_filled = False  # 是否已填補
        self.fill_idx = None  # 填補位置

    def is_inside(self, price: float) -> bool:
        """檢查價格是否在 FVG 區域內"""
        return self.low <= price <= self.high

    def check_filled(self, high: float, low: float) -> bool:
        """檢查 FVG 是否被填補"""
        # FVG 被認為已填補，如果價格進入 FVG 區域
        if self.direction == 'bullish':
            # 看漲 FVG：價格回落到 FVG 區域
            return low <= self.high
        else:
            # 看跌 FVG：價格回升到 FVG 區域
            return high >= self.low


class FVGDetector:
    """Fair Value Gap 檢測器"""

    def __init__(self, min_size_pct: float = 0.001, max_age: int = 100):
        """
        初始化 FVG 檢測器

        Args:
            min_size_pct: FVG 最小大小（百分比，默認 0.1%）
            max_age: FVG 最大保留期數（默認 100 根 K 線）
        """
        self.min_size_pct = min_size_pct
        self.max_age = max_age

    def detect_fvgs(self, df: pd.DataFrame) -> tuple:
        """
        檢測所有 Fair Value Gaps

        邏輯：
        - 看漲 FVG: 三根 K 線中，第一根的 high < 第三根的 low
        - 看跌 FVG: 三根 K 線中，第一根的 low > 第三根的 high

        Args:
            df: OHLC 數據

        Returns:
            (bullish_fvgs, bearish_fvgs): 看漲和看跌 FVG 列表
        """
        bullish_fvgs = []
        bearish_fvgs = []

        # 遍歷數據，檢查每三根 K 線
        for i in range(len(df) - 2):
            bar1_high = df['high'].iloc[i]
            bar1_low = df['low'].iloc[i]
            bar3_high = df['high'].iloc[i + 2]
            bar3_low = df['low'].iloc[i + 2]

            # 檢測看漲 FVG
            if bar1_high < bar3_low:
                gap_size = (bar3_low - bar1_high) / bar1_high
                if gap_size >= self.min_size_pct:
                    fvg = FairValueGap(
                        start_idx=i,
                        high=bar3_low,
                        low=bar1_high,
                        direction='bullish',
                        timestamp=df.index[i]
                    )
                    bullish_fvgs.append(fvg)

            # 檢測看跌 FVG
            if bar1_low > bar3_high:
                gap_size = (bar1_low - bar3_high) / bar3_high
                if gap_size >= self.min_size_pct:
                    fvg = FairValueGap(
                        start_idx=i,
                        high=bar1_low,
                        low=bar3_high,
                        direction='bearish',
                        timestamp=df.index[i]
                    )
                    bearish_fvgs.append(fvg)

        return bullish_fvgs, bearish_fvgs

    def update_fvg_status(self, df: pd.DataFrame,
                         bullish_fvgs: List[FairValueGap],
                         bearish_fvgs: List[FairValueGap]) -> None:
        """
        更新 FVG 填補狀態

        Args:
            df: OHLC 數據
            bullish_fvgs: 看漲 FVG 列表
            bearish_fvgs: 看跌 FVG 列表
        """
        # 檢查每個 FVG 是否被填補
        for fvg in bullish_fvgs:
            if not fvg.is_filled:
                for i in range(fvg.start_idx + 3, len(df)):
                    if fvg.check_filled(df['high'].iloc[i], df['low'].iloc[i]):
                        fvg.is_filled = True
                        fvg.fill_idx = i
                        break

        for fvg in bearish_fvgs:
            if not fvg.is_filled:
                for i in range(fvg.start_idx + 3, len(df)):
                    if fvg.check_filled(df['high'].iloc[i], df['low'].iloc[i]):
                        fvg.is_filled = True
                        fvg.fill_idx = i
                        break

    def get_nearest_unfilled_fvg(self, current_idx: int,
                                 bullish_fvgs: List[FairValueGap],
                                 bearish_fvgs: List[FairValueGap],
                                 df: pd.DataFrame) -> tuple:
        """
        獲取最近的未填補 FVG

        Args:
            current_idx: 當前索引
            bullish_fvgs: 看漲 FVG 列表
            bearish_fvgs: 看跌 FVG 列表
            df: OHLC 數據

        Returns:
            (nearest_bullish_fvg, nearest_bearish_fvg)
        """
        current_price = df['close'].iloc[current_idx]

        # 過濾：只保留未填補且在有效期內的 FVG
        valid_bullish = [
            fvg for fvg in bullish_fvgs
            if not fvg.is_filled
            and fvg.start_idx < current_idx
            and (current_idx - fvg.start_idx) <= self.max_age
        ]

        valid_bearish = [
            fvg for fvg in bearish_fvgs
            if not fvg.is_filled
            and fvg.start_idx < current_idx
            and (current_idx - fvg.start_idx) <= self.max_age
        ]

        # 找到最近的看漲 FVG
        nearest_bullish = None
        min_dist = float('inf')
        for fvg in valid_bullish:
            dist = abs(current_price - (fvg.high + fvg.low) / 2)
            if dist < min_dist:
                min_dist = dist
                nearest_bullish = fvg

        # 找到最近的看跌 FVG
        nearest_bearish = None
        min_dist = float('inf')
        for fvg in valid_bearish:
            dist = abs(current_price - (fvg.high + fvg.low) / 2)
            if dist < min_dist:
                min_dist = dist
                nearest_bearish = fvg

        return nearest_bullish, nearest_bearish

    def calculate_features(self, df: pd.DataFrame, current_idx: int) -> dict:
        """
        計算當前位置的 FVG 特徵

        Args:
            df: OHLC 數據
            current_idx: 當前索引

        Returns:
            dict: 包含 3 個特徵的字典
            {
                'in_bullish_fvg': int,        # 是否在看漲 FVG 內 (0/1)
                'in_bearish_fvg': int,        # 是否在看跌 FVG 內 (0/1)
                'nearest_fvg_direction': int  # 最近未填補 FVG 方向 (-1/0/1)
            }
        """
        if current_idx < 10:
            # 數據不足
            return {
                'in_bullish_fvg': 0,
                'in_bearish_fvg': 0,
                'nearest_fvg_direction': 0
            }

        # 獲取回看數據
        lookback_start = max(0, current_idx - 200)
        df_lookback = df.iloc[lookback_start:current_idx+1].copy()

        # 檢測 FVG
        bullish_fvgs, bearish_fvgs = self.detect_fvgs(df_lookback)

        # 更新 FVG 狀態
        self.update_fvg_status(df_lookback, bullish_fvgs, bearish_fvgs)

        # 獲取最近的未填補 FVG
        nearest_bullish, nearest_bearish = self.get_nearest_unfilled_fvg(
            len(df_lookback) - 1, bullish_fvgs, bearish_fvgs, df_lookback
        )

        current_price = df.iloc[current_idx]['close']

        # 計算特徵
        features = {
            'in_bullish_fvg': 0,
            'in_bearish_fvg': 0,
            'nearest_fvg_direction': 0
        }

        # 檢查是否在 FVG 內
        if nearest_bullish and nearest_bullish.is_inside(current_price):
            features['in_bullish_fvg'] = 1

        if nearest_bearish and nearest_bearish.is_inside(current_price):
            features['in_bearish_fvg'] = 1

        # 確定最近 FVG 方向
        if nearest_bullish and not nearest_bearish:
            features['nearest_fvg_direction'] = 1
        elif nearest_bearish and not nearest_bullish:
            features['nearest_fvg_direction'] = -1
        elif nearest_bullish and nearest_bearish:
            # 兩者都存在，選擇更近的
            dist_bullish = abs(current_price - (nearest_bullish.high + nearest_bullish.low) / 2)
            dist_bearish = abs(current_price - (nearest_bearish.high + nearest_bearish.low) / 2)
            features['nearest_fvg_direction'] = 1 if dist_bullish < dist_bearish else -1

        return features

    def analyze_full_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """分析整個數據集的 FVG"""
        print(f"🔍 分析 Fair Value Gaps...")
        print(f"   數據範圍: {df.index[0]} 到 {df.index[-1]}")
        print(f"   總 K 線數: {len(df):,}")

        # 檢測所有 FVG
        bullish_fvgs, bearish_fvgs = self.detect_fvgs(df)

        # 更新填補狀態
        self.update_fvg_status(df, bullish_fvgs, bearish_fvgs)

        # 統計
        unfilled_bullish = sum(1 for fvg in bullish_fvgs if not fvg.is_filled)
        unfilled_bearish = sum(1 for fvg in bearish_fvgs if not fvg.is_filled)

        print(f"   檢測到看漲 FVG: {len(bullish_fvgs)} (未填補: {unfilled_bullish})")
        print(f"   檢測到看跌 FVG: {len(bearish_fvgs)} (未填補: {unfilled_bearish})")
        print(f"✅ Fair Value Gaps 分析完成！\n")

        # 創建結果 DataFrame
        result = df.copy()
        result['in_bullish_fvg'] = 0
        result['in_bearish_fvg'] = 0
        result['nearest_fvg_direction'] = 0

        # 計算每個位置的特徵
        for i in range(len(df)):
            features = self.calculate_features(df, i)
            result['in_bullish_fvg'].iloc[i] = features['in_bullish_fvg']
            result['in_bearish_fvg'].iloc[i] = features['in_bearish_fvg']
            result['nearest_fvg_direction'].iloc[i] = features['nearest_fvg_direction']

        return result


def test_fvg():
    """測試 FVG 模塊"""
    print("=" * 60)
    print("  🧪 測試 Fair Value Gaps 檢測模塊")
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
    fvg_detector = FVGDetector(min_size_pct=0.001, max_age=100)

    # 分析整個測試數據集
    result = fvg_detector.analyze_full_dataset(df_test)

    # 顯示統計
    print("📊 FVG 統計:")
    in_bullish_count = result['in_bullish_fvg'].sum()
    in_bearish_count = result['in_bearish_fvg'].sum()
    print(f"   價格在看漲 FVG 內: {in_bullish_count} 次")
    print(f"   價格在看跌 FVG 內: {in_bearish_count} 次")

    direction_counts = result['nearest_fvg_direction'].value_counts().sort_index()
    print(f"\n   最近 FVG 方向分布:")
    for direction, count in direction_counts.items():
        dir_name = {-1: "看跌", 0: "無", 1: "看漲"}[direction]
        print(f"      {dir_name}: {count}")

    print("\n✅ 測試完成！\n")

    # 測試單點特徵計算
    print("🎯 測試單點特徵計算（最後一根 K 線）:")
    features = fvg_detector.calculate_features(df_test, len(df_test) - 1)
    print(f"   在看漲 FVG 內: {features['in_bullish_fvg']}")
    print(f"   在看跌 FVG 內: {features['in_bearish_fvg']}")
    print(f"   最近 FVG 方向: {features['nearest_fvg_direction']}")

    return result


if __name__ == "__main__":
    result = test_fvg()
