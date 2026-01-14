"""
Market Structure 檢測模塊

實現 ICT (Inner Circle Trader) 市場結構概念：
1. 趨勢識別（上升/下降/震盪）
2. BOS (Break of Structure) - 趨勢延續
3. ChoCh (Change of Character) - 趨勢轉變

作者：PPO Trading Team
日期：2026-01-14
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


class MarketStructure:
    """
    市場結構分析器

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

        # 狀態追蹤
        self.current_trend = 0  # -1: 下降, 0: 震盪, 1: 上升
        self.last_structure_change_bar = 0  # 上次結構改變的 K 線位置
        self.last_swing_high = None
        self.last_swing_low = None

    def identify_swing_points(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        識別 Swing Highs 和 Swing Lows

        Swing High: 當前高點高於前後 N 根 K 線的高點
        Swing Low: 當前低點低於前後 N 根 K 線的低點

        Args:
            df: OHLC 數據，包含 'high' 和 'low' 列

        Returns:
            (swing_highs, swing_lows): 兩個 Series，非 swing point 的位置為 NaN
        """
        highs = df['high'].values
        lows = df['low'].values
        n = len(df)
        w = self.swing_window

        # 初始化結果陣列
        swing_highs = np.full(n, np.nan)
        swing_lows = np.full(n, np.nan)

        # 識別 swing points（需要前後各有 w 根 K 線）
        for i in range(w, n - w):
            # Swing High: 當前高點是局部最高點
            if highs[i] == max(highs[i-w:i+w+1]):
                swing_highs[i] = highs[i]

            # Swing Low: 當前低點是局部最低點
            if lows[i] == min(lows[i-w:i+w+1]):
                swing_lows[i] = lows[i]

        return pd.Series(swing_highs, index=df.index), pd.Series(swing_lows, index=df.index)

    def detect_bos_choch(self, df: pd.DataFrame, swing_highs: pd.Series,
                         swing_lows: pd.Series) -> pd.DataFrame:
        """
        檢測 BOS (Break of Structure) 和 ChoCh (Change of Character)

        BOS (趨勢延續):
        - 上升趨勢中，價格突破前一個 swing high
        - 下降趨勢中，價格突破前一個 swing low

        ChoCh (趨勢轉變):
        - 上升趨勢中，價格跌破前一個 swing low
        - 下降趨勢中，價格突破前一個 swing high

        Args:
            df: OHLC 數據
            swing_highs: Swing highs series
            swing_lows: Swing lows series

        Returns:
            DataFrame 包含：
            - trend_state: 趨勢狀態 (-1/0/1)
            - structure_signal: 結構信號 (-1: Bearish ChoCh, 0: 無, 1: Bullish BOS)
            - bars_since_change: 距離上次結構改變的 K 線數
        """
        n = len(df)

        # 初始化結果陣列
        trend_state = np.zeros(n)  # 趨勢狀態
        structure_signal = np.zeros(n)  # 結構信號
        bars_since_change = np.zeros(n)  # 距離上次改變的 K 線數

        # 獲取所有有效的 swing points
        valid_swing_highs = swing_highs.dropna()
        valid_swing_lows = swing_lows.dropna()

        if len(valid_swing_highs) < 2 or len(valid_swing_lows) < 2:
            # 數據不足，返回中性狀態
            return pd.DataFrame({
                'trend_state': trend_state,
                'structure_signal': structure_signal,
                'bars_since_change': bars_since_change
            }, index=df.index)

        # 追蹤變量
        current_trend = 0  # 初始為震盪
        last_change_idx = 0
        last_high = valid_swing_highs.iloc[0]
        last_low = valid_swing_lows.iloc[0]

        # 遍歷所有 K 線
        for i in range(n):
            current_price = df['close'].iloc[i]
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]

            # 更新最近的 swing points
            if not np.isnan(swing_highs.iloc[i]):
                last_high = swing_highs.iloc[i]
            if not np.isnan(swing_lows.iloc[i]):
                last_low = swing_lows.iloc[i]

            # 檢測結構變化
            if current_trend >= 0:  # 上升趨勢或震盪
                # 檢查是否突破前高 (BOS - 看漲)
                if current_high > last_high and i > 0:
                    structure_signal[i] = 1  # Bullish BOS
                    current_trend = 1  # 確認上升趨勢
                    last_change_idx = i

                # 檢查是否跌破前低 (ChoCh - 看跌)
                elif current_low < last_low and i > 0:
                    structure_signal[i] = -1  # Bearish ChoCh
                    current_trend = -1  # 轉為下降趨勢
                    last_change_idx = i

            if current_trend <= 0:  # 下降趨勢或震盪
                # 檢查是否跌破前低 (BOS - 看跌)
                if current_low < last_low and i > 0:
                    structure_signal[i] = -1  # Bearish BOS
                    current_trend = -1  # 確認下降趨勢
                    last_change_idx = i

                # 檢查是否突破前高 (ChoCh - 看漲)
                elif current_high > last_high and i > 0:
                    structure_signal[i] = 1  # Bullish ChoCh
                    current_trend = 1  # 轉為上升趨勢
                    last_change_idx = i

            # 更新狀態
            trend_state[i] = current_trend
            bars_since_change[i] = i - last_change_idx

        return pd.DataFrame({
            'trend_state': trend_state,
            'structure_signal': structure_signal,
            'bars_since_change': bars_since_change
        }, index=df.index)

    def calculate_features(self, df: pd.DataFrame, current_idx: int) -> dict:
        """
        計算當前位置的市場結構特徵（用於強化學習環境）

        Args:
            df: OHLC 數據
            current_idx: 當前 K 線索引

        Returns:
            dict: 包含 3 個特徵的字典
            {
                'trend_state': int,           # -1/0/1
                'structure_signal': int,      # -1/0/1
                'bars_since_structure_change': int  # 距離上次改變的 K 線數
            }
        """
        # 獲取當前位置之前的數據（避免未來洩漏）
        lookback_start = max(0, current_idx - self.lookback)
        df_lookback = df.iloc[lookback_start:current_idx+1].copy()

        if len(df_lookback) < self.swing_window * 2:
            # 數據不足，返回中性狀態
            return {
                'trend_state': 0,
                'structure_signal': 0,
                'bars_since_structure_change': 0
            }

        # 識別 swing points
        swing_highs, swing_lows = self.identify_swing_points(df_lookback)

        # 檢測 BOS/ChoCh
        structure_df = self.detect_bos_choch(df_lookback, swing_highs, swing_lows)

        # 返回當前位置的特徵
        return {
            'trend_state': int(structure_df['trend_state'].iloc[-1]),
            'structure_signal': int(structure_df['structure_signal'].iloc[-1]),
            'bars_since_structure_change': int(structure_df['bars_since_change'].iloc[-1])
        }

    def analyze_full_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        分析整個數據集的市場結構（用於回測或視覺化）

        Args:
            df: 完整的 OHLC 數據

        Returns:
            DataFrame: 包含市場結構特徵的完整數據
        """
        print(f"🔍 分析市場結構...")
        print(f"   數據範圍: {df.index[0]} 到 {df.index[-1]}")
        print(f"   總 K 線數: {len(df):,}")

        # 識別 swing points
        swing_highs, swing_lows = self.identify_swing_points(df)
        print(f"   發現 Swing Highs: {swing_highs.notna().sum()}")
        print(f"   發現 Swing Lows: {swing_lows.notna().sum()}")

        # 檢測 BOS/ChoCh
        structure_df = self.detect_bos_choch(df, swing_highs, swing_lows)

        # 統計
        bos_count = (structure_df['structure_signal'] == 1).sum()
        choch_count = (structure_df['structure_signal'] == -1).sum()

        print(f"   檢測到 Bullish BOS: {bos_count}")
        print(f"   檢測到 Bearish ChoCh: {choch_count}")
        print(f"✅ 市場結構分析完成！\n")

        # 合併結果
        result = df.copy()
        result['swing_high'] = swing_highs
        result['swing_low'] = swing_lows
        result['trend_state'] = structure_df['trend_state']
        result['structure_signal'] = structure_df['structure_signal']
        result['bars_since_structure_change'] = structure_df['bars_since_change']

        return result


def test_market_structure():
    """測試 Market Structure 模塊"""
    print("=" * 60)
    print("  🧪 測試 Market Structure 檢測模塊")
    print("=" * 60)

    # 載入數據
    import sys
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
    ms = MarketStructure(lookback=50)

    # 分析整個測試數據集
    result = ms.analyze_full_dataset(df_test)

    # 顯示統計
    print("📊 趨勢分布:")
    trend_counts = result['trend_state'].value_counts().sort_index()
    for trend, count in trend_counts.items():
        trend_name = {-1: "下降趨勢", 0: "震盪", 1: "上升趨勢"}[trend]
        print(f"   {trend_name}: {count} 根 K 線 ({count/len(result)*100:.1f}%)")

    print("\n📊 結構信號統計:")
    signal_counts = result['structure_signal'].value_counts().sort_index()
    for signal, count in signal_counts.items():
        signal_name = {-1: "Bearish ChoCh", 0: "無信號", 1: "Bullish BOS"}[signal]
        print(f"   {signal_name}: {count}")

    print("\n✅ 測試完成！\n")

    # 測試單點特徵計算
    print("🎯 測試單點特徵計算（最後一根 K 線）:")
    features = ms.calculate_features(df_test, len(df_test) - 1)
    print(f"   趨勢狀態: {features['trend_state']}")
    print(f"   結構信號: {features['structure_signal']}")
    print(f"   距離上次改變: {features['bars_since_structure_change']} 根 K 線")

    return result


if __name__ == "__main__":
    # 執行測試
    result = test_market_structure()
