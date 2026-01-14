"""
特徵整合模塊

將所有 ICT 特徵整合成完整的狀態向量供 RL 環境使用

整合的特徵類別：
1. Market Structure (3 features)
2. Order Blocks (4 features)
3. Fair Value Gaps (3 features)
4. Liquidity (3 features)
5. Volume & Price (5 features)
6. Multi-Timeframe (2 features)

總計：20 個特徵

作者：PPO Trading Team
日期：2026-01-14
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from pathlib import Path

# 導入所有特徵檢測器
try:
    # 相對導入（用於包內導入）
    from .market_structure import MarketStructure
    from .order_blocks import OrderBlockDetector
    from .fvg import FVGDetector
    from .liquidity import LiquidityDetector
    from .volume import VolumeAnalyzer
    from .multi_timeframe import MultiTimeframeAnalyzer
except ImportError:
    # 絕對導入（用於直接運行測試）
    from market_structure import MarketStructure
    from order_blocks import OrderBlockDetector
    from fvg import FVGDetector
    from liquidity import LiquidityDetector
    from volume import VolumeAnalyzer
    from multi_timeframe import MultiTimeframeAnalyzer


class FeatureAggregator:
    """特徵整合器 - 將所有 ICT 特徵組合成完整狀態向量"""

    def __init__(self, config: Dict = None):
        """
        初始化特徵整合器

        Args:
            config: 配置字典（可選）
        """
        # 使用默認配置或自定義配置
        if config is None:
            config = self._get_default_config()

        self.config = config

        # 初始化所有特徵檢測器
        self.market_structure = MarketStructure(
            lookback=config.get('market_structure_lookback', 50)
        )

        self.order_blocks = OrderBlockDetector(
            lookback=config.get('order_block_lookback', 20),
            min_size_pct=config.get('order_block_min_size', 0.002)
        )

        self.fvg_detector = FVGDetector(
            min_size_pct=config.get('fvg_min_size', 0.001),
            max_age=config.get('fvg_max_age', 100)
        )

        self.liquidity_detector = LiquidityDetector(
            lookback=config.get('liquidity_lookback', 50),
            sweep_threshold=config.get('liquidity_sweep_threshold', 0.001)
        )

        self.volume_analyzer = VolumeAnalyzer(
            volume_window=config.get('volume_window', 20),
            swing_window=config.get('volume_swing_window', 50)
        )

        self.mtf_analyzer = MultiTimeframeAnalyzer()

        # 特徵名稱列表（用於調試和記錄）
        self.feature_names = self._get_feature_names()

    def _get_default_config(self) -> Dict:
        """獲取默認配置"""
        return {
            'market_structure_lookback': 50,
            'order_block_lookback': 20,
            'order_block_min_size': 0.002,
            'fvg_min_size': 0.001,
            'fvg_max_age': 100,
            'liquidity_lookback': 50,
            'liquidity_sweep_threshold': 0.001,
            'volume_window': 20,
            'volume_swing_window': 50
        }

    def _get_feature_names(self) -> List[str]:
        """獲取所有特徵名稱"""
        return [
            # Market Structure (3)
            'trend_state',
            'structure_signal',
            'bars_since_structure_change',
            # Order Blocks (4)
            'dist_to_bullish_ob',
            'dist_to_bearish_ob',
            'in_bullish_ob',
            'in_bearish_ob',
            # Fair Value Gaps (3)
            'in_bullish_fvg',
            'in_bearish_fvg',
            'nearest_fvg_direction',
            # Liquidity (3)
            'liquidity_above',
            'liquidity_below',
            'liquidity_sweep',
            # Volume & Price (5)
            'volume_ratio',
            'price_momentum',
            'vwap_momentum',
            'price_position_in_range',
            'zone_classification',
            # Multi-Timeframe (2)
            'trend_5m',
            'trend_15m'
        ]

    def get_state_vector(self, df: pd.DataFrame, current_idx: int) -> np.ndarray:
        """
        計算當前位置的完整狀態向量

        Args:
            df: OHLCV 數據
            current_idx: 當前索引

        Returns:
            np.ndarray: 20 維狀態向量
        """
        # 收集所有特徵
        features = {}

        # 1. Market Structure
        ms_features = self.market_structure.calculate_features(df, current_idx)
        features.update(ms_features)

        # 2. Order Blocks
        ob_features = self.order_blocks.calculate_features(df, current_idx)
        features.update(ob_features)

        # 3. Fair Value Gaps
        fvg_features = self.fvg_detector.calculate_features(df, current_idx)
        features.update(fvg_features)

        # 4. Liquidity
        liq_features = self.liquidity_detector.calculate_features(df, current_idx)
        features.update(liq_features)

        # 5. Volume & Price
        vol_features = self.volume_analyzer.calculate_features(df, current_idx)
        features.update(vol_features)

        # 6. Multi-Timeframe
        mtf_features = self.mtf_analyzer.calculate_features(df, current_idx)
        features.update(mtf_features)

        # 按照特徵名稱順序組裝成向量
        state_vector = np.array([features[name] for name in self.feature_names], dtype=np.float32)

        return state_vector

    def get_feature_dict(self, df: pd.DataFrame, current_idx: int) -> Dict:
        """
        獲取當前位置的完整特徵字典（用於調試和分析）

        Args:
            df: OHLCV 數據
            current_idx: 當前索引

        Returns:
            Dict: 包含所有特徵的字典
        """
        features = {}

        # 收集所有特徵
        features.update(self.market_structure.calculate_features(df, current_idx))
        features.update(self.order_blocks.calculate_features(df, current_idx))
        features.update(self.fvg_detector.calculate_features(df, current_idx))
        features.update(self.liquidity_detector.calculate_features(df, current_idx))
        features.update(self.volume_analyzer.calculate_features(df, current_idx))
        features.update(self.mtf_analyzer.calculate_features(df, current_idx))

        return features

    def get_state_dimension(self) -> int:
        """獲取狀態空間維度"""
        return len(self.feature_names)

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        正規化狀態向量（可選，目前不使用）

        大多數特徵已經在各自的檢測器中進行了正規化或標準化
        此方法保留用於未來可能的全局正規化需求

        Args:
            state: 原始狀態向量

        Returns:
            np.ndarray: 正規化後的狀態向量
        """
        # 目前直接返回原始狀態
        return state


def test_feature_aggregator():
    """測試特徵整合器"""
    print("=" * 60)
    print("  🧪 測試特徵整合器")
    print("=" * 60)

    # 載入數據
    project_root = Path(__file__).parent.parent.parent

    print("\n📂 載入測試數據...")
    data_path = project_root / "data" / "raw" / "BTCUSDT_1m_train_latest.csv"
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # 使用最近 1000 根 K 線進行測試
    df_test = df.iloc[-1000:].copy()
    print(f"   測試數據: {len(df_test)} 根 K 線")
    print(f"   時間範圍: {df_test.index[0]} 到 {df_test.index[-1]}\n")

    # 初始化特徵整合器
    print("🔧 初始化特徵整合器...")
    aggregator = FeatureAggregator()
    print(f"   狀態空間維度: {aggregator.get_state_dimension()}")
    print(f"   特徵列表:")
    for i, name in enumerate(aggregator.feature_names):
        print(f"      {i+1}. {name}")

    # 測試單點特徵提取
    print("\n🎯 測試單點特徵提取（最後一根 K 線）:")
    test_idx = len(df_test) - 1

    # 獲取特徵字典
    features_dict = aggregator.get_feature_dict(df_test, test_idx)
    print("\n   特徵字典:")
    for category, start_idx in [
        ("Market Structure", 0),
        ("Order Blocks", 3),
        ("Fair Value Gaps", 7),
        ("Liquidity", 10),
        ("Volume & Price", 13),
        ("Multi-Timeframe", 18)
    ]:
        print(f"\n   {category}:")
        feature_subset = list(aggregator.feature_names[start_idx:start_idx+10])
        for fname in feature_subset:
            if fname in features_dict:
                print(f"      {fname}: {features_dict[fname]}")

    # 獲取狀態向量
    print("\n📊 狀態向量:")
    state_vector = aggregator.get_state_vector(df_test, test_idx)
    print(f"   形狀: {state_vector.shape}")
    print(f"   數據類型: {state_vector.dtype}")
    print(f"   向量內容: {state_vector}")

    # 驗證狀態向量與特徵字典一致性
    print("\n✅ 驗證狀態向量與特徵字典一致性:")
    all_match = True
    for i, fname in enumerate(aggregator.feature_names):
        if not np.isclose(state_vector[i], features_dict[fname], atol=1e-5):
            print(f"   ❌ 不匹配: {fname} - 向量值 {state_vector[i]}, 字典值 {features_dict[fname]}")
            all_match = False

    if all_match:
        print("   ✅ 所有特徵值匹配！")

    # 測試多個時間點
    print("\n📈 測試多個時間點的狀態向量生成:")
    test_indices = [100, 300, 500, 700, 900]
    for idx in test_indices:
        state = aggregator.get_state_vector(df_test, idx)
        print(f"   索引 {idx}: 狀態向量形狀 {state.shape}, 均值 {state.mean():.3f}, 標準差 {state.std():.3f}")

    print("\n✅ 特徵整合器測試完成！")
    print("\n🎉 所有 20 個 ICT 特徵已成功整合！")


if __name__ == "__main__":
    test_feature_aggregator()
