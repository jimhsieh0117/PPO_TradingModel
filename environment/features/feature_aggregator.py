"""
特徵整合模塊（向量化優化版）

將所有 ICT 特徵整合成完整的狀態向量供 RL 環境使用

整合的特徵類別：
1. Market Structure (3 features)
2. Order Blocks (4 features)
3. Fair Value Gaps (3 features)
4. Liquidity (3 features)
5. Volume & Price (5 features)
6. Multi-Timeframe (2 features)

總計：20 個特徵

優化：
- 所有模塊都支持向量化預計算
- 預計算後查詢 O(1)
- 使用 NumPy 數組存儲特徵緩存

作者：PPO Trading Team
日期：2026-01-14
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from pathlib import Path
from tqdm import tqdm

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
    """特徵整合器 - 將所有 ICT 特徵組合成完整狀態向量（向量化優化版）"""

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

        # === 優化：特徵緩存 ===
        self._cache_valid = False
        self._feature_cache = None  # [n_steps, 20] numpy array

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
            'trend_15m',
            # Volatility (1)
            'atr_normalized',
            # Time (2)
            'hour_sin',
            'hour_cos'
        ]

    def precompute_all_features(self, df: pd.DataFrame, verbose: bool = True) -> None:
        """
        向量化預計算整個數據集的所有特徵

        優化版本：
        - 每個模塊獨立向量化預計算
        - 最後組裝成完整緩存數組
        - 時間複雜度從 O(n × L²) 降低到 O(n)

        Args:
            df: 完整的 OHLCV 數據
            verbose: 是否顯示進度
        """
        n = len(df)
        n_features = len(self.feature_names)
        self._feature_cache = np.zeros((n, n_features), dtype=np.float32)

        if verbose:
            print(f"[FeatureAggregator] Precomputing {n:,} steps x {n_features} features...")

        # 1. 預計算 Multi-Timeframe（最重要的優化）
        if verbose:
            print("   [1/6] Multi-Timeframe...")
        self.mtf_analyzer.precompute_all_features(df)

        # 2. 預計算 Market Structure
        if verbose:
            print("   [2/6] Market Structure...")
        self.market_structure.precompute_all_features(df)

        # 3. 預計算 Order Blocks
        if verbose:
            print("   [3/6] Order Blocks...")
        self.order_blocks.precompute_all_features(df)

        # 4. 預計算 FVG
        if verbose:
            print("   [4/6] Fair Value Gaps...")
        self.fvg_detector.precompute_all_features(df)

        # 5. 預計算 Liquidity
        if verbose:
            print("   [5/6] Liquidity...")
        self.liquidity_detector.precompute_all_features(df)

        # 6. 預計算 Volume
        if verbose:
            print("   [6/6] Volume & Price...")
        self.volume_analyzer.precompute_all_features(df)

        # 7. 組裝所有特徵到緩存數組
        if verbose:
            print("   Assembling feature cache...")

        # 使用向量化方式組裝（避免逐點循環）
        # Market Structure (3 features)
        self._feature_cache[:, 0] = self.market_structure._trend_cache
        self._feature_cache[:, 1] = self.market_structure._signal_cache
        self._feature_cache[:, 2] = self.market_structure._bars_since_cache

        # Order Blocks (4 features)
        self._feature_cache[:, 3] = self.order_blocks._dist_bullish_cache
        self._feature_cache[:, 4] = self.order_blocks._dist_bearish_cache
        self._feature_cache[:, 5] = self.order_blocks._in_bullish_cache
        self._feature_cache[:, 6] = self.order_blocks._in_bearish_cache

        # FVG (3 features)
        self._feature_cache[:, 7] = self.fvg_detector._in_bullish_cache
        self._feature_cache[:, 8] = self.fvg_detector._in_bearish_cache
        self._feature_cache[:, 9] = self.fvg_detector._nearest_dir_cache

        # Liquidity (3 features)
        self._feature_cache[:, 10] = self.liquidity_detector._liq_above_cache
        self._feature_cache[:, 11] = self.liquidity_detector._liq_below_cache
        self._feature_cache[:, 12] = self.liquidity_detector._liq_sweep_cache

        # Volume & Price (5 features)
        self._feature_cache[:, 13] = self.volume_analyzer._volume_ratio_cache
        self._feature_cache[:, 14] = self.volume_analyzer._price_momentum_cache
        self._feature_cache[:, 15] = self.volume_analyzer._vwap_momentum_cache
        self._feature_cache[:, 16] = self.volume_analyzer._price_position_cache
        self._feature_cache[:, 17] = self.volume_analyzer._zone_class_cache

        # Multi-Timeframe (2 features)
        self._feature_cache[:, 18] = self.mtf_analyzer._trend_5m_cache
        self._feature_cache[:, 19] = self.mtf_analyzer._trend_15m_cache

        # ATR Normalized (1 feature)
        self._feature_cache[:, 20] = self.volume_analyzer._atr_normalized_cache

        # Time features (2 features) — 從 DatetimeIndex 計算
        hours = df.index.hour + df.index.minute / 60.0  # fractional hour
        self._feature_cache[:, 21] = np.sin(2 * np.pi * hours / 24).astype(np.float32)
        self._feature_cache[:, 22] = np.cos(2 * np.pi * hours / 24).astype(np.float32)

        self._cache_valid = True
        if verbose:
            print(f"[FeatureAggregator] Done! Cache shape: {self._feature_cache.shape}")

    def get_cached_state_vector(self, current_idx: int) -> np.ndarray:
        """
        從緩存獲取狀態向量（O(1) 操作）

        Args:
            current_idx: 當前索引

        Returns:
            np.ndarray: 20 維狀態向量
        """
        if not self._cache_valid:
            raise RuntimeError("特徵緩存未初始化，請先調用 precompute_all_features()")

        return self._feature_cache[current_idx].copy()

    def get_state_vector(self, df: pd.DataFrame, current_idx: int) -> np.ndarray:
        """
        計算當前位置的完整狀態向量

        優化版本：優先使用緩存，否則回退到原始計算

        Args:
            df: OHLCV 數據
            current_idx: 當前索引

        Returns:
            np.ndarray: 20 維狀態向量
        """
        # 優先使用緩存（極快）
        if self._cache_valid and self._feature_cache is not None:
            return self.get_cached_state_vector(current_idx)

        # 回退到原始計算（兼容舊代碼）
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
    print("  Testing Feature Aggregator (Vectorized)")
    print("=" * 60)

    import time

    # 載入數據
    project_root = Path(__file__).parent.parent.parent

    print("\n   Loading test data...")
    data_path = project_root / "data" / "raw" / "BTCUSDT_1m_train_latest.csv"

    if not data_path.exists():
        # 嘗試其他數據文件
        data_files = list((project_root / "data" / "raw").glob("BTCUSDT_1m_full_*.csv"))
        if data_files:
            data_path = sorted(data_files)[-1]
        else:
            print("   No test data found, skipping test")
            return

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # 使用 10000 根 K 線進行測試
    df_test = df.iloc[-10000:].copy()
    print(f"   Test data: {len(df_test)} bars")
    print(f"   Time range: {df_test.index[0]} to {df_test.index[-1]}\n")

    # 初始化特徵整合器
    print("   Initializing FeatureAggregator...")
    aggregator = FeatureAggregator()
    print(f"   State dimension: {aggregator.get_state_dimension()}")

    # 測試向量化預計算
    print("\n   Testing vectorized precomputation...")
    start = time.time()
    aggregator.precompute_all_features(df_test, verbose=True)
    elapsed = time.time() - start
    print(f"\n   Total precompute time: {elapsed:.2f}s")
    print(f"   Speed: {len(df_test) / elapsed:.0f} steps/second")

    # 測試緩存查詢速度
    print("\n   Testing cache query speed...")
    start = time.time()
    for i in range(len(df_test)):
        _ = aggregator.get_cached_state_vector(i)
    elapsed = time.time() - start
    print(f"   Cache query time ({len(df_test)} queries): {elapsed:.3f}s")
    print(f"   Speed: {len(df_test) / elapsed:.0f} queries/second")

    # 驗證特徵值
    print("\n   Sample feature vector (last bar):")
    state = aggregator.get_cached_state_vector(len(df_test) - 1)
    for i, name in enumerate(aggregator.feature_names):
        print(f"      {name}: {state[i]:.4f}")

    print("\n   OK: Vectorized FeatureAggregator test passed!")


if __name__ == "__main__":
    test_feature_aggregator()
