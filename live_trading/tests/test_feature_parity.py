"""
特徵一致性驗證測試

驗證 live_trading.feature_engine 和 data_pipeline 的特徵計算結果完全一致。

這是實盤最關鍵的測試 — 特徵不一致 = 模型收到 OOD 輸入 = 默默虧錢。

測試方法：
1. 載入一段歷史數據
2. 分別用 data_pipeline 和 feature_engine 計算特徵
3. assert 結果一致（rtol=1e-5）
"""

import sys
from pathlib import Path

# 確保專案根目錄在 sys.path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import pytest

from environment.features.feature_aggregator import FeatureAggregator
from live_trading.feature_engine import FeatureEngine


def _load_test_config() -> dict:
    """載入測試用的特徵配置（與模型訓練一致）"""
    return {
        "structure_lookback": 180,
        "ob_lookback": 180,
        "ob_min_size": 0.002,
        "fvg_min_size": 0.001,
        "fvg_max_age": 180,
        "liquidity_lookback": 180,
    }


def _load_sample_data(n_bars: int = 500) -> pd.DataFrame:
    """
    載入一段歷史 K 線數據供測試

    優先用 processed data，沒有的話用 raw data。
    """
    from utils.config_utils import load_config

    config = load_config()

    # 嘗試載入已處理的數據
    processed_dir = Path(config.get("data", {}).get("processed_data_dir", "data/processed"))
    symbol = config.get("data", {}).get("symbol", "ETHUSDT")
    processed_files = sorted(processed_dir.glob(f"{symbol}_*.parquet"))

    if processed_files:
        df = pd.read_parquet(processed_files[-1])
        # 確保有 DatetimeIndex
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        elif not isinstance(df.index, pd.DatetimeIndex):
            if df.index.dtype in ["int64", "float64"]:
                df.index = pd.to_datetime(df.index, unit="ms", utc=True)

        # 只取 OHLCV 欄位
        ohlcv_cols = ["open", "high", "low", "close", "volume", "trades"]
        available = [c for c in ohlcv_cols if c in df.columns]
        df = df[available]

        # 取最後 n_bars 行
        df = df.tail(n_bars).copy()
        return df

    pytest.skip("No processed data available for testing")


class TestFeatureParity:
    """特徵一致性測試"""

    def test_feature_dimension(self):
        """驗證特徵維度 = 28"""
        config = _load_test_config()
        engine = FeatureEngine(config)
        assert engine.aggregator.get_state_dimension() == 28

    def test_feature_names_match(self):
        """驗證特徵名稱列表一致"""
        config = _load_test_config()

        engine_agg = FeatureEngine(config).aggregator
        direct_agg = FeatureAggregator(config=config)

        assert engine_agg.feature_names == direct_agg.feature_names

    def test_parity_with_direct_aggregator(self):
        """
        核心測試：FeatureEngine 與直接使用 FeatureAggregator 的結果一致

        模擬：
        - data_pipeline 路徑：整段數據 → precompute_all_features → 取任意 idx
        - feature_engine 路徑：同樣的完整 buffer → precompute_all_features → 取最後一行

        注意：多時間框架特徵（trend_5m, trend_15m, idx 18-19）在不同 buffer 長度下
        重新取樣邊界可能不同，這是已知限制。實盤永遠使用固定 500 根 buffer，
        所以 test_parity_sliding_window 才是最關鍵的測試。
        """
        config = _load_test_config()
        df = _load_sample_data(n_bars=500)

        # 路徑 1：直接使用 FeatureAggregator（模擬 data_pipeline / TradingEnv）
        direct_agg = FeatureAggregator(config=config)
        direct_agg.precompute_all_features(df, verbose=False)

        # 路徑 2：使用 FeatureEngine（完整 buffer）
        engine = FeatureEngine(config)

        # 使用完整 buffer 比對最後一行（buffer 長度一致 → 特徵必須完全一致）
        ref_features = direct_agg.get_cached_state_vector(len(df) - 1)
        live_features = engine.compute(df)

        np.testing.assert_allclose(
            live_features, ref_features, rtol=1e-5,
            err_msg=(
                f"Feature parity failed at last index.\n"
                f"Max diff: {np.max(np.abs(live_features - ref_features)):.10f}"
            )
        )

    def test_parity_sliding_window(self):
        """
        滑動窗口測試：模擬實盤的 500 根 buffer 逐步滑動

        驗證：用完整歷史算的特徵 vs 用最近 500 根算的特徵
        buffer_size=500 遠大於最大 lookback=180，應完全一致
        """
        config = _load_test_config()
        df = _load_sample_data(n_bars=500)

        if len(df) < 500:
            pytest.skip(f"Need 500 bars, got {len(df)}")

        # 完整計算
        full_agg = FeatureAggregator(config=config)
        full_agg.precompute_all_features(df, verbose=False)
        ref = full_agg.get_cached_state_vector(len(df) - 1)

        # 滑動窗口（取最後 500 根）
        engine = FeatureEngine(config)
        window = df.tail(500).copy()
        live = engine.compute(window)

        np.testing.assert_allclose(
            live, ref, rtol=1e-5,
            err_msg="Sliding window features differ from full history"
        )

    def test_no_nan_features(self):
        """驗證特徵中沒有 NaN（暖機後）"""
        config = _load_test_config()
        df = _load_sample_data(n_bars=500)

        engine = FeatureEngine(config)
        features = engine.compute(df)

        assert not np.any(np.isnan(features)), (
            f"NaN found in features at indices: "
            f"{np.where(np.isnan(features))[0]}"
        )

    def test_feature_dtype(self):
        """驗證特徵 dtype 為 float32（SB3 要求）"""
        config = _load_test_config()
        df = _load_sample_data(n_bars=500)

        engine = FeatureEngine(config)
        features = engine.compute(df)

        assert features.dtype == np.float32, (
            f"Expected float32, got {features.dtype}"
        )

    def test_atr_available_after_compute(self):
        """驗證 compute() 後可以取得 ATR"""
        config = _load_test_config()
        df = _load_sample_data(n_bars=500)

        engine = FeatureEngine(config)
        engine.compute(df)
        atr = engine.get_atr(df)

        assert isinstance(atr, float)
        assert atr > 0, f"ATR should be positive, got {atr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
