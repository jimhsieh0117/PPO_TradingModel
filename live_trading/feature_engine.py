"""
即時特徵計算引擎 — 複用 FeatureAggregator

職責：
- 將 K 線 buffer 轉換為 28 維市場特徵向量
- 直接使用 FeatureAggregator，確保與訓練環境的特徵完全一致
- 每根 K 線只需最後一行的特徵（當前 bar）

設計原則：
- 特徵一致性是最高優先級 — 訓練和實盤的特徵必須完全相同
- 使用 precompute_all_features() 在完整 buffer 上計算
  → 保證所有 lookback window 正確填充
  → 只取最後一行作為當前特徵
- buffer_size=500 遠大於任何特徵的最大 lookback（180）
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from environment.features.feature_aggregator import FeatureAggregator

logger = logging.getLogger("live_trading.feature_engine")

# 期望的特徵維度
EXPECTED_FEATURE_DIM = 28


class FeatureEngine:
    """
    即時特徵計算引擎

    Usage:
        engine = FeatureEngine(feature_config)
        features = engine.compute(buffer_df)  # → np.ndarray [28]
    """

    def __init__(self, feature_config: Dict):
        """
        初始化特徵引擎

        Args:
            feature_config: 特徵配置（從模型的訓練 config.yaml 的 features 區塊讀取）
                必須包含：structure_lookback, ob_lookback, ob_min_size,
                          fvg_min_size, fvg_max_age, liquidity_lookback 等
        """
        self.aggregator = FeatureAggregator(config=feature_config)

        # 驗證特徵維度
        actual_dim = self.aggregator.get_state_dimension()
        if actual_dim != EXPECTED_FEATURE_DIM:
            raise ValueError(
                f"Feature dimension mismatch: expected {EXPECTED_FEATURE_DIM}, "
                f"got {actual_dim}. Model expects exactly {EXPECTED_FEATURE_DIM} "
                f"market features."
            )

        self._compute_count = 0
        self._last_compute_time = 0.0

        # 每日特徵快照用
        self._snapshot_dir: Optional[Path] = None

        logger.info(
            f"FeatureEngine initialized | "
            f"feature_dim={actual_dim} | "
            f"feature_names={self.aggregator.feature_names}"
        )

    def compute(self, buffer_df: pd.DataFrame) -> np.ndarray:
        """
        從 K 線 buffer 計算當前 bar 的 28 維市場特徵

        Args:
            buffer_df: K 線 DataFrame（DatetimeIndex，至少 warmup_bars 行）
                       columns: open, high, low, close, volume, trades

        Returns:
            np.ndarray shape [28], dtype=float32

        Raises:
            ValueError: buffer 為空或特徵維度不對
        """
        if buffer_df.empty:
            raise ValueError("Cannot compute features: buffer is empty")

        t0 = time.time()

        # 確保 DatetimeIndex
        if not isinstance(buffer_df.index, pd.DatetimeIndex):
            if "timestamp" in buffer_df.columns:
                buffer_df = buffer_df.set_index("timestamp")
            else:
                raise ValueError(
                    "buffer_df must have DatetimeIndex or 'timestamp' column"
                )

        # 在完整 buffer 上預計算所有特徵
        self.aggregator.precompute_all_features(buffer_df, verbose=False)

        # 取最後一行（= 當前 bar 的特徵）
        last_idx = len(buffer_df) - 1
        features = self.aggregator.get_cached_state_vector(last_idx)

        # 驗證
        if features.shape[0] != EXPECTED_FEATURE_DIM:
            raise ValueError(
                f"Feature shape mismatch: expected [{EXPECTED_FEATURE_DIM}], "
                f"got [{features.shape[0]}]"
            )

        # NaN 檢查（NaN 會導致模型輸出不可預測）
        if np.any(np.isnan(features)):
            nan_indices = np.where(np.isnan(features))[0]
            nan_names = [self.aggregator.feature_names[i] for i in nan_indices]
            logger.error(f"NaN detected in features: {nan_names}")
            # 用 0 替代 NaN（安全 fallback）
            features = np.nan_to_num(features, nan=0.0)

        self._compute_count += 1
        self._last_compute_time = time.time() - t0

        if self._compute_count % 100 == 0:
            logger.debug(
                f"Feature compute #{self._compute_count} | "
                f"time={self._last_compute_time:.3f}s | "
                f"buffer_len={len(buffer_df)}"
            )

        return features

    def get_atr(self, buffer_df: pd.DataFrame) -> float:
        """
        取得當前 ATR 值（供止損計算用）

        必須在 compute() 之後呼叫（依賴已計算的快取）

        Returns:
            當前 bar 的 ATR 原始值
        """
        if not self.aggregator._cache_valid:
            logger.warning("get_atr called before compute() — computing now")
            self.compute(buffer_df)

        # ATR 在 volume_analyzer 內部快取
        atr_cache = self.aggregator.volume_analyzer._atr_cache
        return float(atr_cache[-1])

    # ================================================================
    # 特徵一致性驗證
    # ================================================================

    def verify_parity(self, buffer_df: pd.DataFrame,
                      reference_features: np.ndarray,
                      rtol: float = 1e-5) -> bool:
        """
        驗證即時特徵與參考特徵是否一致

        Args:
            buffer_df: K 線 buffer
            reference_features: 離線計算的參考特徵 [28]
            rtol: 相對容差

        Returns:
            True = 一致, False = 不一致
        """
        live_features = self.compute(buffer_df)
        is_close = np.allclose(live_features, reference_features, rtol=rtol)

        if not is_close:
            diff = np.abs(live_features - reference_features)
            max_diff_idx = np.argmax(diff)
            max_diff_name = self.aggregator.feature_names[max_diff_idx]
            logger.error(
                f"Feature parity FAILED | "
                f"max_diff={diff[max_diff_idx]:.8f} "
                f"at '{max_diff_name}' (idx={max_diff_idx}) | "
                f"live={live_features[max_diff_idx]:.8f} "
                f"ref={reference_features[max_diff_idx]:.8f}"
            )
        else:
            logger.info("Feature parity check PASSED")

        return is_close

    # ================================================================
    # 每日快照
    # ================================================================

    def setup_daily_snapshot(self, log_dir: str) -> None:
        """設定每日特徵快照目錄"""
        self._snapshot_dir = Path(log_dir) / "feature_snapshots"
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)

    def save_daily_snapshot(self, buffer_df: pd.DataFrame) -> Optional[str]:
        """
        儲存當前 buffer 的完整特徵矩陣（.npy）

        供離線比對與事後除錯使用（ARCHITECTURE.md 3.4 節要求）

        Returns:
            儲存路徑，或 None（未設定目錄）
        """
        if self._snapshot_dir is None:
            return None

        if not self.aggregator._cache_valid:
            self.compute(buffer_df)

        timestamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
        path = self._snapshot_dir / f"features_{timestamp}.npy"

        np.save(str(path), self.aggregator._feature_cache)
        logger.info(
            f"Daily feature snapshot saved: {path} "
            f"(shape={self.aggregator._feature_cache.shape})"
        )
        return str(path)

    # ================================================================
    # 統計
    # ================================================================

    def get_stats(self) -> dict:
        return {
            "compute_count": self._compute_count,
            "last_compute_time_ms": round(self._last_compute_time * 1000, 1),
        }
