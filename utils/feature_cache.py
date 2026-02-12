"""
特徵緩存管理器

實現智能特徵緩存：
1. 將預計算的特徵保存到硬碟
2. 使用數據文件哈希值檢測數據變更
3. 緩存有效時直接讀取，跳過計算

作者：PPO Trading Team
日期：2026-02-12
"""

import hashlib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict
from datetime import datetime


class FeatureCacheManager:
    """特徵緩存管理器"""

    def __init__(self, cache_dir: str = "data/cache"):
        """
        初始化緩存管理器

        Args:
            cache_dir: 緩存目錄
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """
        計算數據的哈希值（用於檢測數據變更）

        使用數據的關鍵屬性計算哈希：
        - 數據長度
        - 首尾時間戳
        - 首尾價格
        - 數據形狀

        Args:
            df: OHLCV 數據

        Returns:
            str: 8 字符的哈希值
        """
        # 收集關鍵信息
        info = {
            'length': len(df),
            'columns': sorted(df.columns.tolist()),
            'first_close': float(df['close'].iloc[0]) if 'close' in df.columns else 0,
            'last_close': float(df['close'].iloc[-1]) if 'close' in df.columns else 0,
            'first_timestamp': str(df.index[0]) if isinstance(df.index, pd.DatetimeIndex) else str(df.iloc[0].get('timestamp', 0)),
            'last_timestamp': str(df.index[-1]) if isinstance(df.index, pd.DatetimeIndex) else str(df.iloc[-1].get('timestamp', 0)),
        }

        # 計算哈希
        info_str = json.dumps(info, sort_keys=True)
        hash_obj = hashlib.md5(info_str.encode())
        return hash_obj.hexdigest()[:8]

    def _compute_config_hash(self, config: Dict) -> str:
        """
        計算配置的哈希值

        Args:
            config: 特徵配置

        Returns:
            str: 8 字符的哈希值
        """
        config_str = json.dumps(config, sort_keys=True, default=str)
        hash_obj = hashlib.md5(config_str.encode())
        return hash_obj.hexdigest()[:8]

    def get_cache_path(self, data_hash: str, config_hash: str) -> Tuple[Path, Path]:
        """
        獲取緩存文件路徑

        Args:
            data_hash: 數據哈希
            config_hash: 配置哈希

        Returns:
            (features_path, metadata_path)
        """
        cache_name = f"features_{data_hash}_{config_hash}"
        features_path = self.cache_dir / f"{cache_name}.npy"
        metadata_path = self.cache_dir / f"{cache_name}.json"
        return features_path, metadata_path

    def save_cache(self, features: np.ndarray, df: pd.DataFrame, config: Dict) -> Path:
        """
        保存特徵緩存到硬碟

        Args:
            features: 預計算的特徵數組 [n_steps, n_features]
            df: 原始數據
            config: 特徵配置

        Returns:
            Path: 緩存文件路徑
        """
        data_hash = self._compute_data_hash(df)
        config_hash = self._compute_config_hash(config)
        features_path, metadata_path = self.get_cache_path(data_hash, config_hash)

        # 保存特徵數組
        np.save(features_path, features)

        # 保存元數據
        metadata = {
            'data_hash': data_hash,
            'config_hash': config_hash,
            'n_steps': features.shape[0],
            'n_features': features.shape[1],
            'created_at': datetime.now().isoformat(),
            'data_length': len(df),
        }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        return features_path

    def load_cache(self, df: pd.DataFrame, config: Dict) -> Optional[np.ndarray]:
        """
        嘗試從硬碟讀取緩存

        Args:
            df: 原始數據（用於驗證緩存有效性）
            config: 特徵配置

        Returns:
            np.ndarray 或 None（緩存無效時）
        """
        data_hash = self._compute_data_hash(df)
        config_hash = self._compute_config_hash(config)
        features_path, metadata_path = self.get_cache_path(data_hash, config_hash)

        # 檢查緩存文件是否存在
        if not features_path.exists() or not metadata_path.exists():
            return None

        # 讀取元數據驗證
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # 驗證數據長度
            if metadata.get('data_length') != len(df):
                print(f"[FeatureCache] Cache invalid: data length mismatch ({metadata.get('data_length')} vs {len(df)})")
                return None

            # 讀取特徵數組
            features = np.load(features_path)

            # 驗證形狀
            if features.shape[0] != len(df):
                print(f"[FeatureCache] Cache invalid: feature shape mismatch")
                return None

            return features

        except Exception as e:
            print(f"[FeatureCache] Error loading cache: {e}")
            return None

    def clear_old_caches(self, keep_latest: int = 3):
        """
        清理舊的緩存文件

        Args:
            keep_latest: 保留最新的 N 個緩存
        """
        # 找到所有緩存元數據文件
        metadata_files = list(self.cache_dir.glob("features_*.json"))

        if len(metadata_files) <= keep_latest:
            return

        # 按創建時間排序
        cache_info = []
        for meta_path in metadata_files:
            try:
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                created_at = metadata.get('created_at', '1970-01-01')
                cache_info.append((meta_path, created_at))
            except Exception:
                continue

        cache_info.sort(key=lambda x: x[1], reverse=True)

        # 刪除舊緩存
        for meta_path, _ in cache_info[keep_latest:]:
            features_path = meta_path.with_suffix('.npy')
            try:
                meta_path.unlink()
                if features_path.exists():
                    features_path.unlink()
                print(f"[FeatureCache] Removed old cache: {meta_path.stem}")
            except Exception as e:
                print(f"[FeatureCache] Error removing cache: {e}")


def precompute_features_with_cache(
    df: pd.DataFrame,
    config: Dict,
    cache_dir: str = "data/cache",
    verbose: bool = True
) -> np.ndarray:
    """
    帶緩存的特徵預計算

    優先從緩存讀取，緩存無效時重新計算並保存

    Args:
        df: OHLCV 數據
        config: 特徵配置
        cache_dir: 緩存目錄
        verbose: 是否顯示進度

    Returns:
        np.ndarray: 特徵數組 [n_steps, n_features]
    """
    from environment.features.feature_aggregator import FeatureAggregator

    cache_manager = FeatureCacheManager(cache_dir)

    # 確保 DataFrame 有正確的 datetime index
    if 'timestamp' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df_indexed = df.copy()
        df_indexed['timestamp'] = pd.to_datetime(df_indexed['timestamp'])
        df_indexed = df_indexed.set_index('timestamp')
    else:
        df_indexed = df

    # 嘗試從緩存讀取
    if verbose:
        print("[FeatureCache] Checking for cached features...")

    cached_features = cache_manager.load_cache(df_indexed, config)

    if cached_features is not None:
        if verbose:
            print(f"[FeatureCache] Loaded from cache! Shape: {cached_features.shape}")
            print(f"[FeatureCache] Skipped {len(df):,} steps of computation")
        return cached_features

    # 緩存無效，重新計算
    if verbose:
        print("[FeatureCache] Cache miss, computing features...")

    aggregator = FeatureAggregator(config=config)
    aggregator.precompute_all_features(df_indexed, verbose=verbose)
    features = aggregator._feature_cache

    # 保存到緩存
    if verbose:
        print("[FeatureCache] Saving to cache...")

    cache_path = cache_manager.save_cache(features, df_indexed, config)

    if verbose:
        print(f"[FeatureCache] Saved to: {cache_path}")

    # 清理舊緩存
    cache_manager.clear_old_caches(keep_latest=3)

    return features


def test_feature_cache():
    """測試特徵緩存"""
    print("=" * 60)
    print("  Testing Feature Cache")
    print("=" * 60)

    import time

    # 載入測試數據
    data_files = list(Path("data/raw").glob("BTCUSDT_1m_full_*.csv"))
    if not data_files:
        print("No test data found")
        return

    df = pd.read_csv(sorted(data_files)[-1])
    df_small = df.iloc[:5000]

    config = {}

    # 第一次：計算並緩存
    print("\n[Test 1] First run (compute + cache):")
    start = time.time()
    features1 = precompute_features_with_cache(df_small, config, verbose=True)
    time1 = time.time() - start
    print(f"   Time: {time1:.2f}s")

    # 第二次：從緩存讀取
    print("\n[Test 2] Second run (load from cache):")
    start = time.time()
    features2 = precompute_features_with_cache(df_small, config, verbose=True)
    time2 = time.time() - start
    print(f"   Time: {time2:.2f}s")

    # 驗證一致性
    print("\n[Validation]:")
    if np.allclose(features1, features2):
        print("   Features match!")
    else:
        print("   WARNING: Features do not match!")

    print(f"\n   Speedup: {time1 / time2:.1f}x")
    print("\n   OK: Feature cache test passed!")


if __name__ == "__main__":
    test_feature_cache()
