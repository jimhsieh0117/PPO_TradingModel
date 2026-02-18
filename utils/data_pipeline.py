"""
數據管線模組 - 增量下載 + 處理後數據存儲

功能：
1. 掃描 data/raw 中已有的 Parquet 檔案
2. 計算缺少的日期範圍，僅下載缺少的部分（增量下載）
3. 合併原始數據 → 計算 ICT 特徵 → 存儲為處理後 Parquet
4. 後續請求直接從處理後快取載入（透過 data_hash + feature_config_hash 驗證）
5. 返回包含 OHLCV + 20 ICT 特徵的 DataFrame
"""

import hashlib
import json
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

# 確保能 import data.download_data
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OHLCV_COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'trades']

FEATURE_COLUMNS = [
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
    'hour_cos',
    # Market Regime (3)
    'adx_normalized',
    'volatility_regime',
    'trend_strength',
]


def extract_features(df: pd.DataFrame) -> np.ndarray:
    """
    從包含 OHLCV + 特徵的 DataFrame 中提取 20 維特徵數組。

    Args:
        df: 含有 FEATURE_COLUMNS 的 DataFrame（由 ensure_data_ready / load_full_data 返回）

    Returns:
        np.ndarray: shape [n_rows, 20]
    """
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing feature columns: {missing}")
    return df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)


# ---------------------------------------------------------------------------
# Backward compat (kept but deprecated)
# ---------------------------------------------------------------------------

def _date_to_tag(date_str: str) -> str:
    """將日期字串轉為檔名用的標籤，例如 '2020-01-01 00:00:00' → '20200101'"""
    dt = datetime.strptime(date_str.strip(), "%Y-%m-%d %H:%M:%S")
    return dt.strftime("%Y%m%d")


def _build_expected_filename(symbol: str, start_date: str, end_date: str,
                             interval: str = "1m") -> str:
    """(Deprecated) 根據日期範圍生成預期的數據文件名。"""
    start_tag = _date_to_tag(start_date)
    end_tag = _date_to_tag(end_date)
    return f"{symbol}_{interval}_{start_tag}_{end_tag}.parquet"


# ---------------------------------------------------------------------------
# Private: incremental download helpers
# ---------------------------------------------------------------------------

_RAW_PATTERN = re.compile(
    r"^(?P<symbol>[A-Z]+)_(?P<interval>\w+)_(?P<start>\d{8})_(?P<end>\d{8})\.parquet$"
)


def _scan_existing_raw_data(
    raw_dir: Path, symbol: str, interval: str
) -> Optional[pd.DataFrame]:
    """
    掃描 raw_dir 中所有匹配 {symbol}_{interval}_{YYYYMMDD}_{YYYYMMDD}.parquet 的檔案，
    讀取、合併、去重後返回。
    """
    matched_files = []
    for f in raw_dir.glob(f"{symbol}_{interval}_*.parquet"):
        m = _RAW_PATTERN.match(f.name)
        if m and m.group("symbol") == symbol and m.group("interval") == interval:
            matched_files.append(f)

    if not matched_files:
        return None

    print(f"   [SCAN] Found {len(matched_files)} raw file(s) for {symbol}/{interval}")

    dfs = []
    for f in matched_files:
        try:
            df = pd.read_parquet(f)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            elif df.index.name == 'timestamp':
                df = df.reset_index()
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            dfs.append(df)
        except Exception as e:
            print(f"   [WARN] Failed to read {f.name}: {e}")

    if not dfs:
        return None

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.drop_duplicates(subset='timestamp', keep='last')
    merged = merged.sort_values('timestamp').reset_index(drop=True)
    print(f"   [SCAN] Merged: {len(merged):,} rows, "
          f"{merged['timestamp'].iloc[0]} ~ {merged['timestamp'].iloc[-1]}")
    return merged


def _determine_missing_ranges(
    existing_df: Optional[pd.DataFrame],
    required_start: str,
    required_end: str,
) -> list[tuple[str, str]]:
    """
    比較已有數據與要求的日期範圍，返回缺少的 (start_date_str, end_date_str) 列表。
    日期字串格式: 'YYYY-MM-DD'
    """
    req_start = pd.Timestamp(datetime.strptime(required_start.strip(), "%Y-%m-%d %H:%M:%S"))
    req_end = pd.Timestamp(datetime.strptime(required_end.strip(), "%Y-%m-%d %H:%M:%S"))

    if existing_df is None or existing_df.empty:
        return [(req_start.strftime("%Y-%m-%d"), req_end.strftime("%Y-%m-%d"))]

    existing_min = existing_df['timestamp'].min()
    existing_max = existing_df['timestamp'].max()

    gaps = []

    # Front gap: required start is before existing data
    if req_start < existing_min - pd.Timedelta(minutes=2):
        gap_end = (existing_min - pd.Timedelta(minutes=1)).strftime("%Y-%m-%d")
        gaps.append((req_start.strftime("%Y-%m-%d"), gap_end))

    # Back gap: required end is after existing data
    # 容忍度 1 天：避免尾端幾小時的缺失反覆觸發下載
    if req_end > existing_max + pd.Timedelta(days=1):
        gap_start = (existing_max + pd.Timedelta(minutes=1)).strftime("%Y-%m-%d")
        gaps.append((gap_start, req_end.strftime("%Y-%m-%d")))

    return gaps


def _download_and_merge(
    existing_df: Optional[pd.DataFrame],
    missing_ranges: list[tuple[str, str]],
    symbol: str,
    interval: str,
    raw_dir: Path,
) -> pd.DataFrame:
    """
    下載缺少的日期範圍，與已有數據合併，保存為單一 Parquet 並清理碎片。
    """
    from data.download_data import BinanceDataDownloader

    downloader = BinanceDataDownloader(symbol=symbol, interval=interval)
    new_dfs = []

    for start_str, end_str in missing_ranges:
        print(f"\n   [DOWNLOAD] Fetching {symbol} {interval}: {start_str} ~ {end_str}")
        df = downloader.download_by_date_range(start_str, end_str)
        if df.index.name == 'timestamp':
            df = df.reset_index()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        new_dfs.append(df)

    # Merge with existing
    all_parts = []
    if existing_df is not None and not existing_df.empty:
        all_parts.append(existing_df)
    all_parts.extend(new_dfs)

    merged = pd.concat(all_parts, ignore_index=True)
    merged = merged.drop_duplicates(subset='timestamp', keep='last')
    merged = merged.sort_values('timestamp').reset_index(drop=True)

    # Build filename from actual data range
    start_tag = merged['timestamp'].iloc[0].strftime("%Y%m%d")
    end_tag = merged['timestamp'].iloc[-1].strftime("%Y%m%d")
    new_name = f"{symbol}_{interval}_{start_tag}_{end_tag}.parquet"

    # Delete old fragmented files (same symbol/interval)
    for f in raw_dir.glob(f"{symbol}_{interval}_*.parquet"):
        m = _RAW_PATTERN.match(f.name)
        if m and f.name != new_name:
            print(f"   [CLEANUP] Removing old file: {f.name}")
            f.unlink()

    # Save merged
    save_path = raw_dir / new_name
    merged.to_parquet(save_path, index=False)
    print(f"   [OK] Raw data saved: {save_path.name} ({len(merged):,} rows)")

    return merged


# ---------------------------------------------------------------------------
# Private: processed data management
# ---------------------------------------------------------------------------

def _compute_data_hash(df: pd.DataFrame) -> str:
    """基於長度 + 首尾時間戳 + 首尾價格計算哈希。"""
    info = {
        'length': len(df),
        'first_ts': str(df['timestamp'].iloc[0]),
        'last_ts': str(df['timestamp'].iloc[-1]),
        'first_close': float(df['close'].iloc[0]),
        'last_close': float(df['close'].iloc[-1]),
    }
    info_str = json.dumps(info, sort_keys=True)
    return hashlib.md5(info_str.encode()).hexdigest()[:8]


def _compute_config_hash(feature_config: dict) -> str:
    """計算特徵配置哈希（包含特徵列表版本，確保特徵變更時自動失效）。"""
    combined = {
        'config': feature_config,
        'feature_columns': FEATURE_COLUMNS,
    }
    config_str = json.dumps(combined, sort_keys=True, default=str)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def _ensure_processed_data(
    raw_df: pd.DataFrame,
    symbol: str,
    interval: str,
    feature_config: dict,
    processed_dir: Path,
) -> pd.DataFrame:
    """
    檢查是否有有效的處理後 Parquet 快取。
    若快取有效則直接載入；否則重新計算特徵並存儲。

    Returns:
        DataFrame with columns: timestamp + OHLCV_COLUMNS + FEATURE_COLUMNS
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = processed_dir / f"{symbol}_{interval}.parquet"
    meta_path = processed_dir / f"{symbol}_{interval}.meta.json"

    data_hash = _compute_data_hash(raw_df)
    config_hash = _compute_config_hash(feature_config)

    # Check cache validity
    if parquet_path.exists() and meta_path.exists():
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            if meta.get('data_hash') == data_hash and meta.get('feature_config_hash') == config_hash:
                print(f"   [CACHE HIT] Loading processed data from {parquet_path.name}")
                cached = pd.read_parquet(parquet_path)
                cached['timestamp'] = pd.to_datetime(cached['timestamp'])
                print(f"   [OK] {len(cached):,} rows, {len(FEATURE_COLUMNS)} features")
                return cached
            else:
                reason = []
                if meta.get('data_hash') != data_hash:
                    reason.append("data changed")
                if meta.get('feature_config_hash') != config_hash:
                    reason.append("feature config changed")
                print(f"   [CACHE MISS] {', '.join(reason)} — recomputing features...")
        except Exception as e:
            print(f"   [CACHE MISS] Error reading cache metadata: {e}")
    else:
        print(f"   [CACHE MISS] No processed cache found, computing features...")

    # Compute features
    from environment.features.feature_aggregator import FeatureAggregator

    # FeatureAggregator expects DatetimeIndex
    df_indexed = raw_df.copy()
    df_indexed['timestamp'] = pd.to_datetime(df_indexed['timestamp'])
    df_indexed = df_indexed.set_index('timestamp')

    print(f"   [COMPUTE] Computing {len(FEATURE_COLUMNS)} ICT features for {len(df_indexed):,} rows...")
    aggregator = FeatureAggregator(config=feature_config)
    aggregator.precompute_all_features(df_indexed, verbose=True)
    features_array = aggregator._feature_cache  # [n_rows, 20]

    # Build combined DataFrame: timestamp + OHLCV + features
    combined = raw_df[['timestamp'] + OHLCV_COLUMNS].copy()
    for i, col_name in enumerate(FEATURE_COLUMNS):
        combined[col_name] = features_array[:, i]

    # Save parquet
    combined.to_parquet(parquet_path, index=False)

    # Save metadata
    meta = {
        'symbol': symbol,
        'interval': interval,
        'data_hash': data_hash,
        'feature_config_hash': config_hash,
        'n_rows': len(combined),
        'n_features': len(FEATURE_COLUMNS),
        'data_range': [str(combined['timestamp'].iloc[0]), str(combined['timestamp'].iloc[-1])],
        'feature_columns': FEATURE_COLUMNS,
        'created_at': datetime.now().isoformat(),
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"   [OK] Processed data saved: {parquet_path.name} ({len(combined):,} rows)")
    return combined


def _get_processed_data(config: dict) -> pd.DataFrame:
    """
    核心邏輯：掃描已有原始數據 → 增量下載 → 確保處理後快取。

    Returns:
        DataFrame with columns: timestamp + OHLCV + 20 ICT features
    """
    data_config = config.get('data', {})
    trading_config = config.get('trading', {})
    feature_config = config.get('features', {})

    symbol = data_config.get('symbol', trading_config.get('symbol', 'BTCUSDT'))
    interval = trading_config.get('timeframe', '1m')
    start_date = data_config.get('start_date', '2020-01-01 00:00:00')
    end_date = data_config.get('end_date', '2025-12-31 23:59:59')
    raw_dir = Path(data_config.get('raw_data_dir', 'data/raw'))
    processed_dir = Path(data_config.get('processed_data_dir', 'data/processed'))

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[DATA PIPELINE] Incremental Data Pipeline")
    print(f"{'='*60}")
    print(f"   Symbol:    {symbol}")
    print(f"   Interval:  {interval}")
    print(f"   Required:  {start_date} ~ {end_date}")

    # Step 1: Scan existing raw data
    print(f"\n[Step 1] Scanning existing raw data...")
    existing_df = _scan_existing_raw_data(raw_dir, symbol, interval)

    # Step 2: Determine missing ranges
    print(f"\n[Step 2] Checking for missing date ranges...")
    gaps = _determine_missing_ranges(existing_df, start_date, end_date)

    if gaps:
        total_gaps = ", ".join(f"{s}~{e}" for s, e in gaps)
        print(f"   [GAPS] Missing ranges: {total_gaps}")
    else:
        print(f"   [OK] All required data is available locally")

    # Step 3: Download missing data if needed
    if gaps:
        print(f"\n[Step 3] Downloading missing data...")
        raw_df = _download_and_merge(existing_df, gaps, symbol, interval, raw_dir)
    else:
        raw_df = existing_df
        print(f"\n[Step 3] No download needed")

    # Step 4: Ensure processed data (OHLCV + features)
    print(f"\n[Step 4] Ensuring processed data with features...")
    combined_df = _ensure_processed_data(raw_df, symbol, interval, feature_config, processed_dir)

    return combined_df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ensure_data_ready(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    確保訓練/測試數據就緒（含 OHLCV + 20 ICT 特徵）。

    流程：
    1. 增量下載（僅補足缺少的日期範圍）
    2. 計算/載入處理後快取（OHLCV + 特徵）
    3. 按 test_start_date 分割 train/test

    Args:
        config: 完整配置字典

    Returns:
        (train_df, test_df): 兩者均包含 timestamp + OHLCV + 20 feature 列
    """
    combined_df = _get_processed_data(config)

    # Split by test_start_date
    data_config = config.get('data', {})
    test_start_date = data_config.get('test_start_date', '2025-01-01 00:00:00')
    split_dt = pd.Timestamp(
        datetime.strptime(test_start_date.strip(), "%Y-%m-%d %H:%M:%S")
    )

    train_df = combined_df[combined_df['timestamp'] < split_dt].copy().reset_index(drop=True)
    test_df = combined_df[combined_df['timestamp'] >= split_dt].copy().reset_index(drop=True)

    print(f"\n[DATA PIPELINE] Split complete:")
    print(f"   Train: {len(train_df):,} rows ({train_df['timestamp'].iloc[0]} ~ {train_df['timestamp'].iloc[-1]})")
    print(f"   Test:  {len(test_df):,} rows ({test_df['timestamp'].iloc[0]} ~ {test_df['timestamp'].iloc[-1]})")
    print(f"   Columns: {len(train_df.columns)} (timestamp + {len(OHLCV_COLUMNS)} OHLCV + {len(FEATURE_COLUMNS)} features)")

    return train_df, test_df


def load_full_data(config: dict) -> pd.DataFrame:
    """
    載入完整數據（不分割），用於 WFA 等需要全量數據的場景。

    Returns:
        DataFrame with columns: timestamp + OHLCV + 20 feature columns
    """
    return _get_processed_data(config)
