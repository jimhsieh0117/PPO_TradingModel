"""
數據管線模組 - 自動化數據檢查、下載、分割

訓練前自動確保數據就緒：
1. 檢查 data/raw 是否已有匹配日期範圍的 Parquet 文件
2. 若無 → 呼叫 BinanceDataDownloader 下載
3. 按 test_start_date 分割 train/test
4. 返回 (train_df, test_df)
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# 確保能 import data.download_data
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def _date_to_tag(date_str: str) -> str:
    """將日期字串轉為檔名用的標籤，例如 '2020-01-01 00:00:00' → '20200101'"""
    dt = datetime.strptime(date_str.strip(), "%Y-%m-%d %H:%M:%S")
    return dt.strftime("%Y%m%d")


def _build_expected_filename(symbol: str, start_date: str, end_date: str,
                             interval: str = "1m") -> str:
    """
    根據日期範圍生成預期的數據文件名（不含路徑）。
    例如：BTCUSDT_1m_20200101_20251231.parquet
    """
    start_tag = _date_to_tag(start_date)
    end_tag = _date_to_tag(end_date)
    return f"{symbol}_{interval}_{start_tag}_{end_tag}.parquet"


def _find_existing_data(raw_dir: Path, expected_name: str) -> Path | None:
    """檢查是否已存在完整數據文件（優先 Parquet，兼容舊 CSV）"""
    full_path = raw_dir / expected_name
    if full_path.exists() and full_path.stat().st_size > 0:
        return full_path
    # 向後兼容：檢查是否有舊的 CSV 版本
    csv_name = expected_name.replace('.parquet', '.csv')
    csv_path = raw_dir / csv_name
    if csv_path.exists() and csv_path.stat().st_size > 0:
        print(f"   ℹ️  找到舊版 CSV: {csv_name}，將轉換為 Parquet...")
        df = pd.read_csv(csv_path)
        df.to_parquet(full_path, index=False)
        print(f"   ✅ 已轉換為: {expected_name}")
        return full_path
    return None


def _download_data(symbol: str, start_date: str, end_date: str, raw_dir: Path,
                   expected_name: str, interval: str = "1m") -> Path:
    """
    下載數據並保存為指定文件名。

    Returns:
        保存的完整數據文件路徑
    """
    from data.download_data import BinanceDataDownloader

    # 解析日期
    start_str = datetime.strptime(start_date.strip(), "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")
    end_str = datetime.strptime(end_date.strip(), "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")

    downloader = BinanceDataDownloader(symbol=symbol, interval=interval)

    # 下載
    full_df = downloader.download_by_date_range(start_str, end_str)

    # 驗證
    downloader.validate_data(full_df)
    downloader.generate_summary(full_df)

    # 保存為 Parquet（壓縮 + 快速讀取）
    save_path = raw_dir / expected_name
    full_df.to_parquet(save_path, index=False)
    print(f"   💾 數據已保存 (Parquet): {save_path}")

    return save_path


def _load_and_split(data_path: Path, test_start_date: str) -> tuple:
    """
    載入數據（Parquet 或 CSV）並按日期分割 train/test。

    Returns:
        (train_df, test_df)
    """
    print(f"\n[DATA] 載入數據: {data_path}")
    if str(data_path).endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    # 確保 timestamp 列存在
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    elif df.index.name == 'timestamp':
        df.index = pd.to_datetime(df.index)

    print(f"   總 K 線數: {len(df):,}")
    print(f"   時間範圍: {df.index[0]} ~ {df.index[-1]}")

    # 按日期分割
    split_dt = pd.Timestamp(
        datetime.strptime(test_start_date.strip(), "%Y-%m-%d %H:%M:%S")
    )

    train_df = df[df.index < split_dt].copy()
    test_df = df[df.index >= split_dt].copy()

    # 重置索引（train.py 期望 timestamp 作為列而非索引）
    train_df.reset_index(inplace=True)
    test_df.reset_index(inplace=True)

    print(f"\n[DATA] 數據分割完成:")
    print(f"   訓練集: {len(train_df):,} 根K線 ({train_df['timestamp'].iloc[0]} ~ {train_df['timestamp'].iloc[-1]})")
    print(f"   測試集: {len(test_df):,} 根K線 ({test_df['timestamp'].iloc[0]} ~ {test_df['timestamp'].iloc[-1]})")

    return train_df, test_df


def ensure_data_ready(config: dict) -> tuple:
    """
    確保訓練/測試數據就緒。

    流程：
    1. 讀取 config 中的數據日期範圍
    2. 檢查 data/raw 是否已有匹配的 CSV
    3. 如果沒有 → 自動下載
    4. 按 test_start_date 分割 train/test
    5. 返回 (train_df, test_df)

    Args:
        config: 完整配置字典

    Returns:
        (train_df, test_df): 訓練集和測試集 DataFrame
    """
    data_config = config.get('data', {})
    trading_config = config.get('trading', {})

    symbol = data_config.get('symbol', trading_config.get('symbol', 'BTCUSDT'))
    interval = trading_config.get('timeframe', '1m')
    start_date = data_config.get('start_date', '2020-01-01 00:00:00')
    end_date = data_config.get('end_date', '2025-12-31 23:59:59')
    test_start_date = data_config.get('test_start_date', '2025-01-01 00:00:00')
    raw_dir = Path(data_config.get('raw_data_dir', 'data/raw'))

    raw_dir.mkdir(parents=True, exist_ok=True)

    expected_name = _build_expected_filename(symbol, start_date, end_date, interval)

    print(f"\n{'='*60}")
    print(f"[DATA] 數據管線檢查")
    print(f"{'='*60}")
    print(f"   交易對: {symbol}")
    print(f"   時間框架: {interval}")
    print(f"   日期範圍: {start_date} ~ {end_date}")
    print(f"   測試集起始: {test_start_date}")
    print(f"   預期文件: {expected_name}")

    # Step 1: 檢查是否已有數據
    existing = _find_existing_data(raw_dir, expected_name)

    if existing:
        print(f"\n   ✅ 數據已存在: {existing}")
    else:
        print(f"\n   ⚠️  數據不存在，開始下載...")
        existing = _download_data(symbol, start_date, end_date, raw_dir, expected_name, interval)

    # Step 2: 載入並分割
    train_df, test_df = _load_and_split(existing, test_start_date)

    return train_df, test_df
