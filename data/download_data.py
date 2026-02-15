#!/usr/bin/env python3
"""
數據下載腳本 - 從 Binance Futures API 下載 BTCUSDT 1分K 數據

功能：
1. 下載指定日期範圍的 BTCUSDT 永續合約 1分K 數據
2. 自動分割為訓練集和測試集（按日期或比例）
3. 保存為 CSV 格式
4. 顯示下載進度
5. 驗證數據完整性
6. 支持自動重試

用法：
  python download_data.py                           # 預設：最近 6 個月
  python download_data.py --start 2020-01-01 --end 2025-12-31
  python download_data.py --start 2020-01-01 --end 2025-12-31 --test-start 2025-01-01

作者：PPO Trading Team
日期：2026-01-14（v2: 支援日期範圍下載）
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from binance.um_futures import UMFutures
from tqdm import tqdm
import time

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class BinanceDataDownloader:
    """Binance 數據下載器"""

    def __init__(self, symbol="BTCUSDT", interval="1m"):
        """
        初始化下載器

        Args:
            symbol: 交易對符號（默認 BTCUSDT）
            interval: K線間隔（默認 1m）
        """
        self.symbol = symbol
        self.interval = interval
        self.client = UMFutures()

        # 創建數據目錄
        self.raw_dir = project_root / "data" / "raw"
        self.processed_dir = project_root / "data" / "processed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        print(f"🚀 初始化 Binance 數據下載器")
        print(f"   交易對: {self.symbol}")
        print(f"   K線間隔: {self.interval}")
        print(f"   數據保存路徑: {self.raw_dir}")

    def get_klines(self, start_time, end_time, limit=1500, max_retries=5):
        """
        獲取 K線數據（帶自動重試）

        Args:
            start_time: 開始時間戳（毫秒）
            end_time: 結束時間戳（毫秒）
            limit: 每次請求的數量限制（最大 1500）
            max_retries: 最大重試次數

        Returns:
            K線數據列表
        """
        for attempt in range(max_retries):
            try:
                klines = self.client.klines(
                    symbol=self.symbol,
                    interval=self.interval,
                    startTime=start_time,
                    endTime=end_time,
                    limit=limit
                )
                return klines
            except Exception as e:
                err_str = str(e).lower()
                if attempt < max_retries - 1:
                    # 若為 rate limit (429)，等待更久
                    if '429' in err_str or 'too many' in err_str or 'rate' in err_str:
                        wait = 30  # rate limit 冷卻 30 秒
                        print(f"\n   ⚠️ Rate limited! 等待 {wait}s 冷卻...")
                    else:
                        wait = 2 ** attempt  # exponential backoff
                        print(f"\n   retry {attempt+1}/{max_retries}: {e}, wait {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"\n   API failed after {max_retries} retries: {e}")
                    return None

    def download_data(self, months=6):
        """
        下載指定月數的歷史數據（從當前時間倒推）

        Args:
            months: 下載的月數（默認 6 個月）

        Returns:
            DataFrame: 下載的數據
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=months * 30)
        return self.download_by_date_range(
            start_time.strftime('%Y-%m-%d'),
            end_time.strftime('%Y-%m-%d')
        )

    def download_by_date_range(self, start_date: str, end_date: str):
        """
        下載指定日期範圍的歷史數據

        Args:
            start_date: 開始日期，格式 'YYYY-MM-DD'
            end_date: 結束日期，格式 'YYYY-MM-DD'

        Returns:
            DataFrame: 下載的數據
        """
        start_time = datetime.strptime(start_date, '%Y-%m-%d')
        end_time = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1) - timedelta(seconds=1)

        total_days = (end_time - start_time).days
        print(f"\n📊 開始下載 {self.symbol} 數據...")
        print(f"   日期範圍: {start_date} ~ {end_date} ({total_days} 天)")

        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        estimated_klines = total_days * 24 * 60
        print(f"   預計 K 線數: ~{estimated_klines:,} 根\n")

        all_klines = []
        current_start = start_ms
        limit = 1500
        interval_ms = 60 * 1000

        pbar = tqdm(total=estimated_klines, desc="下載進度", unit="根K線")

        consecutive_failures = 0
        max_consecutive_failures = 10

        while current_start < end_ms:
            batch_end = min(current_start + (limit * interval_ms), end_ms)

            klines = self.get_klines(current_start, batch_end, limit)

            if klines is None or len(klines) == 0:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print(f"\n❌ 連續 {max_consecutive_failures} 次失敗，中止下載")
                    break
                current_start = batch_end + 1
                time.sleep(1)
                continue

            consecutive_failures = 0
            all_klines.extend(klines)
            pbar.update(len(klines))

            current_start = int(klines[-1][0]) + 1
            time.sleep(0.15)  # 避免觸發 Binance rate limit

            if len(klines) < limit and current_start < end_ms:
                continue
            elif len(klines) < limit:
                break

        pbar.close()

        print(f"\n✅ 下載完成！共獲得 {len(all_klines):,} 根 K 線\n")

        df = self._klines_to_dataframe(all_klines)
        return df

    def _klines_to_dataframe(self, klines):
        """
        將 K線數據轉換為 DataFrame

        Args:
            klines: K線數據列表

        Returns:
            DataFrame
        """
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        # 轉換數據類型
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

        # 轉換價格和成交量為浮點數
        price_cols = ['open', 'high', 'low', 'close']
        volume_cols = ['volume', 'quote_volume', 'taker_buy_base', 'taker_buy_quote']

        for col in price_cols + volume_cols:
            df[col] = df[col].astype(float)

        df['trades'] = df['trades'].astype(int)

        # 只保留必要的列
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades']]

        # 設置 timestamp 為索引
        df.set_index('timestamp', inplace=True)

        return df

    def split_train_test(self, df, train_months=5, test_months=1):
        """
        按比例分割訓練集和測試集

        Args:
            df: 完整數據
            train_months: 訓練集月數
            test_months: 測試集月數

        Returns:
            train_df, test_df: 訓練集和測試集
        """
        print(f"📂 分割數據集...")
        print(f"   訓練集: {train_months} 個月")
        print(f"   測試集: {test_months} 個月")

        total_rows = len(df)
        train_ratio = train_months / (train_months + test_months)
        split_idx = int(total_rows * train_ratio)

        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        print(f"\n   訓練集: {len(train_df):,} 根K線 ({train_df.index[0]} 到 {train_df.index[-1]})")
        print(f"   測試集: {len(test_df):,} 根K線 ({test_df.index[0]} 到 {test_df.index[-1]})")

        return train_df, test_df

    def split_train_test_by_date(self, df, test_start_date: str):
        """
        按日期分割訓練集和測試集

        Args:
            df: 完整數據
            test_start_date: 測試集開始日期，格式 'YYYY-MM-DD'

        Returns:
            train_df, test_df: 訓練集和測試集
        """
        split_dt = pd.Timestamp(test_start_date)
        print(f"📂 按日期分割數據集...")
        print(f"   分割日期: {test_start_date}")

        train_df = df[df.index < split_dt].copy()
        test_df = df[df.index >= split_dt].copy()

        print(f"\n   訓練集: {len(train_df):,} 根K線 ({train_df.index[0]} 到 {train_df.index[-1]})")
        print(f"   測試集: {len(test_df):,} 根K線 ({test_df.index[0]} 到 {test_df.index[-1]})")

        return train_df, test_df

    def save_data(self, train_df, test_df, full_df):
        """
        保存數據到 CSV 文件

        Args:
            train_df: 訓練集
            test_df: 測試集
            full_df: 完整數據集
        """
        print(f"\n💾 保存數據...")

        # 生成文件名（帶時間戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        train_file = self.raw_dir / f"{self.symbol}_{self.interval}_train_{timestamp}.csv"
        test_file = self.raw_dir / f"{self.symbol}_{self.interval}_test_{timestamp}.csv"
        full_file = self.raw_dir / f"{self.symbol}_{self.interval}_full_{timestamp}.csv"

        # 保存
        train_df.to_csv(train_file)
        test_df.to_csv(test_file)
        full_df.to_csv(full_file)

        print(f"   ✅ 訓練集: {train_file}")
        print(f"   ✅ 測試集: {test_file}")
        print(f"   ✅ 完整數據: {full_file}")

        # 創建符號鏈接指向最新文件（方便使用）
        latest_train = self.raw_dir / f"{self.symbol}_{self.interval}_train_latest.csv"
        latest_test = self.raw_dir / f"{self.symbol}_{self.interval}_test_latest.csv"
        latest_full = self.raw_dir / f"{self.symbol}_{self.interval}_full_latest.csv"

        # 刪除舊的符號鏈接（如果存在）
        for link in [latest_train, latest_test, latest_full]:
            if link.exists():
                link.unlink()

        # 創建新的符號鏈接
        try:
            latest_train.symlink_to(train_file.name)
            latest_test.symlink_to(test_file.name)
            latest_full.symlink_to(full_file.name)
            print(f"\n   🔗 已創建符號鏈接指向最新數據")
        except OSError:
            # Windows 上可能沒有符號鏈接權限，改用複製
            import shutil
            shutil.copy(train_file, latest_train)
            shutil.copy(test_file, latest_test)
            shutil.copy(full_file, latest_full)
            print(f"\n   📋 已複製最新數據文件")

        return train_file, test_file, full_file

    def validate_data(self, df):
        """
        驗證數據品質

        Args:
            df: 數據 DataFrame

        Returns:
            bool: 驗證是否通過
        """
        print(f"\n🔍 驗證數據品質...")

        issues = []

        # 檢查缺失值
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            issues.append(f"發現 {null_counts.sum()} 個缺失值: {null_counts[null_counts > 0].to_dict()}")

        # 檢查時間序列連續性（1分K 應該每分鐘一根）
        time_diffs = df.index.to_series().diff()
        expected_diff = pd.Timedelta(minutes=1)
        gaps = time_diffs[time_diffs > expected_diff * 1.5]  # 允許一些誤差

        if len(gaps) > 0:
            issues.append(f"發現 {len(gaps)} 個時間間隙（可能缺少 K 線）")
            print(f"   ⚠️  時間間隙位置: {gaps.head()}")

        # 檢查價格異常值
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if (df[col] <= 0).any():
                issues.append(f"{col} 列存在零或負數價格")

            # 檢查極端值（使用 IQR 方法）
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 3*IQR) | (df[col] > Q3 + 3*IQR)]

            if len(outliers) > 0:
                issues.append(f"{col} 列發現 {len(outliers)} 個極端值")

        # 檢查 OHLC 邏輯
        invalid_ohlc = df[
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        ]

        if len(invalid_ohlc) > 0:
            issues.append(f"發現 {len(invalid_ohlc)} 根 K 線的 OHLC 邏輯錯誤")

        # 檢查成交量
        if (df['volume'] < 0).any():
            issues.append("成交量存在負數")

        # 輸出結果
        if len(issues) == 0:
            print("   ✅ 數據驗證通過！沒有發現問題。")
            return True
        else:
            print("   ⚠️  數據驗證發現以下問題:")
            for i, issue in enumerate(issues, 1):
                print(f"      {i}. {issue}")
            return False

    def generate_summary(self, df):
        """
        生成數據摘要統計

        Args:
            df: 數據 DataFrame
        """
        print(f"\n📈 數據摘要統計:")
        print(f"   總 K 線數: {len(df):,}")
        print(f"   時間範圍: {df.index[0]} 到 {df.index[-1]}")
        print(f"   持續天數: {(df.index[-1] - df.index[0]).days} 天")
        print(f"\n   價格統計:")
        print(f"      最高價: ${df['high'].max():,.2f}")
        print(f"      最低價: ${df['low'].min():,.2f}")
        print(f"      平均價: ${df['close'].mean():,.2f}")
        print(f"      價格標準差: ${df['close'].std():,.2f}")
        print(f"\n   成交量統計:")
        print(f"      總成交量: {df['volume'].sum():,.2f}")
        print(f"      平均成交量: {df['volume'].mean():,.2f}")
        print(f"      最大成交量: {df['volume'].max():,.2f}")


def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description='PPO Trading Model - 數據下載工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""範例:
  python download_data.py                                            # 最近 6 個月
  python download_data.py --months 12                                # 最近 12 個月
  python download_data.py --start 2020-01-01 --end 2025-12-31       # 指定日期範圍
  python download_data.py --start 2020-01-01 --end 2025-12-31 --test-start 2025-01-01
        """
    )
    parser.add_argument('--start', type=str, default=None,
                        help='開始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None,
                        help='結束日期 (YYYY-MM-DD)')
    parser.add_argument('--months', type=int, default=6,
                        help='倒推月數 (預設: 6，僅在未指定 --start/--end 時生效)')
    parser.add_argument('--test-start', type=str, default=None,
                        help='測試集開始日期 (YYYY-MM-DD)，未指定則按比例分割')
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                        help='交易對符號 (預設: BTCUSDT)')
    return parser.parse_args()


def main():
    """主函數"""
    args = parse_args()

    print("=" * 60)
    print("  📊 PPO Trading Model - 數據下載工具")
    print("=" * 60)

    downloader = BinanceDataDownloader(symbol=args.symbol, interval="1m")

    # 下載數據
    if args.start and args.end:
        full_df = downloader.download_by_date_range(args.start, args.end)
    else:
        full_df = downloader.download_data(months=args.months)

    # 驗證數據
    downloader.validate_data(full_df)

    # 生成摘要
    downloader.generate_summary(full_df)

    # 分割訓練集和測試集
    if args.test_start:
        train_df, test_df = downloader.split_train_test_by_date(full_df, args.test_start)
    else:
        train_df, test_df = downloader.split_train_test(full_df, train_months=5, test_months=1)

    # 保存數據
    train_file, test_file, full_file = downloader.save_data(train_df, test_df, full_df)

    print("\n" + "=" * 60)
    print("  🎉 數據下載完成！")
    print("=" * 60)
    print(f"\n💡 下一步:")
    print(f"   1. 檢查數據品質: 查看上面的驗證結果")
    print(f"   2. 開始訓練: python train.py")
    print(f"\n🚀 讓我們繼續賺大錢！💰\n")


if __name__ == "__main__":
    main()
