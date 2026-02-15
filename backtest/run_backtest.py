"""
Run backtesting.py evaluation for a trained PPO model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import yaml
from backtesting import Backtest

from backtest.strategy import PPOTradingStrategy


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def find_latest_csv(data_dir: Path, pattern: str) -> Path:
    candidates = []
    for path in data_dir.glob(pattern):
        if path.name.endswith("_latest.csv"):
            continue
        try:
            if path.is_file() and path.stat().st_size > 0:
                candidates.append(path)
        except OSError:
            continue
    if not candidates:
        raise FileNotFoundError(f"No data files found in {data_dir} with pattern {pattern}")
    return sorted(candidates)[-1]


def load_test_data(config: Dict, data_path: Optional[str]) -> Tuple[pd.DataFrame, Path]:
    # 如果明確指定了 data_path，直接使用
    if data_path:
        path = Path(data_path)
        if str(path).endswith('.parquet'):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.set_index("timestamp")
        df = df.sort_index()
        return df, path

    # 使用 data pipeline 自動取得測試數據（含 OHLCV + 特徵）
    from utils.data_pipeline import ensure_data_ready

    data_config = config.get("data", {})
    _train_df, test_df = ensure_data_ready(config)

    # test_df 來自 pipeline，timestamp 是列不是索引
    if "timestamp" in test_df.columns:
        test_df["timestamp"] = pd.to_datetime(test_df["timestamp"], errors="coerce")
        test_df = test_df.set_index("timestamp")
    test_df = test_df.sort_index()

    # 使用處理後資料路徑作為日誌顯示
    processed_dir = Path(data_config.get("processed_data_dir", "data/processed"))
    symbol = data_config.get("symbol", "BTCUSDT")
    interval = config.get("trading", {}).get("timeframe", "1m")
    path = processed_dir / f"{symbol}_{interval}.parquet"

    return test_df, path


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    column_map = {col.lower(): col for col in df.columns}
    required = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in column_map]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    ohlcv = df[[column_map[col] for col in required]].copy()
    ohlcv = ohlcv.rename(columns={
        column_map["open"]: "Open",
        column_map["high"]: "High",
        column_map["low"]: "Low",
        column_map["close"]: "Close",
        column_map["volume"]: "Volume",
    })
    return ohlcv


def find_latest_run_dir(models_dir: Path) -> Path:
    run_dirs = [path for path in models_dir.glob("run_*") if path.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run_* directories found in {models_dir}")
    return sorted(run_dirs)[-1]


def resolve_model_path(run_dir: Path, model_path: Optional[str]) -> Path:
    if model_path:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        return path

    for name in [
        "ppo_trading_model_best.zip",
        "ppo_trading_model_final.zip",
        "ppo_trading_model_interrupted.zip",
    ]:
        candidate = run_dir / name
        if candidate.exists():
            return candidate

    checkpoints = sorted((run_dir / "checkpoints").glob("*.zip"))
    if checkpoints:
        return checkpoints[-1]

    raise FileNotFoundError(f"No model found in {run_dir}")


def max_consecutive(flags) -> int:
    best = 0
    current = 0
    for flag in flags:
        if flag:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def count_stop_losses(trades: pd.DataFrame) -> int:
    if trades.empty or "SL" not in trades.columns:
        return 0
    sl = trades["SL"]
    exit_price = trades["ExitPrice"]
    size = trades["Size"]
    valid = sl.notna()
    if not valid.any():
        return 0

    tol = 1e-8
    long_hit = (size > 0) & (exit_price <= sl + tol)
    short_hit = (size < 0) & (exit_price >= sl - tol)
    return int((valid & (long_hit | short_hit)).sum())


def build_metrics(stats: pd.Series, trades: pd.DataFrame, data_index: pd.Index = None) -> Dict:
    wins = trades["PnL"] > 0 if not trades.empty else []
    losses = trades["PnL"] < 0 if not trades.empty else []

    avg_duration = stats.get("Avg. Trade Duration")
    avg_duration_str = str(avg_duration) if avg_duration is not None else "n/a"

    # 計算回測時間範圍
    start_date = None
    end_date = None
    duration_days = 0.0
    avg_trades_per_day = 0.0

    if data_index is not None and len(data_index) > 0:
        start_date = data_index[0]
        end_date = data_index[-1]
        if hasattr(start_date, 'strftime'):
            duration = end_date - start_date
            duration_days = duration.total_seconds() / 86400  # 轉換為天數
            total_trades = int(stats.get("# Trades", 0))
            avg_trades_per_day = total_trades / duration_days if duration_days > 0 else 0.0

    metrics = {
        "backtest_start": start_date.strftime("%Y-%m-%d %H:%M") if start_date and hasattr(start_date, 'strftime') else "n/a",
        "backtest_end": end_date.strftime("%Y-%m-%d %H:%M") if end_date and hasattr(end_date, 'strftime') else "n/a",
        "backtest_duration_days": round(duration_days, 2),
        "avg_trades_per_day": round(avg_trades_per_day, 2),
        "total_return_pct": float(stats.get("Return [%]", 0.0)),
        "annualized_return_pct": float(stats.get("Return (Ann.) [%]", 0.0)),
        "sharpe_ratio": float(stats.get("Sharpe Ratio", 0.0)),
        "max_drawdown_pct": float(stats.get("Max. Drawdown [%]", 0.0)),
        "total_trades": int(stats.get("# Trades", 0)),
        "win_rate_pct": float(stats.get("Win Rate [%]", 0.0)),
        "profit_factor": float(stats.get("Profit Factor", 0.0)),
        "avg_holding_time": avg_duration_str,
        "avg_holding_bars": float((trades["ExitBar"] - trades["EntryBar"]).mean())
        if not trades.empty else 0.0,
        "max_consecutive_wins": max_consecutive(wins),
        "max_consecutive_losses": max_consecutive(losses),
        "stop_loss_count": count_stop_losses(trades),
    }
    return metrics


def main() -> None:
    import time

    parser = argparse.ArgumentParser(description="Run PPO backtest with backtesting.py")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--data", help="Test CSV path (optional)")
    parser.add_argument("--model", help="Model path (optional)")
    parser.add_argument("--run-dir", help="Run directory for outputs (optional)")
    parser.add_argument("--output-dir", help="Output directory (optional)")
    args = parser.parse_args()

    print("=" * 60)
    print("  PPO Trading Model - Backtest")
    print("=" * 60)

    config = load_config(args.config)

    print("\n[1/4] Loading test data...")
    df_raw, data_path = load_test_data(config, args.data)
    bt_data = normalize_ohlcv(df_raw)
    print(f"      Loaded {len(bt_data):,} bars from {data_path.name}")

    models_dir = Path(config.get("training", {}).get("model_save_dir", "models"))
    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run_dir(models_dir)
    model_path = resolve_model_path(run_dir, args.model)

    backtest_config = config.get("backtest", {})
    trading_config = config.get("trading", {})

    print(f"\n[2/4] Loading model: {model_path.name}")
    PPOTradingStrategy.model_path = str(model_path)
    PPOTradingStrategy.feature_config = config.get("features", {})
    PPOTradingStrategy.position_size_pct = float(trading_config.get("position_size_pct", 0.15))
    PPOTradingStrategy.stop_loss_pct = float(trading_config.get("stop_loss_pct", 0.015))

    # 手續費 + 滑點（backtesting.py 的 commission 為百分比，與滑點單位一致）
    base_commission = float(backtest_config.get("commission", 0.0004))
    slippage = float(trading_config.get("slippage", 0.0))
    effective_commission = base_commission + slippage
    print(f"      Commission: {base_commission:.4%} + Slippage: {slippage:.4%} = {effective_commission:.4%}")

    bt = Backtest(
        bt_data,
        PPOTradingStrategy,
        cash=float(backtest_config.get("initial_capital", 10000)),
        commission=effective_commission,
        trade_on_close=True,
        exclusive_orders=True,
    )

    print("\n[3/4] Running backtest...")
    start_time = time.time()
    stats = bt.run()
    elapsed = time.time() - start_time
    bars_per_sec = len(bt_data) / elapsed if elapsed > 0 else 0
    print(f"      Completed in {elapsed:.2f}s ({bars_per_sec:,.0f} bars/sec)")

    print("\n[4/4] Saving results...")
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "backtest_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    trades = stats.get("_trades", pd.DataFrame())
    equity_curve = stats.get("_equity_curve", pd.DataFrame())

    if backtest_config.get("html_report", True):
        bt.plot(results=stats, filename=str(output_dir / "backtest.html"), open_browser=False)

    if backtest_config.get("trades_csv", True) and not trades.empty:
        trades.to_csv(output_dir / "trades.csv", index=False)

    if backtest_config.get("plots", True) and not equity_curve.empty:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(equity_curve.index, equity_curve["Equity"], color="#1f77b4", linewidth=1.5)
        ax.set_title("Equity Curve")
        ax.set_xlabel("Time")
        ax.set_ylabel("Equity")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "equity_curve.png", dpi=150)
        plt.close(fig)

    metrics = build_metrics(stats, trades, bt_data.index)
    metrics.update({
        "data_path": str(data_path),
        "model_path": str(model_path),
        "run_dir": str(run_dir),
        "backtest_time_sec": elapsed,
        "bars_per_sec": bars_per_sec,
    })
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=True)

    print("\n" + "=" * 60)
    print("  Backtest Results")
    print("=" * 60)
    print(f"  Period:          {metrics['backtest_start']} ~ {metrics['backtest_end']}")
    print(f"  Duration:        {metrics['backtest_duration_days']:.1f} days")
    print("-" * 60)
    print(f"  Total Return:    {metrics['total_return_pct']:+.2f}%")
    print(f"  Ann. Return:     {metrics['annualized_return_pct']:+.2f}%")
    print(f"  Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:    {metrics['max_drawdown_pct']:.2f}%")
    print("-" * 60)
    print(f"  Total Trades:    {metrics['total_trades']}")
    print(f"  Avg Trades/Day:  {metrics['avg_trades_per_day']:.1f}")
    print(f"  Win Rate:        {metrics['win_rate_pct']:.1f}%")
    print(f"  Profit Factor:   {metrics['profit_factor']:.2f}")
    print(f"  Avg Holding:     {metrics['avg_holding_time']}")
    print(f"  Avg Hold Bars:   {metrics['avg_holding_bars']:.1f}")
    print("-" * 60)
    print(f"  Max Consec Wins: {metrics['max_consecutive_wins']}")
    print(f"  Max Consec Loss: {metrics['max_consecutive_losses']}")
    print(f"  Stop Losses:     {metrics['stop_loss_count']}")
    print("-" * 60)
    print(f"  Backtest Time:   {metrics['backtest_time_sec']:.1f}s ({metrics['bars_per_sec']:,.0f} bars/sec)")
    print(f"  Data:            {metrics['data_path']}")
    print(f"  Model:           {metrics['model_path']}")
    print("=" * 60)
    print(f"\n  Output: {output_dir}")


if __name__ == "__main__":
    main()
