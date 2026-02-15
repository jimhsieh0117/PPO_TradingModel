"""
Walk Forward Analysis (WFA) - 滾動窗口驗證

使用方法：
    python wfa.py
    python wfa.py --config config.yaml

流程：
1. 載入完整數據集（一次）
2. 預計算全部特徵（一次，最大時間優化）
3. 生成 fold 排程
4. 並行訓練 + 回測每個 fold
5. 彙總結果、判定策略是否穩健
"""

from __future__ import annotations

import json
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import yaml
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

# Ensure project root is importable
_project_root = Path(__file__).parent
sys.path.insert(0, str(_project_root))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data loading (full dataset, no split)
# ---------------------------------------------------------------------------

def load_full_dataset(config: dict) -> pd.DataFrame:
    """Load the full dataset (with features) as a single DataFrame.

    Delegates to data_pipeline.load_full_data() which handles incremental
    download and processed-data caching.
    """
    from utils.data_pipeline import load_full_data
    return load_full_data(config)


# ---------------------------------------------------------------------------
# Fold schedule
# ---------------------------------------------------------------------------

def generate_fold_schedule(
    data_start: pd.Timestamp,
    data_end: pd.Timestamp,
    train_months: int,
    test_months: int,
    step_months: int,
) -> List[Dict]:
    """Generate a list of rolling-window folds."""
    folds: List[Dict] = []
    current = data_start

    while True:
        train_start = current
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)

        if test_end > data_end:
            break

        folds.append(
            {
                "fold_id": len(folds) + 1,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
        current += pd.DateOffset(months=step_months)

    return folds


# ---------------------------------------------------------------------------
# Data / feature slicing
# ---------------------------------------------------------------------------

def slice_data_and_features(
    full_df: pd.DataFrame,
    full_features: np.ndarray,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Slice DataFrame and pre-computed features by date range."""
    mask = (full_df["timestamp"] >= start_dt) & (full_df["timestamp"] < end_dt)
    idx = mask.values
    return full_df.loc[idx].copy().reset_index(drop=True), full_features[idx]


# ---------------------------------------------------------------------------
# Single-fold worker (runs in child process)
# ---------------------------------------------------------------------------

def _run_single_fold(args: tuple) -> Dict:
    """Train + backtest a single fold.  Designed for ProcessPoolExecutor."""

    fold_info, full_data_path, full_features_path, config, wfa_dir = args
    fold_id = fold_info["fold_id"]
    fold_dir = Path(wfa_dir) / f"fold_{fold_id:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # Redirect stdout to per-fold log (keep terminal clean)
    log_fp = open(fold_dir / "fold_log.txt", "w", encoding="utf-8")
    _orig_stdout = sys.stdout
    sys.stdout = log_fp

    try:
        # --- Load full dataset from disk (each process loads independently) ---
        full_df = pd.read_parquet(full_data_path)
        full_df["timestamp"] = pd.to_datetime(full_df["timestamp"])
        full_features = np.load(full_features_path)

        # --- Slice train / test ---
        train_df, train_features = slice_data_and_features(
            full_df, full_features, fold_info["train_start"], fold_info["train_end"]
        )
        test_df, test_features = slice_data_and_features(
            full_df, full_features, fold_info["test_start"], fold_info["test_end"]
        )

        # Release full data from memory
        del full_df, full_features

        if len(train_df) < 1000:
            return {"fold_id": fold_id, "status": "error",
                    "error": f"Not enough train data: {len(train_df)} bars"}
        if len(test_df) < 100:
            return {"fold_id": fold_id, "status": "error",
                    "error": f"Not enough test data: {len(test_df)} bars"}

        # ===================== TRAIN =====================
        # Limit numpy/torch threads to avoid contention
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"

        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from environment.trading_env import TradingEnv
        from agent.callbacks import TrainingMetricsCallback

        trading_cfg = config.get("trading", {})
        training_cfg = config.get("training", {})
        backtest_cfg = config.get("backtest", {})
        ppo_cfg = config.get("ppo", {})
        wfa_cfg = config.get("wfa", {})

        n_envs = wfa_cfg.get("n_cpu_per_fold", 2)
        total_timesteps = wfa_cfg.get("total_timesteps", 500_000)

        def make_env(rank: int):
            def _init():
                env = TradingEnv(
                    df=train_df,
                    initial_balance=backtest_cfg.get("initial_capital", 1_000_000),
                    leverage=trading_cfg.get("leverage", 1),
                    position_size_pct=trading_cfg.get("position_size_pct", 1.0),
                    stop_loss_pct=trading_cfg.get("stop_loss_pct", 0.015),
                    max_daily_drawdown=trading_cfg.get("daily_drawdown_limit", 0.10),
                    trading_fee=trading_cfg.get("taker_fee", 0.0004),
                    slippage=trading_cfg.get("slippage", 0.0),
                    episode_length=training_cfg.get("episode_length", 480),
                    feature_config=config.get("features", {}),
                    reward_config=config.get("reward", {}),
                    precomputed_features=train_features,
                    atr_stop_multiplier=trading_cfg.get("atr_stop_multiplier", 2.0),
                    trailing_stop=trading_cfg.get("trailing_stop", True),
                )
                env.reset(seed=42 + rank + fold_id * 100)
                return env
            return _init

        env = DummyVecEnv([make_env(i) for i in range(n_envs)])

        model = PPO(
            policy="MlpPolicy",
            env=env,
            device=ppo_cfg.get("device", "cpu"),
            learning_rate=ppo_cfg.get("learning_rate", 3e-4),
            n_steps=ppo_cfg.get("n_steps", 4096),
            batch_size=ppo_cfg.get("batch_size", 64),
            n_epochs=ppo_cfg.get("n_epochs", 8),
            gamma=ppo_cfg.get("gamma", 0.95),
            gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
            clip_range=ppo_cfg.get("clip_range", 0.2),
            ent_coef=ppo_cfg.get("ent_coef", 0.05),
            vf_coef=ppo_cfg.get("vf_coef", 0.5),
            max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
            verbose=0,
        )

        metrics_cb = TrainingMetricsCallback(
            log_path=str(fold_dir / "training_log.csv"),
            best_model_path=str(fold_dir / "ppo_best.zip"),
            best_log_path=str(fold_dir / "best_model_log.csv"),
            episode_length=training_cfg.get("episode_length", 480),
            initial_capital=backtest_cfg.get("initial_capital", 1_000_000),
            max_daily_drawdown=trading_cfg.get("daily_drawdown_limit", 0.10),
            enable_detailed_logging=False,
            verbose=0,
        )

        model.learn(total_timesteps=total_timesteps, callback=metrics_cb,
                     progress_bar=False)

        best_path = fold_dir / "ppo_best.zip"
        if not best_path.exists():
            model.save(str(best_path))

        env.close()

        # ===================== BACKTEST =====================
        from backtesting import Backtest
        from backtest.strategy import PPOTradingStrategy
        from backtest.run_backtest import normalize_ohlcv, build_metrics

        test_bt = test_df.copy()
        if "timestamp" in test_bt.columns:
            test_bt["timestamp"] = pd.to_datetime(test_bt["timestamp"])
            test_bt = test_bt.set_index("timestamp")
        test_bt = test_bt.sort_index()
        bt_data = normalize_ohlcv(test_bt)

        PPOTradingStrategy.model_path = str(best_path)
        PPOTradingStrategy.feature_config = config.get("features", {})
        PPOTradingStrategy.position_size_pct = float(
            trading_cfg.get("position_size_pct", 1.0)
        )
        PPOTradingStrategy.stop_loss_pct = float(
            trading_cfg.get("stop_loss_pct", 0.015)
        )
        PPOTradingStrategy.atr_stop_multiplier = float(
            trading_cfg.get("atr_stop_multiplier", 2.0)
        )
        PPOTradingStrategy.trailing_stop = bool(
            trading_cfg.get("trailing_stop", True)
        )
        PPOTradingStrategy.precomputed_features = test_features

        base_comm = float(backtest_cfg.get("commission", 0.0004))
        slippage = float(trading_cfg.get("slippage", 0.0))

        bt = Backtest(
            bt_data,
            PPOTradingStrategy,
            cash=float(backtest_cfg.get("initial_capital", 1_000_000)),
            commission=base_comm + slippage,
            trade_on_close=True,
            exclusive_orders=True,
            finalize_trades=True,
        )
        stats = bt.run()
        trades = stats.get("_trades", pd.DataFrame())

        metrics = build_metrics(stats, trades, bt_data.index)
        metrics["fold_id"] = fold_id
        metrics["train_start"] = str(fold_info["train_start"].date())
        metrics["train_end"] = str(fold_info["train_end"].date())
        metrics["test_start"] = str(fold_info["test_start"].date())
        metrics["test_end"] = str(fold_info["test_end"].date())
        metrics["train_bars"] = len(train_df)
        metrics["test_bars"] = len(test_df)
        metrics["status"] = "ok"

        with open(fold_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        return metrics

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        return {
            "fold_id": fold_id,
            "status": "error",
            "error": str(exc),
            "traceback": tb,
        }

    finally:
        sys.stdout = _orig_stdout
        log_fp.close()


# ---------------------------------------------------------------------------
# Aggregation & reporting
# ---------------------------------------------------------------------------

def aggregate_results(
    fold_metrics: List[Dict], criteria: Dict
) -> Dict:
    """Compile fold results and check pass/fail criteria."""
    ok = [m for m in fold_metrics if m.get("status") == "ok"]
    failed = [m for m in fold_metrics if m.get("status") != "ok"]

    if not ok:
        return {
            "verdict": "FAIL",
            "reason": "All folds failed",
            "total_folds": len(fold_metrics),
            "completed_folds": 0,
        }

    def _safe(vals):
        """Filter out NaN/None values."""
        return [v for v in vals if v is not None and not np.isnan(v)]

    returns = _safe([m.get("total_return_pct", 0) for m in ok])
    sharpes = _safe([m.get("sharpe_ratio", 0) for m in ok])
    drawdowns = _safe([m.get("max_drawdown_pct", 0) for m in ok])
    win_rates = _safe([m.get("win_rate_pct", 0) for m in ok])
    trades_per_day = _safe([m.get("avg_trades_per_day", 0) for m in ok])

    profitable = sum(1 for r in returns if r > 0)
    profitable_ratio = profitable / len(ok) if ok else 0
    avg_sharpe = float(np.mean(sharpes)) if sharpes else 0.0
    worst_dd = float(min(drawdowns)) if drawdowns else 0.0

    # Criteria checks
    min_ratio = criteria.get("min_profitable_folds_ratio", 0.67)
    min_sharpe = criteria.get("min_avg_sharpe", 1.3)
    max_dd = criteria.get("max_fold_drawdown_pct", -10.0)

    checks = {
        "profitable_ratio": {
            "pass": profitable_ratio >= min_ratio,
            "detail": f"{profitable_ratio:.1%} {'>=':} {min_ratio:.1%}",
        },
        "avg_sharpe": {
            "pass": avg_sharpe >= min_sharpe,
            "detail": f"{avg_sharpe:.2f} {'>=':} {min_sharpe:.2f}",
        },
        "max_drawdown": {
            "pass": worst_dd >= max_dd,
            "detail": f"{worst_dd:.2f}% {'>=':} {max_dd:.1f}%",
        },
    }

    verdict = "PASS" if all(c["pass"] for c in checks.values()) else "FAIL"

    return {
        "verdict": verdict,
        "checks": checks,
        "total_folds": len(fold_metrics),
        "completed_folds": len(ok),
        "failed_folds": len(failed),
        "profitable_folds": profitable,
        "profitable_ratio": round(profitable_ratio, 4),
        "avg_return_pct": round(float(np.mean(returns)), 2),
        "median_return_pct": round(float(np.median(returns)), 2),
        "best_return_pct": round(float(max(returns)), 2),
        "worst_return_pct": round(float(min(returns)), 2),
        "total_return_pct": round(float(sum(returns)), 2),
        "avg_sharpe": round(avg_sharpe, 2),
        "avg_win_rate_pct": round(float(np.mean(win_rates)), 1),
        "avg_trades_per_day": round(float(np.mean(trades_per_day)), 1),
        "worst_drawdown_pct": round(worst_dd, 2),
        "fold_details": ok,
        "fold_errors": [{"fold_id": m["fold_id"], "error": m.get("error", "")}
                        for m in failed],
    }


def print_wfa_report(summary: Dict) -> None:
    """Pretty-print the WFA results."""
    v = summary["verdict"]
    tag = "PASS" if v == "PASS" else "FAIL"

    print("\n" + "=" * 70)
    print(f"  Walk Forward Analysis Results   [{tag}]")
    print("=" * 70)
    print(f"  Folds:         {summary['completed_folds']} completed"
          f"  /  {summary['total_folds']} total"
          f"  ({summary['failed_folds']} failed)")
    print(f"  Profitable:    {summary['profitable_folds']}"
          f" / {summary['completed_folds']}"
          f"  ({summary['profitable_ratio']:.1%})")
    print("-" * 70)
    print(f"  Avg Return:    {summary['avg_return_pct']:+.2f}%")
    print(f"  Median Return: {summary['median_return_pct']:+.2f}%")
    print(f"  Best / Worst:  {summary['best_return_pct']:+.2f}%"
          f"  /  {summary['worst_return_pct']:+.2f}%")
    print(f"  Cumulative:    {summary['total_return_pct']:+.2f}%")
    print(f"  Avg Sharpe:    {summary['avg_sharpe']:.2f}")
    print(f"  Avg Win Rate:  {summary['avg_win_rate_pct']:.1f}%")
    print(f"  Avg Trades/Day:{summary['avg_trades_per_day']:.1f}")
    print(f"  Worst DD:      {summary['worst_drawdown_pct']:.2f}%")
    print("-" * 70)
    print("  Criteria checks:")
    for name, chk in summary.get("checks", {}).items():
        status = "PASS" if chk["pass"] else "FAIL"
        print(f"    [{status}] {name}: {chk['detail']}")
    print("=" * 70)

    # Per-fold table
    details = summary.get("fold_details", [])
    if details:
        print(f"\n  {'Fold':>4}  {'Test Period':<25} {'Return':>8}"
              f"  {'Sharpe':>7}  {'Trades':>6}  {'WinRate':>7}  {'MaxDD':>7}")
        print("  " + "-" * 68)
        for m in sorted(details, key=lambda x: x["fold_id"]):
            period = f"{m['test_start']} ~ {m['test_end']}"
            print(f"  {m['fold_id']:>4}  {period:<25}"
                  f" {m['total_return_pct']:>+7.2f}%"
                  f"  {m['sharpe_ratio']:>7.2f}"
                  f"  {m.get('total_trades', 0):>6}"
                  f"  {m['win_rate_pct']:>6.1f}%"
                  f"  {m['max_drawdown_pct']:>6.2f}%")
        print()

    # Errors
    errors = summary.get("fold_errors", [])
    if errors:
        print("  Fold errors:")
        for e in errors:
            print(f"    Fold {e['fold_id']}: {e['error']}")
        print()


def generate_wfa_plot(summary: Dict, output_path: Path) -> None:
    """Generate a bar chart of per-fold returns."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        details = sorted(summary.get("fold_details", []),
                         key=lambda x: x["fold_id"])
        if not details:
            return

        fold_ids = [m["fold_id"] for m in details]
        returns = [m["total_return_pct"] for m in details]
        colors = ["#2ca02c" if r > 0 else "#d62728" for r in returns]

        fig, ax = plt.subplots(figsize=(max(12, len(fold_ids) * 0.5), 5))
        ax.bar(range(len(fold_ids)), returns, color=colors, width=0.7)
        ax.set_xticks(range(len(fold_ids)))
        ax.set_xticklabels(fold_ids, fontsize=7)
        ax.set_xlabel("Fold")
        ax.set_ylabel("Return (%)")
        ax.set_title(
            f"WFA Per-Fold Returns  |  Avg: {summary['avg_return_pct']:+.2f}%"
            f"  |  Sharpe: {summary['avg_sharpe']:.2f}"
            f"  |  [{summary['verdict']}]"
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"  Plot saved: {output_path}")
    except Exception as exc:
        print(f"  Plot generation failed: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Walk Forward Analysis for PPO")
    parser.add_argument("--config", default="config.yaml", help="Config file")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  Walk Forward Analysis (WFA)")
    print("=" * 70)

    config = load_config(args.config)
    wfa_cfg = config.get("wfa", {})

    train_months = wfa_cfg.get("train_window_months", 12)
    test_months = wfa_cfg.get("test_window_months", 2)
    step_months = wfa_cfg.get("step_months", 2)
    max_workers = wfa_cfg.get("max_parallel_folds", 3)
    total_timesteps = wfa_cfg.get("total_timesteps", 500_000)
    criteria = wfa_cfg.get("pass_criteria", {})

    # ---- 1. Load full dataset (ONCE) ----
    print("\n[1/5] Loading full dataset...")
    full_df = load_full_dataset(config)
    data_start = full_df["timestamp"].min()
    data_end = full_df["timestamp"].max()
    print(f"      {len(full_df):,} bars  |  {data_start} ~ {data_end}")

    # ---- 2. Extract pre-computed features from pipeline data ----
    print("\n[2/5] Extracting pre-computed features...")
    from utils.data_pipeline import extract_features

    full_features = extract_features(full_df)
    print(f"      Feature shape: {full_features.shape}")

    # ---- 3. Generate fold schedule ----
    print("\n[3/5] Generating fold schedule...")
    folds = generate_fold_schedule(
        data_start, data_end, train_months, test_months, step_months
    )
    print(f"      {len(folds)} folds"
          f"  (train={train_months}mo, test={test_months}mo, step={step_months}mo)")

    if not folds:
        print("      ERROR: No valid folds. Check date range and window sizes.")
        return

    for f in folds[:3]:
        print(f"      Fold {f['fold_id']:>2}: "
              f"Train {str(f['train_start'].date())} ~ {str(f['train_end'].date())} "
              f"| Test {str(f['test_start'].date())} ~ {str(f['test_end'].date())}")
    if len(folds) > 3:
        print(f"      ... ({len(folds) - 3} more folds)")

    # ---- Create WFA output directory ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    wfa_dir = Path(config.get("training", {}).get("model_save_dir", "models")) / f"wfa_{ts}"
    wfa_dir.mkdir(parents=True, exist_ok=True)

    # Save full data & features to disk for worker processes
    tmp_data_path = wfa_dir / "_full_data.parquet"
    tmp_feat_path = wfa_dir / "_full_features.npy"
    full_df.to_parquet(tmp_data_path, index=False)
    np.save(tmp_feat_path, full_features)

    # Free memory in main process
    del full_df, full_features

    # Save config snapshot
    import shutil
    shutil.copy(args.config, wfa_dir / "config.yaml")

    # ---- 4. Run folds in parallel ----
    print(f"\n[4/5] Running {len(folds)} folds"
          f"  ({max_workers} parallel, {total_timesteps:,} steps each)...")
    print(f"      Output: {wfa_dir}")
    print()

    worker_args = [
        (fold, str(tmp_data_path), str(tmp_feat_path), config, str(wfa_dir))
        for fold in folds
    ]

    fold_metrics: List[Dict] = []
    t0 = time.time()

    from tqdm import tqdm

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_run_single_fold, wa): wa[0]["fold_id"]
            for wa in worker_args
        }

        pbar = tqdm(total=len(folds), desc="WFA Folds", unit="fold",
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

        for future in as_completed(future_map):
            fid = future_map[future]
            try:
                result = future.result()
                fold_metrics.append(result)
                status = result.get("status", "error")
                if status == "ok":
                    ret = result.get("total_return_pct", 0) or 0
                    sr = result.get("sharpe_ratio", 0) or 0
                    wr = result.get("win_rate_pct", 0) or 0
                    # Replace NaN with 0 for display
                    ret = 0.0 if np.isnan(ret) else ret
                    sr = 0.0 if np.isnan(sr) else sr
                    wr = 0.0 if np.isnan(wr) else wr
                    pbar.set_postfix_str(
                        f"Fold {fid:>2} {ret:>+.1f}% Sharpe={sr:.1f}")
                    tqdm.write(
                        f"  [OK] Fold {fid:>2}/{len(folds)}  "
                        f"Return: {ret:>+7.2f}%  "
                        f"Sharpe: {sr:>6.2f}  "
                        f"WinRate: {wr:>5.1f}%")
                else:
                    err = result.get("error", "unknown")
                    tqdm.write(f"  [ERR] Fold {fid:>2}/{len(folds)}  {err}")
            except Exception as exc:
                tqdm.write(f"  [ERR] Fold {fid:>2}/{len(folds)}  {exc}")
                fold_metrics.append(
                    {"fold_id": fid, "status": "error", "error": str(exc)}
                )
            pbar.update(1)

        pbar.close()

    elapsed = time.time() - t0
    print(f"\n      All folds finished in {elapsed:.1f}s"
          f"  ({elapsed / len(folds):.1f}s avg per fold)")

    # Clean up temp files
    tmp_data_path.unlink(missing_ok=True)
    tmp_feat_path.unlink(missing_ok=True)

    # ---- 5. Aggregate & report ----
    print("\n[5/5] Aggregating results...")
    summary = aggregate_results(fold_metrics, criteria)
    summary["wfa_config"] = {
        "train_window_months": train_months,
        "test_window_months": test_months,
        "step_months": step_months,
        "total_timesteps": total_timesteps,
        "max_parallel_folds": max_workers,
    }
    summary["elapsed_seconds"] = round(elapsed, 1)

    # Save summary
    with open(wfa_dir / "wfa_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    # Print report
    print_wfa_report(summary)

    # Generate plot
    generate_wfa_plot(summary, wfa_dir / "wfa_returns.png")

    print(f"\n  Results saved: {wfa_dir}")
    print(f"  Summary: {wfa_dir / 'wfa_summary.json'}")


if __name__ == "__main__":
    main()
