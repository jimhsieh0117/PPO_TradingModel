import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _smooth(series: pd.Series, window: int = 10) -> pd.Series:
    if series.empty:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def _plot_lines(
    df: pd.DataFrame,
    columns: list,
    title: str,
    ylabel: str,
    output_path: Path,
    smooth_window: int = 10,
) -> None:
    plt.figure(figsize=(10, 6))
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col]
        plt.plot(series, alpha=0.3, label=f"{col} (raw)")
        plt.plot(_smooth(series, smooth_window), label=f"{col} (smooth)")
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_training_plots(log_csv_path: str, output_dir: str) -> None:
    log_path = Path(log_csv_path)
    out_dir = Path(output_dir)
    _ensure_dir(out_dir)

    df = pd.read_csv(log_path)
    if df.empty:
        raise ValueError("training_log.csv is empty")

    sns.set_theme(style="whitegrid")

    _plot_lines(
        df,
        ["episode_reward_mean", "episode_reward_max", "episode_reward_min", "episode_reward_std"],
        "Reward Curves",
        "Reward",
        out_dir / "01_reward_curves.png",
    )

    _plot_lines(
        df,
        ["policy_loss", "value_loss", "entropy_loss", "total_loss"],
        "Loss Curves",
        "Loss",
        out_dir / "02_loss_curves.png",
    )

    _plot_lines(
        df,
        ["clip_fraction", "approx_kl", "explained_variance", "learning_rate"],
        "PPO Metrics",
        "Value",
        out_dir / "03_ppo_metrics.png",
    )

    _plot_lines(
        df,
        ["total_trades_per_episode", "long_ratio", "short_ratio", "hold_ratio", "close_ratio"],
        "Trading Behavior",
        "Value",
        out_dir / "04_trading_behavior.png",
    )

    _plot_lines(
        df,
        ["episode_profit", "win_rate", "profit_factor"],
        "Profit Metrics",
        "Value",
        out_dir / "05_profit_metrics.png",
    )

    _plot_lines(
        df,
        ["sharpe_ratio", "max_drawdown", "stop_loss_count", "daily_drawdown_violations"],
        "Risk Metrics",
        "Value",
        out_dir / "06_risk_metrics.png",
    )

    _plot_lines(
        df,
        ["episode_length", "episode_completion_rate", "avg_equity_curve_slope"],
        "Episode Stats",
        "Value",
        out_dir / "07_episode_stats.png",
    )

    _plot_lines(
        df,
        ["action_dist_0", "action_dist_1", "action_dist_2", "action_dist_3"],
        "Action Distribution",
        "Probability",
        out_dir / "08_action_distribution.png",
    )

    if "episode_return_pct" in df.columns:
        base = 10000.0
        returns = df["episode_return_pct"].fillna(0.0).to_numpy() / 100.0
        equity = base * np.cumprod(1.0 + returns)
        plt.figure(figsize=(10, 6))
        plt.plot(equity, label="Equity Curve")
        plt.title("Equity Curve Samples")
        plt.xlabel("Episode")
        plt.ylabel("Equity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "09_equity_curve_samples.png")
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training plots.")
    parser.add_argument("--log", required=True, help="Path to training_log.csv")
    parser.add_argument("--out", required=True, help="Output directory for plots")
    args = parser.parse_args()

    generate_training_plots(args.log, args.out)


if __name__ == "__main__":
    main()
