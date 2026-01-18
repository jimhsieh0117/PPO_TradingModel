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


def _plot_single_metric(
    timesteps: np.ndarray,
    values: np.ndarray,
    title: str,
    ylabel: str,
    output_path: Path,
    smooth_window: int = 10,
) -> None:
    """
    繪製單一指標的圖表（深色主題）

    Args:
        timesteps: X 軸數據（訓練步數）
        values: Y 軸數據（指標值）
        title: 圖表標題
        ylabel: Y 軸標籤
        output_path: 輸出路徑
        smooth_window: 滾動平均窗口
    """
    # 設定深色主題
    plt.style.use('dark_background')

    fig, ax = plt.subplots(figsize=(12, 6))

    # 原始數據（半透明）
    ax.plot(timesteps, values, alpha=0.3, linewidth=1, label='Raw', color='#66B2FF')

    # 滾動平均（清晰）
    if len(values) >= smooth_window:
        smoothed = pd.Series(values).rolling(window=smooth_window, min_periods=1).mean()
        ax.plot(timesteps, smoothed, linewidth=2, label=f'Smooth (window={smooth_window})', color='#FF6B6B')

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Timesteps', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, facecolor='#1e1e1e')
    plt.close()


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


def plot_training_metrics(log_csv_path: str, output_dir: str = None, smooth_window: int = 10) -> None:
    """
    為每個訓練指標生成獨立的圖表（深色主題）

    Args:
        log_csv_path: training_log.csv 的路徑
        output_dir: 輸出目錄，預設為 log 所在目錄下的 train_log_png
        smooth_window: 滾動平均窗口（預設 10 個 episodes）
    """
    log_path = Path(log_csv_path)
    if not log_path.exists():
        raise FileNotFoundError(f"找不到訓練日誌: {log_csv_path}")

    # 如果未指定輸出目錄，使用 log 所在目錄下的 train_log_png
    if output_dir is None:
        output_dir = log_path.parent / "train_log_png"
    else:
        output_dir = Path(output_dir)

    _ensure_dir(output_dir)

    print(f"\n[PLOT] 開始生成訓練監控圖表...")
    print(f"   輸入: {log_csv_path}")
    print(f"   輸出: {output_dir}")
    print(f"   滾動窗口: {smooth_window} episodes\n")

    # 讀取訓練日誌
    df = pd.read_csv(log_path)
    if df.empty:
        raise ValueError("training_log.csv 是空的")

    timesteps = df['timesteps'].values

    # 定義所有要繪製的指標（指標名稱, 顯示標題, Y軸標籤）
    metrics = [
        # 獎勵指標
        ('episode_reward_mean', 'Episode Reward Mean', 'Reward'),
        ('episode_reward_std', 'Episode Reward Std', 'Std'),
        ('episode_reward_max', 'Episode Reward Max', 'Reward'),
        ('episode_reward_min', 'Episode Reward Min', 'Reward'),
        ('cumulative_reward', 'Cumulative Reward', 'Total Reward'),

        # 損失函數
        ('policy_loss', 'Policy Loss', 'Loss'),
        ('value_loss', 'Value Loss', 'Loss'),
        ('entropy_loss', 'Entropy Loss', 'Loss'),
        ('total_loss', 'Total Loss', 'Loss'),

        # PPO 指標
        ('clip_fraction', 'Clip Fraction', 'Fraction'),
        ('approx_kl', 'Approx KL Divergence', 'KL'),
        ('explained_variance', 'Explained Variance', 'Variance'),
        ('learning_rate', 'Learning Rate', 'LR'),

        # 交易行為
        ('total_trades_per_episode', 'Total Trades per Episode', 'Count'),
        ('long_ratio', 'Long Action Ratio', 'Ratio'),
        ('short_ratio', 'Short Action Ratio', 'Ratio'),
        ('hold_ratio', 'Hold Action Ratio', 'Ratio'),
        ('close_ratio', 'Close Action Ratio', 'Ratio'),
        ('avg_holding_time', 'Average Holding Time', 'Bars'),

        # 盈利指標
        ('episode_profit', 'Episode Profit (USDT)', 'USDT'),
        ('episode_return_pct', 'Episode Return (%)', 'Return (%)'),
        ('win_rate', 'Win Rate', 'Win Rate'),
        ('profit_factor', 'Profit Factor', 'Factor'),

        # 風險指標
        ('sharpe_ratio', 'Sharpe Ratio', 'Ratio'),
        ('max_drawdown', 'Max Drawdown', 'Drawdown (%)'),
        ('stop_loss_count', 'Stop Loss Count', 'Count'),
        ('daily_drawdown_violations', 'Daily Drawdown Violations', 'Count'),

        # Episode 統計
        ('episode_length', 'Episode Length', 'Steps'),
        ('episode_completion_rate', 'Episode Completion Rate', 'Rate'),
        ('action_entropy', 'Action Entropy (Exploration)', 'Entropy'),

        # 動作分佈
        ('action_dist_0', 'Action Distribution - Close (0)', 'Probability'),
        ('action_dist_1', 'Action Distribution - Long (1)', 'Probability'),
        ('action_dist_2', 'Action Distribution - Short (2)', 'Probability'),
        ('action_dist_3', 'Action Distribution - Hold (3)', 'Probability'),
    ]

    # 逐個生成圖表
    generated_count = 0
    for idx, (col_name, title, ylabel) in enumerate(metrics, start=1):
        if col_name not in df.columns:
            print(f"   [SKIP] {col_name} (欄位不存在)")
            continue

        values = df[col_name].values

        # 跳過全為 NaN 的欄位
        if pd.isna(values).all():
            print(f"   [SKIP] {col_name} (全為 NaN)")
            continue

        output_path = output_dir / f"{idx:02d}_{col_name}.png"

        try:
            _plot_single_metric(
                timesteps=timesteps,
                values=values,
                title=title,
                ylabel=ylabel,
                output_path=output_path,
                smooth_window=smooth_window
            )
            generated_count += 1
            print(f"   [OK] [{generated_count:02d}] {output_path.name}")
        except Exception as e:
            print(f"   [ERROR] 生成 {col_name} 失敗: {e}")

    print(f"\n[DONE] 圖表生成完成！共生成 {generated_count} 張圖片")
    print(f"[SAVE] 保存位置: {output_dir}\n")


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
