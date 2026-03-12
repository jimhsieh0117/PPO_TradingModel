"""
PPO Trading Model - Optuna 超參數優化腳本

使用方法：
    # Phase 1：優化 PPO 核心參數
    python optimize.py --phase phase1_ppo --n-trials 50

    # Phase 2：固定 Phase 1 最佳，優化獎勵函數
    python optimize.py --phase phase2_reward --n-trials 40

    # Phase 3：精煉最敏感參數
    python optimize.py --phase phase3_combined --n-trials 30

    # 中斷後繼續
    python optimize.py --phase phase1_ppo --n-trials 50 --resume

    # 自訂訓練步數
    python optimize.py --phase phase1_ppo --n-trials 30 --timesteps 300000
"""

import argparse
import copy
import gc
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import optuna
import yaml
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3.common.callbacks import BaseCallback

from utils.config_utils import load_config, deep_merge
from utils.data_pipeline import ensure_data_ready, extract_features


# ============================================================
# 搜索空間定義
# ============================================================
SEARCH_SPACES: Dict[str, Dict[str, tuple]] = {
    "phase1_ppo": {
        "ppo.learning_rate": ("log_float", 3e-5, 5e-4),
        "ppo.ent_coef": ("float", 0.1, 0.3),
        "ppo.gamma": ("float", 0.9, 0.99),
        "ppo.n_epochs": ("int", 4, 16),
        "ppo.batch_size": ("categorical", [32, 64, 128, 256]),
        "ppo.vf_coef": ("float", 0.3, 1.0),
    },
    "phase2_reward": {
        "reward.pnl_reward_scale": ("int", 300, 1000, 50),
        "reward.floating_reward_scale": ("int", 10, 80, 5),
        "reward.take_profit_multiplier": ("float", 1.0, 2.0),
        "reward.stop_loss_extra_penalty": ("float", 1.0, 8.0),
        "reward.holding_bonus_max": ("float", 0.5, 3.0),
    },
    "phase3_combined": {
        # Phase 1+2 最敏感參數，範圍從結果縮窄
        # 預設包含常見敏感參數，使用者可編輯
        "ppo.learning_rate": ("log_float", 5e-5, 3e-4),
        "ppo.ent_coef": ("float", 0.1, 0.25),
        "reward.pnl_reward_scale": ("int", 400, 800, 50),
        "reward.floating_reward_scale": ("int", 15, 60, 5),
    },
}


# ============================================================
# Optuna Pruning Callback for SB3
# ============================================================
class OptunaPruningCallback(BaseCallback):
    """每 N 步回報 mean episode reward 給 Optuna，支援 pruning。"""

    def __init__(self, trial: optuna.Trial, report_interval: int = 50000):
        super().__init__(verbose=0)
        self.trial = trial
        self.report_interval = report_interval
        self._last_report_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_report_step < self.report_interval:
            return True

        self._last_report_step = self.num_timesteps

        # 從 SB3 internal logger 取得 mean reward
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
        else:
            mean_reward = 0.0

        self.trial.report(mean_reward, step=self.num_timesteps)

        if self.trial.should_prune():
            raise optuna.TrialPruned(
                f"Trial pruned at step {self.num_timesteps} "
                f"(mean_reward={mean_reward:.2f})"
            )

        return True


# ============================================================
# 工具函數
# ============================================================
def suggest_param(trial: optuna.Trial, name: str, spec: tuple) -> Any:
    """根據搜索空間定義，向 trial 建議參數值。"""
    param_type = spec[0]

    if param_type == "log_float":
        return trial.suggest_float(name, spec[1], spec[2], log=True)
    elif param_type == "float":
        return trial.suggest_float(name, spec[1], spec[2])
    elif param_type == "int":
        step = spec[3] if len(spec) > 3 else 1
        return trial.suggest_int(name, spec[1], spec[2], step=step)
    elif param_type == "categorical":
        return trial.suggest_categorical(name, spec[1])
    else:
        raise ValueError(f"Unknown param type: {param_type}")


def apply_trial_params(config: dict, trial: optuna.Trial, phase: str) -> dict:
    """將 trial 建議的參數套用到 config（dot-notation 設定巢狀值）。"""
    config = copy.deepcopy(config)
    search_space = SEARCH_SPACES[phase]

    for dotted_key, spec in search_space.items():
        value = suggest_param(trial, dotted_key, spec)

        # dot-notation → 巢狀 dict 設定
        keys = dotted_key.split(".")
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    return config


def normalize(value: float, low: float, high: float) -> float:
    """將數值正規化到 [0, 1] 範圍，超出範圍則 clip。"""
    if high == low:
        return 0.5
    return max(0.0, min(1.0, (value - low) / (high - low)))


def compute_composite_score(metrics: dict) -> float:
    """
    計算加權複合分數（機構交易員評估方式）。

    score = 0.40 × norm(sharpe, -2, 5)
           + 0.25 × norm(return%, -20, 50)
           + 0.20 × norm(-|MDD|, -30, 0)
           + 0.15 × norm(profit_factor, 0, 3)

    硬約束：total_trades < 10 → -1.0
    """
    total_trades = metrics.get("total_trades", 0)
    if total_trades < 10:
        return -1.0

    sharpe = metrics.get("sharpe_ratio", 0.0)
    total_return = metrics.get("total_return_pct", 0.0)
    max_dd = metrics.get("max_drawdown_pct", 0.0)  # 已是負數
    profit_factor = metrics.get("profit_factor", 0.0)

    score = (
        0.40 * normalize(sharpe, -2.0, 5.0)
        + 0.25 * normalize(total_return, -20.0, 50.0)
        + 0.20 * normalize(max_dd, -30.0, 0.0)
        + 0.15 * normalize(profit_factor, 0.0, 3.0)
    )

    return round(score, 4)


def export_best_params(study: optuna.Study, output_dir: Path, phase: str) -> None:
    """將最佳參數輸出為 config_local.yaml 相容格式。"""
    best = study.best_trial
    params_nested: Dict[str, Any] = {}

    for dotted_key, value in best.params.items():
        keys = dotted_key.split(".")
        d = params_nested
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    out_path = output_dir / "best_params.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(params_nested, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"\n[EXPORT] 最佳參數已輸出: {out_path}")
    print(f"  Score: {best.value:.4f}")
    print(f"  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")


def export_optimization_report(study: optuna.Study, output_dir: Path) -> None:
    """生成 Optuna 視覺化報告（HTML）。"""
    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate,
            plot_slice,
        )

        figs = {
            "optimization_history": plot_optimization_history(study),
            "param_importances": plot_param_importances(study),
            "parallel_coordinate": plot_parallel_coordinate(study),
            "slice_plot": plot_slice(study),
        }

        html_parts = [
            "<html><head><title>Optuna Optimization Report</title></head><body>",
            f"<h1>Optimization Report - {study.study_name}</h1>",
            f"<p>Best Score: {study.best_value:.4f}</p>",
            f"<p>Total Trials: {len(study.trials)}</p>",
        ]

        for name, fig in figs.items():
            html_parts.append(f"<h2>{name.replace('_', ' ').title()}</h2>")
            html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))

        html_parts.append("</body></html>")

        report_path = output_dir / "optimization_report.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html_parts))

        print(f"[REPORT] HTML 報告已生成: {report_path}")

    except Exception as e:
        print(f"[WARN] HTML 報告生成失敗（需要 plotly）: {e}")


# ============================================================
# 目標函數
# ============================================================
def _create_env_direct(train_df, precomputed_features, config, n_cpu):
    """
    直接建立訓練環境，跳過 train.py 中的 extract_features 重複計算。
    使用 DummyVecEnv 避免每 trial spawn 子進程的開銷。
    """
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.utils import set_random_seed
    from environment.trading_env import TradingEnv

    trading_config = config.get('trading', {})
    training_config = config.get('training', {})
    backtest_config = config.get('backtest', {})
    misc_config = config.get('misc', {})

    seed = misc_config.get('random_seed', None)
    if seed is not None:
        set_random_seed(seed)

    n_envs = max(1, n_cpu)

    def make_env(rank: int):
        def _init():
            env = TradingEnv(
                df=train_df,
                initial_balance=backtest_config.get('initial_capital', 10000.0),
                leverage=trading_config.get('leverage', 10),
                position_size_pct=trading_config.get('position_size_pct', 0.15),
                stop_loss_pct=trading_config.get('stop_loss_pct', 0.015),
                max_daily_drawdown=trading_config.get('daily_drawdown_limit', 0.10),
                trading_fee=trading_config.get('taker_fee', 0.0004),
                slippage=trading_config.get('slippage', 0.0),
                episode_length=training_config.get('episode_length', 1440),
                feature_config=config.get('features', {}),
                reward_config=config.get('reward', {}),
                precomputed_features=precomputed_features,
                atr_stop_multiplier=trading_config.get('atr_stop_multiplier', 2.0),
                trailing_stop=trading_config.get('trailing_stop', True),
            )
            if seed is not None:
                env.reset(seed=seed + rank)
            return env
        return _init

    return DummyVecEnv([make_env(i) for i in range(n_envs)])


def _run_backtest_fast(config, trial_dir, test_df_cached, bt_data_cached, test_features_cached):
    """
    快速回測：使用預載的測試數據，跳過重複的 data I/O 和 feature extraction。
    """
    import time as _time
    from backtesting import Backtest
    from backtest.strategy import PPOTradingStrategy
    from backtest.run_backtest import resolve_model_path, build_metrics, normalize_ohlcv

    trial_dir = Path(trial_dir)
    model_path = resolve_model_path(trial_dir, None)

    trading_config = config.get("trading", {})
    backtest_config = config.get("backtest", {})

    PPOTradingStrategy.model_path = str(model_path)
    PPOTradingStrategy.feature_config = config.get("features", {})
    PPOTradingStrategy.precomputed_features = test_features_cached
    PPOTradingStrategy.position_size_pct = float(trading_config.get("position_size_pct", 0.15))
    PPOTradingStrategy.stop_loss_pct = float(trading_config.get("stop_loss_pct", 0.015))
    PPOTradingStrategy.atr_stop_multiplier = float(trading_config.get("atr_stop_multiplier", 2.0))
    PPOTradingStrategy.trailing_stop = bool(trading_config.get("trailing_stop", True))
    PPOTradingStrategy.use_lstm = bool(config.get('lstm', {}).get('enabled', False))
    PPOTradingStrategy.episode_length = int(config.get('training', {}).get('episode_length', 480))
    PPOTradingStrategy.max_holding_steps = int(config.get('reward', {}).get('max_holding_steps', 9999))

    base_commission = float(backtest_config.get("commission", 0.0004))
    slippage = float(trading_config.get("slippage", 0.0))
    effective_commission = base_commission + slippage

    bt = Backtest(
        bt_data_cached,
        PPOTradingStrategy,
        cash=float(backtest_config.get("initial_capital", 10000)),
        commission=effective_commission,
        trade_on_close=True,
        exclusive_orders=True,
    )

    stats = bt.run()

    # 儲存 metrics.json（部分 trial 目錄保留時需要）
    import pandas as pd
    trades = stats.get("_trades", pd.DataFrame())
    metrics = build_metrics(stats, trades, bt_data_cached.index)

    output_dir = trial_dir / "backtest_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    import json as _json
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        _json.dump(metrics, f, indent=2, ensure_ascii=True)

    return metrics


def create_objective(
    base_config: dict,
    train_df,
    train_features: np.ndarray,
    test_df,
    bt_data,
    test_features: np.ndarray,
    phase: str,
    total_timesteps: int,
    n_cpu: int,
    output_dir: Path,
    top_n_models: int = 5,
):
    """
    建立 Optuna 目標函數 closure。

    所有數據和特徵在外部預算一次，透過 closure 傳入，避免每 trial 重複計算。
    """

    # 追蹤 top N 模型分數，用於清理
    saved_models: list = []

    def objective(trial: optuna.Trial) -> float:
        trial_start = time.time()
        trial_dir = output_dir / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        env = None
        model = None

        try:
            # 1. 套用 trial 參數
            config = apply_trial_params(base_config, trial, phase)

            # 2. 降低訓練規模
            config["training"]["total_timesteps"] = total_timesteps
            config["misc"]["n_cpu"] = n_cpu
            config["training"]["enable_detailed_logging"] = False
            config["backtest"]["html_report"] = False
            config["backtest"]["plots"] = False
            config["backtest"]["trades_csv"] = False
            config["ppo"]["verbose"] = 0

            # 3. 建立環境（使用預算特徵 + DummyVecEnv，無子進程開銷）
            env = _create_env_direct(train_df, train_features, config, n_cpu)

            # 4. 建立模型（verbose=0）
            from train import create_ppo_model

            model = create_ppo_model(env, config)
            model.verbose = 0

            # 5. 訓練（搭配 pruning callback）
            pruning_cb = OptunaPruningCallback(trial, report_interval=50000)
            model.learn(
                total_timesteps=total_timesteps,
                callback=pruning_cb,
                progress_bar=True,
            )

            # 6. 儲存模型（回測需要 .zip）
            model_path = trial_dir / "ppo_trading_model_best.zip"
            model.save(str(model_path))

            # 7. 快速回測（使用預載數據，無重複 I/O）
            metrics = _run_backtest_fast(
                config, trial_dir, test_df, bt_data, test_features
            )

            # 8. 計算複合分數
            score = compute_composite_score(metrics)

            # 記錄到 trial
            trial.set_user_attr("sharpe_ratio", metrics.get("sharpe_ratio", 0.0))
            trial.set_user_attr("total_return_pct", metrics.get("total_return_pct", 0.0))
            trial.set_user_attr("max_drawdown_pct", metrics.get("max_drawdown_pct", 0.0))
            trial.set_user_attr("profit_factor", metrics.get("profit_factor", 0.0))
            trial.set_user_attr("total_trades", metrics.get("total_trades", 0))
            trial.set_user_attr("win_rate_pct", metrics.get("win_rate_pct", 0.0))

            elapsed = time.time() - trial_start

            print(
                f"  Trial {trial.number:3d} | "
                f"Score: {score:+.4f} | "
                f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f} | "
                f"Return: {metrics.get('total_return_pct', 0):+.1f}% | "
                f"MDD: {metrics.get('max_drawdown_pct', 0):.1f}% | "
                f"PF: {metrics.get('profit_factor', 0):.2f} | "
                f"Trades: {metrics.get('total_trades', 0)} | "
                f"Time: {elapsed:.0f}s"
            )

            # 9. 清理非 top N 模型目錄
            saved_models.append((score, trial.number, trial_dir))
            saved_models.sort(key=lambda x: x[0], reverse=True)

            if len(saved_models) > top_n_models:
                _, _, old_dir = saved_models.pop()
                if old_dir.exists():
                    shutil.rmtree(old_dir, ignore_errors=True)

            return score

        except optuna.TrialPruned:
            # 清理 pruned trial 目錄
            if trial_dir.exists():
                shutil.rmtree(trial_dir, ignore_errors=True)
            raise

        except Exception as e:
            print(f"  Trial {trial.number:3d} | FAILED: {e}")
            # 清理失敗的 trial
            if trial_dir.exists():
                shutil.rmtree(trial_dir, ignore_errors=True)
            return -1.0

        finally:
            # 10. 釋放資源
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            del model
            del env
            gc.collect()

    return objective


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="PPO Trading Model - Optuna 超參數優化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  python optimize.py --phase phase1_ppo --n-trials 50
  python optimize.py --phase phase2_reward --n-trials 40
  python optimize.py --phase phase1_ppo --n-trials 50 --resume
  python optimize.py --phase phase1_ppo --n-trials 30 --timesteps 300000
        """,
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=list(SEARCH_SPACES.keys()),
        help="優化階段 (phase1_ppo / phase2_reward / phase3_combined)",
    )
    parser.add_argument(
        "--n-trials", type=int, default=50, help="Optuna trials 數量 (default: 50)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500000,
        help="每個 trial 訓練步數 (default: 500000)",
    )
    parser.add_argument(
        "--n-cpu",
        type=int,
        default=6,
        help="每個 trial 的並行環境數 (default: 6)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="從既有 study 繼續（需同 phase + symbol）",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="基礎配置檔路徑 (default: config.yaml)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="保留 top N 模型，其餘刪除 (default: 5)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  PPO Trading Model - Optuna Hyperparameter Optimization")
    print("=" * 60)
    print(f"  Phase:      {args.phase}")
    print(f"  Trials:     {args.n_trials}")
    print(f"  Timesteps:  {args.timesteps:,}")
    print(f"  N CPU:      {args.n_cpu}")
    print(f"  Resume:     {args.resume}")
    print("=" * 60)

    # 1. 載入配置
    print("\n[1/4] 載入配置...")
    config = load_config(args.config)
    symbol = config.get("data", {}).get("symbol", "BTCUSDT")

    # 2. 載入數據 + 預算特徵（全局一次）
    print("\n[2/4] 載入數據與預算特徵（全局一次）...")
    train_df, test_df = ensure_data_ready(config)
    print(f"  訓練數據: {len(train_df):,} bars")
    print(f"  測試數據: {len(test_df):,} bars")

    # 預算特徵（避免每 trial 重複計算）
    import pandas as pd
    train_features = extract_features(train_df)
    test_features = extract_features(test_df)
    print(f"  訓練特徵: {train_features.shape}")
    print(f"  測試特徵: {test_features.shape}")

    # 預處理回測用 OHLCV（避免每 trial 重複轉換）
    from backtest.run_backtest import normalize_ohlcv
    test_df_indexed = test_df.copy()
    if "timestamp" in test_df_indexed.columns:
        test_df_indexed["timestamp"] = pd.to_datetime(test_df_indexed["timestamp"], errors="coerce")
        test_df_indexed = test_df_indexed.set_index("timestamp")
    test_df_indexed = test_df_indexed.sort_index()
    bt_data = normalize_ohlcv(test_df_indexed)
    print(f"  回測 OHLCV: {bt_data.shape}")

    # 3. 建立 Optuna Study
    print("\n[3/4] 建立 Optuna Study...")
    output_dir = Path("optimized_param")
    output_dir.mkdir(parents=True, exist_ok=True)

    study_name = f"ppo_{symbol}_{args.phase}"
    storage_path = output_dir / f"study_{symbol}_{args.phase}.db"
    storage_url = f"sqlite:///{storage_path}"

    pruner = MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=100000,
        interval_steps=50000,
    )
    sampler = TPESampler(seed=42)

    if args.resume and storage_path.exists():
        print(f"  繼續既有 Study: {storage_path}")
        study = optuna.load_study(
            study_name=study_name,
            storage=storage_url,
            pruner=pruner,
            sampler=sampler,
        )
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        print(f"  已完成 {completed} trials")
    else:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction="maximize",
            pruner=pruner,
            sampler=sampler,
            load_if_exists=args.resume,
        )

    # 搜索空間摘要
    space = SEARCH_SPACES[args.phase]
    print(f"\n  搜索空間 ({args.phase}):")
    for k, v in space.items():
        print(f"    {k}: {v}")

    # 4. 執行優化
    print(f"\n[4/4] 開始優化（{args.n_trials} trials）...")
    print("-" * 100)

    objective = create_objective(
        base_config=config,
        train_df=train_df,
        train_features=train_features,
        test_df=test_df_indexed,
        bt_data=bt_data,
        test_features=test_features,
        phase=args.phase,
        total_timesteps=args.timesteps,
        n_cpu=args.n_cpu,
        output_dir=output_dir,
        top_n_models=args.top_n,
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    try:
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    except KeyboardInterrupt:
        print("\n\n[WARN] 優化被中斷，儲存目前結果...")

    # 結果摘要
    print("\n" + "=" * 60)
    print("  Optimization Results")
    print("=" * 60)

    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    failed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.FAIL
    ]

    print(f"  Completed: {len(completed_trials)}")
    print(f"  Pruned:    {len(pruned_trials)}")
    print(f"  Failed:    {len(failed_trials)}")

    if completed_trials:
        print(f"\n  Best Score: {study.best_value:.4f}")
        print(f"  Best Trial: #{study.best_trial.number}")

        # Top 5
        sorted_trials = sorted(
            completed_trials, key=lambda t: t.value, reverse=True
        )[:5]
        print(f"\n  Top 5 Trials:")
        print(f"  {'#':>4}  {'Score':>8}  {'Sharpe':>7}  {'Return':>8}  {'MDD':>7}  {'PF':>6}  {'Trades':>6}")
        print(f"  {'-'*4}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*6}  {'-'*6}")
        for t in sorted_trials:
            attrs = t.user_attrs
            print(
                f"  {t.number:4d}  {t.value:+8.4f}  "
                f"{attrs.get('sharpe_ratio', 0):7.2f}  "
                f"{attrs.get('total_return_pct', 0):+7.1f}%  "
                f"{attrs.get('max_drawdown_pct', 0):6.1f}%  "
                f"{attrs.get('profit_factor', 0):6.2f}  "
                f"{attrs.get('total_trades', 0):6d}"
            )

        # 輸出最佳參數
        export_best_params(study, output_dir, args.phase)

        # 生成 HTML 報告
        export_optimization_report(study, output_dir)

    print(f"\n  Study DB: {storage_path}")
    print(f"  輸出目錄: {output_dir}")
    print("\n  下一步：")
    print(f"    cp {output_dir}/best_params.yaml config_local.yaml")
    print(f"    python train.py")
    print(f"    python wfa.py")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
