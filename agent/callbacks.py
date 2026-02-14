import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TrainingMetricsCallback(BaseCallback):
    def __init__(
        self,
        log_path: str,
        best_model_path: str,
        best_log_path: str,
        episode_length: int,
        initial_capital: float,
        max_daily_drawdown: float,
        enable_detailed_logging: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.log_path = log_path
        self.best_model_path = Path(best_model_path)
        self.best_log_path = Path(best_log_path)
        self.episode_length = episode_length
        self.initial_capital = initial_capital
        self.max_daily_drawdown = max_daily_drawdown
        self.enable_detailed_logging = enable_detailed_logging

        self._file = None
        self._writer = None
        self._headers: List[str] = []
        self._best_return_pct = float("-inf")

        self._episode_rewards: List[float] = []
        self._episode_lengths: List[int] = []
        self._episode_action_counts: List[np.ndarray] = []
        self._episode_holding_steps: List[int] = []

        self._episode_counter = 0
        self._cumulative_reward = 0.0
        self._episode_reward_history: List[float] = []
        self._episode_return_history: List[float] = []
        self._last_train_metrics: Dict[str, Optional[float]] = {}

        # 效能優化：每 N 個 episodes 才 flush 一次
        self._flush_interval = 20
        self._episodes_since_last_flush = 0

    def _init_tracking(self, n_envs: int) -> None:
        self._episode_rewards = [0.0 for _ in range(n_envs)]
        self._episode_lengths = [0 for _ in range(n_envs)]
        self._episode_action_counts = [np.zeros(4, dtype=np.int64) for _ in range(n_envs)]
        self._episode_holding_steps = [0 for _ in range(n_envs)]

    def _setup_csv(self) -> None:
        log_dir = Path(self.log_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        # 基本指標（總是記錄）
        basic_headers = [
            "timestamp",
            "timesteps",
            "episode",
            "episode_reward",
            "episode_profit",
            "episode_return_pct",
            "total_trades_per_episode",
        ]

        # 詳細指標（可選）
        detailed_headers = [
            "episode_reward_mean",
            "episode_reward_std",
            "episode_reward_max",
            "episode_reward_min",
            "cumulative_reward",
            "policy_loss",
            "value_loss",
            "entropy_loss",
            "total_loss",
            "clip_fraction",
            "approx_kl",
            "explained_variance",
            "learning_rate",
            "long_ratio",
            "short_ratio",
            "hold_ratio",
            "close_ratio",
            "avg_holding_time",
            "win_rate",
            "profit_factor",
            "sharpe_ratio",
            "max_drawdown",
            "stop_loss_count",
            "daily_drawdown_violations",
            "episode_length",
            "episode_completion_rate",
            "avg_equity_curve_slope",
            "action_entropy",
            "action_dist_0",
            "action_dist_1",
            "action_dist_2",
            "action_dist_3",
        ]

        # 根據設定選擇要記錄的指標
        if self.enable_detailed_logging:
            self._headers = basic_headers + detailed_headers
        else:
            self._headers = basic_headers

        self._file = open(self.log_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self._headers)
        self._writer.writeheader()
        self._file.flush()

    def _setup_best_log(self) -> None:
        if self.best_log_path.exists():
            return
        self.best_log_path.parent.mkdir(parents=True, exist_ok=True)
        headers = [
            "timestamp",
            "timesteps",
            "episode",
            "new_best_return_pct",
            "previous_best_return_pct",
            "overwritten_model",
            "best_model_path",
        ]
        with open(self.best_log_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()

    def _log_best_event(
        self,
        row: Dict,
        new_best: float,
        previous_best: Optional[float],
        overwritten_model: str,
    ) -> None:
        data = {
            "timestamp": row.get("timestamp"),
            "timesteps": row.get("timesteps"),
            "episode": row.get("episode"),
            "new_best_return_pct": new_best,
            "previous_best_return_pct": previous_best,
            "overwritten_model": overwritten_model,
            "best_model_path": str(self.best_model_path),
        }
        with open(self.best_log_path, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=data.keys())
            writer.writerow(data)

    def _maybe_save_best(self, row: Dict) -> None:
        episode_return_pct = row.get("episode_return_pct")
        sharpe_ratio = row.get("sharpe_ratio")
        max_drawdown = row.get("max_drawdown")
        if episode_return_pct is None or sharpe_ratio is None or max_drawdown is None:
            return
        # v9: 放寬門檻，只保留 drawdown 安全閥（原 sharpe>1.3 導致從未保存）
        if float(max_drawdown) > 0.50:
            return

        current_return = float(episode_return_pct)
        if current_return <= self._best_return_pct:
            return

        previous_best = None if self._best_return_pct == float("-inf") else self._best_return_pct
        overwritten_model = str(self.best_model_path) if self.best_model_path.exists() else ""
        self.model.save(str(self.best_model_path))
        self._best_return_pct = current_return
        self._log_best_event(row, current_return, previous_best, overwritten_model)

    def _get_logger_value(self, key: str) -> Optional[float]:
        logger = getattr(self.model, "logger", None)
        if logger is None:
            return None
        if hasattr(logger, "name_to_value"):
            return logger.name_to_value.get(key)
        if hasattr(logger, "_name_to_value"):
            return logger._name_to_value.get(key)
        return None

    def _update_train_metrics(self) -> None:
        policy_loss = self._get_logger_value("train/policy_loss")
        if policy_loss is None:
            policy_loss = self._get_logger_value("train/policy_gradient_loss")
        value_loss = self._get_logger_value("train/value_loss")
        entropy_loss = self._get_logger_value("train/entropy_loss")
        total_loss = self._get_logger_value("train/loss")
        if total_loss is None:
            parts = [v for v in (policy_loss, value_loss, entropy_loss) if v is not None]
            total_loss = float(np.sum(parts)) if parts else None

        self._last_train_metrics = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "total_loss": total_loss,
            "clip_fraction": self._get_logger_value("train/clip_fraction"),
            "approx_kl": self._get_logger_value("train/approx_kl"),
            "explained_variance": self._get_logger_value("train/explained_variance"),
            "learning_rate": self._get_logger_value("train/learning_rate"),
        }

    def _record_episode(self, env_idx: int, info: Dict) -> None:
        episode_reward = self._episode_rewards[env_idx]
        episode_length = self._episode_lengths[env_idx]

        total_trades = int(info.get("total_trades", 0))
        equity = info.get("equity")
        episode_profit = float(equity - self.initial_capital) if equity is not None else None
        episode_return_pct = info.get("total_return_pct")
        if episode_return_pct is None and episode_profit is not None:
            episode_return_pct = (episode_profit / self.initial_capital) * 100.0

        self._episode_counter += 1
        self._cumulative_reward += episode_reward
        self._episode_reward_history.append(episode_reward)
        if episode_return_pct is not None:
            self._episode_return_history.append(float(episode_return_pct))

        # === 詳細指標計算（僅在啟用時執行）===
        if self.enable_detailed_logging:
            action_counts = self._episode_action_counts[env_idx]
            action_total = int(action_counts.sum())

            if action_total > 0:
                action_dist = action_counts / action_total
                action_entropy = -float(np.sum([p * math.log(p) for p in action_dist if p > 0]))
            else:
                action_dist = np.zeros(4, dtype=float)
                action_entropy = 0.0

            avg_holding_time = (
                float(self._episode_holding_steps[env_idx]) / max(total_trades, 1)
                if total_trades >= 0
                else 0.0
            )

            reward_array = np.array(self._episode_reward_history, dtype=float)
            reward_mean = float(reward_array.mean()) if reward_array.size else 0.0
            reward_std = float(reward_array.std()) if reward_array.size else 0.0
            reward_max = float(reward_array.max()) if reward_array.size else 0.0
            reward_min = float(reward_array.min()) if reward_array.size else 0.0

            returns_window = np.array(self._episode_return_history[-20:], dtype=float)
            if returns_window.size >= 2:
                sharpe_ratio = float(
                    returns_window.mean() / (returns_window.std() + 1e-8) * math.sqrt(252)
                )
            else:
                sharpe_ratio = 0.0

            max_drawdown = info.get("max_drawdown")
            daily_drawdown_violations = 1 if (
                max_drawdown is not None and max_drawdown > self.max_daily_drawdown
            ) else 0

            episode_completion_rate = 1 if episode_length >= self.episode_length else 0
            avg_equity_curve_slope = (
                float(episode_return_pct) / episode_length
                if episode_return_pct is not None and episode_length > 0
                else 0.0
            )
        else:
            # 簡化模式：不計算詳細指標
            action_dist = None
            action_entropy = None
            avg_holding_time = None
            reward_mean = None
            reward_std = None
            reward_max = None
            reward_min = None
            sharpe_ratio = None
            max_drawdown = None
            daily_drawdown_violations = None
            episode_completion_rate = None
            avg_equity_curve_slope = None

        # 基本指標（總是記錄）
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "timesteps": int(self.num_timesteps),
            "episode": int(self._episode_counter),
            "episode_reward": float(episode_reward),
            "episode_profit": episode_profit,
            "episode_return_pct": episode_return_pct,
            "total_trades_per_episode": total_trades,
        }

        # 詳細指標（僅在啟用時記錄）
        if self.enable_detailed_logging:
            detailed_data = {
                "episode_reward_mean": reward_mean,
                "episode_reward_std": reward_std,
                "episode_reward_max": reward_max,
                "episode_reward_min": reward_min,
                "cumulative_reward": float(self._cumulative_reward),
                "policy_loss": self._last_train_metrics.get("policy_loss"),
                "value_loss": self._last_train_metrics.get("value_loss"),
                "entropy_loss": self._last_train_metrics.get("entropy_loss"),
                "total_loss": self._last_train_metrics.get("total_loss"),
                "clip_fraction": self._last_train_metrics.get("clip_fraction"),
                "approx_kl": self._last_train_metrics.get("approx_kl"),
                "explained_variance": self._last_train_metrics.get("explained_variance"),
                "learning_rate": self._last_train_metrics.get("learning_rate"),
                "long_ratio": float(action_dist[1]),
                "short_ratio": float(action_dist[2]),
                "hold_ratio": float(action_dist[3]),
                "close_ratio": float(action_dist[0]),
                "avg_holding_time": avg_holding_time,
                "win_rate": info.get("win_rate"),
                "profit_factor": info.get("profit_factor"),
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "stop_loss_count": info.get("stop_loss_count"),
                "daily_drawdown_violations": daily_drawdown_violations,
                "episode_length": episode_length,
                "episode_completion_rate": episode_completion_rate,
                "avg_equity_curve_slope": avg_equity_curve_slope,
                "action_entropy": action_entropy,
                "action_dist_0": float(action_dist[0]),
                "action_dist_1": float(action_dist[1]),
                "action_dist_2": float(action_dist[2]),
                "action_dist_3": float(action_dist[3]),
            }
            row.update(detailed_data)

        if self._writer is not None:
            self._writer.writerow(row)

            # 效能優化：每 20 個 episodes 才 flush 一次
            self._episodes_since_last_flush += 1
            if self._episodes_since_last_flush >= self._flush_interval:
                self._file.flush()
                self._episodes_since_last_flush = 0

        self._maybe_save_best(row)

    def _on_training_start(self) -> None:
        n_envs = self.model.get_env().num_envs
        self._init_tracking(n_envs)
        self._setup_csv()
        self._setup_best_log()

    def _on_rollout_end(self) -> None:
        self._update_train_metrics()

    def _on_step(self) -> bool:
        actions = self.locals.get("actions")
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        infos = self.locals.get("infos") or []

        actions = np.atleast_1d(actions)
        rewards = np.atleast_1d(rewards)
        dones = np.atleast_1d(dones)

        for idx in range(len(dones)):
            self._episode_rewards[idx] += float(rewards[idx])
            self._episode_lengths[idx] += 1

            action = int(actions[idx]) if idx < len(actions) else None
            if action is not None and 0 <= action <= 3:
                self._episode_action_counts[idx][action] += 1

            info = infos[idx] if idx < len(infos) else {}
            if info.get("position", 0) != 0:
                self._episode_holding_steps[idx] += 1

            if dones[idx]:
                self._record_episode(idx, info)
                self._episode_rewards[idx] = 0.0
                self._episode_lengths[idx] = 0
                self._episode_action_counts[idx] = np.zeros(4, dtype=np.int64)
                self._episode_holding_steps[idx] = 0

        return True

    def _on_training_end(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
