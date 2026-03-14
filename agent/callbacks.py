import csv
import math
import shutil
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
        best_model_rolling_window: int = 20,
        best_model_top_n: int = 3,
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
        self._rolling_window = best_model_rolling_window

        self._file = None
        self._writer = None
        self._headers: List[str] = []
        self._best_rolling_mean = float("-inf")

        self._episode_rewards: List[float] = []
        self._episode_lengths: List[int] = []
        self._episode_action_counts: List[np.ndarray] = []
        self._episode_holding_steps: List[int] = []

        self._episode_counter = 0
        self._cumulative_reward = 0.0
        self._episode_reward_history: List[float] = []
        self._episode_return_history: List[float] = []
        self._last_train_metrics: Dict[str, Optional[float]] = {}

        # 複合評分用歷史
        self._episode_pf_history: List[float] = []
        self._episode_mdd_history: List[float] = []

        # Top-N 模型管理
        self._top_n = best_model_top_n
        self._top_n_models: List[Dict] = []  # [{score, path, timestep, episode}]

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
            "scoring_method",
            "new_best_rolling_mean",
            "previous_best_rolling_mean",
            "composite_score",
            "rolling_sharpe",
            "rolling_pf",
            "rolling_mdd",
            "rolling_return",
            "rank",
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
            "scoring_method": "rolling_mean_return",
            "new_best_rolling_mean": new_best,
            "previous_best_rolling_mean": previous_best,
            "composite_score": None,
            "rolling_sharpe": None,
            "rolling_pf": None,
            "rolling_mdd": None,
            "rolling_return": new_best,
            "rank": None,
            "overwritten_model": overwritten_model,
            "best_model_path": str(self.best_model_path),
        }
        with open(self.best_log_path, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=data.keys())
            writer.writerow(data)

    def _log_best_event_composite(self, row: Dict, entry: Dict) -> None:
        w = self._rolling_window
        rolling_return = float(np.mean(self._episode_return_history[-w:]))
        rolling_pf = float(np.mean(self._episode_pf_history[-w:]))
        rolling_mdd = float(np.mean(self._episode_mdd_history[-w:]))
        returns_arr = np.array(self._episode_return_history[-w:], dtype=float)
        rolling_sharpe = float(
            returns_arr.mean() / (returns_arr.std() + 1e-8) * math.sqrt(252)
        )

        # 找到此 entry 的排名
        rank = next(
            (i + 1 for i, m in enumerate(self._top_n_models) if m["path"] == entry["path"]),
            None,
        )

        data = {
            "timestamp": row.get("timestamp"),
            "timesteps": row.get("timesteps"),
            "episode": row.get("episode"),
            "scoring_method": "composite",
            "new_best_rolling_mean": rolling_return,
            "previous_best_rolling_mean": None,
            "composite_score": entry["score"],
            "rolling_sharpe": rolling_sharpe,
            "rolling_pf": rolling_pf,
            "rolling_mdd": rolling_mdd,
            "rolling_return": rolling_return,
            "rank": rank,
            "overwritten_model": "",
            "best_model_path": entry["path"],
        }
        with open(self.best_log_path, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=data.keys())
            writer.writerow(data)

    def _compute_composite_score(self) -> Optional[float]:
        w = self._rolling_window
        if (len(self._episode_return_history) < w
                or len(self._episode_pf_history) < w
                or len(self._episode_mdd_history) < w):
            return None

        rolling_return = float(np.mean(self._episode_return_history[-w:]))
        rolling_pf = float(np.mean(self._episode_pf_history[-w:]))
        rolling_mdd = float(np.mean(self._episode_mdd_history[-w:]))

        returns_arr = np.array(self._episode_return_history[-w:], dtype=float)
        rolling_sharpe = float(
            returns_arr.mean() / (returns_arr.std() + 1e-8) * math.sqrt(252)
        )

        return (
            0.40 * rolling_sharpe
            + 0.25 * rolling_pf * 10
            - 0.20 * rolling_mdd * 100
            + 0.15 * rolling_return
        )

    def _update_top_n(self, score: float, row: Dict) -> None:
        entry = {
            "score": score,
            "timestep": int(self.num_timesteps),
            "episode": int(self._episode_counter),
        }

        if len(self._top_n_models) < self._top_n:
            rank = len(self._top_n_models) + 1
            path = str(self.best_model_path.parent / f"ppo_trading_model_best_top{rank}.zip")
            entry["path"] = path
            self.model.save(path)
            self._top_n_models.append(entry)
            self._top_n_models.sort(key=lambda x: x["score"], reverse=True)
        elif score > self._top_n_models[-1]["score"]:
            entry["path"] = self._top_n_models[-1]["path"]
            self.model.save(entry["path"])
            self._top_n_models[-1] = entry
            self._top_n_models.sort(key=lambda x: x["score"], reverse=True)
        else:
            return  # 沒進 top-N

        # 排序後重新命名檔案以匹配排名（使用暫存檔名避免互換衝突）
        parent = self.best_model_path.parent
        needs_rename = []
        for i, model_entry in enumerate(self._top_n_models):
            expected_path = str(parent / f"ppo_trading_model_best_top{i + 1}.zip")
            if model_entry["path"] != expected_path:
                needs_rename.append((i, model_entry))

        if needs_rename:
            # Pass 1: 全部先移到暫存名
            for i, model_entry in needs_rename:
                tmp_path = str(parent / f"ppo_trading_model_best_top{i + 1}.zip.tmp")
                old_path = Path(model_entry["path"])
                if old_path.exists():
                    shutil.move(str(old_path), tmp_path)
                model_entry["path"] = tmp_path

            # Pass 2: 暫存名移到最終名
            for i, model_entry in needs_rename:
                final_path = str(parent / f"ppo_trading_model_best_top{i + 1}.zip")
                tmp_path = model_entry["path"]
                if Path(tmp_path).exists():
                    shutil.move(tmp_path, final_path)
                model_entry["path"] = final_path

        # 同步 best.zip = top1（向後相容）
        top1_path = self._top_n_models[0]["path"]
        shutil.copy2(top1_path, str(self.best_model_path))

        self._log_best_event_composite(row, entry)

    def _maybe_save_best(self, row: Dict) -> None:
        max_drawdown = row.get("max_drawdown")
        if max_drawdown is None:
            return
        # 安全閥：單 episode 回撤超過 50% 不保存
        if float(max_drawdown) > 0.50:
            return

        # 嘗試複合評分（需要足夠歷史的 PF/MDD 資料）
        score = self._compute_composite_score()
        if score is not None and self._top_n > 0:
            self._update_top_n(score, row)
            return

        # Fallback：原始 rolling mean return 邏輯
        if len(self._episode_return_history) < self._rolling_window:
            return
        rolling_mean = float(np.mean(self._episode_return_history[-self._rolling_window:]))
        if rolling_mean <= self._best_rolling_mean:
            return

        previous_best = None if self._best_rolling_mean == float("-inf") else self._best_rolling_mean
        overwritten_model = str(self.best_model_path) if self.best_model_path.exists() else ""
        self.model.save(str(self.best_model_path))
        self._best_rolling_mean = rolling_mean
        self._log_best_event(row, rolling_mean, previous_best, overwritten_model)

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

        # 累積 PF/MDD 歷史（無論 enable_detailed_logging 開關，info dict 始終可用）
        pf = info.get("profit_factor")
        mdd = info.get("max_drawdown")
        if pf is not None:
            self._episode_pf_history.append(float(pf))
        if mdd is not None:
            self._episode_mdd_history.append(float(mdd))
            # 確保 row 中有 max_drawdown（簡化模式下不在 row 中）
            if "max_drawdown" not in row:
                row["max_drawdown"] = mdd

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
