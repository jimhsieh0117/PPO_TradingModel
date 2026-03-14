"""
模型推論模組 — 載入 PPO 模型 + predict

職責：
- 啟動時載入模型（一次性）
- MD5 checksum 校驗（防誤覆蓋）
- 執行 deterministic 推論
- 支援 MLP / LSTM 模式

安全要點：
- deterministic=True 永遠不變（實盤不做探索）
- device="cpu"（推論不需 GPU，避免 MPS/CUDA 差異）
- 模型載入後不再更新（更換需停機重啟）
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("live_trading.inference")


class InferenceEngine:
    """
    PPO 模型推論引擎

    Usage:
        engine = InferenceEngine(
            model_path="models/.../ppo_trading_model_best.zip",
            expected_md5="9d3e0d2ca248e1ffb2ac326b1f9ea751",
        )
        action = engine.predict(obs)  # obs: np.ndarray [33]
    """

    # Action 名稱對應表（與 TradingEnv 一致）
    ACTION_NAMES = {0: "CLOSE", 1: "LONG", 2: "SHORT", 3: "HOLD"}

    def __init__(self, model_path: str, expected_md5: str = "",
                 use_lstm: bool = False, deterministic: bool = True):
        """
        載入 PPO 模型

        Args:
            model_path: 模型 .zip 檔案路徑
            expected_md5: 預期的 MD5（空字串 = 跳過校驗）
            use_lstm: 是否為 LSTM 模型
            deterministic: 是否 deterministic（實盤必須 True）

        Raises:
            FileNotFoundError: 模型檔案不存在
            ValueError: MD5 不符
        """
        self.model_path = Path(model_path)
        self.use_lstm = use_lstm
        self.deterministic = deterministic

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # MD5 校驗
        if expected_md5:
            actual_md5 = self._compute_md5()
            if actual_md5 != expected_md5:
                raise ValueError(
                    f"Model MD5 mismatch!\n"
                    f"  Expected: {expected_md5}\n"
                    f"  Actual:   {actual_md5}\n"
                    f"  File:     {self.model_path}\n"
                    f"模型檔案可能被意外覆蓋，拒絕啟動。"
                )
            logger.info(f"Model MD5 verified: {actual_md5}")
        else:
            logger.warning("Model MD5 check skipped (expected_md5 is empty)")

        # 載入模型
        if use_lstm:
            from sb3_contrib import RecurrentPPO
            self._model = RecurrentPPO.load(str(self.model_path), device="cpu")
            self._lstm_states = None
            self._episode_starts = np.array([True])
            logger.info("Loaded RecurrentPPO (LSTM) model")
        else:
            from stable_baselines3 import PPO
            self._model = PPO.load(str(self.model_path), device="cpu")
            logger.info("Loaded PPO (MLP) model")

        # 驗證觀察空間
        obs_dim = self._model.observation_space.shape[0]
        logger.info(
            f"Model loaded: {self.model_path.name} | "
            f"obs_dim={obs_dim} | "
            f"action_space={self._model.action_space} | "
            f"deterministic={self.deterministic}"
        )

        self._predict_count = 0

    def predict(self, obs: np.ndarray) -> int:
        """
        執行模型推論

        Args:
            obs: 觀察向量 np.ndarray shape [33], dtype=float32

        Returns:
            action: int (0=CLOSE, 1=LONG, 2=SHORT, 3=HOLD)
        """
        # 確保 dtype 和 shape
        if obs.dtype != np.float32:
            obs = obs.astype(np.float32)

        if self.use_lstm:
            action, self._lstm_states = self._model.predict(
                obs,
                state=self._lstm_states,
                episode_start=self._episode_starts,
                deterministic=self.deterministic,
            )
            self._episode_starts = np.array([False])
        else:
            action, _ = self._model.predict(obs, deterministic=self.deterministic)

        action = int(action)
        self._predict_count += 1

        if self._predict_count % 60 == 0:
            logger.debug(
                f"Predict #{self._predict_count} | "
                f"action={action} ({self.ACTION_NAMES.get(action, '?')})"
            )

        return action

    @property
    def obs_dimension(self) -> int:
        """模型期望的觀察空間維度"""
        return self._model.observation_space.shape[0]

    def _compute_md5(self) -> str:
        """計算模型檔案的 MD5"""
        md5 = hashlib.md5()
        with open(self.model_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)
        return md5.hexdigest()

    def get_stats(self) -> dict:
        return {
            "model_path": str(self.model_path),
            "predict_count": self._predict_count,
            "use_lstm": self.use_lstm,
            "deterministic": self.deterministic,
        }
