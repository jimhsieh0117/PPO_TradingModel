"""
交易日誌系統 — JSONL 格式 + 審計軌跡

職責：
- trades.jsonl: 每筆交易記錄（開倉、平倉、止損觸發）
- decisions.jsonl: 每根 K 線的決策記錄（模型輸入/輸出、風控結果）
- errors.log: 系統錯誤日誌（含 stack trace）

設計原則：
- JSONL 格式：每行一筆記錄，方便 pandas 讀取和 grep
- 包含 model_obs_hash：可與歷史回測比對，驗證特徵一致性
- RotatingFileHandler：自動輪轉，防止磁碟爆滿
- 所有時間戳為 UTC ISO-8601
"""

import hashlib
import json
import logging
import logging.handlers
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Action 名稱對應表（與 TradingEnv 一致）
ACTION_NAMES = {0: "CLOSE", 1: "LONG", 2: "SHORT", 3: "HOLD"}


class TradingLogger:
    """
    交易日誌管理器

    Usage:
        tlogger = TradingLogger(log_dir="live_trading/logs")
        tlogger.log_trade(trade_record)
        tlogger.log_decision(decision_record)
    """

    def __init__(self, log_dir: str = "live_trading/logs",
                 max_size_mb: int = 50, backup_count: int = 10,
                 console_level: str = "INFO"):
        """
        初始化日誌系統

        Args:
            log_dir: 日誌目錄
            max_size_mb: 單檔最大大小（MB），超過自動輪轉
            backup_count: 保留的備份檔案數量
            console_level: 終端機輸出的日誌等級
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.max_bytes = max_size_mb * 1024 * 1024
        self.backup_count = backup_count

        # JSONL 檔案路徑
        self._trades_path = self.log_dir / "trades.jsonl"
        self._decisions_path = self.log_dir / "decisions.jsonl"

        # 設定 Python logging（系統日誌 + 錯誤日誌）
        self._setup_system_logger(console_level)

        self.logger = logging.getLogger("live_trading")
        self.logger.info(f"TradingLogger initialized | log_dir={self.log_dir}")

    def _setup_system_logger(self, console_level: str) -> None:
        """設定系統層級的 Python logging"""
        root_logger = logging.getLogger("live_trading")
        root_logger.setLevel(logging.DEBUG)

        # 避免重複 handler
        if root_logger.handlers:
            return

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, console_level.upper()))
        console_fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_fmt)
        root_logger.addHandler(console_handler)

        # Error file handler（只記錄 WARNING 以上）
        error_path = self.log_dir / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_path,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.WARNING)
        error_fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
        error_handler.setFormatter(error_fmt)
        root_logger.addHandler(error_handler)

        # Debug file handler（完整日誌）
        debug_path = self.log_dir / "debug.log"
        debug_handler = logging.handlers.RotatingFileHandler(
            debug_path,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding="utf-8",
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(error_fmt)
        root_logger.addHandler(debug_handler)

    # ================================================================
    # 交易記錄（trades.jsonl）
    # ================================================================

    def log_trade(self, trade: Dict[str, Any]) -> None:
        """
        記錄一筆交易

        Args:
            trade: 交易記錄 dict，至少包含：
                - action: int (0-3)
                - symbol: str
                - 其他欄位見 ARCHITECTURE.md Section 3.10
        """
        record = {
            "timestamp": _utc_now_iso(),
            "action": trade.get("action"),
            "action_name": ACTION_NAMES.get(trade.get("action", -1), "UNKNOWN"),
            "symbol": trade.get("symbol"),
            "side": trade.get("side"),
            "entry_price": trade.get("entry_price"),
            "exit_price": trade.get("exit_price"),
            "quantity": trade.get("quantity"),
            "sl_price": trade.get("sl_price"),
            "pnl": trade.get("pnl"),
            "pnl_pct": trade.get("pnl_pct"),
            "fee": trade.get("fee"),
            "balance_after": trade.get("balance_after"),
            "reason": trade.get("reason"),  # e.g. "max_holding_steps", "stop_loss"
            "order_id": trade.get("order_id"),
            "model_obs_hash": trade.get("model_obs_hash"),
        }
        self._append_jsonl(self._trades_path, record)
        self.logger.info(
            f"TRADE | {record['action_name']} {record['symbol']} "
            f"qty={record['quantity']} price={record.get('entry_price') or record.get('exit_price')} "
            f"pnl={record.get('pnl')}"
        )

    # ================================================================
    # 決策記錄（decisions.jsonl）
    # ================================================================

    def log_decision(self, bar_close: float, action: int,
                     executed: bool, risk_passed: bool,
                     risk_block_reason: Optional[str],
                     position_before: int, position_after: int,
                     features: Optional[np.ndarray] = None,
                     obs: Optional[np.ndarray] = None) -> None:
        """
        記錄每根 K 線的模型決策

        Args:
            bar_close: K 線收盤價
            action: 模型輸出的動作 (0-3)
            executed: 是否實際執行
            risk_passed: 風控是否通過
            risk_block_reason: 風控拒絕原因
            position_before: 執行前持倉狀態
            position_after: 執行後持倉狀態
            features: 33 維觀察向量（可選）
            obs: 完整觀察向量（用於計算 hash）
        """
        record = {
            "timestamp": _utc_now_iso(),
            "bar_close": bar_close,
            "action_raw": action,
            "action_name": ACTION_NAMES.get(action, "UNKNOWN"),
            "action_executed": executed,
            "risk_check_passed": risk_passed,
            "risk_block_reason": risk_block_reason,
            "position_before": position_before,
            "position_after": position_after,
        }

        if features is not None:
            record["features_snapshot"] = features.tolist()

        if obs is not None:
            record["model_obs_hash"] = _obs_hash(obs)

        self._append_jsonl(self._decisions_path, record)

    # ================================================================
    # 風控事件
    # ================================================================

    def log_risk_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """記錄風控事件（供 risk_manager 呼叫）"""
        self.logger.warning(f"RISK EVENT | {event_type}: {details}")
        record = {
            "timestamp": _utc_now_iso(),
            "type": "risk_event",
            "event": event_type,
            **details,
        }
        self._append_jsonl(self._trades_path, record)

    # ================================================================
    # 內部方法
    # ================================================================

    def _append_jsonl(self, path: Path, record: Dict) -> None:
        """Append 一筆 JSON 記錄到 JSONL 檔案"""
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write JSONL to {path}: {e}")


def _utc_now_iso() -> str:
    """UTC ISO-8601 時間戳"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _obs_hash(obs: np.ndarray) -> str:
    """觀察向量的 MD5 hash（取前 8 碼，供比對用）"""
    return hashlib.md5(obs.tobytes()).hexdigest()[:8]
