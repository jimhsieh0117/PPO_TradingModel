"""
狀態快照持久化 — 防止崩潰後風控失效

職責：
- 每次狀態更新後寫入 state_snapshot.json
- 啟動時讀取上次的狀態快照（恢復 consecutive_losses、daily_pnl 等）
- 防止崩潰重啟後風控計數器歸零

設計原則：
- 原子寫入：先寫暫存檔再 rename，防止寫到一半崩潰產生損壞的 JSON
- 只保存風控相關的統計數據，不保存持倉狀態
  （持倉狀態啟動時從交易所 API 同步，以交易所為準）
- 快照檔案人類可讀（formatted JSON）
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("live_trading.state_snapshot")

DEFAULT_SNAPSHOT_PATH = "live_trading/state_snapshot.json"


class StateSnapshot:
    """
    狀態快照管理器

    Usage:
        snapshot = StateSnapshot()

        # 啟動時恢復
        prev = snapshot.load()
        if prev:
            state.consecutive_losses = prev["consecutive_losses"]
            state.daily_pnl = prev["daily_pnl"]

        # 每次狀態更新後保存
        snapshot.save({
            "consecutive_losses": state.consecutive_losses,
            "daily_pnl": state.daily_pnl,
            "total_pnl": state.total_pnl,
            "trade_count": state.trade_count,
        })
    """

    def __init__(self, path: str = DEFAULT_SNAPSHOT_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, state_data: Dict[str, Any]) -> bool:
        """
        原子寫入狀態快照

        Args:
            state_data: 要保存的狀態數據

        Returns:
            True = 成功, False = 失敗
        """
        snapshot = {
            "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "version": 1,
            **state_data,
        }

        try:
            # 原子寫入：先寫暫存檔再 rename
            dir_path = self.path.parent
            fd, tmp_path = tempfile.mkstemp(
                suffix=".tmp", prefix="state_", dir=str(dir_path)
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(snapshot, f, indent=2, ensure_ascii=False, default=str)
                os.replace(tmp_path, str(self.path))
            except Exception:
                # 清理暫存檔
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise

            logger.debug(f"State snapshot saved to {self.path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save state snapshot: {e}")
            return False

    def load(self) -> Optional[Dict[str, Any]]:
        """
        讀取上次的狀態快照

        Returns:
            狀態數據 dict，檔案不存在或損壞時回傳 None
        """
        if not self.path.exists():
            logger.info("No previous state snapshot found — starting fresh")
            return None

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)

            saved_at = data.get("saved_at", "unknown")
            logger.info(f"Loaded state snapshot from {self.path} (saved_at={saved_at})")

            # 驗證必要欄位
            if "version" not in data:
                logger.warning("State snapshot missing version field — ignoring")
                return None

            return data

        except json.JSONDecodeError as e:
            logger.error(f"State snapshot corrupted: {e} — starting fresh")
            return None
        except Exception as e:
            logger.error(f"Failed to load state snapshot: {e}")
            return None

    def get_recoverable_fields(self) -> Dict[str, Any]:
        """
        取得可恢復的風控欄位（帶預設值）

        Returns:
            dict with keys:
                - consecutive_losses: int
                - daily_pnl: float
                - total_pnl: float
                - trade_count: int
                - daily_reset_date: str (YYYY-MM-DD)
        """
        data = self.load()
        if data is None:
            return self._default_fields()

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        snapshot_date = data.get("daily_reset_date", "")

        result = {
            "consecutive_losses": data.get("consecutive_losses", 0),
            "total_pnl": data.get("total_pnl", 0.0),
            "trade_count": data.get("trade_count", 0),
        }

        # 如果快照是今天的，恢復 daily 數據；否則重置
        if snapshot_date == today:
            result["daily_pnl"] = data.get("daily_pnl", 0.0)
            result["daily_reset_date"] = snapshot_date
            logger.info(
                f"Recovered daily stats from today's snapshot: "
                f"daily_pnl={result['daily_pnl']:.2f}, "
                f"consecutive_losses={result['consecutive_losses']}"
            )
        else:
            result["daily_pnl"] = 0.0
            result["daily_reset_date"] = today
            logger.info(
                f"Snapshot from {snapshot_date or 'unknown'}, "
                f"today is {today} — daily stats reset"
            )

        return result

    @staticmethod
    def _default_fields() -> Dict[str, Any]:
        """預設值（首次啟動或快照損壞時使用）"""
        return {
            "consecutive_losses": 0,
            "daily_pnl": 0.0,
            "total_pnl": 0.0,
            "trade_count": 0,
            "daily_reset_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        }
