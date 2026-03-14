"""
重試機制 — 指數退避 + 可配置策略

用途：
- WebSocket 斷線重連
- 冪等 API 呼叫重試（GET 查詢）
- 非冪等操作（POST 下單）不應使用重試

設計原則：
- 只對可恢復的暫時性錯誤重試
- 指數退避避免打爆 API rate limit
- 最大重試次數和最大等待時間有上限
"""

import logging
import time
from functools import wraps
from typing import Callable, Optional, Tuple, Type

import requests

logger = logging.getLogger(__name__)

# 預設可重試的異常類型
DEFAULT_RETRYABLE_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    requests.ConnectionError,
    requests.Timeout,
    ConnectionResetError,
    OSError,
)


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    on_retry: Optional[Callable] = None,
) -> Callable:
    """
    指數退避重試 decorator

    Args:
        max_retries: 最大重試次數
        base_delay: 初始等待秒數
        max_delay: 最大等待秒數
        backoff_factor: 退避倍數（每次乘以此值）
        retryable_exceptions: 可重試的異常類型
        on_retry: 重試時的回調函數 fn(attempt, exception, delay)

    Usage:
        @retry_with_backoff(max_retries=3)
        def fetch_data():
            return requests.get(url)
    """
    if retryable_exceptions is None:
        retryable_exceptions = DEFAULT_RETRYABLE_EXCEPTIONS

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(
                            f"[Retry] {func.__name__} failed after "
                            f"{max_retries} retries: {e}"
                        )
                        raise

                    logger.warning(
                        f"[Retry] {func.__name__} attempt {attempt + 1}/{max_retries} "
                        f"failed: {e}. Retrying in {delay:.1f}s..."
                    )

                    if on_retry:
                        on_retry(attempt + 1, e, delay)

                    time.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)

            # 理論上不會到這裡
            raise last_exception  # type: ignore

        return wrapper
    return decorator


class ReconnectManager:
    """
    WebSocket 重連管理器

    管理重連嘗試計數和退避延遲，
    與具體的 WebSocket 實作分離。

    Usage:
        mgr = ReconnectManager()
        while not connected:
            delay = mgr.next_delay()
            time.sleep(delay)
            try:
                connect()
                mgr.reset()
            except Exception:
                if mgr.should_give_up():
                    raise
    """

    def __init__(self, base_delay: float = 1.0, max_delay: float = 60.0,
                 backoff_factor: float = 2.0, max_attempts: int = 50):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.max_attempts = max_attempts

        self._attempt = 0
        self._current_delay = base_delay

    def next_delay(self) -> float:
        """取得下一次重連的等待秒數"""
        delay = self._current_delay
        self._attempt += 1
        self._current_delay = min(
            self._current_delay * self.backoff_factor,
            self.max_delay,
        )
        return delay

    def reset(self) -> None:
        """重連成功後重置計數器"""
        self._attempt = 0
        self._current_delay = self.base_delay

    def should_give_up(self) -> bool:
        """是否已達最大重試次數"""
        return self._attempt >= self.max_attempts

    @property
    def attempt_count(self) -> int:
        return self._attempt
