"""
Config 工具模組 - 統一配置讀取 + 覆蓋機制 + 裝置偵測

功能：
1. load_config()：讀取 config.yaml，自動合併 config_local.yaml 覆蓋
2. deep_merge()：遞迴合併字典（local 覆蓋 base）
3. detect_device()：自動偵測最佳訓練裝置 (cuda / mps / cpu)

使用方式：
    from utils.config_utils import load_config
    config = load_config()  # 自動合併 config_local.yaml
"""

import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def deep_merge(base: Dict, override: Dict) -> Dict:
    """
    遞迴合併兩個字典，override 的值覆蓋 base 的值。
    
    規則：
    - 如果兩邊都是 dict → 遞迴合併
    - 否則 → override 的值直接覆蓋
    
    Args:
        base: 基礎配置字典
        override: 覆蓋配置字典
    
    Returns:
        合併後的新字典（不修改原始字典）
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def detect_device() -> str:
    """
    自動偵測最佳訓練裝置。
    
    優先順序：
    1. CUDA (NVIDIA GPU) — Windows / Linux
    2. MPS (Apple Silicon GPU) — macOS
    3. CPU — fallback
    
    Returns:
        "cuda", "mps", 或 "cpu"
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[Device] 偵測到 CUDA GPU: {gpu_name}")
            return "cuda"
        if sys.platform == "darwin" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("[Device] 偵測到 Apple MPS (Metal Performance Shaders)")
            return "mps"
    except ImportError:
        pass

    print("[Device] 使用 CPU")
    return "cpu"


def load_config(config_path: str = "config.yaml") -> dict:
    """
    讀取配置文件，自動合併 config_local.yaml 覆蓋。
    
    流程：
    1. 讀取 config_path（預設 config.yaml）
    2. 檢查同目錄下是否有 config_local.yaml
    3. 若有 → deep_merge 覆蓋（config_local 的值優先）
    4. 若 device == "auto" → 自動偵測最佳裝置
    
    Args:
        config_path: 配置文件路徑（預設 "config.yaml"）
    
    Returns:
        合併後的配置字典
    """
    config_file = Path(config_path)

    # 讀取主配置
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 檢查並合併 config_local.yaml
    local_config_path = config_file.parent / "config_local.yaml"
    if local_config_path.exists():
        with open(local_config_path, "r", encoding="utf-8") as f:
            local_config = yaml.safe_load(f)
        if local_config:
            config = deep_merge(config, local_config)
            print(f"[Config] 已合併本地配置: {local_config_path}")
    
    # 處理 device: "auto"
    ppo_config = config.get("ppo", {})
    if ppo_config.get("device", "cpu") == "auto":
        detected = detect_device()
        config["ppo"]["device"] = detected

    return config
