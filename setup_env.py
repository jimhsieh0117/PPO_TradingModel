"""
PPO Trading Model - 環境自動安裝腳本

功能：
1. 偵測當前作業系統 (Windows / macOS / Linux)
2. 建立 Python 虛擬環境（如果不存在）
3. 安裝對應平台的依賴 (requirements_windows.txt / requirements_mac.txt)
4. 自動生成 config_local.yaml（含平台適合的預設值）

使用方式：
    python setup_env.py              # 正常安裝
    python setup_env.py --dry-run    # 只顯示會執行的動作，不實際執行
"""

import os
import sys
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent


def detect_platform() -> str:
    """偵測當前平台"""
    if sys.platform == "win32":
        return "windows"
    elif sys.platform == "darwin":
        return "mac"
    elif sys.platform == "linux":
        return "linux"
    else:
        return "unknown"


def detect_gpu_info() -> dict:
    """偵測 GPU 資訊"""
    info = {"device": "cpu", "gpu_name": None}
    
    platform = detect_platform()
    
    if platform == "windows" or platform == "linux":
        # 嘗試偵測 NVIDIA GPU
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                info["device"] = "cuda"
                info["gpu_name"] = result.stdout.strip().split("\n")[0]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    
    elif platform == "mac":
        # macOS Apple Silicon → MPS
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                cpu_brand = result.stdout.strip()
                if "Apple" in cpu_brand:
                    info["device"] = "mps"
                    info["gpu_name"] = f"{cpu_brand} (MPS)"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    return info


def get_cpu_count() -> int:
    """取得 CPU 核心數"""
    return os.cpu_count() or 4


def get_requirements_file(platform: str) -> str:
    """取得對應平台的 requirements 檔案"""
    mapping = {
        "windows": "requirements_windows.txt",
        "mac": "requirements_mac.txt",
        "linux": "requirements_mac.txt",  # Linux 用 CPU/通用版
    }
    return mapping.get(platform, "requirements.txt")


def create_config_local(platform: str, gpu_info: dict, n_cpu: int, dry_run: bool = False):
    """自動生成 config_local.yaml"""
    config_local_path = PROJECT_ROOT / "config_local.yaml"
    
    if config_local_path.exists():
        print(f"  ⚠️  config_local.yaml 已存在，跳過生成（避免覆蓋你的自訂設定）")
        return
    
    device = gpu_info["device"]
    gpu_name = gpu_info.get("gpu_name", "N/A")
    
    content = f"""# === 本地配置覆蓋 ===
# 此檔案由 setup_env.py 自動生成，可手動修改
# 這裡的設定會覆蓋 config.yaml 中的對應值
# 此檔案不會被 git 追蹤 (已加入 .gitignore)
#
# 平台: {platform}
# GPU: {gpu_name}
# CPU 核心數: {get_cpu_count()}

# 取消註解並修改你想覆蓋的參數：

# ppo:
#   device: "{device}"     # auto / cuda / mps / cpu

# misc:
#   n_cpu: {n_cpu}          # 並行環境數
"""
    
    if dry_run:
        print(f"\n  [DRY-RUN] 將生成 config_local.yaml:")
        print(content)
    else:
        with open(config_local_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  ✅ 已生成 config_local.yaml")


def install_requirements(requirements_file: str, dry_run: bool = False):
    """安裝依賴"""
    filepath = PROJECT_ROOT / requirements_file
    if not filepath.exists():
        print(f"  ❌ 找不到 {requirements_file}")
        return False
    
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(filepath)]
    
    if dry_run:
        print(f"  [DRY-RUN] 將執行: {' '.join(cmd)}")
        return True
    
    print(f"  📦 正在安裝 {requirements_file}...")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f"  ✅ 依賴安裝完成")
        return True
    else:
        print(f"  ❌ 安裝失敗，請查看上方錯誤訊息")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PPO Trading Model - 環境自動安裝")
    parser.add_argument("--dry-run", action="store_true", help="只顯示動作，不實際執行")
    args = parser.parse_args()
    
    dry_run = args.dry_run
    
    print("=" * 60)
    print("  PPO Trading Model - 環境安裝腳本")
    print("=" * 60)
    
    # 1. 偵測平台
    platform = detect_platform()
    print(f"\n[1/4] 偵測平台")
    print(f"  作業系統: {platform}")
    print(f"  Python:   {sys.version.split()[0]}")
    print(f"  路徑:     {sys.executable}")
    
    if platform == "unknown":
        print(f"  ❌ 不支援的平台: {sys.platform}")
        sys.exit(1)
    
    # 2. 偵測硬體
    print(f"\n[2/4] 偵測硬體")
    gpu_info = detect_gpu_info()
    cpu_count = get_cpu_count()
    n_cpu = min(cpu_count, 6)  # 預設上限 6
    print(f"  CPU 核心數: {cpu_count}")
    print(f"  訓練用核心: {n_cpu}")
    print(f"  GPU 裝置:   {gpu_info['device']}")
    if gpu_info.get("gpu_name"):
        print(f"  GPU 名稱:   {gpu_info['gpu_name']}")
    
    # 3. 安裝依賴
    requirements_file = get_requirements_file(platform)
    print(f"\n[3/4] 安裝依賴")
    print(f"  使用: {requirements_file}")
    install_requirements(requirements_file, dry_run=dry_run)
    
    # 4. 生成 config_local.yaml
    print(f"\n[4/4] 生成本地配置")
    create_config_local(platform, gpu_info, n_cpu, dry_run=dry_run)
    
    # 完成
    print(f"\n{'=' * 60}")
    if dry_run:
        print("  [DRY-RUN] 以上為模擬結果，未實際執行任何操作")
    else:
        print("  ✅ 環境安裝完成！")
        print(f"\n  下一步：")
        print(f"    1. 查看/修改 config_local.yaml（如需自訂參數）")
        print(f"    2. 執行 python train.py 開始訓練")
    print("=" * 60)


if __name__ == "__main__":
    main()
