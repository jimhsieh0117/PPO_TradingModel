"""
Config 同步腳本 - 將 config_local.yaml 的參數合併回 config.yaml（保留註解）

使用場景：
    經過調參實驗找到更好的參數後，用此腳本將 config_local.yaml 的設定
    同步回 config.yaml（共用配置），這樣其他開發者/機器可以透過 git pull 取得。

使用方式：
    python sync_config.py                # 互動模式，顯示差異後確認
    python sync_config.py --yes          # 直接同步，不詢問
    python sync_config.py --dry-run      # 只顯示差異，不修改
"""

import shutil
import sys
from pathlib import Path

import yaml
from ruamel.yaml import YAML


def find_diffs(base: dict, override: dict, path: str = "") -> list:
    """找出兩個字典之間的差異"""
    diffs = []
    for key, value in override.items():
        current_path = f"{path}.{key}" if path else key
        if key not in base:
            diffs.append(("新增", current_path, None, value))
        elif isinstance(value, dict) and isinstance(base.get(key), dict):
            diffs.extend(find_diffs(base[key], value, current_path))
        elif base[key] != value:
            diffs.append(("修改", current_path, base[key], value))
    return diffs


def update_recursive(target, source):
    """遞迴更新 ruamel.yaml CommentedMap，只改值不動註解。"""
    updated = []
    for key, value in source.items():
        if key in target:
            if isinstance(value, dict) and hasattr(target[key], 'items'):
                updated.extend(update_recursive(target[key], value))
            else:
                old_val = target[key]
                target[key] = value
                updated.append((key, old_val, value))
        else:
            target[key] = value
            updated.append((key, None, value))
    return updated


def main():
    import argparse
    parser = argparse.ArgumentParser(description="同步 config_local.yaml → config.yaml（保留註解）")
    parser.add_argument("--yes", "-y", action="store_true", help="直接同步，不詢問")
    parser.add_argument("--dry-run", action="store_true", help="只顯示差異")
    parser.add_argument("--config", default="config.yaml", help="主配置文件路徑")
    parser.add_argument("--local", default="config_local.yaml", help="本地配置文件路徑")
    args = parser.parse_args()

    config_path = Path(args.config)
    local_path = Path(args.local)

    print("=" * 60)
    print("  Config 同步工具（保留註解）")
    print("=" * 60)

    if not config_path.exists():
        print(f"  [ERROR] 找不到 {config_path}")
        sys.exit(1)

    if not local_path.exists():
        print(f"  [ERROR] 找不到 {local_path}")
        sys.exit(1)

    # 用標準 yaml 讀取兩份配置（比較差異用）
    with open(config_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    with open(local_path, "r", encoding="utf-8") as f:
        local_config = yaml.safe_load(f)

    if not local_config:
        print(f"  {local_path} 為空，沒有需要同步的項目")
        sys.exit(0)

    # 找出差異
    diffs = find_diffs(base_config, local_config)

    if not diffs:
        print(f"\n  兩份配置完全一致，無需同步")
        sys.exit(0)

    # 顯示差異
    print(f"\n  找到 {len(diffs)} 項差異：\n")
    for action, path, old_val, new_val in diffs:
        if action == "新增":
            print(f"  [+] {path}: {new_val}")
        else:
            print(f"  [~] {path}: {old_val} -> {new_val}")

    if args.dry_run:
        print(f"\n  [DRY-RUN] 以上為差異，未修改任何檔案")
        sys.exit(0)

    # 確認同步
    if not args.yes:
        answer = input(f"\n  是否將以上變更同步到 {config_path}？(y/N) ").strip().lower()
        if answer != "y":
            print("  取消同步")
            sys.exit(0)

    # 用 ruamel.yaml round-trip 讀取 config.yaml（保留註解）
    ryaml = YAML()
    ryaml.preserve_quotes = True

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = ryaml.load(f)

    # 遞迴更新值（只改值，不動註解結構）
    update_recursive(config_data, local_config)

    # 備份原始檔案
    backup_path = config_path.with_suffix(".yaml.bak")
    shutil.copy2(config_path, backup_path)
    print(f"\n  備份: {backup_path}")

    # 寫回（保留註解）
    with open(config_path, "w", encoding="utf-8") as f:
        ryaml.dump(config_data, f)

    print(f"  已同步 {len(diffs)} 項變更到 {config_path}（註解已保留）")
    print(f"  如需還原: cp {backup_path} {config_path}")


if __name__ == "__main__":
    main()
