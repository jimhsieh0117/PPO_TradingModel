"""
Config 同步腳本 - 將 config_local.yaml 的參數合併回 config.yaml

使用場景：
    經過調參實驗找到更好的參數後，用此腳本將 config_local.yaml 的設定
    同步回 config.yaml（共用配置），這樣其他開發者/機器可以透過 git pull 取得。

使用方式：
    python sync_config.py                # 互動模式，顯示差異後確認
    python sync_config.py --yes          # 直接同步，不詢問
    python sync_config.py --dry-run      # 只顯示差異，不修改
"""

import sys
from pathlib import Path

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """遞迴合併字典"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="同步 config_local.yaml → config.yaml")
    parser.add_argument("--yes", "-y", action="store_true", help="直接同步，不詢問")
    parser.add_argument("--dry-run", action="store_true", help="只顯示差異")
    parser.add_argument("--config", default="config.yaml", help="主配置文件路徑")
    parser.add_argument("--local", default="config_local.yaml", help="本地配置文件路徑")
    args = parser.parse_args()
    
    config_path = Path(args.config)
    local_path = Path(args.local)
    
    print("=" * 60)
    print("  Config 同步工具")
    print("=" * 60)
    
    # 檢查檔案
    if not config_path.exists():
        print(f"  ❌ 找不到 {config_path}")
        sys.exit(1)
    
    if not local_path.exists():
        print(f"  ❌ 找不到 {local_path}")
        print(f"  💡 請先建立 config_local.yaml 或執行 python setup_env.py")
        sys.exit(1)
    
    # 讀取配置
    with open(config_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)
    
    with open(local_path, "r", encoding="utf-8") as f:
        local_config = yaml.safe_load(f)
    
    if not local_config:
        print(f"  ℹ️  {local_path} 為空或全部被註解，沒有需要同步的項目")
        sys.exit(0)
    
    # 找出差異
    diffs = find_diffs(base_config, local_config)
    
    if not diffs:
        print(f"\n  ✅ 兩份配置完全一致，無需同步")
        sys.exit(0)
    
    # 顯示差異
    print(f"\n  找到 {len(diffs)} 項差異：\n")
    for action, path, old_val, new_val in diffs:
        if action == "新增":
            print(f"  ➕ {path}: {new_val}")
        else:
            print(f"  📝 {path}: {old_val} → {new_val}")
    
    if args.dry_run:
        print(f"\n  [DRY-RUN] 以上為差異，未修改任何檔案")
        sys.exit(0)
    
    # 確認同步
    if not args.yes:
        answer = input(f"\n  是否將以上變更同步到 {config_path}？(y/N) ").strip().lower()
        if answer != "y":
            print("  ❌ 取消同步")
            sys.exit(0)
    
    # 執行同步：讀取原始 config.yaml 的文字內容，用 deep_merge 合併後寫回
    merged = deep_merge(base_config, local_config)
    
    # 保留原始 config.yaml 的註解結構，使用 yaml.dump 寫回
    # 注意：yaml.dump 會丟失原始註解，但能保證格式正確
    with open(config_path, "r", encoding="utf-8") as f:
        original_content = f.read()
    
    # 備份原始檔案
    backup_path = config_path.with_suffix(".yaml.bak")
    with open(backup_path, "w", encoding="utf-8") as f:
        f.write(original_content)
    print(f"\n  💾 已備份原始配置到 {backup_path}")
    
    # 寫入合併後的配置
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(merged, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"  ✅ 已同步 {len(diffs)} 項變更到 {config_path}")
    print(f"\n  ⚠️  注意：原始的 YAML 註解可能已丟失，請檢查 {config_path}")
    print(f"  💡 如需還原：cp {backup_path} {config_path}")


if __name__ == "__main__":
    main()
