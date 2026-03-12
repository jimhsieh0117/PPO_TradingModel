"""
從 Optuna SQLite 資料庫中取得各 phase 最佳參數，合併後輸出為 best_params_{SYMBOL}.yaml

使用方式：
    # 自動偵測所有 .db 檔案，合併 phase1~3 最佳參數
    python optimized_param/export_best_params.py

    # 指定 symbol（預設自動從 db 檔名偵測）
    python optimized_param/export_best_params.py --symbol ETHUSDT

    # 指定輸出目錄
    python optimized_param/export_best_params.py --output-dir optimized_param

輸出：
    optimized_param/best_params_ETHUSDT.yaml
"""

import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict

import optuna
import yaml

# 靜音 Optuna 日誌
optuna.logging.set_verbosity(optuna.logging.WARNING)

PHASE_ORDER = ["phase1_ppo", "phase2_reward", "phase3_combined"]


def discover_studies(db_dir: Path) -> dict:
    """
    掃描目錄下所有 study_*.db，回傳 {symbol: {phase: db_path}} 結構。

    檔名格式：study_{SYMBOL}_{PHASE}.db
    例如：study_ETHUSDT_phase1_ppo.db
    """
    pattern = re.compile(r"^study_([A-Z0-9]+)_(phase\d+_\w+)\.db$")
    result = defaultdict(dict)

    for db_file in sorted(db_dir.glob("study_*.db")):
        m = pattern.match(db_file.name)
        if m:
            symbol, phase = m.group(1), m.group(2)
            result[symbol][phase] = db_file

    return dict(result)


def load_best_params(db_path: Path, study_name: str) -> dict:
    """從單一 study db 載入最佳 trial 的參數與指標。"""
    storage = f"sqlite:///{db_path}"
    study = optuna.load_study(study_name=study_name, storage=storage)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        return {}

    best = study.best_trial
    return {
        "params": dict(best.params),
        "score": best.value,
        "trial_number": best.number,
        "user_attrs": dict(best.user_attrs),
        "n_completed": len(completed),
        "n_total": len(study.trials),
    }


def merge_params_to_nested(all_params: dict) -> dict:
    """
    將 dot-notation 參數（如 'ppo.learning_rate'）轉為巢狀 dict。
    後面的 phase 會覆蓋前面的（phase3 > phase2 > phase1）。
    """
    nested = {}
    for dotted_key, value in all_params.items():
        keys = dotted_key.split(".")
        d = nested
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return nested


def export_symbol(symbol: str, phase_dbs: dict, output_dir: Path) -> Path:
    """處理單一 symbol：載入各 phase 最佳參數 → 合併 → 輸出 yaml。"""
    print(f"\n{'='*60}")
    print(f"  Symbol: {symbol}")
    print(f"{'='*60}")

    merged_params = {}
    all_details = {}

    for phase in PHASE_ORDER:
        if phase not in phase_dbs:
            print(f"  [{phase}] 未找到 .db 檔案，跳過")
            continue

        db_path = phase_dbs[phase]
        study_name = f"ppo_{symbol}_{phase}"
        info = load_best_params(db_path, study_name)

        if not info:
            print(f"  [{phase}] 無已完成的 trial，跳過")
            continue

        print(f"  [{phase}] Best Trial #{info['trial_number']} "
              f"| Score: {info['score']:.4f} "
              f"| Trials: {info['n_completed']}/{info['n_total']}")

        for k, v in info["params"].items():
            print(f"    {k}: {v}")

        attrs = info["user_attrs"]
        if attrs:
            print(f"    --- 回測指標 ---")
            for k in ["sharpe_ratio", "total_return_pct", "max_drawdown_pct",
                       "profit_factor", "total_trades", "win_rate_pct"]:
                if k in attrs:
                    print(f"    {k}: {attrs[k]}")

        # 合併（後面 phase 覆蓋前面）
        merged_params.update(info["params"])
        all_details[phase] = info

    if not merged_params:
        print(f"\n  [WARN] {symbol} 無任何可用參數，跳過輸出")
        return None

    # 轉為巢狀 dict（config_local.yaml 相容格式）
    nested = merge_params_to_nested(merged_params)

    # 加入註解用的 metadata
    metadata = {
        "_metadata": {
            "symbol": symbol,
            "description": f"Optuna best params for {symbol} (merged from {len(all_details)} phases)",
            "phases": {},
        }
    }
    for phase, info in all_details.items():
        metadata["_metadata"]["phases"][phase] = {
            "best_trial": info["trial_number"],
            "score": round(info["score"], 4),
            "trials": f"{info['n_completed']}/{info['n_total']}",
        }

    # 輸出
    out_path = output_dir / f"best_params_{symbol}.yaml"

    # 先寫 metadata 作為註解
    lines = [f"# Optuna Best Parameters for {symbol}"]
    for phase, info in all_details.items():
        lines.append(
            f"# {phase}: Trial #{info['trial_number']} "
            f"| Score: {info['score']:.4f} "
            f"| Sharpe: {info['user_attrs'].get('sharpe_ratio', 'N/A'):.2f} "
            f"| Return: {info['user_attrs'].get('total_return_pct', 'N/A'):+.1f}%"
        )
    lines.append("#")
    lines.append(f"# 使用方式：將此檔內容合併到 config_local.yaml 中對應欄位")
    lines.append("")

    yaml_content = yaml.dump(nested, allow_unicode=True, default_flow_style=False, sort_keys=False)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        f.write(yaml_content)

    print(f"\n  [OK] 已輸出: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="從 Optuna SQLite 資料庫匯出各 phase 最佳參數",
    )
    parser.add_argument(
        "--symbol", type=str, default=None,
        help="指定 symbol（預設自動偵測所有 symbol）"
    )
    parser.add_argument(
        "--db-dir", type=str, default=None,
        help="存放 .db 檔案的目錄（預設為腳本所在目錄）"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="輸出目錄（預設同 db-dir）"
    )
    args = parser.parse_args()

    # 預設目錄：腳本所在位置
    script_dir = Path(__file__).parent.resolve()
    db_dir = Path(args.db_dir) if args.db_dir else script_dir
    output_dir = Path(args.output_dir) if args.output_dir else db_dir

    print(f"掃描目錄: {db_dir}")
    discoveries = discover_studies(db_dir)

    if not discoveries:
        print(f"[ERROR] 在 {db_dir} 中未找到任何 study_*.db 檔案")
        sys.exit(1)

    print(f"找到 {len(discoveries)} 個 symbol: {', '.join(discoveries.keys())}")

    # 篩選 symbol
    if args.symbol:
        if args.symbol not in discoveries:
            print(f"[ERROR] 找不到 symbol: {args.symbol}")
            print(f"  可用: {', '.join(discoveries.keys())}")
            sys.exit(1)
        discoveries = {args.symbol: discoveries[args.symbol]}

    # 逐 symbol 處理
    output_files = []
    for symbol, phase_dbs in sorted(discoveries.items()):
        result = export_symbol(symbol, phase_dbs, output_dir)
        if result:
            output_files.append(result)

    # 總結
    print(f"\n{'='*60}")
    print(f"  匯出完成：{len(output_files)} 個檔案")
    for f in output_files:
        print(f"    {f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
