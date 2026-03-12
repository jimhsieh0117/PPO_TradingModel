# Optuna 超參數優化 - 使用指南

## 基本用法

```bash
# Phase 1：優化 PPO 核心參數（learning_rate, ent_coef, gamma 等）
python optimize.py --phase phase1_ppo --n-trials 50

# Phase 2：固定 Phase 1 最佳參數，優化獎勵函數
python optimize.py --phase phase2_reward --n-trials 40

# Phase 3：精煉 Phase 1+2 最敏感參數（縮小範圍）
python optimize.py --phase phase3_combined --n-trials 30
```

## 進階選項

```bash
# 中斷後繼續（自動載入 SQLite study）
python optimize.py --phase phase1_ppo --n-trials 50 --resume

# 自訂訓練步數（預設 500k）
python optimize.py --phase phase1_ppo --n-trials 30 --timesteps 300000

# 調整並行環境數（預設 2）
python optimize.py --phase phase1_ppo --n-trials 50 --n-cpu 4

# 少量 trials 快速測試
python optimize.py --phase phase1_ppo --n-trials 3 --timesteps 100000
```

## 輸出檔案

```
optimized_param/
  study_{symbol}_{phase}.db       # SQLite（可 resume）
  best_params.yaml                # 最佳參數，格式相容 config_local.yaml
  optimization_report.html        # Optuna 視覺化報告
  trial_XXXX/                     # Top N 模型目錄（其餘自動清理）
```

## 使用最佳參數

```bash
# 將最佳參數複製為 config_local.yaml
cp optimized_param/best_params.yaml config_local.yaml

# 正式訓練
python train.py

# WFA 驗證
python wfa.py
```

## 匯出最佳參數
```
1. 自動偵測所有 study_{SYMBOL}_{PHASE}.db
2. 依序載入 phase1 → phase2 → phase3 最佳參數
3. 合併（後面 phase 覆蓋前面，例如 phase3 的 ppo.learning_rate 會蓋掉 phase1 的）
4. 輸出 best_params_ETHUSDT.yaml，巢狀格式直接相容 config_local.yaml
```

```bash
# 使用
python optimized_param/export_best_params.py

# 只匯出特定 symbol
python optimized_param/export_best_params.py --symbol ETHUSDT
```