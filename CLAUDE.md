
# PPO Trading Model - 專案規格文檔

> 本文檔用於快速了解專案的核心設計。詳細內容見 `docs/` 目錄。當前專案同時在 macOS 和 Windows 上開發，請注意環境差異。部分模型可能不在同一台電腦中。請用繁體中文回應使用者
將會用Codex審查你的程式碼
## 專案概述

**目標**：使用 PPO 強化學習 + ICT 策略特徵，開發可自適應各種市場環境的加密貨幣 1m 高頻交易機器人。

**開發者**：具有 ICT 交易經驗的大學生 | **定位**：機構交易員視角的自動化交易系統

### 核心原則
1. **訓練與回測環境必須最接近真實交易**：手續費、滑點、止損觸發邏輯、PnL 計算等必須與實盤一致
2. **以機構交易員角度評估表現**：不看絕對報酬，看 Sharpe、MDD、Profit Factor、勝率等風控指標
3. **推論時永遠 `deterministic=True`**：實盤不做探索，所有學習在離線環境完成
4. **不做 Online Learning**：學習永遠使用真實市場數據但不用真實資金

### 部署演進路徑
1. **現階段**：離線訓練 → 回測驗證 → 部署固定模型
2. **穩定後**：每 1-2 週收集新數據離線重訓 → A/B 對比 → 驗證後上線
3. **進階**：多 Regime 模型 + 根據市場狀態自動切換

---

## 交易設定

| 參數 | 值 | 說明 |
|------|-----|------|
| 交易標的 | 幣安永續合約 | 支援 BTCUSDT / ETHUSDT / SOLUSDT / WIFUSDT |
| 時間框架 | 1分K | 短線交易 |
| 槓桿 | 1x | 全倉無槓桿 |
| 倉位比例 | 40% | 每次開倉使用 40% 資金 |
| 止損 | 2x ATR | 動態止損 + 追蹤止損（2% fallback） |
| 手續費 | 0.06% taker | 開+平 = 每筆 0.12%（訓練必須啟用） |
| 初始資金 | 1,000,000 USDT | 回測用 |

---

## 模型設計

- **動作空間**：Discrete(4) — 平倉 / 做多 / 做空 / 持有
- **觀察空間**：33 維 = 28 市場特徵 + 5 持倉狀態（詳見 `docs/FEATURE_SPEC.md`）
- **網路架構**：MLP 256×128（可選 LSTM，config `lstm.enabled`）
- **框架**：Stable-Baselines3 PPO / sb3-contrib RecurrentPPO

### 關鍵訓練參數
| 參數 | 值 | 參數 | 值 |
|------|-----|------|-----|
| Episode 長度 | 720 steps (12hr) | 學習率 | 0.0001 |
| n_steps | 2048 | Batch Size | 64 |
| Entropy Coef | 0.08 | Gamma | 0.95 |
| Clip Range | 0.18 | 總訓練步數 | 4M steps |
| 並行環境 | 6 (SubprocVecEnv) | 最大持倉 | 120 steps |

---

## 獎勵函數（v9.0+）

```python
# 平倉：已實現 PnL（主導信號）
reward = (realized_pnl / balance) * 750 * (1.8 if profit else 1.0)
# 每步：浮動 PnL（輔助）
reward += floating_pnl_pct * 25
# 止損額外懲罰
if stop_loss: reward -= 6.0
# 盈利持倉品質獎勵（60 步達最大 3.0）
if in_profit: reward += 3.0 * min(holding_time / 60, 1.0)
# 快速重入懲罰（10 步內重新開倉）
if rapid_reentry: reward -= 2.0
# EMA Reward Normalization 穩定 critic
reward = normalize_reward(reward)
```

**設計要點**：已實現 PnL 主導（750）、浮動獎勵輔助（25）、止盈不對稱（1.8x）、快速重入懲罰（2.0）、手續費為天然交易頻率約束

---

## 數據管線

**核心 API**（`utils/data_pipeline.py`）：
- `ensure_data_ready(config)` → `(train_df, test_df)`（train.py, run_backtest.py 使用）
- `load_full_data(config)` → 完整 DataFrame（wfa.py 使用）
- `extract_features(df)` → `np.ndarray [n, 28]`

**特性**：增量下載（只補缺口）+ 處理後快取（data_hash + feature_config_hash 驗證）

---

## 重要注意事項

### 已知限制
- 滑點未模擬：實盤會有額外成本
- 1分K 噪音：極短線容易過擬合（最佳步數 ~1M，超過即過擬合）
- ICT 主觀性：Order Block / FVG 定義有多種解釋

### 歷史重大 Bug（已修復）
1. **Config key 不匹配** (v0.8)：`feature_aggregator.py` 讀 `market_structure_lookback`，config 定義為 `structure_lookback`
2. **policy_kwargs 未傳遞** (v0.8)：所有 v0.7 前模型用 SB3 默認 [64, 64] 而非 config 設定的 [128, 128]

### 改進方向
- 降低最大回撤（調整 `atr_stop_multiplier`、加 reward 回撤懲罰）
- WFA 通過率優化（目前 41.4% 盈利 fold，需 67%）
- 加入更長時間框架（30m, 1h）
- 多幣種訓練提高泛化能力

---

## 延伸文檔

| 文檔 | 內容 |
|------|------|
| `docs/FEATURE_SPEC.md` | 33 維特徵完整規格與計算模組 |
| `docs/TRAINING_METRICS.md` | 40+ 監控指標（7 大類）與 9 張圖表說明 |
| `docs/CHANGELOG.md` | 版本記錄（v0.1 ~ v0.9） |

---

*最後更新：2026-03-18*
