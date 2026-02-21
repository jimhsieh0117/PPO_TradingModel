
# PPO Trading Model - 專案規格文檔

> 本文檔用於快速了解專案的完整設計與實現細節

## 📋 專案概述

**目標**：使用 PPO（Proximal Policy Optimization）強化學習算法，基於 ICT（Inner Circle Trader）策略，開發一個加密貨幣短線交易機器人。

**定位**：機構交易員視角的自動化交易系統

---

## 🎯 交易設定

### 基本參數
| 參數 | 值 | 說明 |
|------|-----|------|
| 交易標的 | BTCUSDT | 幣安永續合約 |
| 時間框架 | 1分K | 短線交易 |
| 槓桿倍數 | 1x | 無槓桿（v9 配置） |
| 倉位大小 | 100% | 全倉操作 |
| 實際敞口 | 100% | 1.0 × 1x |
| 止損設定 | 2x ATR | 動態止損 + 追蹤止損（1.5% 為 fallback） |
| 單日回撤限制 | 10% | 觸發停止交易 |
| 初始資金 | 1,000,000 USDT | 回測資金 |

### 交易成本
- **手續費**：0.04% (taker)，開倉+平倉各一次 = 每筆 0.08%
- **滑點**：暫不模擬
- **同時持倉**：最多 1 個倉位

### 風險分析
- 單次止損損失：100% × 1x × 1.5% = **1.5% 賬戶**
- 每 episode (~100 筆交易) 手續費累計約 8%
- 價格 2% 波動 = **2% 賬戶變動**

---

## 🤖 PPO 模型設計

### 動作空間（離散）
```python
Action Space: Discrete(4)
{
    0: "平倉" - 關閉當前持倉（如果有）
    1: "做多" - 開多單，15% 資金，10x 槓桿
    2: "做空" - 開空單，15% 資金，10x 槓桿
    3: "持有" - 不做任何操作（重要：避免過度交易）
}
```

### 狀態空間特徵（31 維 = 26 市場特徵 + 5 持倉狀態）

#### 1. 市場結構 (Market Structure) - 3 個
- `trend_state`, `structure_signal`, `bars_since_structure_change`

#### 2. Order Blocks - 4 個
- `dist_to_bullish_ob`, `dist_to_bearish_ob`, `in_bullish_ob`, `in_bearish_ob`

#### 3. Fair Value Gaps - 3 個
- `in_bullish_fvg`, `in_bearish_fvg`, `nearest_fvg_direction`

#### 4. Liquidity - 3 個
- `liquidity_above`, `liquidity_below`, `liquidity_sweep`

#### 5. Volume & Price - 5 個
- `volume_ratio`, `price_momentum`, `vwap_momentum`, `price_position_in_range`, `zone_classification`

#### 6. Multi-Timeframe - 2 個
- `trend_5m`, `trend_15m`

#### 7. 波動率（v0.6 新增）- 1 個
- `atr_normalized`: ATR / close，當前相對波動率水平

#### 8. 時間特徵（v0.6 新增）- 2 個
- `hour_sin`: sin(2π × hour/24)，捕捉亞洲/歐美時段週期
- `hour_cos`: cos(2π × hour/24)

#### 9. 市場 Regime（v0.8 新增）- 3 個
- `adx_normalized`: ADX(14) / 100，趨勢強度 [0, 1]（>0.4 = 強趨勢，<0.2 = 盤整）
- `volatility_regime`: ATR 在過去 480 根 K 線中的相對位置 [0, 1]（rolling min/max 正規化）
- `trend_strength`: (close - EMA200) / ATR，裁切到 [-1, 1]（正 = 多頭趨勢，負 = 空頭趨勢）

#### 10. 持倉狀態 - 5 個
- `position_state`: 持倉方向 {-1, 0, 1}
- `floating_pnl_pct`: 浮動盈虧百分比
- `holding_time_norm`: 持倉時間正規化 (0~1)
- `distance_to_stop_loss`: 距止損距離 (0~1)
- `equity_change_pct`: Episode 權益變化

**總計：31 個特徵（26 市場 + 5 持倉）** ✅

---

## 💰 獎勵函數設計 (v9.0+)

### 當前參數設定
```yaml
reward:
  pnl_reward_scale: 700           # 已實現盈虧縮放（主導信號，含滑點後調高）
  take_profit_multiplier: 1.3     # 止盈獎勵倍數（盈利平倉 × 1.3，虧損 × 1.0）
  floating_reward_scale: 30       # 浮動盈虧縮放（v9: 120→30，已實現 PnL 主導）
  stop_loss_extra_penalty: 3.5    # 止損額外懲罰
  holding_bonus_max: 1.5          # 盈利持倉品質獎勵（持倉30步達最大值）
  holding_bonus_steps: 30         # 達到最大獎勵的持倉步數
  rapid_reentry_penalty: 0.0      # v9: 移除，手續費即為天然懲罰
  episode_profit_bonus: 100       # Episode 結算獎勵縮放
  # === 空倉機會成本（可選）===
  idle_penalty_enabled: false     # 啟用空倉機會成本
  idle_penalty_atr_threshold: 10  # 觸發門檻：bar 變動 > Nx ATR
  idle_penalty_scale: 0.3         # 懲罰縮放（小值，避免過度交易）
  idle_penalty_cooldown: 5        # 平倉後 N 步內不觸發
  # === 低波動持倉獎勵（可選）===
  low_vol_hold_bonus: 0.0         # 低波動空倉每步獎勵
  low_vol_threshold: 0.3          # volatility_regime < 此值時觸發
```

### 獎勵公式
```python
# 平倉時：已實現盈虧獎勵（主要信號）
if profit:
    reward = (realized_pnl / balance) * 700 * 1.3  # 止盈 1.3x
else:
    reward = (realized_pnl / balance) * 700 * 1.0  # 止損 1.0x

# 每步（持倉時）：浮動盈虧信號（輔助，權重低）
reward += floating_pnl_pct * 30

# 止損懲罰
if stop_loss: reward -= 3.5

# 盈利持倉品質獎勵（鼓勵持有盈利倉位）
if in_profit and holding_time <= 30:
    reward += 1.5 * (holding_time / 30)

# EMA Reward Normalization（v9 重新啟用）
reward = normalize_reward(reward)  # 穩定 critic target
```

### 設計原則
- ✅ 已實現 PnL 為主導信號（pnl_reward_scale=700）
- ✅ 浮動獎勵僅為輔助（30，之前 120 導致 7:1 失衡）
- ✅ 止盈不對稱倍數（1.3x）鼓勵讓利潤奔跑
- ✅ 啟用 EMA reward normalization 穩定 value network
- ✅ 手續費作為天然交易頻率約束
- ✅ 空倉機會成本可選啟用（解決 WFA 零交易 fold 問題）

---

## 🛠️ 技術架構

### 框架選擇
| 組件 | 框架/庫 | 版本建議 | 用途 |
|------|---------|----------|------|
| 深度學習 | PyTorch | >= 2.0 | 神經網絡後端 |
| 強化學習 | Stable-Baselines3 | >= 2.0 | PPO 算法實現 |
| LSTM 支援 | sb3-contrib | >= 2.0 | RecurrentPPO（LSTM 策略） |
| 環境接口 | Gymnasium | >= 0.29 | RL 環境標準 |
| 數據獲取 | binance-connector-python | latest | 幣安 API |
| 回測框架 | backtesting.py | latest | 策略回測 |
| 數據處理 | pandas, numpy | - | 數據處理 |
| 視覺化 | matplotlib, seaborn | - | 訓練曲線圖 |

### 為什麼選擇 Stable-Baselines3？
- ✅ PyTorch 底層，代碼清晰
- ✅ PPO 實現成熟穩定
- ✅ 2025 年持續維護
- ✅ 內建 Callback 系統
- ✅ Hugging Face 團隊，安全可靠

### LSTM 模式（v0.8 新增）

透過 `config.yaml` 中 `lstm.enabled: true` 切換：

| 模式 | Policy | 說明 |
|------|--------|------|
| MLP（默認） | `MlpPolicy` | 標準 PPO，只看靜態快照 |
| LSTM | `MlpLstmPolicy` | RecurrentPPO，有時間記憶，能感知 regime 轉換 |

```yaml
lstm:
  enabled: false        # true = RecurrentPPO, false = 普通 PPO
  lstm_hidden_size: 128
  n_lstm_layers: 1
```

**注意**：
- LSTM 比 MLP 慢約 2-3x
- 回測時需正確傳遞 LSTM hidden state（`strategy.py` 已處理）
- `n_steps × n_envs % batch_size == 0` 必須成立

---

## 📊 訓練與評估

### 數據管線（v0.5 增量架構）

**核心設計**：增量下載 + 處理後數據快取，避免重複下載與重複計算特徵。

**流程**：
```
config.yaml 日期範圍
    ↓
_scan_existing_raw_data()    → 掃描 data/raw/*.parquet
    ↓
_determine_missing_ranges()  → 比較已有 vs 需求，找出缺口
    ↓
_download_and_merge()        → 僅下載缺少部分，合併為單一 parquet
    ↓
_ensure_processed_data()     → 檢查 data/processed/ 快取（data_hash + feature_config_hash）
    ↓                          快取命中 → 直接載入；未命中 → 計算 26 市場特徵
ensure_data_ready()          → 按 test_start_date 分割 train/test
```

**關鍵 API**（`utils/data_pipeline.py`）：
| 函式 | 用途 | 使用者 |
|------|------|--------|
| `ensure_data_ready(config)` | 返回 `(train_df, test_df)`，含 OHLCV + 26 特徵 | `train.py`, `run_backtest.py` |
| `load_full_data(config)` | 返回完整 DataFrame（不分割） | `wfa.py` |
| `extract_features(df)` | 從 DataFrame 提取 `np.ndarray [n, 26]` | `train.py`, `wfa.py` |

**處理後資料格式**：
- 檔案：`data/processed/BTCUSDT_1m.parquet`（~150-200 MB）
- 欄位：`timestamp` + 6 OHLCV + 26 市場特徵 = 33 欄
- 快取驗證：`data/processed/BTCUSDT_1m.meta.json`（data_hash + feature_config_hash）

**增量下載行為**：
- 延長 `end_date` 5 天 → 僅下載缺少的 5 天數據
- 第二次執行相同配置 → 直接從處理後快取載入（秒級啟動）
- 變更 `features:` 配置 → 自動重新計算特徵（原始數據不重下載）

**數據概況**：
- **數據來源**：Binance Futures API (永續合約)
- **全部數據**：2020-01-01 ~ 2026-02-14（~3,220,000 根 1分K）
- **訓練數據**：2020-01-01 ~ 2024-12-31（~2,630,000 根 1分K）
- **測試數據**：2025-01-01 ~ 2026-02-14（~590,000 根 1分K）
- **數據分割**：時間序列分割（`test_start_date`），避免未來洩漏

### 訓練設置（當前配置）
| 參數 | 值 | 說明 |
|------|-----|------|
| Episode 長度 | 480 steps | 8 小時 = 1 個訓練回合 |
| 總訓練步數 | 2,500,000 | ~1M 為最佳檢查點 |
| 更新頻率 | 2048 steps | PPO 更新間隔 |
| 學習率 | 0.0001 | 穩定學習 |
| Batch Size | 64 | 小批量訓練 |
| N epochs | 8 | 每次更新的訓練輪數 |
| Gamma | 0.95 | 折扣因子 |
| GAE Lambda | 0.95 | 優勢估計參數 |
| Entropy Coef | 0.1 | 探索係數 |
| VF Coef | 0.5 | v9: 2.0→0.5 恢復默認 |
| 手續費 | 0.04% | Taker 費率（必須啟用）|
| 並行環境 | 6 | SubprocVecEnv |
| 網路架構 | 128×128 | MlpPolicy / MlpLstmPolicy |
| LSTM | 可選 | config `lstm.enabled`，128 hidden，1 層 |

### 訓練過程監控指標 ⭐（專業級）

為了**全面監控模型收斂情況**並找出改進方向，我們記錄以下所有指標：

#### 1. 獎勵指標（Reward Metrics）
- `episode_reward_mean`: 平均每個 episode 的總獎勵
- `episode_reward_std`: 獎勵標準差（評估穩定性）
- `episode_reward_max`: 最大單集獎勵
- `episode_reward_min`: 最小單集獎勵
- `cumulative_reward`: 累積總獎勵

**用途**：判斷模型是否在學習獲利策略

#### 2. 損失函數（Loss Metrics）
- `policy_loss`: 策略網絡損失（越小越好）
- `value_loss`: 價值網絡損失（越小越好）
- `entropy_loss`: 熵損失（探索程度）
- `total_loss`: 總損失

**用途**：監控神經網絡是否過擬合或學習停滯

#### 3. PPO 特定指標（PPO-Specific）
- `clip_fraction`: 被裁剪的比例（0.1-0.3 正常）
- `approx_kl`: 近似 KL 散度（< 0.05 正常）
- `explained_variance`: 解釋方差（越接近 1 越好）
- `learning_rate`: 當前學習率（如果使用學習率調度）

**用途**：判斷 PPO 更新是否過大/過小，是否需要調整超參數

#### 4. 交易行為指標（Trading Behavior）
- `total_trades_per_episode`: 每個 episode 的交易次數
- `long_ratio`: 做多動作比例
- `short_ratio`: 做空動作比例
- `hold_ratio`: 持有動作比例
- `close_ratio`: 平倉動作比例
- `avg_holding_time`: 平均持倉時間（K 線數）

**用途**：判斷模型是否過度交易或過於保守

#### 5. 盈利與風險指標（Profit & Risk）
- `episode_profit`: 每個 episode 的淨利潤（USDT）
- `episode_return_pct`: 每個 episode 的報酬率（%）
- `win_rate`: 勝率（獲利交易 / 總交易）
- `profit_factor`: 盈虧比（總獲利 / 總虧損）
- `sharpe_ratio`: 夏普比率（滾動計算）
- `max_drawdown`: 最大回撤（%）
- `stop_loss_count`: 止損次數
- `daily_drawdown_violations`: 單日回撤超過 10% 的次數

**用途**：評估模型的實際盈利能力和風險控制

#### 6. Episode 統計（Episode Stats）
- `episode_length`: 每個 episode 的步數
- `episode_completion_rate`: episode 正常結束比例（非提前終止）
- `avg_equity_curve_slope`: 權益曲線斜率（趨勢）

**用途**：判斷訓練是否穩定

#### 7. 探索 vs 利用（Exploration vs Exploitation）
- `action_entropy`: 動作選擇的熵（高 = 探索多）
- `action_distribution`: 各動作的選擇頻率分布

**用途**：確保模型不會過早收斂到次優策略

---

### 視覺化輸出（Training Plots）

每個模型訓練後會生成以下圖表：

```
plots/
├── 01_reward_curves.png          # 獎勵曲線（mean, max, min, std）
├── 02_loss_curves.png             # 三大損失曲線
├── 03_ppo_metrics.png             # clip_fraction, KL, explained_variance
├── 04_trading_behavior.png        # 動作分布和交易次數
├── 05_profit_metrics.png          # 利潤、勝率、profit_factor
├── 06_risk_metrics.png            # 夏普比率、最大回撤、止損次數
├── 07_episode_stats.png           # episode 長度和完成率
├── 08_action_distribution.png     # 動作選擇分布
└── 09_equity_curve_samples.png    # 隨機抽樣幾個 episode 的權益曲線
```

**每張圖都包含滾動平均線（smoothing）** 以便觀察趨勢

---

### 回測評估指標（Backtesting Metrics）

訓練完成後，使用 `backtesting.py` 進行回測，記錄以下指標：

1. **收益指標**
   - 總報酬率 (Total Return)
   - 年化報酬率 (Annualized Return)
   - 累積權益曲線 (Equity Curve)

2. **風險指標**
   - 夏普比率 (Sharpe Ratio)
   - 最大回撤 (Max Drawdown)
   - 最大回撤持續時間

3. **交易指標**
   - 交易次數 (Total Trades)
   - 勝率 (Win Rate)
   - 平均盈虧比 (Profit Factor)
   - **止損次數** (Stop Loss Count)

4. **其他**
   - 平均持倉時間
   - 最大連續虧損次數
   - 最大連續獲利次數

---

## 🔬 訓練發現與最佳實踐

### 過擬合分析 (2026-02-13)

經過多次訓練實驗，發現以下重要規律：

| 訓練步數 | 模型 | 測試報酬 | 交易/天 | 結論 |
|----------|------|---------|---------|------|
| 300k | run_20260213_145702 | -0.18% | 54 | 欠擬合，模型學會減少交易 |
| **1M** | **run_20260213_160140** | **+3.27%** | **64** | **最佳檢查點** |
| 2M | run_20260213_162800 | -1.85% | 91 | 過擬合，交易頻率上升 |

### 過擬合診斷指標

1. **交易頻率上升**：從 64 → 91 trades/day（警告信號）
2. **訓練指標持續改善，但測試表現下降**
3. **Explained Variance 持續上升但測試 Sharpe 下降**

### 關鍵 TensorBoard 指標

| 指標 | 1M (PPO_65) | 2M (PPO_66) | 說明 |
|------|-------------|-------------|------|
| Explained Variance | 0.043 → 0.214 | 0.100 → 0.242 | EV 持續上升但過度擬合 |
| Value Loss | 0.578 → 0.796 | 峰值後下降 | 2M 的 value loss 下降是過擬合信號 |
| 測試報酬 | +3.27% | -1.85% | 訓練更久反而更差 |

### 最佳模型檢查點

**推薦模型**：`run_20260213_160140/ppo_trading_model_best.zip`

| 指標 | 值 |
|------|-----|
| 總報酬 | +3.27% |
| 年化報酬 | +47.1% |
| 夏普比率 | 5.87 |
| 最大回撤 | -0.70% |
| 勝率 | 41.7% |
| 盈虧比 | 1.27 |
| 每日交易 | 64 筆 |

### 訓練建議

1. **最佳訓練步數**：約 1M steps（針對當前數據量）
2. **早停策略**：監控 Explained Variance，當 EV > 0.20 且趨勢平緩時考慮停止
3. **交易頻率監控**：若 trades/day 突然上升，可能是過擬合信號
4. **手續費重要性**：啟用 0.04% 手續費後，模型學會降低交易頻率

### 手續費對策略的影響

| 設定 | 無手續費 | 有手續費 (0.04%) |
|------|---------|-----------------|
| 每日交易 | 136 筆 | 64 筆 |
| 模型行為 | 高頻交易 | 選擇性交易 |
| 盈利能力 | 虛假高報酬 | 真實可行報酬 |

**結論**：訓練時必須啟用手續費，否則模型會學到無法在實盤獲利的策略。

---

### WFA 分析演進 (2026-02-20)

#### 訓練步數對 WFA 的決定性影響

| WFA Run | total_timesteps/fold | 盈利 fold | 零交易 fold | avg Sharpe | 結論 |
|---------|---------------------|-----------|------------|------------|------|
| wfa_043646 | 1,000,000 | 4/29 (13.8%) | **20/29** | -3.42 | 完全失敗，模型學會不交易 |
| **wfa_101954** | **3,000,000** | **12/29 (41.4%)** | **4/29** | **0.55** | **顯著改善，但仍未達標** |

> 單純增加訓練步數（1M → 3M），盈利 fold 從 4 個跳到 12 個，零交易從 20 個降至 4 個。

#### 最新 WFA（wfa_20260220_101954）- 各時期表現

| 時期 | fold | 報酬 | Sharpe | 交易/天 | 診斷 |
|------|------|------|--------|---------|------|
| 2021 牛市（Q1-Q2） | 1-3 | +15～+21% | 7～15 | 22～45 | ✅ 趨勢市場表現最佳 |
| 2021 波動期（Q3-Q4） | 4-7 | -1.3～+1.1% | -1～+0.9 | 39～67 | ⚠️ 交易頻率過高，損益接近零 |
| 2022 熊市回調 | 8-9 | +0.9～+3.1% | 1.2～4.2 | 31～40 | ✅ 仍能盈利 |
| 2022-2023 低波動 | 10-13 | -0.4～+0.6% | -1.4～+3.9 | 2～16 | ⚠️ 交易頻率大幅降低，報酬微薄 |
| 2023 橫盤期 | 14-16 | -0.5～-1.5% | -7.7～-3.5 | 4～11 | ❌ WR 34-35%，策略失效 |
| 2023-2024 轉折期 | 17-19 | 0～-3.9% | NaN～-5.0 | 0～2.8 | ❌ 零交易或嚴重虧損 |
| 2024 初期 | 20-22 | -1.9～0% | -5.0～NaN | 0～3.2 | ❌ 仍無法適應 2024 市場 |
| 2024 末牛市 | 23-24 | +1.4～+3.6% | 2.8～5.1 | 0.7～7.7 | ✅ 大幅降低頻率，精準交易 |
| 2025 年 | 25-29 | -3.3～0% | -3.1～NaN | 0～1.9 | ❌ 未來數據期，策略尚未泛化 |

---

## 📁 專案結構

```
PPO_TradingModel/
├── README.md                      # 專案說明
├── CLAUDE.md                      # 本文檔（專案規格）
├── requirements.txt               # Python 依賴
│
├── data/                          # 數據目錄
│   ├── raw/                       # 原始 OHLCV parquet（增量下載管理）
│   ├── processed/                 # 處理後數據（OHLCV + 26 市場特徵 parquet + meta.json）
│   ├── cache/                     # 舊版特徵快取（feature_cache.py 使用，strategy.py fallback）
│   └── download_data.py           # Binance API 數據下載器
│
├── environment/                   # Gymnasium 環境
│   ├── __init__.py
│   ├── trading_env.py             # 主環境類（31 維觀察空間）
│   └── features/                  # 特徵計算模塊
│       ├── __init__.py
│       ├── feature_aggregator.py  # 特徵聚合器（26 維市場特徵組裝）
│       ├── market_structure.py    # 市場結構特徵
│       ├── order_blocks.py        # Order Blocks 檢測
│       ├── fvg.py                 # Fair Value Gaps
│       ├── liquidity.py           # 流動性檢測
│       └── volume.py              # 成交量 + ATR + Regime 特徵
│
├── agent/                         # PPO 代理
│   ├── __init__.py
│   ├── ppo_agent.py               # PPO 訓練邏輯
│   ├── callbacks.py               # 訓練回調（記錄）
│   └── reward.py                  # 獎勵函數
│
├── backtest/                      # 回測模塊
│   ├── __init__.py
│   ├── run_backtest.py            # backtesting.py 整合
│   └── strategy.py                # 將 PPO 轉換為回測策略
│
├── utils/                         # 工具函數
│   ├── __init__.py
│   ├── data_pipeline.py           # 數據管線（增量下載 + 處理後快取，核心 API）
│   ├── feature_cache.py           # 舊版特徵快取（strategy.py fallback 仍使用）
│   ├── logger.py                  # 日誌系統
│   └── visualization.py           # 視覺化工具
│
├── models/                        # 模型儲存目錄
│   └── run_YYYYMMDD_HHMMSS/       # 每次訓練的輸出
│       ├── config.json            # 超參數配置
│       ├── model_best.zip         # 最佳模型檢查點
│       ├── model_final.zip        # 最終模型
│       ├── training_log.csv       # 訓練記錄（逐 step）
│       ├── plots/                 # 訓練曲線圖（專業級監控）
│       │   ├── 01_reward_curves.png
│       │   ├── 02_loss_curves.png
│       │   ├── 03_ppo_metrics.png
│       │   ├── 04_trading_behavior.png
│       │   ├── 05_profit_metrics.png
│       │   ├── 06_risk_metrics.png
│       │   ├── 07_episode_stats.png
│       │   ├── 08_action_distribution.png
│       │   └── 09_equity_curve_samples.png
│       └── backtest_results/      # 回測結果
│           ├── backtest.html      # backtesting.py 輸出
│           ├── equity_curve.png   # 權益曲線
│           ├── trades.csv         # 所有交易記錄
│           └── metrics.json       # 評估指標摘要
│
├── train.py                       # 訓練入口腳本
├── wfa.py                         # Walk Forward Analysis（滾動窗口驗證）
├── evaluate.py                    # 評估入口腳本
└── config.yaml                    # 全局配置文件
```

---

## 🚀 下一步行動

### 第一階段：環境搭建
1. ✅ 安裝依賴套件
2. ✅ 下載 5 個月 + 1 個月 BTCUSDT 1分K 數據
3. ✅ 驗證數據完整性

### 第二階段：特徵工程
1. ✅ 實現 ICT 特徵檢測算法
   - Market Structure (BOS/ChoCh)
   - Order Blocks
   - Fair Value Gaps
   - Liquidity Zones
2. ✅ 多時間框架整合（5m, 15m）
3. ✅ 特徵正規化與預處理

### 第三階段：環境與代理
1. ✅ 實現 Gymnasium Trading Environment
2. ✅ 實現獎勵函數
3. ✅ 配置 PPO 代理（SB3）
4. ✅ 設置訓練 Callbacks

### 第四階段：訓練與優化
1. ✅ 初始訓練（觀察收斂性）
2. ✅ 超參數調優
3. ✅ 特徵重要性分析

### 第五階段：回測與評估
1. ✅ backtesting.py 整合
2. ✅ 生成評估報告
3. ✅ 分析失敗案例

### 第六階段：（未來）
1. ⏳ 實盤接入準備
2. ⏳ API 延遲處理
3. ⏳ 風險監控系統

---

## ⚠️ 重要注意事項

### 風險聲明
- 本專案用於**學習和研究目的**
- 加密貨幣交易風險極高，可能導致全部資金損失
- **10x 槓桿**屬於高風險操作
- 回測表現**不代表**實盤表現

### 已知限制
1. **滑點未模擬**：實盤會有額外成本
2. **1分K 噪音**：極短線容易過擬合
3. **市場環境變化**：訓練數據若遇極端行情可能失效
4. **ICT 主觀性**：Order Block / FVG 定義有多種解釋

### 歷史重大 Bug（已修復）
1. **Config key 不匹配** (v0.8 修復)：`feature_aggregator.py` 讀 `market_structure_lookback` / `order_block_lookback`，但 config.yaml 定義為 `structure_lookback` / `ob_lookback`。所有 v0.7 及之前的模型都使用了默認 lookback 值（50/20），而非 config 中設定的 180。
2. **policy_kwargs 未傳遞** (v0.8 修復)：`train.py` 和 `wfa.py` 從未將 `policy_network` / `value_network` 傳給 `PPO()`。所有 v0.7 及之前的模型都使用 SB3 默認的 `[64, 64]` 網路架構，而非 config 中設定的 `[128, 128]`。

### 建議改進方向
- 加入更長時間框架（30m, 1h）
- ~~使用 ATR 動態止損替代固定百分比~~ ✅ v0.6 已實現
- ~~實現追蹤止損（trailing stop）~~ ✅ v0.6 已實現
- 加入最大持倉時間限制（詳見「當前瓶頸診斷」）
- 多幣種訓練（BTC, ETH）提高泛化能力
- WFA 通過率優化（目前 41.4% 盈利 fold，需 67%；詳見「當前瓶頸診斷」）

---

## 📝 版本記錄

- **v0.8** (2026-02-20): 市場 Regime 特徵 + LSTM 支援 + Bug 修復
  - ✅ 新增 3 個市場 Regime 特徵：`adx_normalized`、`volatility_regime`、`trend_strength`
  - ✅ 觀察空間更新：28 維 → 31 維（26 市場特徵 + 5 持倉狀態）
  - ✅ LSTM/RecurrentPPO 支援：config `lstm.enabled` 切換 MLP/LSTM，train.py + wfa.py + strategy.py 同步
  - ✅ 新增 `sb3-contrib` 依賴（RecurrentPPO）
  - ✅ 空倉機會成本懲罰機制（`idle_penalty_*` 參數，可選啟用）
  - ✅ 低波動持倉獎勵機制（`low_vol_hold_bonus`，可選啟用）
  - ✅ **修復 config key 不匹配 bug**：`feature_aggregator.py` 讀 `market_structure_lookback` 但 config 定義為 `structure_lookback`，導致所有 lookback 設定被靜默忽略、永遠使用默認值 50/20
  - ✅ **修復 policy_kwargs 未傳遞 bug**：train.py/wfa.py 從未將 `policy_network`/`value_network` 傳給 PPO()，所有舊模型用 SB3 默認 [64, 64]
  - ✅ 效能優化：`volatility_regime` 計算從 O(n × window) rolling rank 改為 O(n) rolling min/max
  - ⚠️ 所有舊模型不相容（觀察空間維度變更 + 特徵計算修正），需重新訓練

- **v0.7** (2026-02-17): 修復特徵正規化縮放問題（ATR 正規化取代價格百分比）
  - ✅ 修復距離特徵價格相依問題：`/ current_price * 100` → `/ ATR`
  - ✅ 影響特徵：`dist_to_bullish_ob`, `dist_to_bearish_ob`, `liquidity_above`, `liquidity_below`
  - ✅ 問題根因：相同 500 點距離在 $25K 顯示為 2%，在 $95K 顯示為 0.53%，導致 4x 失真
  - ✅ FVG 檢測閾值動態化：固定 `min_size_pct` → ATR 相對閾值 `min_size_atr=0.3`
  - ✅ 哨兵值更新：OB 距離 10.0→50.0，流動性距離 5.0→50.0（以 ATR 為單位）
  - ✅ FeatureAggregator 計算順序調整：Volume 移至第 1 步（產生 ATR），再傳給 OB/FVG/Liquidity
  - ✅ 清除舊快取（data/processed/ + data/cache/），強制重算特徵
  - ⚠️ 舊模型與新特徵尺度不相容，需重新訓練

- **v0.6** (2026-02-16): ATR 動態止損 + 波動率/時間特徵
  - ✅ ATR 動態止損：取代固定 1.5%，使用 2x ATR 倍數適應市場波動
  - ✅ 追蹤止損：止損價只朝有利方向移動，鎖住已有利潤
  - ✅ 新增 `atr_normalized` 特徵：讓模型感知當前波動率水平
  - ✅ 新增 `hour_sin`, `hour_cos` 時間特徵：捕捉亞洲/歐美時段交易模式
  - ✅ 觀察空間更新：25 維 → 28 維（23 ICT + 5 持倉）
  - ✅ VolumeAnalyzer 新增 ATR 預計算（原始 + 正規化）
  - ✅ 環境、回測策略、WFA 全部同步更新
  - ✅ config.yaml 新增 `atr_stop_multiplier`, `trailing_stop` 參數

- **v0.5** (2026-02-16): 數據管線重寫（增量下載 + 處理後數據快取）
  - ✅ 重寫 `utils/data_pipeline.py`：增量下載架構，僅補足缺少的日期範圍
  - ✅ 新增處理後數據快取：`data/processed/{symbol}_{interval}.parquet`（OHLCV + 20 特徵）
  - ✅ 快取驗證機制：透過 `data_hash` + `feature_config_hash` 自動判斷是否需要重算
  - ✅ 新增公開 API：`load_full_data()`, `extract_features()`
  - ✅ 更新 `train.py`：改用 `extract_features()` 取代 `precompute_features_with_cache()`
  - ✅ 更新 `wfa.py`：`load_full_dataset()` 委派給 `load_full_data()`，移除舊版直接存取
  - ✅ 更新 `backtest/run_backtest.py`：移除 `_build_expected_filename` 依賴
  - ✅ 碎片整合：多個 raw parquet 自動合併為單一檔案，舊碎片自動清理

- **v0.4** (2026-02-14): v9.0 綜合改進（程式碼深度分析 + 訓練數據診斷）
  - ✅ 修復 `vf_coef` 2.0→0.5（SB3 默認值，value loss 不再主導梯度）
  - ✅ 重新啟用 EMA reward normalization（穩定 critic target）
  - ✅ 放寬 best model 保存門檻（移除 sharpe>1.3 要求）
  - ✅ 重新平衡 reward：`floating_reward_scale` 120→30
  - ✅ 移除 `holding_bonus_max` 和 `rapid_reentry_penalty`
  - ✅ 觀察空間更新至 25 維（新增 5 維持倉狀態）
  - ✅ 診斷出浮動獎勵:已實現獎勵 = 7:1 失衡問題

- **v0.3** (2026-02-13): 過擬合分析與最佳實踐
  - ✅ 更新獎勵函數至 v7.3（pnl_reward_scale=500, floating_reward_scale=150）
  - ✅ 新增過擬合分析章節（1M vs 2M 訓練對比）
  - ✅ 確定最佳訓練步數為 ~1M steps
  - ✅ 記錄手續費對策略的關鍵影響
  - ✅ 新增 TensorBoard 關鍵指標解讀
  - ✅ 最佳模型：run_20260213_160140（+3.27% 報酬，Sharpe 5.87）
  - ✅ 探索 XAUUSDT（黃金永續）可行性：數據量不足（僅 64 天）

- **v0.2** (2026-01-13): 新增專業級訓練監控系統
  - ✅ 新增 40+ 訓練監控指標（7 大類）
  - ✅ 新增 9 張詳細訓練曲線圖
  - ✅ 確認使用 Binance Futures API 實盤歷史數據
  - ✅ 明確止損次數追蹤需求

- **v0.1** (2026-01-13): 初始專案規格文檔
  - 確定 ICT 策略框架
  - 設定 15% 倉位 + 10x 槓桿
  - 選定 Stable-Baselines3 + Gymnasium

---

## 🔍 當前瓶頸診斷（2026-02-20）

> 截至最新 WFA（wfa_20260220_101954），模型的核心問題已從「無法交易」升級為「在特定市場 Regime 策略失效」。以下是系統性診斷與下一步方向。

### 問題：特定市場 Regime 下策略完全失效

模型在 29 個 WFA fold 中表現呈現**強烈的 Regime 相依性**：

```
強趨勢市場（2021 牛市）   → Sharpe 7~15，表現極佳
熊市回調                  → Sharpe 1~4，仍能盈利
橫盤低波動（2023）         → WR 34%，策略幾乎無效
高波動快速行情（2024 Q1） → 零交易或大幅虧損
```

### 根本原因分析

| 問題現象 | 診斷 | 嚴重度 |
|---------|------|-------|
| 4 個 fold 零交易（17, 22, 25, 27） | 訓練數據以 2021 牛市為主，模型無法識別 2023-2025 市場特徵 | 高 |
| 2023 WR 跌至 34-35% | ICT 訊號在低波動橫盤中假突破多，止損頻繁 | 高 |
| 晚期 fold 持倉時間拉長至 50-92 bars | 模型困惑於進出場時機，被動等待導致長時間曝險 | 中 |
| fold 19 連虧 27 次（WR 22.8%） | 2024 年初 BTC 劇烈行情，ATR 止損倍數設定過大 | 中 |
| 早期 fold 交易頻率過高（40-67 次/天） | 2021 市場趨勢明顯，模型過度交易 | 低 |

### 改進方向（優先序）

1. **啟用 `idle_penalty_enabled: true`（scale 0.2-0.3）**
   - 目標：解決剩餘 4 個零交易 fold
   - 風險：可能略微增加高頻交易

2. **加入最大持倉時間限制（max_holding_steps ~60 bars）**
   - 目標：避免晚期 fold 持倉 50-92 bars 的被動式持倉
   - 實作：在 `trading_env.py` 的 step 中超時強制平倉

3. **ATR 止損倍數條件化（高波動期縮小到 1.5x）**
   - 目標：解決 fold 19 大幅連虧問題（2024 BTC 劇烈波動）
   - 可結合 `volatility_regime` 特徵動態調整

4. **訓練數據加權（近期數據加權更高）**
   - 目標：提升對 2023-2025 市場的泛化能力
   - 目前訓練以 2020-2024 均等採樣

### WFA 通過門檻現況

| 標準 | 目標 | 最新結果 | 缺口 |
|------|------|---------|------|
| 盈利 fold 比例 | ≥ 67% (19/29) | 41.4% (12/29) | 差 7 個 fold |
| 平均 Sharpe | ≥ 1.3 | 0.55 | 差 0.75 |
| 最大單 fold 回撤 | ≥ -10% | -3.98% | ✅ 通過 |

---

## 📧 聯繫與協作

本專案由 Claude (Anthropic) 與用戶協作開發。

**使用此文檔方式**：
- 在新對話中提供此文檔，快速恢復上下文
- 作為專案開發的 Single Source of Truth
- 持續更新，記錄重要決策和變更

---

*最後更新：2026-02-20*
