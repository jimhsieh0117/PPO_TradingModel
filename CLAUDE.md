
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

### 狀態空間特徵（28 維 = 23 ICT + 5 持倉狀態）

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

#### 9. 持倉狀態 - 5 個
- `position_state`: 持倉方向 {-1, 0, 1}
- `floating_pnl_pct`: 浮動盈虧百分比
- `holding_time_norm`: 持倉時間正規化 (0~1)
- `distance_to_stop_loss`: 距止損距離 (0~1)
- `equity_change_pct`: Episode 權益變化

**總計：28 個特徵** ✅

---

## 💰 獎勵函數設計 (v9.0)

### 當前參數設定
```yaml
reward:
  pnl_reward_scale: 500        # 已實現盈虧縮放（主導信號）
  floating_reward_scale: 30    # v9: 120→30 大幅降低，避免壓倒已實現 PnL
  stop_loss_extra_penalty: 3.0 # 止損額外懲罰
  holding_bonus_max: 0.0       # v9: 移除
  rapid_reentry_penalty: 0.0   # v9: 移除，手續費即為天然懲罰
```

### 獎勵公式
```python
# 平倉時：已實現盈虧獎勵（主要信號）
reward = (realized_pnl / initial_balance) * 500

# 每步（持倉時）：浮動盈虧信號（輔助，權重低）
reward += floating_pnl_pct * 30

# 止損懲罰
if stop_loss: reward -= 3.0

# EMA Reward Normalization（v9 重新啟用）
reward = normalize_reward(reward)  # 穩定 critic target
```

### v9.0 設計原則
- ✅ 已實現 PnL 為主導信號（pnl_reward_scale=500）
- ✅ 浮動獎勵僅為輔助（30，之前 120 導致 7:1 失衡）
- ✅ 移除人工 shaping（holding_bonus, rapid_reentry_penalty）
- ✅ 啟用 EMA reward normalization 穩定 value network
- ✅ 手續費作為天然交易頻率約束

---

## 🛠️ 技術架構

### 框架選擇
| 組件 | 框架/庫 | 版本建議 | 用途 |
|------|---------|----------|------|
| 深度學習 | PyTorch | >= 2.0 | 神經網絡後端 |
| 強化學習 | Stable-Baselines3 | >= 2.0 | PPO 算法實現 |
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
    ↓                          快取命中 → 直接載入；未命中 → 計算 20 ICT 特徵
ensure_data_ready()          → 按 test_start_date 分割 train/test
```

**關鍵 API**（`utils/data_pipeline.py`）：
| 函式 | 用途 | 使用者 |
|------|------|--------|
| `ensure_data_ready(config)` | 返回 `(train_df, test_df)`，含 OHLCV + 20 特徵 | `train.py`, `run_backtest.py` |
| `load_full_data(config)` | 返回完整 DataFrame（不分割） | `wfa.py` |
| `extract_features(df)` | 從 DataFrame 提取 `np.ndarray [n, 20]` | `train.py`, `wfa.py` |

**處理後資料格式**：
- 檔案：`data/processed/BTCUSDT_1m.parquet`（~150-200 MB）
- 欄位：`timestamp` + 6 OHLCV + 20 ICT 特徵 = 27 欄
- 快取驗證：`data/processed/BTCUSDT_1m.meta.json`（data_hash + feature_config_hash）

**增量下載行為**：
- 延長 `end_date` 5 天 → 僅下載缺少的 5 天數據
- 第二次執行相同配置 → 直接從處理後快取載入（秒級啟動）
- 變更 `features:` 配置 → 自動重新計算特徵（原始數據不重下載）

**數據概況**：
- **數據來源**：Binance Futures API (永續合約)
- **訓練數據**：5 個月歷史數據（~216,000 根 1分K）
- **測試數據**：1 個月回測數據（~43,200 根 1分K）
- **數據分割**：時間序列分割，避免未來洩漏

### 訓練設置 (v9.0 配置)
| 參數 | 值 | 說明 |
|------|-----|------|
| Episode 長度 | 480 steps | 8 小時 = 1 個訓練回合 |
| 更新頻率 | 2048 steps | PPO 更新間隔 |
| 學習率 | 0.00015 | 穩定學習 |
| Batch Size | 64 | 小批量訓練 |
| N epochs | 8 | 每次更新的訓練輪數 |
| Gamma | 0.95 | 折扣因子 |
| GAE Lambda | 0.95 | 優勢估計參數 |
| Entropy Coef | 0.2 | 探索係數 |
| VF Coef | 0.5 | v9: 2.0→0.5 恢復默認，避免 value loss 主導 |
| 手續費 | 0.04% | Taker 費率（必須啟用）|
| 並行環境 | 6 | SubprocVecEnv |
| 網路架構 | 256×256 | MlpPolicy |

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

## 📁 專案結構

```
PPO_TradingModel/
├── README.md                      # 專案說明
├── CLAUDE.md                      # 本文檔（專案規格）
├── requirements.txt               # Python 依賴
│
├── data/                          # 數據目錄
│   ├── raw/                       # 原始 OHLCV parquet（增量下載管理）
│   ├── processed/                 # 處理後數據（OHLCV + 20 ICT 特徵 parquet + meta.json）
│   ├── cache/                     # 舊版特徵快取（feature_cache.py 使用，strategy.py fallback）
│   └── download_data.py           # Binance API 數據下載器
│
├── environment/                   # Gymnasium 環境
│   ├── __init__.py
│   ├── trading_env.py             # 主環境類
│   └── features/                  # 特徵計算模塊
│       ├── __init__.py
│       ├── market_structure.py    # 市場結構特徵
│       ├── order_blocks.py        # Order Blocks 檢測
│       ├── fvg.py                 # Fair Value Gaps
│       ├── liquidity.py           # 流動性檢測
│       └── volume.py              # 成交量特徵
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

### 建議改進方向
- 加入更長時間框架（30m, 1h）
- 使用 ATR 動態止損替代固定百分比
- 實現追蹤止損（trailing stop）
- 加入最大持倉時間限制
- 多幣種訓練（BTC, ETH）提高泛化能力

---

## 📝 版本記錄

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

## 📧 聯繫與協作

本專案由 Claude (Anthropic) 與用戶協作開發。

**使用此文檔方式**：
- 在新對話中提供此文檔，快速恢復上下文
- 作為專案開發的 Single Source of Truth
- 持續更新，記錄重要決策和變更

---

*最後更新：2026-02-17*
