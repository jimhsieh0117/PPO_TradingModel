
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
| 槓桿倍數 | 10x | 固定槓桿 |
| 倉位大小 | 15% | 每次開倉使用資金比例 |
| 實際敞口 | 150% | 15% × 10x |
| 止損設定 | 1.5% | 價格波動止損 |
| 單日回撤限制 | 10% | 觸發停止交易 |
| 初始資金 | 10,000 USDT | 模擬訓練資金 |

### 交易成本
- **手續費**：0.04% (taker)
- **滑點**：暫不模擬
- **同時持倉**：最多 1 個倉位

### 風險分析
- 單次止損損失：15% × 10x × 1.5% = **2.25% 賬戶**
- 連續 4-5 次止損接近單日回撤限制
- 價格 2% 波動 = **30% 賬戶損失**（理論最大）

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

### 狀態空間特徵（~20 維）

#### 1. 市場結構 (Market Structure) - 3 個
- `trend_state`: {-1: 下降趨勢, 0: 震盪, 1: 上升趨勢}
- `structure_signal`: {-1: Bearish ChoCh, 0: 無變化, 1: Bullish BOS}
- `bars_since_structure_change`: 距離上次結構轉變的 K 線數

#### 2. Order Blocks - 4 個
- `dist_to_bullish_ob`: 距離最近看漲 OB 的百分比距離
- `dist_to_bearish_ob`: 距離最近看跌 OB 的百分比距離
- `in_bullish_ob`: 是否在看漲 OB 內 (0/1)
- `in_bearish_ob`: 是否在看跌 OB 內 (0/1)

#### 3. Fair Value Gaps - 3 個
- `in_bullish_fvg`: 是否在看漲 FVG 內 (0/1)
- `in_bearish_fvg`: 是否在看跌 FVG 內 (0/1)
- `nearest_fvg_direction`: 最近未填補 FVG 方向 {-1, 0, 1}

#### 4. Liquidity - 3 個
- `liquidity_above`: 上方流動性距離（前高）百分比
- `liquidity_below`: 下方流動性距離（前低）百分比
- `liquidity_sweep`: 是否剛發生流動性掃蕩 (0/1)

#### 5. Premium/Discount Zones - 2 個
- `price_position_in_range`: 當前價格在波段中的位置 (0-100%)
- `zone_classification`: {-1: Discount, 0: Equilibrium, 1: Premium}

#### 6. 成交量與價格行為 - 3 個
- `volume_ratio`: 當前成交量 / 平均成交量
- `price_momentum`: 價格變化幅度（正規化）
- `vwap_momentum`: 成交量加權價格動量

#### 7. 多時間框架確認 - 2 個
- `trend_5m`: 5分K 趨勢方向 {-1, 0, 1}
- `trend_15m`: 15分K 趨勢方向 {-1, 0, 1}

**總計：20 個特徵** ✅

---

## 💰 獎勵函數設計

```python
reward = (
    # 1. 即時損益（正規化到初始資金）
    + (pnl / initial_capital) * 100

    # 2. 夏普比率改善（滾動計算）
    + (sharpe_ratio_new - sharpe_ratio_old) * 10

    # 3. 高風險倉位懲罰
    - max(0, position_size_ratio - 0.5) * 20  # 超過 50% 倉位懲罰

    # 4. 止損嚴重懲罰
    - 50 if hit_stop_loss else 0

    # 5. 交易成本
    - trading_fee

    # 6. 持倉時間獎勵（避免頻繁交易）
    + min(holding_time_bars, 10) * 0.1 if profitable else 0

    # 7. 單日回撤限制懲罰
    - 100 if daily_drawdown > 0.10 else 0
)
```

### 獎勵設計原則
- ✅ 獲利正向激勵
- ✅ 夏普比率提升有額外獎勵
- ✅ 過度風險重懲罰
- ✅ 止損觸發嚴重懲罰（引導學習避免止損）
- ✅ 鼓勵持有獲利倉位，減少過度交易

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

### 數據準備
- **數據來源**：Binance Futures API (永續合約)
- **訓練數據**：5 個月歷史數據（~216,000 根 1分K）
- **測試數據**：1 個月回測數據（~43,200 根 1分K）
- **數據分割**：時間序列分割，避免未來洩漏

### 訓練設置
| 參數 | 值 | 說明 |
|------|-----|------|
| Episode 長度 | 1440 steps | 24 小時 = 1 個訓練回合 |
| 更新頻率 | 2048 steps | PPO 更新間隔 (SB3 預設) |
| 學習率 | 3e-4 | 初始值，可調整 |
| Batch Size | 64 | 小批量訓練 |
| N epochs | 10 | 每次更新的訓練輪數 |
| Gamma | 0.99 | 折扣因子 |
| GAE Lambda | 0.95 | 優勢估計參數 |

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

## 📁 專案結構

```
PPO_TradingModel/
├── README.md                      # 專案說明
├── CLAUDE.md                      # 本文檔（專案規格）
├── requirements.txt               # Python 依賴
│
├── data/                          # 數據目錄
│   ├── raw/                       # 原始下載數據
│   ├── processed/                 # 預處理後數據
│   └── download_data.py           # 數據下載腳本
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

*最後更新：2026-01-13*
