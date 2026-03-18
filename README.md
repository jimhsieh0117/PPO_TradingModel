# PPO Trading Model

基於 PPO（Proximal Policy Optimization）強化學習演算法，結合 ICT（Inner Circle Trader）策略特徵，開發的加密貨幣永續合約自動交易系統。涵蓋離線訓練、回測驗證、滾動窗口驗證（WFA）、超參數優化及實盤部署。

## 專案概述

| 項目 | 說明 |
|------|------|
| 演算法 | PPO（Stable-Baselines3 / sb3-contrib RecurrentPPO） |
| 市場 | 幣安永續合約（USDT 保證金） |
| 支援幣種 | BTCUSDT / ETHUSDT / SOLUSDT / WIFUSDT / XRPUSDT |
| 時間框架 | 1 分 K |
| 觀察空間 | 33 維（28 市場特徵 + 5 持倉狀態） |
| 動作空間 | 離散 4 動作：平倉 / 做多 / 做空 / 持有 |
| 止損方式 | 2x ATR 動態止損 + 可選追蹤止損 |
| 網路架構 | MLP 256x128（可選 LSTM） |

---

## 系統架構

```
                    ┌─────────────────────────────────────────┐
                    │            PPO Trading System           │
                    └─────────────────────────────────────────┘
                                      │
          ┌───────────────┬───────────┼───────────┬───────────────┐
          ▼               ▼           ▼           ▼               ▼
    ┌──────────┐   ┌──────────┐ ┌──────────┐ ┌──────────┐  ┌──────────┐
    │  數據層   │   │  訓練層   │ │  評估層   │ │  優化層   │  │  實盤層   │
    │          │   │          │ │          │ │          │  │          │
    │ 下載     │──▶│ PPO 訓練  │──▶│ 回測    │ │ Optuna   │  │ Bot      │
    │ 快取     │   │ 6 並行環境│ │ WFA     │ │ 3 階段   │  │ WebSocket│
    │ 特徵計算  │   │ 40+ 指標 │ │ HTML報告 │ │ 超參搜索  │  │ Telegram │
    └──────────┘   └──────────┘ └──────────┘ └──────────┘  └──────────┘
```

### 核心流程

```
訓練流程：config.yaml → 數據下載/快取 → TradingEnv × 6 → PPO 訓練 → 模型儲存
評估流程：模型 → backtesting.py 回測 → HTML 報告 + 指標 JSON
實盤流程：Binance WebSocket → 特徵計算 → 模型推論 → 風控閘門 → 下單 → Telegram 通知
```

---

## 快速開始

### 1. 安裝依賴

```bash
pip install -r requirements_base.txt
```

### 2. 配置參數

編輯 `config.yaml` 設定交易對、日期範圍與超參數。本機差異設定請建立 `config_local.yaml`（不納入 git 追蹤）：

```yaml
# config_local.yaml
ppo:
  device: "mps"   # 或 "cuda" / "cpu"
misc:
  n_cpu: 8
data:
  symbol: "ETHUSDT"
```

### 3. 訓練

```bash
python train.py
```

若數據不存在，會自動下載。訓練結果儲存於 `models/run_SYMBOL_YYYYMMDD_HHMMSS/`。

### 4. 回測

```bash
python -m backtest.run_backtest
python -m backtest.run_backtest --run-dir models/run_BTCUSDT_20260307_191940
```

### 5. Walk Forward Analysis

```bash
python wfa.py
```

### 6. 超參數優化

```bash
python optimize.py --phase phase1_ppo
python optimize.py --phase phase2_reward
python optimize.py --phase phase3_refine
```

### 7. 實盤部署

```bash
# 設定環境變數
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"
export TELEGRAM_BOT_TOKEN="your_token"
export TELEGRAM_CHAT_ID="your_chat_id"

# 啟動
python -m live_trading.bot
python -m live_trading.bot --dry-run  # 測試初始化但不交易
```

---

## 專案結構

```
PPO_TradingModel/
├── config.yaml                        # 共用訓練配置（git 追蹤）
├── config_local.yaml                  # 本機覆蓋配置（不追蹤）
├── train.py                           # 訓練入口
├── wfa.py                             # Walk Forward Analysis
├── optimize.py                        # Optuna 超參數優化（三階段）
├── setup_env.py                       # 環境初始化（依賴安裝 + 目錄建立）
├── sync_config.py                     # 配置同步工具
│
├── environment/                       # Gymnasium 交易環境
│   ├── trading_env.py                 # 主環境（33 維觀察、4 動作、獎勵函數 v9.0）
│   └── features/                      # ICT 特徵計算模組（7 個）
│       ├── feature_aggregator.py      # 整合器（產生 28 維特徵向量）
│       ├── market_structure.py        # 市場結構：BOS / ChoCh / 趨勢狀態
│       ├── order_blocks.py            # Order Blocks 偵測與距離
│       ├── fvg.py                     # Fair Value Gaps 偵測
│       ├── liquidity.py               # 流動性區域分析
│       ├── volume.py                  # 成交量、價格動量、ATR、ADX
│       └── multi_timeframe.py         # 5m / 15m 多時間框架趨勢
│
├── agent/                             # PPO 代理
│   └── callbacks.py                   # 訓練監控（40+ 指標、Best Model 複合評分）
│
├── backtest/                          # 回測模組
│   ├── run_backtest.py                # 回測入口（自動尋找最新模型）
│   └── strategy.py                    # PPO → backtesting.py Strategy 封裝
│
├── utils/                             # 工具模組
│   ├── data_pipeline.py               # 增量下載 + 特徵快取（核心 API）
│   ├── config_utils.py                # 配置管理（base + local 合併）
│   ├── feature_cache.py               # 特徵快取管理
│   └── visualization.py               # 訓練曲線圖表生成（9 張）
│
├── data/                              # 數據目錄
│   ├── download_data.py               # Binance API 下載器
│   ├── raw/                           # 原始 OHLCV parquet
│   └── processed/                     # 含 28 特徵的處理後 parquet
│
├── live_trading/                      # 實盤交易系統
│   ├── config_live.yaml               # 實盤專用配置
│   ├── bot.py                         # 主控迴圈（WebSocket 事件驅動）
│   ├── data_feed.py                   # WebSocket K 線接收 + 滾動 buffer
│   ├── feature_engine.py              # 即時特徵計算（複用 FeatureAggregator）
│   ├── inference.py                   # 模型推論引擎（MD5 校驗、deterministic）
│   ├── executor.py                    # 下單執行器（Limit IOC + Algo Stop）
│   ├── state.py                       # 持倉狀態管理（Single Source of Truth）
│   ├── risk_manager.py                # 多層風控閘門（8 層）
│   ├── notifier.py                    # Telegram 通知推送
│   ├── command_handler.py             # Telegram 遠端指令（/status, /stop 等）
│   ├── logger.py                      # 交易日誌（trades.jsonl + decisions.jsonl）
│   ├── state_snapshot.py              # 狀態快照持久化（防崩潰）
│   └── utils/
│       ├── binance_client.py          # Binance Futures REST API 封裝
│       └── retry.py                   # 重連管理（指數退避）
│
├── models/                            # 訓練結果
│   └── run_SYMBOL_YYYYMMDD_HHMMSS/
│       ├── ppo_trading_model_best.zip         # 最佳模型（複合評分）
│       ├── ppo_trading_model_best_top1~3.zip  # Top-3 模型
│       ├── ppo_trading_model_final.zip        # 最終模型
│       ├── config.yaml                        # 訓練時配置快照
│       ├── training_log.csv                   # 40+ 指標 CSV
│       ├── best_model_log.csv                 # Best Model 更新記錄
│       ├── train_log_png/                     # 9 張訓練曲線圖
│       └── backtest_results/                  # 回測報告
│
├── docs/                              # 文檔
│   ├── FEATURE_SPEC.md                # 33 維特徵完整規格
│   ├── TRAINING_METRICS.md            # 40+ 監控指標（7 大類）
│   ├── TRAINING_FINDINGS.md           # 過擬合分析與最佳實踐
│   └── CHANGELOG.md                   # 版本記錄（v0.1 ~ v0.9）
│
└── optimized_param/                   # 超參數優化結果
    └── export_best_params.py
```

---

## 觀察空間（33 維）

### 市場特徵（28 維）

| 類別 | 特徵數 | 特徵 |
|------|--------|------|
| 市場結構 | 3 | trend_state, structure_signal, bars_since_structure_change |
| Order Blocks | 4 | dist_to_bullish/bearish_ob, in_bullish/bearish_ob |
| Fair Value Gaps | 3 | in_bullish/bearish_fvg, nearest_fvg_direction |
| 流動性 | 3 | liquidity_above/below, liquidity_sweep |
| 成交量與價格 | 5 | volume_ratio, price_momentum, vwap_momentum, price_position_in_range, zone_classification |
| 多時間框架 | 2 | trend_5m, trend_15m |
| 波動率 | 1 | atr_normalized |
| 時間特徵 | 2 | hour_sin, hour_cos |
| 市場 Regime | 3 | adx_normalized, volatility_regime, trend_strength |
| **小計** | **26** | |

### 持倉狀態（5 維）

| 特徵 | 說明 |
|------|------|
| position_state | {-1, 0, 1}（做空、無倉、做多） |
| floating_pnl_pct | 浮動盈虧百分比 |
| holding_time_norm | 持倉時間正規化 (0~1) |
| distance_to_stop_loss | 距止損距離 (0~1) |
| equity_change_pct | 滾動 480 步窗口權益變化 |

---

## 獎勵函數（v9.0）

```python
# 平倉：已實現 PnL（主導信號）
reward = (realized_pnl / balance) * 700 * (1.3 if profit else 1.0)

# 每步：浮動 PnL（輔助）
reward += floating_pnl_pct * 30

# 止損額外懲罰
if stop_loss: reward -= 3.5

# 盈利持倉品質獎勵（30 步達最大 1.5）
if in_profit: reward += 1.5 * min(holding_time / 30, 1.0)

# EMA Reward Normalization 穩定 critic
reward = normalize_reward(reward)
```

**設計要點**：已實現 PnL 主導（700）、浮動獎勵輔助（30）、止盈不對稱（1.3x）、手續費為天然交易頻率約束

---

## Best Model 選擇（複合評分）

選擇最佳模型時使用複合評分取代單一 rolling mean return，並保存 Top-3 模型供比較：

```
score = 0.40 × rolling_sharpe + 0.25 × rolling_pf × 10
      - 0.20 × rolling_mdd × 100 + 0.15 × rolling_return
```

| 指標 | 權重 | 說明 |
|------|------|------|
| Sharpe Ratio | 40% | 風險調整回報 |
| Profit Factor | 25% | 毛利 / 毛損 |
| Max Drawdown | 20% | 最大回撤（越低越好） |
| Return | 15% | 回報百分比 |

---

## 實盤交易系統

### 架構

```
Binance WebSocket (kline_1m)
  ↓
DataFeed → 滾動 K 線 buffer
  ↓
FeatureEngine → 28 維市場特徵
  ↓
TradingState → 組合 33 維觀察向量
  ↓
InferenceEngine → 模型推論 action（deterministic=True）
  ↓
RiskManager → 8 層風控閘門
  ↓
Executor → Binance API 下單（Limit IOC + Algo Stop SL）
  ↓
Notifier → Telegram 通知 + Logger → 交易日誌
```

### 風控閘門（8 層）

| 層級 | 檢查項目 |
|------|---------|
| Layer 1 | 系統健康（WebSocket 連接、buffer 完整性、API 可用） |
| Layer 2 | 帳戶級別（單日虧損上限、總虧損上限、連續虧損次數） |
| Layer 3 | 倉位級別（最大持倉數、重複開倉防護） |
| Layer 4 | 訂單級別（最大下單金額、滑點保護） |
| Layer 5 | API 斷路器（連續 N 次 API 錯誤觸發冷卻） |
| Layer 6 | Kill Switch（STOP 檔案存在即停止） |
| Layer 7 | Client-side SL（Algo SL 的 backup，每分鐘檢查價格） |
| Layer 8 | reduceOnly 防護（平倉訂單不會意外開新倉） |

### Telegram 指令

| 指令 | 功能 |
|------|------|
| `/status` | 查看系統狀態（餘額、持倉、運行時間） |
| `/position` | 查看當前持倉詳情 |
| `/risk` | 查看風控統計 |
| `/pause` / `/resume` | 暫停 / 恢復交易 |
| `/mode1` / `/mode2` / `/mode3` | 通知模式：靜音 / 僅平倉 / 全部 |
| `/stop` | 優雅停止 bot |
| `/force_close` | 強制平倉（需確認） |

### 安全設計

- API Key 從環境變數讀取，永不硬編碼
- 推論永遠 `deterministic=True`、`device="cpu"`
- 模型載入後不更新（更換需停機重啟）
- 非冪等操作（下單）不自動重試
- 平倉訂單使用 `reduceOnly` 防止幽靈交易
- 每 5 分鐘健康檢查（state vs 交易所同步）
- 崩潰恢復：`state_snapshot.json` 保存風控計數器

---

## 訓練與實盤對齊

| 項目 | 訓練環境 | 實盤 |
|------|---------|------|
| 特徵計算 | `FeatureAggregator`（28 維） | 同一個 `FeatureAggregator` |
| 觀察空間 | `TradingEnv` 33 維 | `TradingState.build_observation()` 33 維 |
| 動作空間 | Discrete(4) | 相同映射 |
| 止損邏輯 | 2x ATR + 追蹤止損 | 2x ATR + Algo Stop + Client-side SL |
| PnL 計算 | 開倉時鎖定持幣數 | 相同 |
| 手續費 | 0.04% taker（開+平 0.08%） | 實盤依 VIP 等級 |
| equity_change_pct | 滾動 480 步窗口 | 相同 |

---

## 關鍵配置

### 訓練配置（config.yaml）

```yaml
trading:
  position_size_pct: 0.99     # 每次使用當前可用資金的比例
  atr_stop_multiplier: 2.0    # 止損 = 2x ATR
  taker_fee: 0.0004           # 0.04%

ppo:
  learning_rate: 0.0001
  n_steps: 2048
  batch_size: 64
  ent_coef: 0.1               # 必須 >= 0.1
  gamma: 0.95
  policy_network: [256, 128]
  value_network: [256, 128]

training:
  episode_length: 480          # 8 小時
  total_timesteps: 1000000
  best_model_top_n: 3          # 保存 Top-3 模型
```

### 實盤配置（live_trading/config_live.yaml）

```yaml
exchange:
  testnet: false
  api_key_env: "BINANCE_API_KEY"
  api_secret_env: "BINANCE_API_SECRET"

trading:
  symbol: "ETHUSDT"
  leverage: 1
  position_size_pct: 0.20

risk:
  atr_stop_multiplier: 2.0
  max_daily_loss_pct: 0.10
  max_total_loss_pct: 0.30
  max_order_value_usdt: 100.0
  circuit_breaker_threshold: 3
```

---

## Walk Forward Analysis

WFA 以滾動訓練窗口驗證樣本外穩健性，避免單次回測過擬合。

```yaml
wfa:
  train_window_months: 6
  test_window_months: 1
  step_months: 1
  total_timesteps: 2000000

  pass_criteria:
    min_profitable_folds_ratio: 0.50
    min_avg_sharpe: 0.5
    max_fold_drawdown_pct: -15.0
```

---

## 超參數優化（三階段）

| 階段 | 目標 | 搜索空間 |
|------|------|---------|
| Phase 1 | PPO 核心參數 | learning_rate, n_steps, batch_size, ent_coef |
| Phase 2 | 獎勵函數 | pnl_reward_scale, floating_reward_scale, stop_loss_penalty |
| Phase 3 | 精煉 | 保持前兩階段最佳，微調敏感參數 |

---

## 關鍵設計決策

### 熵係數必須 >= 0.1
WFA 實驗顯示 `ent_coef < 0.1` 會導致模型收斂至「從不交易」的策略。手續費本身即為過度交易的天然約束。

### 訓練步數 ~1M
1 分鐘 K 線噪音大，超過 1M steps 容易過擬合。

### 推論策略
- `deterministic=True`：實盤永不探索
- `device="cpu"`：避免 GPU 推論差異
- 模型載入後不更新（離線學習原則）

### 部署演進路徑
1. 現階段：離線訓練 → 回測驗證 → 部署固定模型
2. 穩定後：每 1-2 週收集新數據離線重訓 → A/B 對比
3. 進階：多 Regime 模型 + 根據市場狀態自動切換

---

## 依賴套件

- Python 3.10+
- PyTorch >= 2.0
- stable-baselines3 >= 2.0
- sb3-contrib >= 2.0（LSTM 模式）
- gymnasium >= 0.29
- backtesting >= 0.6.5
- pandas, numpy, matplotlib
- requests（實盤 API）

完整版本清單請見 `requirements_base.txt`。

---

## 延伸文檔

| 文檔 | 內容 |
|------|------|
| `docs/FEATURE_SPEC.md` | 33 維特徵完整規格與計算模組 |
| `docs/TRAINING_METRICS.md` | 40+ 監控指標（7 大類）與 9 張圖表說明 |
| `docs/CHANGELOG.md` | 版本記錄（v0.1 ~ v0.9） |
| `CLAUDE.md` | Claude Code 開發規格文檔 |

---

## 免責聲明

本專案僅供學習與研究使用。加密貨幣交易風險極高，過去的回測表現不代表未來實盤結果。
