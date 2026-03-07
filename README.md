# PPO 交易模型

基於 PPO（Proximal Policy Optimization）強化學習演算法，結合 ICT（Inner Circle Trader）策略特徵，開發的加密貨幣永續合約自動交易機器人。

## 專案概述

| 項目 | 說明 |
|------|------|
| 演算法 | PPO（Stable-Baselines3） |
| 市場 | 幣安永續合約（USDT 保證金） |
| 時間框架 | 1 分 K |
| 觀察空間 | 31 維（26 市場特徵 + 5 持倉狀態） |
| 動作空間 | 離散 4 動作：平倉 / 做多 / 做空 / 持有 |
| 止損方式 | 2x ATR 動態止損 + 可選追蹤止損 |

## 主要特性

- **ICT 特徵工程**：市場結構（BOS/ChoCh）、Order Blocks、Fair Value Gaps、流動性區域
- **多時間框架分析**：5m 與 15m 趨勢信號
- **市場 Regime 偵測**：ADX、波動率 Regime、趨勢強度（EMA200）
- **動態止損**：ATR 自適應止損 + 可選追蹤止損（鎖定獲利）
- **Walk Forward Analysis（WFA）**：滾動窗口驗證，偵測過擬合
- **多幣種支援**：可訓練任意幣安合約交易對
- **增量數據管線**：只下載缺少的日期範圍，快取已處理特徵（秒級啟動）

## 快速開始

### 1. 安裝依賴

```bash
pip install -r requirements_base.txt
```

### 2. 配置參數

編輯 `config.yaml` 設定交易對、日期範圍與超參數。本機差異設定（GPU 裝置、CPU 數量等）請建立 `config_local.yaml`（不納入 git 追蹤）：

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
python train.py --config config.yaml
```

若數據不存在，會自動下載。訓練結果儲存於 `models/run_SYMBOL_YYYYMMDD_HHMMSS/`。

> **注意**：若交易對上市時間晚於 `start_date`（如 WIFUSDT），數據管線會自動探測最早可用日期並從該日開始下載。

### 4. 回測

```bash
python -m backtest.run_backtest
python -m backtest.run_backtest --run-dir models/run_BTCUSDT_20260307_191940
```

結果儲存於 `models/<run_dir>/backtest_results/`，包含 HTML 報告、權益曲線、交易明細 CSV 及指標摘要 JSON。

### 5. Walk Forward Analysis

```bash
python wfa.py
```

## 專案結構

```
PPO_TradingModel/
├── config.yaml                    # 共用配置（git 追蹤）
├── config_local.yaml              # 本機覆蓋配置（不追蹤）
├── train.py                       # 訓練入口
├── wfa.py                         # Walk Forward Analysis
├── evaluate.py                    # 評估入口
│
├── environment/                   # Gymnasium 交易環境
│   ├── trading_env.py             # 主環境（31 維觀察空間）
│   └── features/                  # 特徵計算模組
│       ├── feature_aggregator.py
│       ├── market_structure.py
│       ├── order_blocks.py
│       ├── fvg.py
│       ├── liquidity.py
│       └── volume.py
│
├── agent/                         # PPO 代理
│   ├── ppo_agent.py
│   ├── callbacks.py
│   └── reward.py
│
├── backtest/                      # 回測模組
│   ├── run_backtest.py
│   └── strategy.py                # PPO 封裝為 backtesting.py Strategy
│
├── utils/
│   ├── data_pipeline.py           # 增量下載 + 特徵快取（核心 API）
│   ├── feature_cache.py
│   ├── logger.py
│   └── visualization.py
│
├── data/
│   ├── download_data.py           # Binance API 下載器
│   ├── raw/                       # 原始 OHLCV parquet
│   └── processed/                 # 含 26 特徵的處理後 parquet
│
└── models/
    └── run_SYMBOL_YYYYMMDD_HHMMSS/
        ├── ppo_trading_model_best.zip
        ├── ppo_trading_model_final.zip
        ├── plots/                 # 訓練曲線圖（9 張）
        └── backtest_results/
            ├── backtest.html
            ├── equity_curve.png
            ├── trades.csv
            └── metrics.json
```

## 主要配置說明

```yaml
data:
  symbol: "BTCUSDT"
  start_date: "2020-01-01 00:00:00"
  end_date: "2026-02-14 23:59:59"
  test_start_date: "2025-01-01 00:00:00"   # 訓練/測試分割點

trading:
  position_size_pct: 0.99    # 每次使用當前可用資金的比例
  atr_stop_multiplier: 2.0   # 止損 = N × ATR
  trailing_stop: false        # 是否啟用追蹤止損

ppo:
  learning_rate: 0.0001
  n_steps: 2048
  batch_size: 64
  ent_coef: 0.1              # 必須 >= 0.1，否則模型收斂至零交易策略

reward:
  pnl_reward_scale: 700
  floating_reward_scale: 30
  stop_loss_extra_penalty: 3.5
```

## 回測結果（v0.9，測試期 2025-01-01 ~ 2026-02-14）

| 幣種 | 總報酬 | Sharpe | 最大回撤 | 勝率 | Profit Factor | 每日交易 |
|------|--------|--------|---------|------|---------------|---------|
| BTCUSDT | +23.7% | 2.96 | -9.0% | 38.6% | 1.29 | 8.4 |
| ETHUSDT | +332,522%* | 1.34 | -10.0% | 60.9% | 2.31 | 15.9 |
| SOLUSDT | +164.6% | 2.26 | -21.7% | 34.0% | 1.17 | 41.4 |
| WIFUSDT | +177,705%* | 1.65 | -6.3% | 40.4% | 1.72 | 71.5 |

> **\*報酬異常說明**：`position_size_pct=0.99` 在 backtesting.py 為複利模式（每筆使用當前資金比例），高交易頻率 + 正期望值導致指數複利爆炸。數字數學上正確，但受市場流動性限制現實不可行。**BTCUSDT +23.7% 為最可信參考**；其他幣種請以 Sharpe / Profit Factor / 勝率 / 最大回撤作為評估依據。

## 關鍵設計決策

### 倉位大小（position_size_pct = 0.99）
模型以全倉、無槓桿方式運作，每筆交易使用當前可用資金的 99%。回測中這會產生複利效應；短期實盤運作下影響有限。

### 熵係數必須 >= 0.1
WFA 實驗顯示 `ent_coef < 0.1` 會導致模型收斂至「從不交易」的策略。手續費本身即為過度交易的天然約束。

### 訓練與回測環境對齊（v0.9 修復）
| 項目 | 修復內容 |
|------|---------|
| 止損觸發 | 改用盤中最高/最低點，以止損價成交（原用收盤價） |
| PnL 計算 | 使用開倉時鎖定的持幣數（原每步重算） |
| equity_change_pct | 改為滾動 480 步窗口（原全期累計，回測會 OOD） |
| max_holding_steps | 回測策略同步強制平倉 |
| trailing_stop | 回測透過 `trade.sl` 動態更新追蹤止損 |

### Random Seed
只影響訓練（權重初始化、PPO 探索採樣）。推論時使用 `deterministic=True`，與 seed 完全無關。

## Walk Forward Analysis

WFA 以滾動訓練窗口驗證樣本外穩健性，避免單次回測過擬合。

```yaml
wfa:
  train_window_months: 6
  test_window_months: 1
  step_months: 1
  total_timesteps: 2000000

  pass_criteria:
    min_profitable_folds_ratio: 0.67   # 至少 67% fold 盈利
    min_avg_sharpe: 1.3                # 平均 Sharpe > 1.3
    max_fold_drawdown_pct: -10.0       # 單 fold 最大回撤不超過 -10%
```

最新 WFA 結果（3M steps/fold）：29 個 fold 中 12 個盈利（41.4%），平均 Sharpe 0.55。目標：67% 盈利 fold。

## 依賴套件

- Python 3.10+
- PyTorch >= 2.0
- stable-baselines3 >= 2.0
- sb3-contrib >= 2.0（LSTM 模式需要）
- gymnasium >= 0.29
- backtesting >= 0.6.5
- pandas、numpy、matplotlib
- binance-connector-python

完整版本清單請見 `requirements_base.txt`。

## 免責聲明

本專案僅供學習與研究使用。加密貨幣交易風險極高，過去的回測表現不代表未來實盤結果。
