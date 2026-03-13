# Live Trading System - 架構設計文檔

> **狀態**：已審核通過 — 可開始實作（2026-03-14）
>
> **原則**：每一行程式碼都假設「這會用真錢執行」。寧可不交易，也不可錯誤交易。

---

## 1. 系統總覽

```
                          ┌───────────────────────────┐
                          │     Binance Futures API    │
                          │   (Testnet / Production)   │
                          └─────┬──────────────┬───────┘
                                │              │
                         WebSocket            REST
                        (即時 K 線)      (下單 / 查詢)
                                │              │
┌───────────────────────────────┴──────────────┴─────────────────────┐
│                          bot.py (主控迴圈)                          │
│                                                                    │
│  ┌──────────┐  ┌───────────────┐  ┌──────────┐  ┌──────────────┐  │
│  │data_feed │→│feature_engine │→│ inference │→│  executor    │  │
│  │ .py      │  │ .py           │  │ .py       │  │  .py         │  │
│  └──────────┘  └───────────────┘  └──────────┘  └──────────────┘  │
│                                                                    │
│  ┌───────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │ risk_manager  │  │   state      │  │      notifier          │  │
│  │ .py           │  │   .py        │  │      .py               │  │
│  └───────────────┘  └──────────────┘  └────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
                                │
                          ┌─────┴─────┐
                          │  logger   │
                          │  .py      │
                          └───────────┘
```

### 資料流（每根 1m K 線收盤觸發一次）

```
1. data_feed    → 收到新 K 線（WebSocket kline close event）
2. data_feed    → 更新內部 DataFrame buffer（滾動窗口 500 根）
3. feature_engine → 從 buffer 計算 28 維市場特徵
4. state        → 組合 28 維特徵 + 5 維持倉狀態 = 33 維觀察向量
5. inference    → model.predict(obs, deterministic=True) → action
6. risk_manager → 檢查是否允許執行（風控閘門）
7. executor     → 發送下單指令到 Binance
8. state        → 更新持倉狀態
9. logger       → 記錄決策 + 執行結果
10. notifier    → 推送通知（開/平倉 + 異常）
```

---

## 2. 目錄結構

```
live_trading/
├── ARCHITECTURE.md          # 本文件
├── config_live.yaml         # 實盤專用配置（與訓練 config 分離）
│
├── bot.py                   # 主控迴圈（入口點）
├── data_feed.py             # WebSocket 即時數據接收
├── feature_engine.py        # 即時特徵計算（複用 extract_features）
├── inference.py             # 模型推論（載入 + predict）
├── executor.py              # 下單執行（Binance REST API）
├── risk_manager.py          # 風控模組（多層防護）
├── state.py                 # 持倉狀態管理（single source of truth）
├── notifier.py              # 通知推送（Telegram / LINE）
├── logger.py                # 交易日誌 + 審計軌跡
│
├── utils/
│   ├── binance_client.py    # Binance API 封裝（Testnet / Prod 切換）
│   └── retry.py             # 重試機制（指數退避）
│
├── tests/
│   ├── test_feature_parity.py   # 特徵一致性驗證
│   ├── test_state_machine.py    # 狀態機轉換測試
│   └── test_executor_dry.py     # 下單邏輯乾跑測試
│
└── logs/                    # 執行日誌輸出目錄
    ├── trades.jsonl         # 逐筆交易記錄
    ├── decisions.jsonl      # 逐步決策記錄
    └── errors.log           # 錯誤日誌
```

---

## 3. 各模組詳細設計

### 3.1 `config_live.yaml` — 實盤配置

```yaml
# === 交易所連線 ===
exchange:
  name: "binance_futures"
  testnet: true                    # true = Testnet, false = Production
  api_key_env: "BINANCE_API_KEY"   # 從環境變數讀取，永不寫死
  api_secret_env: "BINANCE_API_SECRET"

# === 交易設定 ===
trading:
  symbol: "ETHUSDT"
  leverage: 1                      # 與訓練一致（run_ETHUSDT_20260311_230121: leverage=1）
  position_size_pct: 0.20          # 與訓練一致（run_ETHUSDT_20260311_230121: 0.20）
                                   # 實際曝險 = 1x leverage × 20% = 0.20x
  margin_type: "CROSSED"           # 全倉模式
  taker_fee: 0.0004               # 與訓練一致（記錄用，實盤依 VIP 等級）
  slippage: 0.0                   # 實盤為真實滑點，不額外加（訓練用 0.0005，實盤更小，有利）
  max_holding_steps: 120           # 與訓練一致 — 持倉超過 120 步強制平倉
  dynamic_atr_stop: false          # 與訓練一致 — 不啟用波動率自適應 ATR 倍數

# === 模型 ===
model:
  path: "models/run_ETHUSDT_20260311_230121/ppo_trading_model_best.zip"
  expected_md5: ""                   # 模型檔案 MD5（啟動時校驗，防誤覆蓋）
  use_lstm: false
  deterministic: true              # 實盤永遠 true

# === 風控 ===
risk:
  # --- 止損 ---
  atr_stop_multiplier: 2.0        # 與訓練一致
  trailing_stop: false             # 與訓練一致（run_ETHUSDT_20260311_230121: false）
  stop_loss_pct: 0.02             # 與訓練一致（run_ETHUSDT_20260311_230121: 0.02）

  # --- 帳戶級別 ---
  max_daily_loss_pct: 0.10         # 單日虧損 > 10% → 暫停交易至隔日
  max_total_loss_pct: 0.30         # 總虧損 > 30% → 完全停止，需人工介入
  max_consecutive_losses: 10       # 連續虧損 10 筆 → 暫停 1 小時
  max_open_positions: 1            # 最多同時 1 個倉位

  # --- 訂單級別 ---
  max_order_value_usdt: 50.0       # 單筆最大下單金額（200U × 20% = 40U，留 25% 餘裕）
  min_balance_to_trade: 10.0       # 餘額低於 10U 停止交易

  # --- 滑點保護 ---
  max_slippage_pct: 0.003          # 滑點超過 0.3% 拒絕下單

  # --- API 斷路器 ---
  circuit_breaker_threshold: 3     # 連續 N 次 API 錯誤（429/5xx）觸發冷卻
  circuit_breaker_cooldown: 60     # 冷卻秒數
  circuit_breaker_max_triggers: 3  # 連續觸發 N 次冷卻 → 進入待機模式

# === 數據 ===
data:
  buffer_size: 500                 # K 線 buffer 長度（特徵計算需要歷史數據）
  warmup_bars: 200                 # 啟動時需先收集 200 根 K 線才開始交易

# === 通知 ===
notification:
  enabled: true
  method: "telegram"               # telegram / line / log_only
  telegram_bot_token_env: "TELEGRAM_BOT_TOKEN"
  telegram_chat_id_env: "TELEGRAM_CHAT_ID"
  # 通知觸發條件
  notify_on_trade: true            # 每筆交易通知
  notify_on_error: true            # 錯誤通知
  notify_heartbeat_minutes: 60     # 每小時心跳（確認系統運行中）

# === 日誌 ===
logging:
  level: "INFO"
  log_dir: "live_trading/logs"
  max_log_size_mb: 50
  backup_count: 10

# === 系統行為 ===
system:
  shutdown_action: "keep_with_sl"  # close_all = 平倉退出 / keep_with_sl = 保留倉位（有止損保護）
  state_snapshot: true             # 每次狀態更新後寫入 state_snapshot.json（崩潰後可恢復）
```

### 3.2 `bot.py` — 主控迴圈

**職責**：啟動所有模組、管理生命週期、處理異常

```
啟動流程：
1. 載入 config_live.yaml
2. 驗證所有必要環境變數存在（API key、通知 token）
3. 初始化 Binance client（Testnet / Prod）
4. 載入模型（inference.py）
5. 同步帳戶狀態（餘額 + 現有持倉）
6. 啟動 WebSocket 連線（data_feed.py）
7. 暖機：等待 buffer 蒐集 warmup_bars 根 K 線
8. 進入主迴圈：每根 K 線觸發一次決策

主迴圈（事件驅動，非輪詢）：
  on_kline_close(kline):
    features = feature_engine.compute(buffer)
    obs = state.build_observation(features)

    # max_holding_steps 強制平倉（與訓練一致，120 步）
    if state.position != 0 and state.holding_steps >= max_holding_steps:
        result = executor.force_close(state, reason="max_holding_steps")
        state.update(result)
        logger.log_trade(result)
        notifier.send_trade(result)
        return

    action = inference.predict(obs)
    if risk_manager.allow(action, state):
        result = executor.execute(action, state)
        state.update(result)
        logger.log_trade(result)
        notifier.send_trade(result)
    else:
        logger.log_blocked(action, reason)

graceful shutdown（可配置 shutdown_action）：
  - shutdown_action: "close_all"   → 平掉所有倉位 → 取消未完成訂單 → 記錄 → 退出
  - shutdown_action: "keep_with_sl" → 保留倉位（已有 STOP_MARKET 保護）→ 記錄 → 退出
  - 預設 "keep_with_sl"（避免在不利時機被迫平倉）
```

**關鍵設計決策**：

| 項目 | 決策 | 原因 |
|------|------|------|
| 事件驅動 vs 輪詢 | 事件驅動（WebSocket kline close） | 避免錯過 K 線、減少 API 呼叫 |
| 同步 vs 異步 | **同步**（非 asyncio） | 1m 級別不需要高併發，同步更容易除錯和審計 |
| 單執行緒 vs 多執行緒 | 單執行緒 + WebSocket 回調 | 避免競態條件，持倉狀態是共享狀態 |

### 3.3 `data_feed.py` — 即時數據

**職責**：維護滾動 K 線 buffer，通知主迴圈

```
核心邏輯：
- 連接 Binance WebSocket（wss://fstream.binance.com）
- 訂閱 {symbol}@kline_1m
- 每根 K 線收盤時（kline.is_closed == True）：
  1. 將新 K 線 append 到 buffer（pandas DataFrame）
  2. 保持 buffer 長度 ≤ buffer_size（FIFO 淘汰最舊）
  3. 回調通知 bot.py

斷線處理：
- WebSocket 斷線 → 自動重連（指數退避：1s, 2s, 4s, 8s, 最大 60s）
- 重連後用 REST API 補回缺失的 K 線（klines endpoint）
- 如果缺口 > 5 分鐘 → 通知使用者 + 標記 gap
```

**Buffer 格式**（與 data_pipeline 一致）：
```python
columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades']
# 約 500 行 × 7 欄，dtype 一致
```

**安全要點**：
- Buffer 是唯一數據來源，不依賴交易所歷史 API 做即時決策
- 每根 K 線驗證 timestamp **嚴格遞增**（防止亂序或重複）
  - Binance WebSocket 偶爾會送重複 kline 或時間戳錯位（尤其高波動時）
  - 收到 kline 時檢查 `new_ts > buffer_last_ts`，不滿足則丟棄並記錄 warning
  - 額外檢查間距：正常應為 60s，容許 ±2s 誤差，超出則標記異常

### 3.4 `feature_engine.py` — 即時特徵計算

**職責**：將 K 線 buffer 轉換為 28 維特徵向量

```
核心邏輯：
- 複用 utils/data_pipeline.py 的 extract_features()
- 但不能直接 import（data_pipeline 有下載邏輯）
- 方案：import FeatureAggregator，在完整 buffer 上計算
  → 只取最後一行的特徵（= 當前 bar）

計算方式：
  feature_aggregator = FeatureAggregator(config)
  all_features = feature_aggregator.compute(buffer_df)  # [500, 28]
  current_features = all_features[-1]                   # [28]
```

**特徵一致性驗證**（關鍵！）：

> **這是實盤最容易出 bug 的地方。** 訓練用的特徵和實盤用的特徵如果有微小差異，
> 模型的決策會完全不同，但不會報錯 — 只會默默虧錢。

驗證方案：
1. **啟動時**：載入一段歷史數據，分別用 data_pipeline 和 feature_engine 計算，
   assert 結果相同（`np.allclose(rtol=1e-5)`）
2. **上線前**：用回測數據跑一遍，確認動作序列和回測完全一致
3. **Runtime**：每 100 根 K 線，用 REST API 拉同樣數據做離線計算，比對特徵
4. **每日快照**：UTC 00:00 將當前 buffer 的完整特徵矩陣存為 `.npy`（存至 logs/feature_snapshots/），
   供離線比對與事後除錯使用

### 3.5 `inference.py` — 模型推論

**職責**：載入 PPO 模型、執行推論

```
核心邏輯：
  model = PPO.load(model_path, device="cpu")  # 推論用 CPU 即可
  action, _ = model.predict(obs, deterministic=True)

注意事項：
- deterministic=True 永遠不變（實盤不做探索）
- device="cpu"（推論不需要 GPU，且避免 MPS/CUDA 差異）
- 模型在啟動時載入一次，不需要重複載入
- LSTM 模式需維護 hidden state（如 use_lstm=true）
```

**模型保護**：
- 模型檔案路徑必須指向已驗證的模型（不可自動更新）
- 模型更換流程：停機 → 更換路徑 → 重新啟動 → 暖機
- **Checksum 校驗**：config_live.yaml 中配置 `model.expected_md5`，
  啟動時計算 .zip 的 MD5 並比對，不符則拒絕啟動
  → 防止誤覆蓋模型檔案（尤其 macOS/Windows 雙機同步時）

### 3.6 `executor.py` — 下單執行

**職責**：將 action 轉換為 Binance API 操作

```
Action 對應表（與 TradingEnv 完全一致）：
  0 = 平倉  → 若有持倉，發送反方向市價單
  1 = 做多  → 若空倉先平，再開多
  2 = 做空  → 若多倉先平，再開空
  3 = 持有  → 不操作

下單流程（以 action=1 做多為例）：
  1. 檢查當前持倉（若有空單 → 先平倉）
  2. 計算下單數量：balance * position_size_pct / price → 取最小精度
  3. 發送市價單（MARKET order）
  4. 等待成交確認（最多 5 秒）
  5. 設置止損單（STOP_MARKET order，ATR 動態計算）
  6. 記錄成交價、數量、手續費
```

**安全要點**：

| 風險 | 防護措施 |
|------|---------|
| 重複下單 | 每次下單前先查詢持倉，與 state 比對 |
| 訂單未成交 | 5 秒超時 → 取消 → 記錄異常 → 不重試 |
| 止損單未掛上 | 下單後驗證止損單存在，不存在則立即平倉 |
| 數量精度 | 查詢交易所 exchangeInfo 取得 stepSize / minQty |
| 餘額不足 | 下單前查餘額，不足則跳過 |
| minNotional 不足 | 啟動時查詢 exchangeInfo 取得 minNotional（ETHUSDT 約 $5-10），下單前驗證 `qty × price ≥ minNotional`。100U × 0.20 = 20U，正常足夠，但餘額低時可能不滿足 |
| API 錯誤 | 非冪等操作不重試，記錄後通知人工處理 |
| 部分成交 | 使用市價單（幾乎不會部分成交），若發生則按實際成交量管理 |
| 止損價基準 | 止損計算必須用**實際成交均價**（API 回傳的 avgPrice），而非下單前的市價。差異雖微小但邏輯上正確 |

**止損單管理**：
```
開倉時：
  1. 計算 ATR 止損價：entry_price - atr_stop_multiplier * ATR（多單）
  2. 發送 STOP_MARKET 止損單（reduce-only）
  3. 記錄止損單 orderId

追蹤止損（每根 K 線）：
  1. 計算新止損價
  2. 如果新止損 > 舊止損（多單）：
     優先方案：使用 Binance CANCEL_REPLACE API（原子操作，單次請求完成更新）
     備用方案：先掛新止損 → 確認成功 → 取消舊止損
     （注意：SDK 是否支援 CANCEL_REPLACE 需在 Phase 3 驗證，
      不支援則用備用方案，100U 規模下雙止損單共存風險極低）
  3. 這是實盤與回測最大的差異點 — 回測修改 trade.sl 是免費的，
     實盤每次修改都是一次 API 呼叫

平倉時：
  1. 取消對應的止損單
```

### 3.7 `risk_manager.py` — 風控模組

**職責**：多層風控閘門，任何一層不通過 → 拒絕交易

```
檢查順序（由粗到細）：

Layer 1 — 系統健康
  □ WebSocket 連線正常（最後心跳 < 30 秒前）
  □ K 線 buffer 無缺口
  □ 帳戶 API 可連通

Layer 2 — 帳戶級別
  □ 當日虧損 < max_daily_loss_pct (10%)
  □ 總虧損 < max_total_loss_pct (30%)
  □ 餘額 > min_balance_to_trade (10U)
  □ 連續虧損 < max_consecutive_losses (10 筆)

Layer 3 — 倉位級別
  □ 當前持倉數 < max_open_positions (1)
  □ 不重複開同方向倉位

Layer 4 — 訂單級別
  □ 預估下單金額 < max_order_value_usdt (50U)
  □ 預估滑點 < max_slippage_pct (0.3%)

任何一層 FAIL → 拒絕執行 + 記錄原因
```

**每日重置**：
- UTC 00:00 重置每日虧損計數器
- 連續虧損暫停後，1 小時後自動恢復

**API 斷路器（Circuit Breaker）**：
```
監控 REST API 回應狀態：
- 連續 3 次 HTTP 429 (Too Many Requests) 或 5xx → 觸發冷卻
- 冷卻期 60 秒，期間不發送任何 API 請求
- 冷卻期間有持倉 → 不操作（已有止損單保護）
- 冷卻結束後發送一次測試請求（ping / account info）確認恢復
- 如果連續觸發 3 次冷卻 → 通知人工 + 進入待機模式

目的：防止 API Key 被交易所暫時封禁（Ban），
     一旦被 Ban，所有操作（包括平倉和止損修改）都會失效
```

**緊急停止（Kill Switch）**：
- 建立 `live_trading/STOP` 檔案 → 系統立即停止交易
- 不需要重啟程式，主迴圈每步檢查此檔案是否存在
- 用途：人工緊急介入，比 Ctrl+C 更安全（不會中斷正在執行的 API 呼叫）

### 3.8 `state.py` — 持倉狀態管理

**職責**：Single source of truth — 所有模組從這裡讀取持倉狀態

```python
class TradingState:
    # 帳戶
    balance: float              # 可用餘額
    equity: float               # 總權益（含浮動盈虧）
    initial_balance: float      # 啟動時的餘額（計算回撤用）

    # 持倉
    position: int               # -1=空, 0=無, 1=多
    entry_price: float          # 開倉價
    entry_time: datetime        # 開倉時間
    quantity: float             # 持倉數量
    holding_steps: int          # 持倉 K 線數
    current_sl: float           # 當前止損價
    sl_order_id: str            # 止損單 ID

    # 歷史（build_observation 需要）
    episode_length: int         # 從模型訓練 config 讀取（run_ETHUSDT_20260311_230121 = 720）
    equity_history: deque       # deque(maxlen=episode_length)，滾動 equity 歷史
                                # maxlen 必須與模型訓練時的 episode_length 對齊
                                # 實盤無 reset，靠 deque 自動淘汰最舊值維持窗口
    last_close_step: int        # 上次平倉步數

    # 統計
    daily_pnl: float            # 今日累計盈虧
    total_pnl: float            # 總累計盈虧
    consecutive_losses: int     # 連續虧損筆數
    trade_count: int            # 今日交易筆數
```

**狀態同步**：
```
啟動時：
  1. 從 Binance API 查詢真實持倉（positionRisk endpoint）
  2. 如果已有持倉（上次未正常關閉）→ 同步到 state
  3. 如果 state 與交易所不一致 → 以交易所為準 + 發出警告

運行時：
  1. 每次下單後，用 API 回傳結果更新 state
  2. 每 5 分鐘做一次「健康檢查」：state vs 交易所實際狀態
  3. 不一致 → 通知 + 以交易所為準修正
```

**5 維持倉特徵**（build_observation，與 PPOTradingStrategy 對齊）：

> **注意**：TradingEnv 用 `(equity - initial_balance) / initial_balance`（因為每 episode reset），
> 但 PPOTradingStrategy 已改為滾動窗口（因為回測不 reset）。
> 實盤同樣不 reset，所以 state.py 必須與 **PPOTradingStrategy** 對齊，而非 TradingEnv。

```python
def build_observation(self, market_features: np.ndarray) -> np.ndarray:
    position_features = np.array([
        self.position,                                         # [-1, 0, 1]
        self._floating_pnl_pct(),                              # clip [-1, 1]
        min(self.holding_steps / 120.0, 1.0),                  # [0, 1]
        self._distance_to_stop_loss(),                         # [0, 1]
        self._equity_change_pct(),                             # clip [-1, 1]
    ], dtype=np.float32)
    return np.concatenate([market_features, position_features])

def _equity_change_pct(self) -> float:
    # 滾動窗口：用 deque 最舊的 equity 作為基準（模擬 episode reset）
    # 與 PPOTradingStrategy._get_position_features() 邏輯一致
    if self.equity_history:
        if len(self.equity_history) >= self.episode_length:
            baseline = self.equity_history[0]   # deque(maxlen) 自動淘汰 → [0] 即為窗口起點
        else:
            baseline = self.initial_balance
    else:
        baseline = self.initial_balance
    return float(np.clip((self.equity - baseline) / (baseline + 1e-10), -1.0, 1.0))
```

### 3.9 `notifier.py` — 通知推送

**職責**：重要事件推送到手機

```
通知類型：
  [TRADE] 開倉/平倉通知
    「ETHUSDT 做多 @ 2450.30 | 數量 0.04 | 止損 2401.12」
    「ETHUSDT 平倉 @ 2478.50 | PnL +28.20 (+1.15%) | 持倉 47 分鐘」

  [RISK] 風控觸發通知
    「[警告] 今日虧損達 -8.2%，接近 10% 停損線」
    「[停止] 連續虧損 10 筆，暫停交易 1 小時」

  [ERROR] 系統異常通知
    「[錯誤] WebSocket 斷線，已重連（缺口 2 分鐘）」
    「[嚴重] 止損單掛單失敗，已緊急平倉」

  [HEARTBEAT] 心跳（每小時）
    「系統運行中 | 餘額 98.5U | 今日 PnL +2.1U | 無持倉」

實作選項：
  - Telegram Bot（推薦，免費且穩定）
  - LINE Notify（台灣常用）
  - 純 log（測試用）
```

### 3.10 `logger.py` — 交易日誌

**職責**：完整審計軌跡，可事後分析

```
trades.jsonl（每筆交易一行）：
{
  "timestamp": "2026-03-15T14:30:00Z",
  "action": 1,
  "action_name": "LONG",
  "symbol": "ETHUSDT",
  "entry_price": 2450.30,
  "exit_price": null,
  "quantity": 0.04,
  "sl_price": 2401.12,
  "pnl": null,
  "pnl_pct": null,
  "fee": 0.0392,
  "balance_after": 99.96,
  "model_obs_hash": "a3f2..."  // 觀察向量 hash，供除錯比對
}

decisions.jsonl（每根 K 線一行，完整記錄模型輸入輸出）：
{
  "timestamp": "2026-03-15T14:30:00Z",
  "bar_close": 2451.20,
  "action_raw": 1,
  "action_executed": true,
  "risk_check_passed": true,
  "risk_block_reason": null,
  "position_before": 0,
  "position_after": 1,
  "features_snapshot": [0.12, -0.34, ...]  // 33 維
}
```

---

## 4. 關鍵風險與對策

### 4.1 特徵漂移（Feature Drift）— 最高優先級

| 問題 | 說明 |
|------|------|
| 什麼是特徵漂移 | 實盤計算的特徵與訓練時不一致（即使邏輯相同） |
| 為什麼會發生 | 訓練用完整歷史 DataFrame，實盤用滾動 500 根 buffer |
| 後果 | 模型收到 OOD 輸入 → 決策不可預測 → 虧損 |

**對策**：
1. buffer_size=500 遠大於任何特徵的 lookback period（最大 180）
2. 啟動暖機 200 根，確保所有特徵的滑動窗口填滿
3. `test_feature_parity.py`：用歷史數據驗證兩種計算路徑結果一致
4. Runtime 抽樣比對（每 100 根）

### 4.2 狀態不一致（State Desync）

| 問題 | 說明 |
|------|------|
| 什麼是狀態不一致 | 本地 state 記錄「無持倉」，但交易所實際有持倉 |
| 為什麼會發生 | 網路問題導致平倉指令未到達交易所 |
| 後果 | 持倉無人管理 → 可能大幅虧損 |

**對策**：
1. 每次下單後驗證交易所回傳結果
2. 每 5 分鐘健康檢查，state vs 交易所同步
3. 啟動時主動同步（處理上次異常關閉的殘留倉位）
4. 不一致時**以交易所為準**

### 4.3 止損單風險

| 問題 | 說明 |
|------|------|
| 止損單未掛上 | API 失敗但開倉成功，倉位無止損保護 |
| 止損單被取消 | 交易所維護或規則變更導致止損單失效 |

**對策**：
1. 開倉後**立即驗證**止損單存在
2. 驗證失敗 → 立即平倉（寧可不賺也不裸露風險）
3. 健康檢查同時驗證止損單狀態
4. 追蹤止損：修改止損時用「先掛新 → 確認成功 → 取消舊」順序（不先取消）

### 4.4 網路與斷線

| 問題 | 對策 |
|------|------|
| WebSocket 斷線 | 自動重連 + REST 補缺口 |
| REST API 超時 | 5 秒超時，非冪等操作不重試 |
| 長時間離線（>5min） | 通知人工 + 不自動開新倉 + 現有倉位保留（有止損保護） |
| Binance 維護 | 偵測 503/451 → 進入待機模式 → 恢復後重新暖機 |

### 4.5 資金安全

| 防護層 | 說明 |
|--------|------|
| API Key 權限 | 只開啟「交易」權限，不開「提現」 |
| IP 白名單 | Binance API 設定只允許運行機器的 IP |
| 環境變數 | API Key 從環境變數讀取，不寫入程式碼或 config |
| 最大下單限制 | 50U 硬上限（config 可調） |
| Kill Switch | `STOP` 檔案立即停止 |

---

## 5. 與訓練環境的差異對照

> 這些差異是實盤和回測/訓練的本質區別，需要在程式碼中明確處理

| 項目 | 訓練 / 回測 | 實盤 |
|------|------------|------|
| K 線來源 | 歷史 CSV/Parquet（完整） | WebSocket 逐根收（不可回頭） |
| 特徵計算 | 整段 DataFrame 一次算 | 滾動 buffer 增量算 |
| 下單 | 瞬間成交、無延遲 | API 呼叫、有延遲和失敗可能 |
| 止損 | 回測引擎自動執行 | 需自行掛 STOP_MARKET 單 |
| 追蹤止損 | 修改 trade.sl（免費） | 取消舊單 + 掛新單（API 呼叫） |
| 持倉狀態 | env 內部變數 | 本地 state + 交易所同步 |
| 手續費 | 固定 0.04% | 依 VIP 等級，市價通常 0.04% |
| 滑點 | config 設定值（0.05%） | 真實市場滑點（通常更小） |
| 資金 | 1,000,000 USDT | 100-200 USDT |
| 曝險 | leverage × position_size_pct | 必須與訓練一致 |
| Episode | episode_length 步後重置 | 不重置，持續運行 |

### 5.1 目標模型的關鍵訓練參數（實盤必須對齊）

> 來源：`models/run_ETHUSDT_20260311_230121/config.yaml`

| 參數 | 訓練值 | config_live.yaml 對應 |
|------|--------|----------------------|
| leverage | 1 | `trading.leverage: 1` |
| position_size_pct | 0.20 | `trading.position_size_pct: 0.20` |
| atr_stop_multiplier | 2.0 | `risk.atr_stop_multiplier: 2.0` |
| trailing_stop | false | `risk.trailing_stop: false` |
| stop_loss_pct | 0.02 | `risk.stop_loss_pct: 0.02` |
| dynamic_atr_stop | false | `trading.dynamic_atr_stop: false` |
| episode_length | 720 | `state.episode_length` (deque maxlen) |
| max_holding_steps | 120 | `trading.max_holding_steps: 120` — 超過強制平倉 |
| slippage | 0.0005 | `trading.slippage: 0.0`（實盤更小，有利） |
| taker_fee | 0.0004 | `trading.taker_fee: 0.0004`（實盤依 VIP 等級） |

### 5.2 Episode 不重置的影響

訓練時每 episode_length 步重置一次（此模型 = 720 步）。實盤持續運行，需要注意：

- **equity_change_pct**：
  - TradingEnv 用 `(equity - initial_balance) / initial_balance`，因為每 episode reset 後 initial_balance 就是起點
  - 但實盤/回測不 reset → PPOTradingStrategy 已改為**滾動窗口**（`equity_history[-episode_length]` 作為 baseline）
  - `state.py` 必須與 **PPOTradingStrategy** 對齊（滾動窗口），具體實作見 3.8 節 `_equity_change_pct()`
- **holding_steps**：實盤若持倉超過 episode_length，需確認 holding_time_norm 不會超出訓練分布
  → 已用 `min(holding_steps / 120.0, 1.0)` 飽和，安全
- **max_holding_steps**：此模型設為 120 步，實盤需一致執行強制平倉

---

## 6. 實作順序與驗證計劃

### Phase 0：基礎設施（預估 1-2 天）

```
□ config_live.yaml 定案
□ utils/binance_client.py — Testnet / Prod 切換
□ logger.py — 基礎 JSONL 日誌
□ notifier.py — Telegram 通知（或 log_only fallback）
□ state_snapshot.json 持久化機制 — 每次狀態更新寫入，啟動時恢復
  （防止崩潰後 consecutive_losses 歸零繞過風控）
```

驗證：能連接 Testnet API、能發送 Telegram 通知

### Phase 1：數據管線（預估 1-2 天）

```
□ data_feed.py — WebSocket 連線 + K 線 buffer
□ feature_engine.py — 即時特徵計算
□ tests/test_feature_parity.py — 特徵一致性測試
```

驗證：啟動後持續收 K 線，特徵計算結果與 data_pipeline 完全一致

### Phase 2：推論 + 狀態（預估 1-2 天）

```
□ inference.py — 模型載入 + predict
□ state.py — 持倉狀態管理 + 觀察向量組合
□ tests/test_state_machine.py — 狀態轉換測試
```

驗證：餵入歷史數據，模型輸出的動作序列與回測一致

### Phase 3：下單 + 風控（預估 2-3 天）

```
□ executor.py — 市價單 + 止損單
□ risk_manager.py — 多層風控閘門
□ tests/test_executor_dry.py — 乾跑測試（不真的下單）
```

驗證：在 Testnet 上手動觸發各種場景（開多、開空、平倉、止損、追蹤止損）

### Phase 4：整合 + Testnet 實測（預估 3-5 天）

```
□ bot.py — 整合所有模組
□ Testnet 全自動運行 1-2 週
□ 比對 Testnet 交易結果與同期回測結果
□ 修復所有差異
```

驗證標準：
- **特徵一致**：Runtime 抽樣誤差 < 1e-5
- **動作一致**：同樣的 K 線序列，產生相同動作（>95%）
- **止損正常**：所有持倉都有止損單保護
- **風控有效**：模擬極端情況（餘額歸零、連續虧損）風控正常觸發
- **斷線恢復**：手動斷網 → 恢復後系統正常運作

### Phase 5：小額實盤（100U）

```
□ 切換到 Production API
□ 初始入金 100 USDT
□ 前 3 天密切監控（每筆交易人工確認）
□ 穩定後轉為自動
```

---

## 7. 部署建議

### 選項 A：本地 Mac（開發測試用）

```
優點：方便監控和除錯
缺點：必須保持電腦不休眠、網路中斷風險較高
建議：僅用於 Testnet 階段
```

### 選項 B：雲端 VPS（實盤推薦）

```
推薦：AWS Lightsail / DigitalOcean / Vultr
規格：1 vCPU / 1GB RAM / 25GB SSD（最便宜方案即可，~$5/月）
地區：東京或新加坡（離幣安伺服器近）
部署：
  1. Docker 容器化（環境一致性）
  2. systemd service（crash 自動重啟）
  3. Telegram 通知（遠端監控）
```

### 選項 C：家用機器 24hr（折衷）

```
Windows 機器 + 遠端桌面
優點：免費、算力足夠
缺點：依賴家庭網路穩定性
建議：作為 Testnet 到 VPS 的過渡
```

---

## 8. 未來擴展（暫不實作）

- 多幣種同時交易
- 動態模型切換（根據 market regime）
- Web dashboard（Flask/Streamlit）
- 限價單（降低手續費）
- 資金費率套利
- ~~狀態持久化~~ → 已提升至 Phase 0（審計報告 B2 建議：崩潰後 consecutive_losses 歸零會繞過風控）

---

## 9. 審核結果

> 已通過 Jim + Gemini Pro 審核（2026-03-14）

- [x] **交易所**：Binance Futures（Testnet → Production）
- [x] **同步架構**：單執行緒 + WebSocket 事件驅動
- [x] **風控參數**：max_daily_loss 10% / max_total_loss 30% / max_consecutive_losses 10
- [x] **止損方式**：STOP_MARKET 單 + 追蹤止損（優先 CANCEL_REPLACE，備用先掛新再刪舊）
- [x] **通知方式**：Telegram Bot
- [x] **部署方式**：Testnet on Mac → Production on VPS
- [x] **Kill Switch**：STOP 檔案觸發
- [x] **特徵計算**：滾動 500 根 buffer + 200 根暖機
- [x] **實作順序**：Phase 0→1→2→3→4→5
- [x] **Testnet 測試期**：1-2 週後才轉實盤（需經歷週末行情）

### 審核後新增項目（Gemini 建議，已採納）

| # | 項目 | 位置 | 說明 |
|---|------|------|------|
| 1 | K 線 timestamp 嚴格遞增檢查 | 3.3 data_feed.py | 防止 WebSocket 重複/亂序 kline 污染 buffer |
| 2 | 每日特徵快照 (.npy) | 3.4 feature_engine.py | UTC 0:00 存 buffer 特徵矩陣，供離線比對 |
| 3 | CANCEL_REPLACE 止損更新 | 3.6 executor.py | 原子操作取代兩步操作，Phase 3 驗證 SDK 支援度 |
| 4 | 模型 MD5 checksum 校驗 | 3.5 inference.py | 啟動時驗證模型檔案完整性，防誤覆蓋 |
| 5 | API 斷路器 (Circuit Breaker) | 3.7 risk_manager.py | 防止連續 API 錯誤導致 Key 被 Ban |
| 6 | deque(maxlen=480) 明確化 | 3.8 state.py | 確保 equity 滾動窗口與訓練 episode_length 對齊 |

### 第二輪審核修正（Opus 審核，2026-03-14）

| # | 問題 | 嚴重度 | 修正 |
|---|------|--------|------|
| 1 | equity_change_pct 計算不一致 | 中高 | state.py 對齊 PPOTradingStrategy（滾動窗口），非 TradingEnv。3.8 節加入偽碼 |
| 2 | position_size_pct 曝險不一致 | 高 | 0.99 → 0.20，與訓練 config 一致。同步修正 max_order_value |
| 3 | trailing_stop 不一致 | 高 | true → false，與訓練 config 一致 |
| 4 | stop_loss_pct 不一致 | 中 | 0.015 → 0.02，與訓練 config 一致 |
| 5 | episode_length 硬編碼 480 | 中 | 改為從模型 config 讀取（此模型 = 720） |
| 6 | 狀態持久化未提及 | 低 | 列入 Section 8 未來擴展 |
| 7 | 新增 5.1 訓練參數對照表 | — | 供實作時交叉比對，防止參數不一致 |

### 第三輪修正（AUDIT_REPORT.md 嚴格審計，2026-03-14）

| # | 項目 | 嚴重度 | 修正 |
|---|------|--------|------|
| S1-S3 | position_size_pct / stop_loss_pct / trailing_stop | S | 第二輪已修正 ✅ |
| S4 | slippage / taker_fee 未明確 | 備註 | config_live.yaml 加入明確值 |
| A2b | max_holding_steps 未在 config 和主迴圈 | A | 加入 config + bot.py 強制平倉邏輯 |
| A3 | 止損用實際成交均價 | A | executor 安全要點表新增 |
| A4 | dynamic_atr_stop 確認 false | A | 加入 config 明確標註 |
| B1 | minNotional 最低限制 | B | executor 安全要點表新增檢查 |
| B2 | 狀態持久化 | B→提升 | 從 Section 8 提升至 Phase 0（防風控繞過） |
| B3 | shutdown_action 配置化 | B | 加入 config，預設 keep_with_sl |

### 延後項目（規模擴大時再加）

| 項目 | 原因 |
|------|------|
| BBA 滑點預估 | 100-200U + ETHUSDT 流動性極深，市價單滑點幾乎為 0 |

---

*建立日期：2026-03-14*
*第一輪審核（Gemini）：2026-03-14*
*第二輪審核（Opus）：2026-03-14*
*第三輪審計（AUDIT_REPORT.md）：2026-03-14*
