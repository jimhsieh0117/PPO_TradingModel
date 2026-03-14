# Live Trading Bot — 部署指南

## 環境變數設定

Bot 需要以下環境變數（API Key 永不寫入程式碼或配置檔）：

| 變數名稱 | 用途 |
|----------|------|
| `BINANCE_API_KEY` | Binance API Key |
| `BINANCE_API_SECRET` | Binance API Secret |
| `TELEGRAM_BOT_TOKEN` | Telegram Bot Token（通知用） |
| `TELEGRAM_CHAT_ID` | Telegram Chat ID（通知用） |

### macOS / Linux

編輯 `~/.zshrc`（macOS 預設）或 `~/.bashrc`（Linux）：

```bash
# 在檔案末尾加入
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_api_secret"
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"
```

儲存後執行：

```bash
source ~/.zshrc
```

### Windows

#### 方法 A：系統環境變數（推薦，永久生效）

1. 按 `Win + R` → 輸入 `sysdm.cpl` → 確定
2. 選「進階」分頁 → 「環境變數」
3. 在「使用者變數」區塊點「新增」
4. 依序新增上述 4 個變數

#### 方法 B：PowerShell（當前 session）

```powershell
$env:BINANCE_API_KEY = "your_api_key"
$env:BINANCE_API_SECRET = "your_api_secret"
$env:TELEGRAM_BOT_TOKEN = "your_bot_token"
$env:TELEGRAM_CHAT_ID = "your_chat_id"
```

#### 方法 C：PowerShell（永久寫入使用者環境變數）

```powershell
[Environment]::SetEnvironmentVariable("BINANCE_API_KEY", "your_api_key", "User")
[Environment]::SetEnvironmentVariable("BINANCE_API_SECRET", "your_api_secret", "User")
[Environment]::SetEnvironmentVariable("TELEGRAM_BOT_TOKEN", "your_bot_token", "User")
[Environment]::SetEnvironmentVariable("TELEGRAM_CHAT_ID", "your_chat_id", "User")
```

設定後重新開啟 terminal 生效。

---

## Telegram 通知設定

1. 在 Telegram 搜尋 `@BotFather` → 發送 `/newbot` → 依指示建立 Bot → 取得 **Bot Token**
2. 在瀏覽器開啟 `https://api.telegram.org/bot<你的Token>/getUpdates`
3. 對你的 Bot 發送任意訊息，重新整理頁面，從 JSON 中找到 `"chat":{"id": 123456}` → 這就是 **Chat ID**

---

## 啟動 Bot

### 前置條件

```bash
# 1. 進入專案目錄
cd PPO_TradingModel

# 2. 啟動虛擬環境
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. 安裝依賴（首次）
pip install -r requirements.txt
```

### 啟動指令

```bash
# Dry-run（測試連線 + 模組初始化，不交易）
python live_trading/bot.py --dry-run

# 正式啟動
python live_trading/bot.py

# 指定配置檔
python live_trading/bot.py --config live_trading/config_live.yaml
```

### 停止 Bot

- 按 `Ctrl + C`（graceful shutdown）
- 或建立 kill switch 檔案：`touch live_trading/STOP`（Bot 會在下一根 K 線停止）

---

## 配置切換

### Testnet ↔ Production

編輯 `live_trading/config_live.yaml`：

```yaml
exchange:
  testnet: true   # true = Testnet, false = Production
```

切換時需同步更換環境變數中的 API Key。

### 風控參數熱重載

Bot 運行中修改 `config_live.yaml` 的以下參數會自動生效（每 5 分鐘檢查），無需重啟：

- `risk.max_daily_loss_pct`
- `risk.max_total_loss_pct`
- `risk.max_consecutive_losses`
- `risk.max_order_value_usdt`
- `risk.min_balance_to_trade`
- `risk.max_slippage_pct`
- `system.shutdown_action`

---

## 檔案結構

```
live_trading/
├── bot.py              # 主控迴圈
├── config_live.yaml    # 配置檔
├── bot.pid             # PID lock（防重複啟動）
├── STOP                # Kill switch（建立此檔案停止 Bot）
├── state_snapshot.json # 狀態快照（崩潰恢復用）
├── logs/
│   ├── debug.log       # 完整日誌
│   ├── errors.log      # 錯誤日誌
│   ├── trades.jsonl    # 交易記錄
│   └── decisions.jsonl # 每根 K 線的決策記錄
└── tests/              # 單元測試
```
