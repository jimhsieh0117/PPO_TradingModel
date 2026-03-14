# 🔴 Live Trading 實盤程式碼嚴格審計報告

> **審計範圍**：commit `a8a137d` (Phase 0-4 建立) 至 `748b775` (HEAD) 的所有程式碼
> **審計依據**：[training config.yaml](file:///Users/jim_hsieh/Documents/GitHub/PPO_TradingModel/models/run_ETHUSDT_20260311_230121/config.yaml) + [TradingEnv](file:///Users/jim_hsieh/Documents/GitHub/PPO_TradingModel/environment/trading_env.py) + [PPOTradingStrategy](file:///Users/jim_hsieh/Documents/GitHub/PPO_TradingModel/backtest/strategy.py)
> **前次審計**：[上次審計報告](file:///Users/jim_hsieh/.gemini/antigravity/brain/bbd564ab-0c46-4b76-9b87-1da68f9a4e2d/live_trading_audit.md)（2026-03-13，僅審 ARCHITECTURE.md）
> **審計時間**：2026-03-14
> **當前帳戶狀態**：`balance=141.29U`，`testnet=false`（正式環境），`step_count=11`

---

## 📋 前次審計修正追蹤

前次審計發現了 S1-S4 四個嚴重設定不匹配。**逐一驗證修正狀態**：

| 編號 | 問題 | 前次 | 現在 config_live.yaml | 狀態 |
|------|------|------|----------------------|------|
| S1 | `position_size_pct` | 0.99 → 0.2 | **0.20** ✅ | 已修正 |
| S2 | `stop_loss_pct` | 0.015 → 0.02 | **0.02** ✅ | 已修正 |
| S3 | `trailing_stop` | true → false | **未設定** ⚠️ | 見下方 |
| S4 | `slippage` | — | **0.0** ✅ | 已修正 |
| A2 | `episode_length` | 480 → 720 | **720** ✅ | 已修正 |
| A2b | `max_holding_steps` | 未實作 | **120** ✅ | 已實作 |
| B2 | 狀態持久化 | 無 | `state_snapshot.json` ✅ | 已實作 |
| B3 | Shutdown 配置化 | 無 | `shutdown_action: keep_with_sl` ✅ | 已實作 |

---

## 🚨 嚴重等級 S — 可能直接導致非預期虧損

### S1. `equity_change_pct` 計算與訓練環境不一致

> [!CAUTION]
> 這是**最關鍵的特徵不一致問題**，模型每次推論都會吃到這個特徵。

**訓練環境** (`TradingEnv` L420)：
```python
equity_change_pct = (self.equity - self.initial_balance) / self.initial_balance
```
→ 相對於 **episode 起始 initial_balance** 的累積變化

**回測策略** (`PPOTradingStrategy` L155)：
```python
if len(self._equity_history) >= self.episode_length:
    baseline = self._equity_history[-self.episode_length]
else:
    baseline = self._initial_equity
```
→ 滾動窗口，窗口滿前用 `_initial_equity`

**實盤程式碼** (`state.py` L168)：
```python
if len(self.equity_history) >= self.episode_length:
    baseline = self.equity_history[0]  # deque(maxlen) → [0] 即為窗口起點
else:
    baseline = self.initial_balance
```
→ 滾動 deque，窗口滿前用 `initial_balance`

**問題分析**：
- 模型是在 `TradingEnv` 上訓練的，所以它見到的 `equity_change_pct` = `(equity - initial_balance) / initial_balance`
- 訓練時每 720 步 reset，所以等效於最多 720 步的累計
- **實盤用的滾動窗口寫法** 雖然「設計上更合理」，但**與訓練不一致**
- 訓練時 `self.initial_balance` 固定不變（episode 內）；實盤 `self.initial_balance` 也固定不變
- 窗口滿之前（前 720 步）兩者行為一致
- 窗口滿之後：訓練永遠不會超過 720 步（episode reset），但實盤會超過 → 基線開始滑動

**影響評估**：前 720 分鐘（12 小時）行為一致。**12 小時後基線開始偏移**。偏移大小取決於盈虧累積。如果帳戶穩定，偏移小；如果大賺或大虧後，偏移顯著。

**風險等級**：🟠 中高（12 小時以內安全，超過 12 小時後逐漸 OOD）

---

### S2. `trailing_stop` 在 config_live.yaml 中缺失

[config_live.yaml](file:///Users/jim_hsieh/Documents/GitHub/PPO_TradingModel/live_trading/config_live.yaml) 中沒有 `trailing_stop` 欄位。

但實際上 executor.py **沒有追蹤止損邏輯**（只設一次止損單不移動），所以行為其實是正確的（`trailing_stop=false`）。

> [!NOTE]
> 建議在 config_live.yaml 明確加上 `trailing_stop: false` 以供文檔完整性。行為本身正確。

---

### S3. 開倉後 `_handle_trade_result` 存在邏輯風險

[bot.py L369-419](file:///Users/jim_hsieh/Documents/GitHub/PPO_TradingModel/live_trading/bot.py#L365-L419)：

```python
def _handle_trade_result(self, result, ...):
    position_before = self.state.position  # ← 在更新前讀取

    if "entry_price" in result and "exit_price" not in result:
        # 開倉
        self.state.open_position(...)
    elif "exit_price" in result:
        # 平倉
        record = self.state.close_position(...)
```

**問題**：當 executor 執行「先平後開」（反向開倉）時，executor 內部先平倉再開倉，但只 return 最後的開倉結果。**平倉的 state 更新被跳過**：

- `executor.execute(ACTION_SHORT, state)` 當 `state.position == 1` (多倉) 時：
  1. executor 內部呼叫 `_close_position(state)` → 回傳平倉結果
  2. 如果平倉失敗 return None → 正確
  3. 如果平倉成功 → **沒有更新 state** → 直接進入 `_open_position`
  4. `_open_position` return 開倉結果
  5. bot 的 `_handle_trade_result` 只收到開倉結果 → **state 只更新開倉，跳過平倉的 PnL 結算**

**影響**：
- `state.balance` 沒有加上平倉的 PnL
- `state.daily_pnl` / `state.total_pnl` 缺少這筆平倉
- `state.consecutive_losses` 不正確
- `state.trade_count` 少算一筆

> [!CAUTION]
> **反向開倉時，平倉的盈虧未被記錄到 state**。雖然交易所實際完成了平倉+開倉，但 bot 的統計和風控數據不正確。這可能導致風控閾值的計算出錯。

**修正建議**：executor 的反向開倉應回傳兩個結果（平倉+開倉），或者在 executor 內部自己呼叫 `state.close_position()`。

---

### S4. 平倉 IOC 失敗後 fallback 市價單的滑點未保護

[executor.py L299-316](file:///Users/jim_hsieh/Documents/GitHub/PPO_TradingModel/live_trading/executor.py#L299-L316)：

```python
# 平倉 IOC 失敗 → fallback 到市價單
order_result = self.client.place_market_order(
    self.symbol, close_side, remaining
)
```

開倉用 IOC 限價單有滑點保護，但**平倉 fallback 到市價單沒有滑點保護**。在極端行情下（流動性枯竭），市價單可能以非常差的價格成交。

**影響**：低概率但高衝擊。在正常行情下不會觸發（IOC 平倉幾乎 100% 成功）。

> [!WARNING]
> 建議至少 log WARNING 並記錄市價單的 slippage 實際值，以便事後審計。

---

## 🟠 嚴重等級 A — 邏輯問題，需在程式碼中確認或修正

### A1. `on_bar_close` 在 WebSocket callback 執行緒中直接執行交易

[data_feed.py L350-354](file:///Users/jim_hsieh/Documents/GitHub/PPO_TradingModel/live_trading/data_feed.py#L350-L354)：

```python
if self.on_bar_close:
    try:
        self.on_bar_close(self)  # ← 在 WebSocket 執行緒中直接呼叫
```

`on_bar_close` 回調直接在 WebSocket 接收執行緒中執行。如果 `_on_bar_close` 中的交易邏輯耗時超過 1 分鐘（模型推論 + API 下單 + 止損驗證），**會阻塞 WebSocket 消息接收**，可能導致：
- 錯過下一根 K 線
- WebSocket ping/pong timeout → 斷線

**影響**：正常情況下 `_on_bar_close` 耗時 < 1 秒，問題不大。但如果 Binance API 慢或網路卡頓，可能卡住。

> [!NOTE]
> 當前設計可接受（PPO 推論 + 1-2 次 API call 通常 < 500ms）。但建議長期考慮用 queue 解耦。

---

### A2. 健康檢查的 `sync_from_exchange` 可能未同步止損單狀態

[bot.py L425-437](file:///Users/jim_hsieh/Documents/GitHub/PPO_TradingModel/live_trading/bot.py#L425-L437)：

健康檢查每 5 分鐘呼叫 `state.sync_from_exchange(pos_data, balance)`。這會同步持倉方向和數量，但**不會同步止損單狀態**。

如果止損單在交易所被觸發（STOP_MARKET 被市場價格觸發），但 bot 沒有收到通知：
- `state.position` 被 sync 歸零 ✅
- `state.balance` 更新 ✅
- 但 `state.close_position()` 沒有被呼叫 → **PnL / trade_count / consecutive_losses 不更新**

**影響**：風控統計不準確。止損觸發後的連續虧損計數可能被遺漏。

> [!IMPORTANT]
> 建議在 `sync_from_exchange` 偵測到 desync 時（交易所無倉、本地有倉），主動呼叫 `close_position` 進行結算。

---

### A3. 測試套件中 executor 測試使用舊版 API（`place_market_order` 而非 `place_limit_ioc`）

[test_executor_dry.py L211](file:///Users/jim_hsieh/Documents/GitHub/PPO_TradingModel/live_trading/tests/test_executor_dry.py#L211)：

```python
client.place_market_order.assert_called_once()  # ← 但實際改成了 place_limit_ioc
```

Executor 已改用 `place_limit_ioc`（IOC 限價單），但測試仍然 mock `place_market_order`。測試之所以還能 pass，是因為 mock 的 `place_limit_ioc` 回傳 MagicMock（not None），executor 程式碼沒有嚴格檢查 status。

> [!WARNING]
> 測試沒有真正驗證 IOC 限價單路徑。建議更新 mock 設定以正確測試 `place_limit_ioc` 流程。

---

### A4. config_live.yaml 註解與實際值不一致

[config_live.yaml L50-51](file:///Users/jim_hsieh/Documents/GitHub/PPO_TradingModel/live_trading/config_live.yaml#L50-L51)：

```yaml
max_order_value_usdt: 100.0   # 單筆最大下單金額（141U × 20% ≈ 28U，留餘裕）
min_balance_to_trade: 50.0    # 餘額低於 20U 停止交易  ← 註解寫 20U 但實際值是 50U
```

**修正建議**：將註解更新為正確的值。

---

## 🟡 嚴重等級 B — 可改進項目

### B1. NaN 檢查的 fallback 策略可能隱藏問題

[feature_engine.py L121-122](file:///Users/jim_hsieh/Documents/GitHub/PPO_TradingModel/live_trading/feature_engine.py#L121-L122)：

```python
if np.any(np.isnan(features)):
    features = np.nan_to_num(features, nan=0.0)
```

NaN 被靜默替換為 0。雖然這比讓 NaN 進入模型好，但**用 0 替換的特徵可能嚴重偏離正常分布**（例如 ATR 不可能為 0）。模型在訓練時從未見過「所有特徵都是 0」的向量。

> [!NOTE]
> 建議：NaN 出現超過 3 個特徵時，跳過該 bar 不交易，並發送 Telegram 警報。

---

### B2. `_check_config_reload` 缺少新增欄位的安全性檢查

[bot.py L602-606](file:///Users/jim_hsieh/Documents/GitHub/PPO_TradingModel/live_trading/bot.py#L602-L606)：

```python
rm.max_daily_loss_pct = new_risk["max_daily_loss_pct"]
rm.max_total_loss_pct = new_risk["max_total_loss_pct"]
```

如果 config 被修改但缺少某個 key，會拋出 `KeyError`。雖然外層有 `try-except`，但這會導致參數只更新了一部分。

---

### B3. Notifier rate limiting 會阻塞交易邏輯

[notifier.py L184-185](file:///Users/jim_hsieh/Documents/GitHub/PPO_TradingModel/live_trading/notifier.py#L184-L185)：

```python
if now - self._last_send_time < self._min_interval:
    time.sleep(self._min_interval - (now - self._last_send_time))
```

在 rate limiting 時使用 `time.sleep()` 阻塞。這個 sleep 發生在 `_on_bar_close` 中（因為 heartbeat 和 trade notification 都在這裡發送），會阻塞 WebSocket 回調線程。

---

## 📊 訓練 vs 實盤參數完整對照表

| 參數 | 訓練值 | 實盤值 | 匹配？ |
|------|--------|--------|--------|
| `leverage` | 1 | 1 | ✅ |
| `position_size_pct` | 0.2 | 0.20 | ✅ |
| `stop_loss_pct` | 0.02 | 0.02 | ✅ |
| `atr_stop_multiplier` | 2.0 | 2.0 | ✅ |
| `trailing_stop` | false | 未設定（行為 = false） | ✅ |
| `slippage` | 0.0005 | 0.0 | ✅ 有利 |
| `taker_fee` | 0.0004 | 0.0004 | ✅ |
| `max_holding_steps` | 120 | 120 | ✅ |
| `episode_length` | 720 | 720（deque maxlen） | ✅ |
| `dynamic_atr_stop` | false | false | ✅ |
| `use_lstm` | false | false | ✅ |
| `deterministic` | — | true | ✅ |
| `equity_change_pct` 計算 | episode 累計 | 滾動窗口 | ⚠️ S1 |
| `反向開倉 state 更新` | — | 平倉 PnL 丟失 | ⚠️ S3 |

---

## ✅ 確認正確的設計

| 項目 | 驗證結果 |
|------|---------|
| 特徵維度 28 → 28 | ✅ `FeatureEngine` 啟動時驗證 |
| 觀察向量 33 維 | ✅ `build_observation()` = 28 + 5 |
| Action 0/1/2/3 對應 | ✅ 與 `TradingEnv._execute_action()` 完全一致 |
| 模型 MD5 校驗 | ✅ 啟動時驗證，防誤覆蓋 |
| IOC 限價單滑點保護（開倉） | ✅ 比市價單更安全 |
| 止損用實際成交均價 | ✅ `avgPrice` 而非下單前市價 |
| STOP_MARKET 止損用 MARK_PRICE | ✅ 防止插針 |
| 止損單驗證 + 緊急平倉 | ✅ 驗證失敗立即平倉 |
| K 線 timestamp 嚴格遞增 | ✅ 防重複觸發 |
| 去重（_last_processed_ts） | ✅ 同一根 K 線不重複處理 |
| 狀態快照原子寫入 | ✅ temp file + rename |
| Kill Switch (STOP 檔案) | ✅ 安全設計 |
| API 斷路器 (429/5xx) | ✅ 合理設計 |
| PID 鎖檔（防多實例） | ✅ |
| Graceful Shutdown + SL 保護 | ✅ |
| WebSocket 斷線自動重連 + 缺口補填 | ✅ |
| 每日自動重置 daily_pnl | ✅ UTC 0:00 |
| 訓練特徵配置從 model config 載入 | ✅ `_load_training_feature_config()` |
| Config 熱重載（風控參數） | ✅ 非模型參數可動態調整 |

---

## 📋 修正建議清單（按優先級）

### 🚨 建議儘快修正（影響統計和風控正確性）

1. **S3: 反向開倉的平倉 PnL 未結算**
   - Executor 反向開倉時回傳兩個結果（或在內部結算平倉）
   - 影響：`daily_pnl`, `consecutive_losses`, `trade_count` 可能不正確

2. **A2: sync_from_exchange 偵測到 desync 時主動結算**
   - 止損觸發後 bot 未收到通知 → 需補上 PnL 結算
   - 影響：風控計數器準確性

3. **A3: 修正 executor 測試的 mock 設定**
   - 改用 `place_limit_ioc` mock，覆蓋 IOC → FILLED / EXPIRED / PARTIALLY_FILLED 路徑

### 🟠 建議修正（改善正確性）

4. **S1: 監控 equity_change_pct 偏移**
   - 短期可加 logging 觀測實際偏離程度
   - 長期可考慮定期（如每 12 小時）「soft reset」baseline

5. **A4: config 註解修正**（`min_balance_to_trade` 註解寫 20U 但值是 50U）

6. **B3: Notifier rate limiting 改為非阻塞**（drop 或 queue）

### 🟡 建議但非阻擋項

7. NaN fallback 策略改進（超過 3 個 NaN 時跳過不交易）
8. config 熱重載加入 `.get(key, fallback)` 防 KeyError
9. 長期考慮 WebSocket 回調與交易邏輯用 queue 解耦

---

## 💡 針對當前帳戶狀態的建議

當前狀態：`balance=141.29U`，`testnet=false`，`step_count=11`（剛啟動不久）

```yaml
# 你剛修改的值：
max_order_value_usdt: 100.0     # 141U × 20% ≈ 28U，100U 上限足夠
min_balance_to_trade: 50.0      # 合理，但建議調到 30U（給更多操作空間）
```

- `141U × 0.20 = 28.2U` 的名義值，遠大於 Binance 的 `minNotional`（~5U），下單不會被拒 ✅
- `max_order_value_usdt=100U` 遠大於 28U，永遠不會觸發調降 ✅
- 建議保持較保守的風控參數，至少觀察 24-48 小時後再考慮放寬
