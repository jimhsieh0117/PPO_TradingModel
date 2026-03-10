# 版本記錄

## v0.9 (2026-03-07): 多幣種支援 + 訓練/回測環境對齊 + 數據下載修復

### 訓練與回測環境對齊（五大修復）
- 止損觸發改用盤中高低點（原為收盤價），以止損價成交
- PnL 計算改用開倉時鎖定的 `position_size`（原每步用當前 balance 重算）
- `equity_change_pct` 改為滾動 480 步窗口（原從 episode 開始累計，回測 OOD 問題）
- `max_holding_steps` 同步至回測策略強制平倉
- `trailing_stop=true` 時回測透過 `trade.sl` 動態更新追蹤止損

### 多幣種回測結果（測試期 2025-01-01 ~ 2026-02-14，初始資金 1,000,000 USDT）
| 幣種 | 總報酬 | Sharpe | 最大回撤 | 勝率 | PF | 每日交易 |
|------|--------|--------|---------|------|-----|---------|
| BTCUSDT | +23.7% | 2.96 | -9.0% | 38.6% | 1.29 | 8.4 |
| ETHUSDT | +332,522%* | 1.34 | -10.0% | 60.9% | 2.31 | 15.9 |
| SOLUSDT | +164.6% | 2.26 | -21.7% | 34.0% | 1.17 | 41.4 |
| WIFUSDT | +177,705%* | 1.65 | -6.3% | 40.4% | 1.72 | 71.5 |

> *報酬異常說明：`position_size_pct=0.99` 在 backtesting.py 為複利模式（每筆用當前資金比例），高頻交易 + 正期望值 → 指數複利爆炸。數字數學正確，但受流動性限制現實不可行。**BTCUSDT +23.7% 為最可信參考**；其他幣種以 Sharpe / PF / 勝率 / 最大回撤為評估依據。

### 數據下載修復
- `download_data.py` 新增 `find_earliest_available_date()`：在批量下載前先以單筆 API 探測最早有效時間戳
- 若 `start_date` 早於代幣上市日（如 WIFUSDT 2023 年上市），自動調整起點並印出警告
- `data_pipeline._download_and_merge()` 新增空 DataFrame 防護，跳過無數據區間並在全空時拋出明確錯誤

### 其他
- `trading.symbol` 移除：symbol 統一由 `data.symbol` 管理
- backtesting.py 完整統計在摘要前輸出
- 手續費模型確認：開平倉各收一次 `taker_fee`，另加 `slippage`
- Random seed 只影響訓練（權重初始化、PPO 探索採樣）；推論時 `deterministic=True`，與 seed 無關

---

## v0.8 (2026-02-20): 市場 Regime 特徵 + LSTM 支援 + Bug 修復
- 新增 3 個市場 Regime 特徵：`adx_normalized`、`volatility_regime`、`trend_strength`
- 觀察空間更新：28 維 → 31 維（26 市場特徵 + 5 持倉狀態）
- LSTM/RecurrentPPO 支援：config `lstm.enabled` 切換 MLP/LSTM，train.py + wfa.py + strategy.py 同步
- 新增 `sb3-contrib` 依賴（RecurrentPPO）
- 空倉機會成本懲罰機制（`idle_penalty_*` 參數，可選啟用）
- 低波動持倉獎勵機制（`low_vol_hold_bonus`，可選啟用）
- **修復 config key 不匹配 bug**：`feature_aggregator.py` 讀 `market_structure_lookback` 但 config 定義為 `structure_lookback`，導致所有 lookback 設定被靜默忽略、永遠使用默認值 50/20
- **修復 policy_kwargs 未傳遞 bug**：train.py/wfa.py 從未將 `policy_network`/`value_network` 傳給 PPO()，所有舊模型用 SB3 默認 [64, 64]
- 效能優化：`volatility_regime` 計算從 O(n × window) rolling rank 改為 O(n) rolling min/max
- ⚠️ 所有舊模型不相容（觀察空間維度變更 + 特徵計算修正），需重新訓練

---

## v0.7 (2026-02-17): 修復特徵正規化縮放問題（ATR 正規化取代價格百分比）
- 修復距離特徵價格相依問題：`/ current_price * 100` → `/ ATR`
- 影響特徵：`dist_to_bullish_ob`, `dist_to_bearish_ob`, `liquidity_above`, `liquidity_below`
- 問題根因：相同 500 點距離在 $25K 顯示為 2%，在 $95K 顯示為 0.53%，導致 4x 失真
- FVG 檢測閾值動態化：固定 `min_size_pct` → ATR 相對閾值 `min_size_atr=0.3`
- 哨兵值更新：OB 距離 10.0→50.0，流動性距離 5.0→50.0（以 ATR 為單位）
- FeatureAggregator 計算順序調整：Volume 移至第 1 步（產生 ATR），再傳給 OB/FVG/Liquidity
- 清除舊快取（data/processed/ + data/cache/），強制重算特徵
- ⚠️ 舊模型與新特徵尺度不相容，需重新訓練

---

## v0.6 (2026-02-16): ATR 動態止損 + 波動率/時間特徵
- ATR 動態止損：取代固定 1.5%，使用 2x ATR 倍數適應市場波動
- 追蹤止損：止損價只朝有利方向移動，鎖住已有利潤
- 新增 `atr_normalized` 特徵：讓模型感知當前波動率水平
- 新增 `hour_sin`, `hour_cos` 時間特徵：捕捉亞洲/歐美時段交易模式
- 觀察空間更新：25 維 → 28 維（23 ICT + 5 持倉）
- VolumeAnalyzer 新增 ATR 預計算（原始 + 正規化）
- 環境、回測策略、WFA 全部同步更新
- config.yaml 新增 `atr_stop_multiplier`, `trailing_stop` 參數

---

## v0.5 (2026-02-16): 數據管線重寫（增量下載 + 處理後數據快取）
- 重寫 `utils/data_pipeline.py`：增量下載架構，僅補足缺少的日期範圍
- 新增處理後數據快取：`data/processed/{symbol}_{interval}.parquet`（OHLCV + 20 特徵）
- 快取驗證機制：透過 `data_hash` + `feature_config_hash` 自動判斷是否需要重算
- 新增公開 API：`load_full_data()`, `extract_features()`
- 更新 `train.py`：改用 `extract_features()` 取代 `precompute_features_with_cache()`
- 更新 `wfa.py`：`load_full_dataset()` 委派給 `load_full_data()`，移除舊版直接存取
- 更新 `backtest/run_backtest.py`：移除 `_build_expected_filename` 依賴
- 碎片整合：多個 raw parquet 自動合併為單一檔案，舊碎片自動清理

---

## v0.4 (2026-02-14): v9.0 綜合改進（程式碼深度分析 + 訓練數據診斷）
- 修復 `vf_coef` 2.0→0.5（SB3 默認值，value loss 不再主導梯度）
- 重新啟用 EMA reward normalization（穩定 critic target）
- 放寬 best model 保存門檻（移除 sharpe>1.3 要求）
- 重新平衡 reward：`floating_reward_scale` 120→30
- 移除 `holding_bonus_max` 和 `rapid_reentry_penalty`
- 觀察空間更新至 25 維（新增 5 維持倉狀態）
- 診斷出浮動獎勵:已實現獎勵 = 7:1 失衡問題

---

## v0.3 (2026-02-13): 過擬合分析與最佳實踐
- 更新獎勵函數至 v7.3（pnl_reward_scale=500, floating_reward_scale=150）
- 新增過擬合分析章節（1M vs 2M 訓練對比）
- 確定最佳訓練步數為 ~1M steps
- 記錄手續費對策略的關鍵影響
- 新增 TensorBoard 關鍵指標解讀
- 最佳模型：run_20260213_160140（+3.27% 報酬，Sharpe 5.87）
- 探索 XAUUSDT（黃金永續）可行性：數據量不足（僅 64 天）

---

## v0.2 (2026-01-13): 新增專業級訓練監控系統
- 新增 40+ 訓練監控指標（7 大類）
- 新增 9 張詳細訓練曲線圖
- 確認使用 Binance Futures API 實盤歷史數據
- 明確止損次數追蹤需求

---

## v0.1 (2026-01-13): 初始專案規格文檔
- 確定 ICT 策略框架
- 設定 15% 倉位 + 10x 槓桿
- 選定 Stable-Baselines3 + Gymnasium
