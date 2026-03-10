# 狀態空間特徵規格（31 維 = 26 市場特徵 + 5 持倉狀態）

## 1. 市場結構 (Market Structure) - 3 個
- `trend_state`: 趨勢方向
- `structure_signal`: BOS/ChoCh 結構信號
- `bars_since_structure_change`: 距上次結構變化的 K 線數

## 2. Order Blocks - 4 個
- `dist_to_bullish_ob`: 距多頭 OB 距離（ATR 單位，哨兵值 50.0）
- `dist_to_bearish_ob`: 距空頭 OB 距離（ATR 單位，哨兵值 50.0）
- `in_bullish_ob`: 是否在多頭 OB 內
- `in_bearish_ob`: 是否在空頭 OB 內

## 3. Fair Value Gaps - 3 個
- `in_bullish_fvg`: 是否在多頭 FVG 內
- `in_bearish_fvg`: 是否在空頭 FVG 內
- `nearest_fvg_direction`: 最近 FVG 方向

## 4. Liquidity - 3 個
- `liquidity_above`: 上方流動性距離（ATR 單位，哨兵值 50.0）
- `liquidity_below`: 下方流動性距離（ATR 單位，哨兵值 50.0）
- `liquidity_sweep`: 流動性掃蕩信號

## 5. Volume & Price - 5 個
- `volume_ratio`: 成交量比率
- `price_momentum`: 價格動量
- `vwap_momentum`: VWAP 動量
- `price_position_in_range`: 價格在區間中的位置 [0, 1]
- `zone_classification`: 區域分類

## 6. Multi-Timeframe - 2 個
- `trend_5m`: 5 分鐘趨勢
- `trend_15m`: 15 分鐘趨勢

## 7. 波動率 (v0.6) - 1 個
- `atr_normalized`: ATR / close，當前相對波動率水平

## 8. 時間特徵 (v0.6) - 2 個
- `hour_sin`: sin(2pi * hour/24)，捕捉亞洲/歐美時段週期
- `hour_cos`: cos(2pi * hour/24)

## 9. 市場 Regime (v0.8) - 3 個
- `adx_normalized`: ADX(14) / 100，趨勢強度 [0, 1]（>0.4 = 強趨勢，<0.2 = 盤整）
- `volatility_regime`: ATR 在過去 480 根 K 線中的相對位置 [0, 1]（rolling min/max 正規化）
- `trend_strength`: (close - EMA200) / ATR，裁切到 [-1, 1]（正 = 多頭趨勢，負 = 空頭趨勢）

## 10. 持倉狀態 - 5 個
- `position_state`: 持倉方向 {-1, 0, 1}
- `floating_pnl_pct`: 浮動盈虧百分比
- `holding_time_norm`: 持倉時間正規化 (0~1, 120 步飽和)
- `distance_to_stop_loss`: 距止損距離 (0~1)
- `equity_change_pct`: 滾動 480 步窗口權益變化

**總計：31 個特徵（26 市場 + 5 持倉）**

---

## 特徵計算模組

| 模組 | 檔案 | 產生特徵 |
|------|------|---------|
| Market Structure | `environment/features/market_structure.py` | trend_state, structure_signal, bars_since |
| Order Blocks | `environment/features/order_blocks.py` | dist_to_ob (x2), in_ob (x2) |
| Fair Value Gaps | `environment/features/fvg.py` | in_fvg (x2), nearest_fvg_direction |
| Liquidity | `environment/features/liquidity.py` | liquidity_above/below, sweep |
| Volume & Price | `environment/features/volume.py` | volume_ratio, momentum (x2), ATR, regime |
| Aggregator | `environment/features/feature_aggregator.py` | 組裝 26 維市場特徵 |

**計算順序**：Volume（產生 ATR）→ Market Structure → Order Blocks → FVG → Liquidity
