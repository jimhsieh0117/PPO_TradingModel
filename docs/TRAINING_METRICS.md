# 訓練過程監控指標

為了全面監控模型收斂情況並找出改進方向，記錄以下所有指標。

## 1. 獎勵指標（Reward Metrics）
- `episode_reward_mean`: 平均每個 episode 的總獎勵
- `episode_reward_std`: 獎勵標準差（評估穩定性）
- `episode_reward_max`: 最大單集獎勵
- `episode_reward_min`: 最小單集獎勵
- `cumulative_reward`: 累積總獎勵

**用途**：判斷模型是否在學習獲利策略

## 2. 損失函數（Loss Metrics）
- `policy_loss`: 策略網絡損失（越小越好）
- `value_loss`: 價值網絡損失（越小越好）
- `entropy_loss`: 熵損失（探索程度）
- `total_loss`: 總損失

**用途**：監控神經網絡是否過擬合或學習停滯

## 3. PPO 特定指標（PPO-Specific）
- `clip_fraction`: 被裁剪的比例（0.1-0.3 正常）
- `approx_kl`: 近似 KL 散度（< 0.05 正常）
- `explained_variance`: 解釋方差（越接近 1 越好）
- `learning_rate`: 當前學習率（如果使用學習率調度）

**用途**：判斷 PPO 更新是否過大/過小，是否需要調整超參數

## 4. 交易行為指標（Trading Behavior）
- `total_trades_per_episode`: 每個 episode 的交易次數
- `long_ratio`: 做多動作比例
- `short_ratio`: 做空動作比例
- `hold_ratio`: 持有動作比例
- `close_ratio`: 平倉動作比例
- `avg_holding_time`: 平均持倉時間（K 線數）

**用途**：判斷模型是否過度交易或過於保守

## 5. 盈利與風險指標（Profit & Risk）
- `episode_profit`: 每個 episode 的淨利潤（USDT）
- `episode_return_pct`: 每個 episode 的報酬率（%）
- `win_rate`: 勝率（獲利交易 / 總交易）
- `profit_factor`: 盈虧比（總獲利 / 總虧損）
- `sharpe_ratio`: 夏普比率（滾動計算）
- `max_drawdown`: 最大回撤（%）
- `stop_loss_count`: 止損次數
- `daily_drawdown_violations`: 單日回撤超過 10% 的次數

**用途**：評估模型的實際盈利能力和風險控制

## 6. Episode 統計（Episode Stats）
- `episode_length`: 每個 episode 的步數
- `episode_completion_rate`: episode 正常結束比例（非提前終止）
- `avg_equity_curve_slope`: 權益曲線斜率（趨勢）

**用途**：判斷訓練是否穩定

## 7. 探索 vs 利用（Exploration vs Exploitation）
- `action_entropy`: 動作選擇的熵（高 = 探索多）
- `action_distribution`: 各動作的選擇頻率分布

**用途**：確保模型不會過早收斂到次優策略

---

## 視覺化輸出（Training Plots）

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

每張圖都包含滾動平均線（smoothing）以便觀察趨勢。

---

## 回測評估指標（Backtesting Metrics）

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
   - 止損次數 (Stop Loss Count)

4. **其他**
   - 平均持倉時間
   - 最大連續虧損次數
   - 最大連續獲利次數
