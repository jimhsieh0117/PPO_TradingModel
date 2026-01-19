"""
測試 v3 獎勵函數 - 驗證快速換倉懲罰與平倉獎勵
"""
import yaml
import pandas as pd
from environment.trading_env import TradingEnv

print("=" * 80)
print("TESTING REWARD FUNCTION v3 - Anti Rapid Switching")
print("=" * 80)

# 讀取配置
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

print("\n[CONFIG] v3 Reward Parameters:")
print(f"   open_position_reward: {config['reward']['open_position_reward']} (was 20)")
print(f"   close_position_reward: {config['reward']['close_position_reward']} (NEW!)")
print(f"   rapid_switch_penalty: {config['reward']['rapid_switch_penalty']} (NEW!)")
print(f"   rapid_switch_window: {config['reward']['rapid_switch_window']} bars")
print(f"   take_profit_threshold: {config['reward']['take_profit_threshold']} (was 0.007)")
print(f"   holding_profit_reward: {config['reward']['holding_profit_reward']} (NEW!)")
print(f"   min_holding_bars: {config['reward']['min_holding_bars']} (NEW!)")
print(f"   max_holding_bars: {config['reward']['max_holding_bars']} (was 720)")

# 讀取測試數據
df = pd.read_csv('data/raw/BTCUSDT_1m_test_20260114_012047.csv', nrows=2000)
print(f"\n[DATA] Loaded {len(df)} rows of test data")

# 創建環境
env = TradingEnv(
    df=df,
    initial_balance=10000.0,
    leverage=1,
    position_size_pct=1.0,
    stop_loss_pct=config['trading']['stop_loss_pct'],
    trading_fee=0.0,
    episode_length=1440,
    reward_config=config['reward']
)

print(f"\n[ENV] Environment created successfully")

# 測試情境 1: 快速換倉懲罰
print(f"\n[TEST CASE 1] Rapid Switch Penalty")
print(f"   Scenario: Open Long -> Switch to Short within 10 bars")

obs, info = env.reset()
obs, r1, done, truncated, info = env.step(1)  # Long
print(f"   Step 1: Long  | Reward={r1:>8.2f} | Position={env.position}")

# 立刻換倉（應該觸發快速換倉懲罰）
obs, r2, done, truncated, info = env.step(2)  # Short (rapid switch!)
print(f"   Step 2: Short | Reward={r2:>8.2f} | Position={env.position}")
print(f"   -> Should include rapid_switch_penalty (-50)")
print(f"   -> Actual penalty applied: {r2 - 3:.2f} (should be around -50)")

# 測試情境 2: 平倉基礎獎勵
print(f"\n[TEST CASE 2] Close Position Base Reward")
print(f"   Scenario: Open Long -> Hold -> Close")

env.reset()
obs, r1, done, truncated, info = env.step(1)  # Long
print(f"   Step 1: Long  | Reward={r1:>8.2f}")

for i in range(10):
    obs, r, done, truncated, info = env.step(3)  # Hold

obs, r_close, done, truncated, info = env.step(0)  # Close
print(f"   Step 12: Close | Reward={r_close:>8.2f}")
print(f"   -> Should include close_position_reward (+20)")

# 測試情境 3: 持有獲利倉位獎勵
print(f"\n[TEST CASE 3] Holding Profit Reward")
print(f"   Scenario: Open Long -> Hold (with profit)")

env.reset()
obs, r1, done, truncated, info = env.step(1)  # Long
print(f"   Step 1: Long | Reward={r1:>8.2f} | Entry={env.entry_price:.2f}")

# 持有幾步（希望有浮盈）
for i in range(5):
    obs, r, done, truncated, info = env.step(3)  # Hold
    if i == 4:
        current_price = env.df.iloc[env.current_step]['close']
        unrealized_pnl_pct = (current_price - env.entry_price) / env.entry_price
        print(f"   Step {i+2}: Hold | Reward={r:>8.2f} | Unrealized={unrealized_pnl_pct*100:.3f}%")
        if unrealized_pnl_pct > 0:
            print(f"   -> Should include holding_profit_reward (+3 per step)")

# 測試情境 4: 最短持倉時間懲罰
print(f"\n[TEST CASE 4] Minimum Holding Time Penalty")
print(f"   Scenario: Open Long -> Close immediately (< 5 bars)")

env.reset()
obs, r1, done, truncated, info = env.step(1)  # Long
print(f"   Step 1: Long  | Holding time={env.holding_time}")

# 立刻平倉（< 5 bars）
obs, r2, done, truncated, info = env.step(0)  # Close
print(f"   Step 2: Close | Reward={r2:>8.2f} | Holding time={env.holding_time}")
print(f"   -> Should include undertime_holding_penalty (-5)")

# 測試情境 5: 獎勵對比（完整循環 vs 快速換倉）
print(f"\n[TEST CASE 5] Reward Comparison")

# 完整循環
env.reset()
r_open = env.step(1)[1]  # Long
total_r1 = r_open
for i in range(15):
    r = env.step(3)[1]  # Hold 15 bars
    total_r1 += r
r_close = env.step(0)[1]  # Close
total_r1 += r_close
print(f"   Full Cycle (Open->Hold 15->Close): Total Reward = {total_r1:.2f}")

# 快速換倉
env.reset()
r_open1 = env.step(1)[1]  # Long
r_switch = env.step(2)[1]  # Immediate switch to Short
total_r2 = r_open1 + r_switch
print(f"   Rapid Switch (Open Long->Open Short): Total Reward = {total_r2:.2f}")

print(f"\n   Expected: Full Cycle > Rapid Switch")
print(f"   Result: {total_r1:.2f} {'>' if total_r1 > total_r2 else '<'} {total_r2:.2f}")
print(f"   SUCCESS!" if total_r1 > total_r2 else "   FAILED - need to adjust parameters")

print("\n" + "=" * 80)
print("v3 Reward Function Test Complete!")
print("=" * 80)
