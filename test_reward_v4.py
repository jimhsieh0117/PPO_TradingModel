"""
測試 v4 獎勵函數 - 驗證平衡性
"""
import yaml
import pandas as pd
from environment.trading_env import TradingEnv

print("=" * 80)
print("TESTING REWARD FUNCTION v4 - Balanced Version")
print("=" * 80)

# 讀取配置
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

print("\n[CONFIG] v4 Reward Parameters (vs v3):")
print(f"   open_position_reward: {config['reward']['open_position_reward']} (v3: 3)")
print(f"   close_position_reward: {config['reward']['close_position_reward']} (v3: 20)")
print(f"   rapid_switch_penalty: {config['reward']['rapid_switch_penalty']} (v3: -50)")
print(f"   rapid_switch_window: {config['reward']['rapid_switch_window']} bars (v3: 10)")
print(f"   excessive_hold_penalty: {config['reward']['excessive_hold_penalty']} (NEW!)")
print(f"   excessive_hold_window: {config['reward']['excessive_hold_window']} bars (NEW!)")
print(f"   take_profit_threshold: {config['reward']['take_profit_threshold']} (v3: 0.001)")

# 讀取測試數據
df = pd.read_csv('data/raw/BTCUSDT_1m_test_20260114_012047.csv', nrows=2000)
print(f"\n[DATA] Loaded {len(df)} rows")

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

print(f"\n[ENV] v4 Environment created")

# 測試情境 1: 完整循環 vs 快速換倉 vs 過度 Hold
print(f"\n[TEST CASE 1] Strategy Comparison")

# 策略 A: 完整循環（預期最優）
env.reset()
total_r_full = 0
total_r_full += env.step(1)[1]  # Long (+10)
for i in range(10):  # Hold 10 bars
    total_r_full += env.step(3)[1]  # +2/step if profit
total_r_full += env.step(0)[1]  # Close (+15 + profit bonus)
print(f"   A. Full Cycle (Open->Hold 10->Close): {total_r_full:.2f}")

# 策略 B: 快速換倉（應該次優但可行）
env.reset()
total_r_switch = 0
total_r_switch += env.step(1)[1]  # Long (+10)
for i in range(3):  # Hold only 3 bars
    total_r_switch += env.step(3)[1]
total_r_switch += env.step(2)[1]  # Switch to Short (+10, no penalty if >5 bars)
print(f"   B. Quick Switch (Open->Hold 3->Switch): {total_r_switch:.2f}")

# 策略 C: 立即換倉（應該被懲罰，但不會太嚴重）
env.reset()
total_r_rapid = 0
total_r_rapid += env.step(1)[1]  # Long (+10)
total_r_rapid += env.step(2)[1]  # Immediate Switch (+10 -15 penalty)
print(f"   C. Rapid Switch (Long->Short immediate): {total_r_rapid:.2f}")

# 策略 D: 過度 Hold（應該被懲罰）
env.reset()
total_r_hold = 0
total_r_hold += env.step(1)[1]  # Long (+10)
for i in range(25):  # Hold 25 bars (> 20 threshold)
    r = env.step(3)[1]
    total_r_hold += r
    if i == 24:
        print(f"   D. Excessive Hold (Long->Hold 25): {total_r_hold:.2f}")

# 預期排序
print(f"\n   Expected Order: A > B > C > D")
print(f"   Actual: A={total_r_full:.1f} vs B={total_r_switch:.1f} vs C={total_r_rapid:.1f} vs D={total_r_hold:.1f}")

success_count = 0
if total_r_full > total_r_switch:
    print(f"   [OK] Full Cycle > Quick Switch")
    success_count += 1
else:
    print(f"   [FAIL] Full Cycle should > Quick Switch")

if total_r_switch > total_r_rapid:
    print(f"   [OK] Quick Switch > Rapid Switch")
    success_count += 1
else:
    print(f"   [FAIL] Quick Switch should > Rapid Switch")

if total_r_rapid > total_r_hold:
    print(f"   [OK] Rapid Switch > Excessive Hold")
    success_count += 1
else:
    print(f"   [FAIL] Rapid Switch should > Excessive Hold")

print(f"\n   Result: {success_count}/3 tests passed")

# 測試情境 2: 過度 Hold 懲罰觸發
print(f"\n[TEST CASE 2] Excessive Hold Penalty Trigger")
env.reset()
print(f"   Holding without position for 25 steps...")
for i in range(25):
    obs, r, done, truncated, info = env.step(3)  # Hold
    if i == 19:
        print(f"   Step 20: Reward={r:.2f} (should be 0, before penalty)")
    if i == 20:
        print(f"   Step 21: Reward={r:.2f} (should include -1 penalty)")
    if i == 24:
        print(f"   Step 25: Reward={r:.2f} (should continue penalty)")

# 測試情境 3: 快速換倉懲罰（5分鐘內）
print(f"\n[TEST CASE 3] Rapid Switch Penalty (within 5 bars)")
env.reset()
obs, r1, done, truncated, info = env.step(1)  # Long
print(f"   Step 1: Long  | Reward={r1:.2f}")

# 3 分鐘後換倉（< 5 bars，應該觸發懲罰）
for i in range(3):
    env.step(3)
obs, r2, done, truncated, info = env.step(2)  # Short
print(f"   Step 5: Short (3 bars later) | Reward={r2:.2f}")
print(f"   -> Should include -15 penalty (total ~= 10-15 = -5)")

# 測試情境 4: 不觸發快速換倉懲罰（6分鐘後）
print(f"\n[TEST CASE 4] No Rapid Switch Penalty (after 6 bars)")
env.reset()
obs, r1, done, truncated, info = env.step(1)  # Long
print(f"   Step 1: Long  | Reward={r1:.2f}")

# 6 分鐘後換倉（> 5 bars，不應該觸發懲罰）
for i in range(6):
    env.step(3)
obs, r2, done, truncated, info = env.step(2)  # Short
print(f"   Step 8: Short (6 bars later) | Reward={r2:.2f}")
print(f"   -> Should NOT include penalty (total ~= 10)")

print("\n" + "=" * 80)
print("v4 Reward Function Test Complete!")
print("Key Improvements:")
print("- Balanced open/close rewards (10 vs 15)")
print("- Reduced rapid switch penalty (-50 -> -15)")
print("- Added excessive hold penalty (prevents conservatism)")
print("=" * 80)
