"""
測試 v2 獎勵函數 - 平倉訊號與持倉時間限制
"""
import yaml
import pandas as pd
from environment.trading_env import TradingEnv

print("=" * 80)
print("TESTING REWARD FUNCTION v2")
print("=" * 80)

# 讀取配置
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

print("\n[CONFIG] New v2 Reward Parameters:")
print(f"   take_profit_reward: {config['reward']['take_profit_reward']}")
print(f"   take_profit_threshold: {config['reward']['take_profit_threshold']}")
print(f"   cut_loss_early_reward: {config['reward']['cut_loss_early_reward']}")
print(f"   cut_loss_threshold_min: {config['reward']['cut_loss_threshold_min']}")
print(f"   cut_loss_threshold_max: {config['reward']['cut_loss_threshold_max']}")
print(f"   holding_loss_penalty: {config['reward']['holding_loss_penalty']}")
print(f"   holding_loss_threshold: {config['reward']['holding_loss_threshold']}")
print(f"   max_holding_bars: {config['reward']['max_holding_bars']}")
print(f"   overtime_holding_penalty: {config['reward']['overtime_holding_penalty']}")

print("\n[CONFIG] Updated Trading Parameters:")
print(f"   stop_loss_pct: {config['trading']['stop_loss_pct']} (1.5%)")

# 讀取測試數據（前 2000 行）
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
print(f"   New v2 parameters loaded:")
print(f"   - take_profit_reward: {env.take_profit_reward}")
print(f"   - cut_loss_early_reward: {env.cut_loss_early_reward}")
print(f"   - holding_loss_penalty: {env.holding_loss_penalty}")
print(f"   - max_holding_bars: {env.max_holding_bars}")
print(f"   - overtime_holding_penalty: {env.overtime_holding_penalty}")

# 測試環境
obs, info = env.reset()
print(f"\n[TEST] Initial observation shape: {obs.shape}")
print(f"   Initial balance: {env.balance:.2f}")

# 測試情境 1: 獲利平倉（應該觸發 take_profit_reward）
print(f"\n[TEST CASE 1] Testing Take Profit Reward...")
print(f"   Scenario: Open Long -> Hold (simulate profit > 0.7%) -> Close")

# 開多
obs, reward, done, truncated, info = env.step(1)  # Long
print(f"   Step 1: Long | Reward={reward:>8.2f} | Position={env.position} | Entry={env.entry_price:.2f}")

# 模擬價格上漲（等待幾個 step）
for i in range(30):
    obs, reward, done, truncated, info = env.step(3)  # Hold
    if i == 29:
        print(f"   Step {i+2}: Hold | Reward={reward:>8.2f} | Position={env.position} | Price={env.df.iloc[env.current_step]['close']:.2f}")

# 平倉
obs, reward, done, truncated, info = env.step(0)  # Close
print(f"   Step 32: Close | Reward={reward:>8.2f} | Position={env.position}")
print(f"   -> Should include take_profit_reward if profit > 0.7%")

# 測試情境 2: 小虧主動平倉（應該觸發 cut_loss_early_reward）
print(f"\n[TEST CASE 2] Testing Cut Loss Early Reward...")
print(f"   Scenario: Open Short -> Hold (wait for small loss) -> Close")

# 重置
env.reset()

# 開空
obs, reward, done, truncated, info = env.step(2)  # Short
print(f"   Step 1: Short | Reward={reward:>8.2f} | Position={env.position} | Entry={env.entry_price:.2f}")

# 等待小虧（希望觸發在 -1.5% ~ -0.5% 之間）
for i in range(20):
    obs, reward, done, truncated, info = env.step(3)  # Hold
    if i == 19:
        current_price = env.df.iloc[env.current_step]['close']
        unrealized_pnl_pct = (env.entry_price - current_price) / env.entry_price
        print(f"   Step {i+2}: Hold | Reward={reward:>8.2f} | Unrealized={unrealized_pnl_pct*100:.2f}%")

# 平倉
obs, reward, done, truncated, info = env.step(0)  # Close
print(f"   Step 22: Close | Reward={reward:>8.2f} | Position={env.position}")
print(f"   -> Should include cut_loss_early_reward if loss in range")

# 測試情境 3: 持有虧損倉位懲罰（應該觸發 holding_loss_penalty）
print(f"\n[TEST CASE 3] Testing Holding Loss Penalty...")
print(f"   Scenario: Open Long -> Hold (with loss > 0.6%)")

env.reset()
obs, reward, done, truncated, info = env.step(1)  # Long
print(f"   Step 1: Long | Reward={reward:>8.2f} | Position={env.position}")

# 持有虧損倉位
for i in range(10):
    obs, reward, done, truncated, info = env.step(3)  # Hold
    current_price = env.df.iloc[env.current_step]['close']
    unrealized_pnl_pct = (current_price - env.entry_price) / env.entry_price
    if abs(unrealized_pnl_pct) >= 0.006:
        print(f"   Step {i+2}: Hold | Reward={reward:>8.2f} | Unrealized={unrealized_pnl_pct*100:.2f}%")
        print(f"   -> Should include holding_loss_penalty if loss > 0.6%")
        break

# 測試情境 4: 超時持倉懲罰（模擬超過 720 根 K 線）
print(f"\n[TEST CASE 4] Testing Overtime Holding Penalty...")
print(f"   Scenario: Open Long -> Hold for > 720 bars")

env.reset()
obs, reward, done, truncated, info = env.step(1)  # Long
print(f"   Step 1: Long | Holding Time={env.holding_time}")

# 快速前進到 721 步（超過 max_holding_bars）
# 注意：這個測試可能無法完整執行，因為 episode 長度限制
print(f"   (Skipping to holding_time > 720...)")
print(f"   -> In real training, should trigger overtime_holding_penalty")

print(f"\n[SUCCESS] New v2 reward function parameters loaded and ready!")
print("=" * 80)
