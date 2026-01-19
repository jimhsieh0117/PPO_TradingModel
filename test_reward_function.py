"""
測試新獎勵函數
"""
import yaml
import pandas as pd
from environment.trading_env import TradingEnv

print("=" * 80)
print("TESTING NEW REWARD FUNCTION")
print("=" * 80)

# 讀取配置
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

print("\n[CONFIG] Reward Parameters:")
for key, value in config['reward'].items():
    print(f"   {key}: {value}")

# 讀取測試數據（前 2000 行）
df = pd.read_csv('data/raw/BTCUSDT_1m_test_20260114_012047.csv', nrows=2000)

print(f"\n[DATA] Loaded {len(df)} rows of test data")

# 創建環境
env = TradingEnv(
    df=df,
    initial_balance=10000.0,
    leverage=1,
    position_size_pct=1.0,
    stop_loss_pct=0.03,
    trading_fee=0.0,
    episode_length=1440,
    reward_config=config['reward']
)

print(f"\n[ENV] Environment created successfully")
print(f"   profit_multiplier: {env.profit_multiplier}")
print(f"   loss_multiplier: {env.loss_multiplier}")
print(f"   open_position_reward: {env.open_position_reward}")
print(f"   profitable_open_bonus: {env.profitable_open_bonus}")
print(f"   trading_fee_multiplier: {env.trading_fee_multiplier}")
print(f"   no_position_window: {env.no_position_window}")
print(f"   no_position_penalty: {env.no_position_penalty}")

# 測試環境
obs, info = env.reset()
print(f"\n[TEST] Initial observation shape: {obs.shape}")
print(f"   Initial balance: {env.balance:.2f}")

# 測試幾個動作
print(f"\n[TEST] Testing actions...")
actions = [3, 1, 3, 3, 0, 3, 2, 3, 0]  # Hold, Long, Hold, Hold, Close, Hold, Short, Hold, Close
for i, action in enumerate(actions):
    obs, reward, done, truncated, info = env.step(action)
    action_names = ['Close', 'Long', 'Short', 'Hold']
    print(f"   Step {i+1}: Action={action_names[action]:<5} | Reward={reward:>8.2f} | Position={env.position:>2} | Equity={env.equity:>10.2f}")
    if done or truncated:
        break

print(f"\n[SUCCESS] New reward function is working!")
print("=" * 80)
