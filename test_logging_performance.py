"""
測試詳細記錄 vs 精簡記錄的效能差異
"""
import yaml
import time

print("=" * 80)
print("訓練記錄效能測試")
print("=" * 80)

# 讀取當前配置
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

current_setting = config['training'].get('enable_detailed_logging', True)

print(f"\n當前設定: enable_detailed_logging = {current_setting}")
print("\n說明：")
print("  - enable_detailed_logging: true  → 記錄 34+ 指標（完整監控）")
print("  - enable_detailed_logging: false → 記錄 7 基本指標（效能優化）")
print("\n建議測試流程：")
print("  1. 設定 enable_detailed_logging: false")
print("  2. 修改 total_timesteps: 100000 (縮短訓練)")
print("  3. 執行 python train.py，記錄訓練時間")
print("  4. 設定 enable_detailed_logging: true")
print("  5. 再次執行 python train.py，比較訓練時間")
print("\n預期結果：")
print("  - 精簡模式應該快 2-5%（主要節省 CSV 寫入和數值計算時間）")
print("  - 100000 steps 訓練約 10-20 分鐘（取決於 CPU）")
print("\n" + "=" * 80)

# 顯示當前 CSV headers 會記錄哪些欄位
if current_setting:
    print("\n[當前模式] 詳細記錄 - 記錄以下欄位：")
    print("  基本欄位 (7):  timestamp, timesteps, episode, episode_reward, episode_profit, episode_return_pct, total_trades_per_episode")
    print("  詳細欄位 (31): episode_reward_mean/std/max/min, cumulative_reward, policy_loss, value_loss, ...")
    print("  總計: 38 個欄位")
else:
    print("\n[當前模式] 精簡記錄 - 僅記錄以下欄位：")
    print("  基本欄位 (7):  timestamp, timesteps, episode, episode_reward, episode_profit, episode_return_pct, total_trades_per_episode")
    print("  總計: 7 個欄位")

print("\n" + "=" * 80)
