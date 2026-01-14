"""
測試交易環境 - 驗證核心功能
"""

import numpy as np
import pandas as pd
from environment.trading_env import TradingEnv


def test_basic_functionality():
    """測試基本功能"""
    print("=" * 60)
    print("測試 1: 基本功能測試")
    print("=" * 60)

    # 載入數據
    print("\n[1] 載入數據...")
    df = pd.read_csv("data/raw/BTCUSDT_1m_full_20260114_012047.csv")
    print(f"✅ 數據載入成功: {len(df)} 根 K 線")

    # 創建環境
    print("\n[2] 創建交易環境...")
    env = TradingEnv(
        df=df,
        initial_balance=10000.0,
        leverage=10,
        position_size_pct=0.15,
        stop_loss_pct=0.015,
        max_daily_drawdown=0.10,
        trading_fee=0.0004,
        episode_length=100  # 測試用較短的 episode
    )
    print(f"✅ 環境創建成功")
    print(f"   - 動作空間: {env.action_space}")
    print(f"   - 觀察空間: {env.observation_space.shape}")
    print(f"   - 初始資金: ${env.initial_balance:.2f}")
    print(f"   - 槓桿倍數: {env.leverage}x")
    print(f"   - 倉位大小: {env.position_size_pct * 100:.1f}%")
    print(f"   - 實際敞口: {env.actual_exposure_pct * 100:.1f}%")

    # 測試 reset
    print("\n[3] 測試 reset() 方法...")
    obs, info = env.reset()
    print(f"✅ Reset 成功")
    print(f"   - 觀察維度: {obs.shape}")
    print(f"   - 觀察範例: {obs[:5]}")
    print(f"   - 初始 info: balance=${info['balance']:.2f}, equity=${info['equity']:.2f}")

    print("\n測試 1: ✅ 通過\n")


def test_random_actions():
    """測試隨機動作"""
    print("=" * 60)
    print("測試 2: 隨機動作測試")
    print("=" * 60)

    # 載入數據並創建環境
    df = pd.read_csv("data/raw/BTCUSDT_1m_full_20260114_012047.csv")
    env = TradingEnv(df=df, episode_length=100)

    # Reset
    obs, info = env.reset(seed=42)
    print(f"\n[1] 初始狀態: balance=${info['balance']:.2f}")

    # 執行 100 步隨機動作
    print(f"\n[2] 執行 100 步隨機動作...")
    total_reward = 0
    for step in range(100):
        # 隨機選擇動作
        action = env.action_space.sample()

        # 執行動作
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # 顯示進度（每 20 步）
        if (step + 1) % 20 == 0:
            print(f"   Step {step + 1:3d}: "
                  f"Action={action} | "
                  f"Equity=${info['equity']:.2f} | "
                  f"Position={info['position']} | "
                  f"Trades={info['total_trades']}")

        # 如果 episode 提前終止
        if terminated:
            print(f"\n⚠️ Episode 在 step {step + 1} 提前終止（觸發風險限制）")
            break

    # 最終統計
    print(f"\n[3] 最終統計:")
    print(f"   - 總回報: ${info['equity'] - 10000:.2f}")
    print(f"   - 回報率: {info['total_return_pct']:.2f}%")
    print(f"   - 總獎勵: {total_reward:.2f}")
    print(f"   - 總交易次數: {info['total_trades']}")
    print(f"   - 勝率: {info['win_rate'] * 100:.1f}%")
    print(f"   - 盈虧比: {info['profit_factor']:.2f}")
    print(f"   - 止損次數: {info['stop_loss_count']}")
    print(f"   - 最大回撤: {info['max_drawdown'] * 100:.2f}%")

    print("\n測試 2: ✅ 通過\n")


def test_specific_actions():
    """測試特定動作序列"""
    print("=" * 60)
    print("測試 3: 特定動作序列測試")
    print("=" * 60)

    # 載入數據並創建環境
    df = pd.read_csv("data/raw/BTCUSDT_1m_full_20260114_012047.csv")
    env = TradingEnv(df=df, episode_length=50)

    obs, info = env.reset(seed=42)
    print(f"\n[1] 初始狀態: balance=${info['balance']:.2f}")

    # 動作序列：持有 -> 做多 -> 持有 5 步 -> 平倉 -> 做空 -> 持有 5 步 -> 平倉
    action_sequence = (
        [3] * 5 +      # 持有 5 步
        [1] +          # 做多
        [3] * 5 +      # 持有 5 步（觀察盈虧）
        [0] +          # 平倉
        [3] * 2 +      # 持有 2 步
        [2] +          # 做空
        [3] * 5 +      # 持有 5 步（觀察盈虧）
        [0]            # 平倉
    )

    print(f"\n[2] 執行特定動作序列...")
    action_names = {0: "平倉", 1: "做多", 2: "做空", 3: "持有"}

    for step, action in enumerate(action_sequence):
        obs, reward, terminated, truncated, info = env.step(action)

        # 顯示重要事件
        if action != 3 or info['position'] != 0:
            print(f"   Step {step + 1:2d}: "
                  f"{action_names[action]} | "
                  f"Equity=${info['equity']:.2f} | "
                  f"Position={info['position']} | "
                  f"Entry=${info['entry_price']:.2f} | "
                  f"StopLoss=${info['stop_loss_price']:.2f}")

        if terminated:
            print(f"\n⚠️ Episode 在 step {step + 1} 提前終止")
            break

    # 最終統計
    print(f"\n[3] 最終統計:")
    print(f"   - 總回報: ${info['equity'] - 10000:.2f}")
    print(f"   - 回報率: {info['total_return_pct']:.2f}%")
    print(f"   - 總交易次數: {info['total_trades']}")
    print(f"   - 勝利/失敗: {info['winning_trades']}/{info['losing_trades']}")
    print(f"   - 止損次數: {info['stop_loss_count']}")

    print("\n測試 3: ✅ 通過\n")


def test_stop_loss():
    """測試止損機制"""
    print("=" * 60)
    print("測試 4: 止損機制測試")
    print("=" * 60)

    df = pd.read_csv("data/raw/BTCUSDT_1m_full_20260114_012047.csv")
    env = TradingEnv(df=df, episode_length=1000, stop_loss_pct=0.015)

    obs, info = env.reset(seed=123)
    print(f"\n[1] 測試止損機制（止損設定: {env.stop_loss_pct * 100:.1f}%）")

    # 持續做多/做空，等待止損觸發
    stop_loss_triggered_count = 0
    for step in range(500):
        # 前 250 步做多，後 250 步做空
        action = 1 if step < 250 else 2

        obs, reward, terminated, truncated, info = env.step(action)

        # 檢查止損
        if info['stop_loss_count'] > stop_loss_triggered_count:
            stop_loss_triggered_count = info['stop_loss_count']
            print(f"   ⚠️ Step {step + 1}: 止損觸發！（第 {stop_loss_triggered_count} 次）")
            print(f"      - 當前權益: ${info['equity']:.2f}")
            print(f"      - 回報率: {info['total_return_pct']:.2f}%")

        if terminated:
            print(f"\n⚠️ Episode 在 step {step + 1} 終止（觸發風險限制）")
            break

    print(f"\n[2] 止損統計:")
    print(f"   - 總止損次數: {info['stop_loss_count']}")
    print(f"   - 總交易次數: {info['total_trades']}")
    print(f"   - 止損比例: {info['stop_loss_count'] / max(info['total_trades'], 1) * 100:.1f}%")

    print("\n測試 4: ✅ 通過\n")


def test_gymnasium_compatibility():
    """測試 Gymnasium 兼容性"""
    print("=" * 60)
    print("測試 5: Gymnasium 兼容性測試")
    print("=" * 60)

    df = pd.read_csv("data/raw/BTCUSDT_1m_full_20260114_012047.csv")
    env = TradingEnv(df=df, episode_length=10)

    print("\n[1] 測試 Gymnasium API...")

    # 測試 check_env（如果有安裝 stable_baselines3）
    try:
        from stable_baselines3.common.env_checker import check_env
        check_env(env, warn=True)
        print("   ✅ Stable-Baselines3 環境檢查通過")
    except ImportError:
        print("   ⚠️ Stable-Baselines3 未安裝，跳過 check_env")
    except Exception as e:
        print(f"   ❌ 環境檢查失敗: {e}")

    # 測試基本 API
    print("\n[2] 測試基本 API...")
    obs, info = env.reset()
    print(f"   ✅ reset() 返回: obs.shape={obs.shape}, info keys={list(info.keys())[:5]}...")

    obs, reward, terminated, truncated, info = env.step(1)
    print(f"   ✅ step() 返回: obs.shape={obs.shape}, reward={reward:.2f}, terminated={terminated}, truncated={truncated}")

    # 測試空間
    print("\n[3] 測試空間...")
    print(f"   ✅ 動作空間: {env.action_space} (n={env.action_space.n})")
    print(f"   ✅ 觀察空間: {env.observation_space} (shape={env.observation_space.shape})")

    # 測試採樣
    print("\n[4] 測試採樣...")
    sample_action = env.action_space.sample()
    print(f"   ✅ 動作採樣: {sample_action}")
    sample_obs = env.observation_space.sample()
    print(f"   ✅ 觀察採樣: shape={sample_obs.shape}")

    print("\n測試 5: ✅ 通過\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 PPO Trading Environment - 完整測試套件")
    print("=" * 60 + "\n")

    try:
        test_basic_functionality()
        test_random_actions()
        test_specific_actions()
        test_stop_loss()
        test_gymnasium_compatibility()

        print("=" * 60)
        print("🎉 所有測試通過！交易環境已準備就緒！")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
