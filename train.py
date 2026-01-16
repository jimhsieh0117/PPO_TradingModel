"""
PPO Trading Bot - 訓練入口腳本

這是主訓練腳本，用於訓練 PPO 交易代理。

使用方法：
    python train.py

交易員視角：
- 我們將訓練一個 AI 代理來學習在市場中交易
- 目標：學習識別 ICT 模式並執行有利可圖的交易
- 關鍵：嚴格的風險管理和穩定的回報
"""

import os
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# 強化學習核心
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed

# 我們的交易環境
from environment.trading_env import TradingEnv
from agent.callbacks import TrainingMetricsCallback


def load_config(config_path: str = "config.yaml") -> dict:
    """載入配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_training_data(config: dict) -> pd.DataFrame:
    """
    載入訓練數據

    Args:
        config: 配置字典

    Returns:
        DataFrame: OHLCV 數據
    """
    # 自動尋找最新的完整數據文件
    raw_data_dir = config.get('data', {}).get('raw_data_dir', 'data/raw')
    data_files = list(Path(raw_data_dir).glob('BTCUSDT_1m_full_*.csv'))

    if not data_files:
        raise FileNotFoundError(f"在 {raw_data_dir} 中找不到 BTCUSDT_1m_full_*.csv 文件")

    # 使用最新的文件
    data_path = str(sorted(data_files)[-1])

    print(f"📂 載入訓練數據: {data_path}")
    df = pd.read_csv(data_path)

    print(f"   ✅ 數據載入成功")
    print(f"   - 總 K 線數: {len(df):,}")
    print(f"   - 時間範圍: {df['timestamp'].iloc[0]} 到 {df['timestamp'].iloc[-1]}")
    print(f"   - 列名: {df.columns.tolist()}")

    return df


def create_training_env(df: pd.DataFrame, config: dict):
    """
    創建訓練環境

    Args:
        df: OHLCV 數據
        config: 配置字典

    Returns:
        VecEnv: 向量化環境
    """
    print("\n🏗️ 創建交易環境...")

    # 從配置中獲取交易參數
    trading_config = config.get('trading', {})

    # 獲取訓練配置
    training_config = config.get('training', {})
    backtest_config = config.get('backtest', {})
    backtest_config = config.get('backtest', {})
    backtest_config = config.get('backtest', {})

    misc_config = config.get('misc', {})
    n_envs = int(misc_config.get('n_cpu', 1))
    if n_envs < 1:
        n_envs = 1

    seed = misc_config.get('random_seed', None)
    if seed is not None:
        set_random_seed(seed)

    def make_env(rank: int):
        def _init():
            env = TradingEnv(
                df=df,
                initial_balance=backtest_config.get('initial_capital', 10000.0),
                leverage=trading_config.get('leverage', 10),
                position_size_pct=trading_config.get('position_size_pct', 0.15),
                stop_loss_pct=trading_config.get('stop_loss_pct', 0.015),
                max_daily_drawdown=trading_config.get('daily_drawdown_limit', 0.10),
                trading_fee=trading_config.get('taker_fee', 0.0004),
                episode_length=training_config.get('episode_length', 1440),
                feature_config=config.get('features', {}),
                reward_config=config.get('reward', {})
            )
            if seed is not None:
                env.reset(seed=seed + rank)
            return env
        return _init

    if n_envs == 1:
        env = DummyVecEnv([make_env(0)])
    else:
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    print("   ✅ 環境創建成功")
    print(f"   - 初始資金: ${backtest_config.get('initial_capital', 10000):,.2f}")
    print(f"   - 槓桿倍數: {trading_config.get('leverage', 10)}x")
    print(f"   - 倉位大小: {trading_config.get('position_size_pct', 0.15) * 100:.1f}%")
    print(f"   - 止損百分比: {trading_config.get('stop_loss_pct', 0.015) * 100:.2f}%")
    print(f"   - Episode 長度: {training_config.get('episode_length', 1440)} steps")
    print(f"   - n_envs: {n_envs}")

    return env


def create_ppo_model(env, config: dict):
    """
    創建 PPO 模型

    Args:
        env: 訓練環境
        config: 配置字典

    Returns:
        PPO: PPO 模型實例
    """
    print("\n🤖 創建 PPO 模型...")

    # 從配置中獲取 PPO 參數
    ppo_config = config.get('ppo', {})

    # 獲取 tensorboard 日誌路徑
    training_config = config.get('training', {})
    tensorboard_log = f"./{training_config.get('tensorboard_log', 'tensorboard')}"

    model = PPO(
        policy="MlpPolicy",
        env=env,
        device="cpu",
        learning_rate=ppo_config.get('learning_rate', 3e-4),
        n_steps=ppo_config.get('n_steps', 2048),
        batch_size=ppo_config.get('batch_size', 64),
        n_epochs=ppo_config.get('n_epochs', 10),
        gamma=ppo_config.get('gamma', 0.99),
        gae_lambda=ppo_config.get('gae_lambda', 0.95),
        clip_range=ppo_config.get('clip_range', 0.2),
        ent_coef=ppo_config.get('ent_coef', 0.01),
        vf_coef=ppo_config.get('vf_coef', 0.5),
        max_grad_norm=ppo_config.get('max_grad_norm', 0.5),
        verbose=1,
        tensorboard_log=tensorboard_log
    )

    print("   ✅ PPO 模型創建成功")
    print(f"   - 策略: MlpPolicy")
    print(f"   - 學習率: {ppo_config.get('learning_rate', 3e-4)}")
    print(f"   - Batch Size: {ppo_config.get('batch_size', 64)}")
    print(f"   - N Epochs: {ppo_config.get('n_epochs', 10)}")
    print(f"   - Gamma: {ppo_config.get('gamma', 0.99)}")

    return model


def setup_callbacks(save_dir: str, config: dict):
    """
    設置訓練回調

    Args:
        save_dir: 保存目錄
        config: 配置字典

    Returns:
        CallbackList: 回調列表
    """
    print("\n📊 設置訓練回調...")

    # 創建保存目錄
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)

    callbacks = []

    # 1. Checkpoint Callback - 定期保存模型
    training_config = config.get('training', {})
    backtest_config = config.get('backtest', {})
    trading_config = config.get('trading', {})
    checkpoint_callback = CheckpointCallback(
        save_freq=training_config.get('save_freq', 10000),
        save_path=f"{save_dir}/checkpoints",
        name_prefix="ppo_trading_model",
        save_replay_buffer=False,
        save_vecnormalize=False
    )
    callbacks.append(checkpoint_callback)
    print(f"   ✅ Checkpoint Callback（每 {training_config.get('save_freq', 10000)} 步保存）")

    # 2. Training metrics callback - CSV log
    
    metrics_callback = TrainingMetricsCallback(
        log_path=f"{save_dir}/training_log.csv",
        best_model_path=f"{save_dir}/ppo_trading_model_best.zip",
        best_log_path=f"{save_dir}/best_model_log.csv",
        episode_length=training_config.get('episode_length', 1440),
        initial_capital=backtest_config.get('initial_capital', 10000.0),
        max_daily_drawdown=trading_config.get('daily_drawdown_limit', 0.10),
        verbose=1
    )
    callbacks.append(metrics_callback)
    print("   TrainingMetrics Callback: training log enabled")

    callback_list = CallbackList(callbacks)

    print(f"   ✅ 共設置 {len(callbacks)} 個回調")

    return callback_list


def train_model(
    model,
    total_timesteps: int,
    callbacks,
    save_dir: str
):
    """
    訓練模型

    Args:
        model: PPO 模型
        total_timesteps: 總訓練步數
        callbacks: 回調列表
        save_dir: 保存目錄
    """
    print("\n" + "=" * 60)
    print("🚀 開始訓練 PPO 交易代理")
    print("=" * 60)
    print(f"   總訓練步數: {total_timesteps:,}")
    print(f"   預計 Episodes: ~{total_timesteps // 1440}")
    print(f"   保存目錄: {save_dir}")
    print("=" * 60 + "\n")

    try:
        # 開始訓練
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True  # 顯示訓練進度條
        )

        print("\n" + "=" * 60)
        print("✅ 訓練完成！")
        print("=" * 60)

        # 保存最終模型
        final_model_path = f"{save_dir}/ppo_trading_model_final.zip"
        model.save(final_model_path)
        print(f"\n💾 最終模型已保存: {final_model_path}")

        return True

    except KeyboardInterrupt:
        print("\n⚠️ 訓練被用戶中斷")

        # 保存中斷時的模型
        interrupted_model_path = f"{save_dir}/ppo_trading_model_interrupted.zip"
        model.save(interrupted_model_path)
        print(f"💾 中斷模型已保存: {interrupted_model_path}")

        return False

    except Exception as e:
        print(f"\n❌ 訓練過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

        return False


def main():
    """主訓練流程"""
    print("\n" + "=" * 60)
    print("💰 PPO Trading Bot - 訓練系統")
    print("=" * 60)
    print("基於 ICT 策略的加密貨幣交易機器人")
    print("=" * 60 + "\n")

    # 1. 載入配置
    print("📋 步驟 1/6: 載入配置文件")
    try:
        config = load_config("config.yaml")
        print("   ✅ 配置載入成功\n")
    except Exception as e:
        print(f"   ❌ 配置載入失敗: {e}")
        return

    # 2. 載入訓練數據
    print("📋 步驟 2/6: 載入訓練數據")
    try:
        df_train = load_training_data(config)
    except Exception as e:
        print(f"   ❌ 數據載入失敗: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. 創建訓練環境
    print("📋 步驟 3/6: 創建訓練環境")
    try:
        env = create_training_env(df_train, config)
    except Exception as e:
        print(f"   ❌ 環境創建失敗: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. 創建 PPO 模型
    print("📋 步驟 4/6: 創建 PPO 模型")
    try:
        model = create_ppo_model(env, config)
    except Exception as e:
        print(f"   ❌ 模型創建失敗: {e}")
        return

    # 5. 設置回調
    print("📋 步驟 5/6: 設置訓練回調")
    try:
        # 創建保存目錄（帶時間戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"models/run_{timestamp}"

        callbacks = setup_callbacks(save_dir, config)

        # 保存配置文件副本
        import shutil
        shutil.copy("config.yaml", f"{save_dir}/config.yaml")
        print(f"   ✅ 配置文件已複製到: {save_dir}/config.yaml")

    except Exception as e:
        print(f"   ❌ 回調設置失敗: {e}")
        return

    # 6. 開始訓練
    print("📋 步驟 6/6: 開始訓練")
    try:
        total_timesteps = config.get('training', {}).get('total_timesteps', 100000)

        success = train_model(
            model=model,
            total_timesteps=total_timesteps,
            callbacks=callbacks,
            save_dir=save_dir
        )

        if success:
            print("\n🎉 訓練流程全部完成！")
            print(f"📁 模型保存位置: {save_dir}")
            print("\n下一步：")
            print("   1. 查看訓練日誌")
            print("   2. 分析訓練曲線")
            print("   3. 在測試集上評估模型")
            print("   4. 如果表現良好，可以進行回測")
        else:
            print("\n⚠️ 訓練未正常完成，請檢查上述錯誤信息")

    except Exception as e:
        print(f"   ❌ 訓練失敗: {e}")
        import traceback
        traceback.print_exc()
        return

    # 清理
    env.close()
    print("\n✅ 環境已關閉")


if __name__ == "__main__":
    main()
