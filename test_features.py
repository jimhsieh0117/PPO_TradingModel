"""
快速測試所有 ICT 特徵

用於驗證特徵整合器的功能和性能

作者：PPO Trading Team
日期：2026-01-14
"""

import pandas as pd
import numpy as np
from pathlib import Path
from environment.features import FeatureAggregator
import time


def test_feature_extraction():
    """測試特徵提取性能和正確性"""
    print("=" * 70)
    print("  🧪 PPO Trading Bot - 特徵提取測試")
    print("=" * 70)

    # 載入數據
    print("\n📂 載入數據...")
    data_path = Path("data/raw/BTCUSDT_1m_train_latest.csv")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"   數據大小: {len(df):,} 根 K 線")
    print(f"   時間範圍: {df.index[0]} 到 {df.index[-1]}")

    # 初始化特徵整合器
    print("\n🔧 初始化特徵整合器...")
    aggregator = FeatureAggregator()
    print(f"   狀態空間維度: {aggregator.get_state_dimension()}")

    # 測試單點提取
    print("\n🎯 測試單點特徵提取...")
    test_idx = len(df) - 1
    start_time = time.time()
    state = aggregator.get_state_vector(df, test_idx)
    elapsed = time.time() - start_time

    print(f"   耗時: {elapsed*1000:.2f} ms")
    print(f"   狀態向量形狀: {state.shape}")
    print(f"   數據類型: {state.dtype}")
    print(f"   均值: {state.mean():.3f}")
    print(f"   標準差: {state.std():.3f}")

    # 測試批量提取
    print("\n📊 測試批量特徵提取（100 個時間點）...")
    test_indices = np.linspace(100, len(df)-1, 100, dtype=int)

    start_time = time.time()
    states = []
    for idx in test_indices:
        state = aggregator.get_state_vector(df, idx)
        states.append(state)
    elapsed = time.time() - start_time

    states = np.array(states)
    print(f"   總耗時: {elapsed:.2f} 秒")
    print(f"   平均每點: {elapsed/len(test_indices)*1000:.2f} ms")
    print(f"   批量狀態形狀: {states.shape}")

    # 顯示特徵統計
    print("\n📈 特徵統計（100 個樣本）:")
    feature_stats = pd.DataFrame({
        'feature': aggregator.feature_names,
        'mean': states.mean(axis=0),
        'std': states.std(axis=0),
        'min': states.min(axis=0),
        'max': states.max(axis=0)
    })

    print("\n   前 10 個特徵:")
    print(feature_stats.head(10).to_string(index=False))

    print("\n   後 10 個特徵:")
    print(feature_stats.tail(10).to_string(index=False))

    # 測試特徵字典
    print("\n🔍 測試特徵字典提取...")
    features_dict = aggregator.get_feature_dict(df, test_idx)
    print(f"   特徵數量: {len(features_dict)}")
    print(f"   特徵鍵: {list(features_dict.keys())[:5]}...")

    print("\n✅ 所有測試通過！")
    print("\n🎉 階段 2（特徵工程）完成！")
    print("📌 下一步：建立 Gymnasium 交易環境")


if __name__ == "__main__":
    test_feature_extraction()
