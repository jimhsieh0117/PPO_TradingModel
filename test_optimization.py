"""
優化驗證腳本 - 確保預計算特徵與原始計算一致

測試項目：
1. MultiTimeframeAnalyzer 預計算 vs 原始計算
2. FeatureAggregator 預計算 vs 原始計算
3. 訓練速度對比
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import time
import numpy as np
import pandas as pd
from pathlib import Path


def test_multi_timeframe_consistency():
    """測試多時間框架特徵的一致性"""
    print("=" * 60)
    print("  🧪 測試 MultiTimeframeAnalyzer 預計算一致性")
    print("=" * 60)

    from environment.features.multi_timeframe import MultiTimeframeAnalyzer

    # 載入測試數據
    data_path = Path("data/raw/BTCUSDT_1m_train_latest.csv")
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    df_test = df.iloc[-1000:].copy()  # 使用最後 1000 根 K 線

    print(f"📂 測試數據: {len(df_test)} 根 K 線")

    # 創建兩個分析器：一個使用緩存，一個不使用
    analyzer_cached = MultiTimeframeAnalyzer()
    analyzer_original = MultiTimeframeAnalyzer()

    # 預計算緩存版本
    print("\n🔄 預計算緩存版本...")
    start = time.time()
    analyzer_cached.precompute_all_features(df_test)
    precompute_time = time.time() - start
    print(f"   預計算耗時: {precompute_time:.2f} 秒")

    # 測試隨機位置的一致性
    test_indices = [50, 100, 200, 500, 800, 999]
    all_match = True

    print("\n📊 比較預計算 vs 原始計算:")
    for idx in test_indices:
        # 緩存版本
        cached_result = analyzer_cached.get_cached_features(idx)

        # 原始版本（強制不使用緩存）
        analyzer_original._cache_valid = False
        original_result = analyzer_original.calculate_features(df_test, idx)

        # 比較
        match = (cached_result['trend_5m'] == original_result['trend_5m'] and
                 cached_result['trend_15m'] == original_result['trend_15m'])

        status = "✅" if match else "❌"
        print(f"   索引 {idx}: {status}")
        print(f"      緩存:  trend_5m={cached_result['trend_5m']}, trend_15m={cached_result['trend_15m']}")
        print(f"      原始:  trend_5m={original_result['trend_5m']}, trend_15m={original_result['trend_15m']}")

        if not match:
            all_match = False

    if all_match:
        print("\n✅ MultiTimeframeAnalyzer 預計算驗證通過！")
    else:
        print("\n❌ MultiTimeframeAnalyzer 預計算存在差異！")

    return all_match


def test_feature_aggregator_consistency():
    """測試特徵聚合器的一致性"""
    print("\n" + "=" * 60)
    print("  🧪 測試 FeatureAggregator 預計算一致性")
    print("=" * 60)

    from environment.features.feature_aggregator import FeatureAggregator

    # 載入測試數據
    data_path = Path("data/raw/BTCUSDT_1m_train_latest.csv")
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    df_test = df.iloc[-500:].copy()  # 使用最後 500 根 K 線（減少測試時間）

    print(f"📂 測試數據: {len(df_test)} 根 K 線")

    # 創建聚合器並預計算
    aggregator = FeatureAggregator()

    print("\n🔄 預計算所有特徵...")
    start = time.time()
    aggregator.precompute_all_features(df_test, verbose=True)
    precompute_time = time.time() - start
    print(f"   預計算總耗時: {precompute_time:.2f} 秒")

    # 測試隨機位置的一致性
    test_indices = [50, 100, 200, 300, 400, 499]
    all_match = True
    max_diff = 0.0

    # 創建一個不使用緩存的聚合器來對比
    aggregator_original = FeatureAggregator()

    print("\n📊 比較預計算 vs 原始計算:")
    for idx in test_indices:
        # 緩存版本
        cached_vector = aggregator.get_cached_state_vector(idx)

        # 原始版本
        original_vector = aggregator_original.get_state_vector(df_test, idx)

        # 計算差異
        diff = np.abs(cached_vector - original_vector)
        max_diff_this = diff.max()
        max_diff = max(max_diff, max_diff_this)

        # 允許小誤差（浮點數精度）
        match = np.allclose(cached_vector, original_vector, atol=1e-5)
        status = "✅" if match else "❌"
        print(f"   索引 {idx}: {status} (最大差異: {max_diff_this:.6f})")

        if not match:
            all_match = False
            # 顯示差異詳情
            for i, (c, o) in enumerate(zip(cached_vector, original_vector)):
                if abs(c - o) > 1e-5:
                    print(f"      特徵 {aggregator.feature_names[i]}: 緩存={c:.4f}, 原始={o:.4f}")

    print(f"\n   總體最大差異: {max_diff:.6f}")

    if all_match:
        print("\n✅ FeatureAggregator 預計算驗證通過！")
    else:
        print("\n❌ FeatureAggregator 預計算存在差異！")

    return all_match


def test_speed_improvement():
    """測試速度提升"""
    print("\n" + "=" * 60)
    print("  🚀 測試速度提升")
    print("=" * 60)

    from environment.features.feature_aggregator import FeatureAggregator

    # 載入測試數據
    data_path = Path("data/raw/BTCUSDT_1m_train_latest.csv")
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    df_test = df.iloc[-200:].copy()  # 使用 200 根 K 線

    print(f"📂 測試數據: {len(df_test)} 根 K 線")

    # 測試原始方法速度
    aggregator_original = FeatureAggregator()
    test_count = 100

    print(f"\n⏱️ 測試原始方法（{test_count} 次 get_state_vector）...")
    start = time.time()
    for i in range(test_count):
        idx = 100 + (i % 50)
        _ = aggregator_original.get_state_vector(df_test, idx)
    original_time = time.time() - start
    original_per_call = original_time / test_count * 1000
    print(f"   原始方法: {original_time:.2f} 秒 ({original_per_call:.2f} ms/call)")

    # 測試緩存方法速度
    aggregator_cached = FeatureAggregator()
    aggregator_cached.precompute_all_features(df_test, verbose=False)

    print(f"\n⏱️ 測試緩存方法（{test_count} 次 get_cached_state_vector）...")
    start = time.time()
    for i in range(test_count):
        idx = 100 + (i % 50)
        _ = aggregator_cached.get_cached_state_vector(idx)
    cached_time = time.time() - start
    cached_per_call = cached_time / test_count * 1000
    print(f"   緩存方法: {cached_time:.4f} 秒 ({cached_per_call:.4f} ms/call)")

    # 計算加速比
    speedup = original_per_call / cached_per_call
    print(f"\n🎯 加速比: {speedup:.0f}x")

    return speedup


def main():
    """主測試函數"""
    print("\n" + "=" * 60)
    print("  📋 PPO_TradingModel 優化驗證")
    print("=" * 60 + "\n")

    results = {}

    # 測試 1: MultiTimeframe 一致性
    results['mtf_consistent'] = test_multi_timeframe_consistency()

    # 測試 2: FeatureAggregator 一致性
    results['aggregator_consistent'] = test_feature_aggregator_consistency()

    # 測試 3: 速度提升
    results['speedup'] = test_speed_improvement()

    # 總結
    print("\n" + "=" * 60)
    print("  📊 測試總結")
    print("=" * 60)
    print(f"   MultiTimeframe 一致性: {'✅ 通過' if results['mtf_consistent'] else '❌ 失敗'}")
    print(f"   FeatureAggregator 一致性: {'✅ 通過' if results['aggregator_consistent'] else '❌ 失敗'}")
    print(f"   特徵提取加速比: {results['speedup']:.0f}x")

    if results['mtf_consistent'] and results['aggregator_consistent']:
        print("\n🎉 所有測試通過！優化不會改變訓練結果。")
    else:
        print("\n⚠️ 部分測試失敗，請檢查代碼。")


if __name__ == "__main__":
    main()
