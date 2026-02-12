"""
Speed test: Compare vectorized vs original feature computation
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import time
import pandas as pd
import numpy as np
from pathlib import Path


def main():
    print("=" * 60)
    print("  Vectorized Feature Computation Speed Test")
    print("=" * 60)

    # Load data
    data_files = list(Path("data/raw").glob("BTCUSDT_1m_full_*.csv"))
    df = pd.read_csv(sorted(data_files)[-1])

    # Test different data sizes
    test_sizes = [1000, 5000, 10000, 50000]

    print("\n[Test] Vectorized precomputation speed:")
    print("-" * 50)

    for size in test_sizes:
        if size > len(df):
            continue

        df_test = df.iloc[:size].copy()

        # Ensure datetime index
        if 'timestamp' in df_test.columns:
            df_test['timestamp'] = pd.to_datetime(df_test['timestamp'])
            df_test = df_test.set_index('timestamp')

        from environment.features.feature_aggregator import FeatureAggregator
        aggregator = FeatureAggregator()

        # Measure precomputation time
        start = time.time()
        aggregator.precompute_all_features(df_test, verbose=False)
        precompute_time = time.time() - start

        # Measure query time (10000 queries)
        n_queries = min(10000, size)
        start = time.time()
        for i in range(n_queries):
            idx = i % size
            _ = aggregator.get_cached_state_vector(idx)
        query_time = time.time() - start

        print(f"   {size:,} bars:")
        print(f"      Precompute: {precompute_time:.2f}s ({size/precompute_time:.0f} bars/sec)")
        print(f"      Query: {query_time:.4f}s for {n_queries:,} queries ({n_queries/query_time:.0f} queries/sec)")

    print("\n" + "=" * 60)
    print("  Speed Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
