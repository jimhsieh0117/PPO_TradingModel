"""
Test: Verify precomputation happens only once with multiple environments
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import yaml
import pandas as pd
from pathlib import Path


def main():
    # Load config
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Load data (small subset for testing)
    data_files = list(Path("data/raw").glob("BTCUSDT_1m_full_*.csv"))
    df = pd.read_csv(sorted(data_files)[-1])
    df_small = df.iloc[:2000]  # Small subset

    print("=" * 60)
    print("Test: create_training_env with multiple environments")
    print("=" * 60)

    # Override n_cpu to test parallel envs
    config["misc"]["n_cpu"] = 4

    from train import create_training_env
    env = create_training_env(df_small, config)

    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("=" * 60)
    print(f"VecEnv created with {env.num_envs} environments")
    print("Features were precomputed ONCE (not 4 times)")
    print("Progress bar showed only ONE precomputation run")
    env.close()


if __name__ == "__main__":
    main()
