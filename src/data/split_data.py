"""
데이터 분할 모듈
시나리오 기반으로 Train/Val/Test 분할 (랜덤 셔플)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def split_by_scenario(
    input_csv: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    random_seed: int = 42,
    shuffle: bool = True
):
    """
    시나리오 기반으로 데이터 분할
    
    Args:
        input_csv: 통합 CSV 경로
        output_dir: 출력 디렉토리
        train_ratio: 학습 비율 (default: 0.7)
        val_ratio: 검증 비율 (default: 0.2)
        test_ratio: 테스트 비율 (default: 0.1)
        random_seed: 랜덤 시드 (재현성을 위해)
        shuffle: True면 시나리오를 랜덤하게 섞음 (권장)
    
    Returns:
        None (CSV 파일로 저장)
    """
    # 비율 검증
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("LeakDB Data Split")
    print("=" * 60)
    print(f"Input:  {input_csv}")
    print(f"Output: {output_dir}")
    print(f"Ratios: Train={train_ratio:.0%}, Val={val_ratio:.0%}, Test={test_ratio:.0%}")
    print(f"Shuffle: {shuffle} (seed={random_seed})")
    print()
    
    # 데이터 로드
    print("Loading data...")
    df = pd.read_csv(input_csv)
    print(f"  ✓ Loaded {len(df):,} rows")
    
    # 시나리오 리스트 추출
    scenarios = sorted(df['Scenario'].unique())
    n_scenarios = len(scenarios)
    
    print(f"  ✓ Found {n_scenarios} scenarios")
    print()
    
    # 랜덤 셔플
    if shuffle:
        np.random.seed(random_seed)
        scenarios = np.random.permutation(scenarios)
        print(f"  ✓ Scenarios shuffled with seed={random_seed}")
    
    # 분할 인덱스 계산
    train_end = int(n_scenarios * train_ratio)
    val_end = train_end + int(n_scenarios * val_ratio)
    
    train_scenarios = scenarios[:train_end]
    val_scenarios = scenarios[train_end:val_end]
    test_scenarios = scenarios[val_end:]
    
    print("-" * 60)
    print("Split Summary:")
    print("-" * 60)
    print(f"Train: {len(train_scenarios):4d} scenarios ({len(train_scenarios)/n_scenarios:.1%})")
    if shuffle:
        print(f"       Range: {sorted(train_scenarios)[0]:4d} - {sorted(train_scenarios)[-1]:4d} (shuffled)")
    else:
        print(f"       Range: {train_scenarios[0]:4d} - {train_scenarios[-1]:4d}")
    
    print(f"Val:   {len(val_scenarios):4d} scenarios ({len(val_scenarios)/n_scenarios:.1%})")
    if shuffle:
        print(f"       Range: {sorted(val_scenarios)[0]:4d} - {sorted(val_scenarios)[-1]:4d} (shuffled)")
    else:
        print(f"       Range: {val_scenarios[0]:4d} - {val_scenarios[-1]:4d}")
    
    print(f"Test:  {len(test_scenarios):4d} scenarios ({len(test_scenarios)/n_scenarios:.1%})")
    if shuffle:
        print(f"       Range: {sorted(test_scenarios)[0]:4d} - {sorted(test_scenarios)[-1]:4d} (shuffled)")
    else:
        print(f"       Range: {test_scenarios[0]:4d} - {test_scenarios[-1]:4d}")
    print()
    
    # 데이터 분할
    print("Splitting data...")
    train_df = df[df['Scenario'].isin(train_scenarios)]
    val_df = df[df['Scenario'].isin(val_scenarios)]
    test_df = df[df['Scenario'].isin(test_scenarios)]
    
    # 저장
    print("Saving files...")
    train_path = output_dir / 'train.csv'
    val_path = output_dir / 'val.csv'
    test_path = output_dir / 'test.csv'
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"  ✓ Train: {len(train_df):,} rows → {train_path}")
    print(f"  ✓ Val:   {len(val_df):,} rows → {val_path}")
    print(f"  ✓ Test:  {len(test_df):,} rows → {test_path}")
    print()
    
    # 라벨 분포 분석
    print("-" * 60)
    print("Label Distribution:")
    print("-" * 60)
    
    def print_label_stats(df, name):
        normal = (df['Label'] == 0).sum()
        leak = (df['Label'] == 1).sum()
        total = len(df)
        leak_ratio = leak / total * 100 if total > 0 else 0
        
        print(f"{name:6s}: Normal={normal:,} ({normal/total:.1%}), "
              f"Leak={leak:,} ({leak_ratio:.2f}%)")
    
    print_label_stats(train_df, "Train")
    print_label_stats(val_df, "Val")
    print_label_stats(test_df, "Test")
    print()
    
    # 노드 분포 확인
    print("-" * 60)
    print("Node Distribution:")
    print("-" * 60)
    train_nodes = sorted(train_df['Node'].unique())
    val_nodes = sorted(val_df['Node'].unique())
    test_nodes = sorted(test_df['Node'].unique())
    
    print(f"Train nodes: {train_nodes}")
    print(f"Val nodes:   {val_nodes}")
    print(f"Test nodes:  {test_nodes}")
    
    # 모든 split에 동일한 노드가 있는지 확인
    if set(train_nodes) == set(val_nodes) == set(test_nodes):
        print("  ✓ All splits contain the same nodes")
    else:
        print("  ⚠ Warning: Node distribution differs across splits")
    print()
    
    # 파일 크기
    print("-" * 60)
    print("File Sizes:")
    print("-" * 60)
    for path in [train_path, val_path, test_path]:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"{path.name:12s}: {size_mb:6.2f} MB")
    print()
    
    # 시나리오 ID 저장 (나중에 추적 용이)
    scenario_map_path = output_dir / 'scenario_split.txt'
    with open(scenario_map_path, 'w') as f:
        f.write(f"Random seed: {random_seed}\n")
        f.write(f"Shuffle: {shuffle}\n\n")
        f.write(f"Train scenarios ({len(train_scenarios)}):\n")
        f.write(f"{sorted(train_scenarios.tolist())}\n\n")
        f.write(f"Val scenarios ({len(val_scenarios)}):\n")
        f.write(f"{sorted(val_scenarios.tolist())}\n\n")
        f.write(f"Test scenarios ({len(test_scenarios)}):\n")
        f.write(f"{sorted(test_scenarios.tolist())}\n")
    
    print(f"  ✓ Scenario mapping saved: {scenario_map_path}")
    print()
    print("=" * 60)
    print("Split complete!")
    print("=" * 60)


def load_split_info(output_dir: str) -> dict:
    """
    분할 정보 로드
    
    Args:
        output_dir: split_by_scenario()로 생성한 출력 디렉토리
    
    Returns:
        dict with keys: 'train_scenarios', 'val_scenarios', 'test_scenarios'
    """
    scenario_map_path = Path(output_dir) / 'scenario_split.txt'
    
    if not scenario_map_path.exists():
        raise FileNotFoundError(f"Scenario split file not found: {scenario_map_path}")
    
    with open(scenario_map_path, 'r') as f:
        content = f.read()
    
    # Parse scenarios
    import ast
    
    train_start = content.find("Train scenarios")
    val_start = content.find("Val scenarios")
    test_start = content.find("Test scenarios")
    
    train_line = content[train_start:val_start].split('\n')[1]
    val_line = content[val_start:test_start].split('\n')[1]
    test_line = content[test_start:].split('\n')[1]
    
    return {
        'train_scenarios': ast.literal_eval(train_line),
        'val_scenarios': ast.literal_eval(val_line),
        'test_scenarios': ast.literal_eval(test_line)
    }


# ===== 사용 예시 =====
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Split LeakDB data by scenario')
    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV file (unified)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Train ratio (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='Val ratio (default: 0.2)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                        help='Test ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--no-shuffle', action='store_true',
                        help='Disable random shuffling')
    
    args = parser.parse_args()
    
    split_by_scenario(
        input_csv=args.input,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        shuffle=not args.no_shuffle
    )
    
    # 예시 실행 커맨드:
    # python -m src.data.split_data \
    #     --input data/processed/net1_cmh_full.csv \
    #     --output data/processed/net1 \
    #     --train-ratio 0.7 \
    #     --val-ratio 0.2 \
    #     --test-ratio 0.1 \
    #     --seed 42