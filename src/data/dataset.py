"""
LeakDB PyTorch Dataset
- 슬라이딩 윈도우로 시계열 분할
- (Scenario, Node)별로 독립적인 시계열 처리
- FCVAE 모델 학습을 위한 데이터 제공
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


class LeakDBDataset(Dataset):
    """
    LeakDB PyTorch Dataset
    슬라이딩 윈도우로 시계열 분할
    """

    def __init__(
        self,
        csv_file: str,
        window_size: int = 96,
        stride: int = 1
    ):
        """
        Args:
            csv_file: 전처리된 CSV 경로 (train_processed.csv, val_processed.csv, test_processed.csv)
            window_size: 윈도우 크기 (기본값: 96 = 48시간, 30분 간격)
            stride: 슬라이딩 윈도우 스트라이드 (기본값: 1)
        """
        self.csv_file = Path(csv_file)
        self.window_size = window_size
        self.stride = stride

        # 데이터 로드
        print(f"Loading dataset from {csv_file}...")
        self.df = pd.read_csv(csv_file, engine='pyarrow')

        # 컬럼 확인
        required_columns = ['Scenario', 'Node', 'Timestamp', 'Value', 'ValueScaled', 'Label']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}\n"
                f"Available columns: {list(self.df.columns)}"
            )

        print(f"  Loaded {len(self.df):,} rows")
        print(f"  Scenarios: {self.df['Scenario'].nunique()}")
        print(f"  Nodes: {self.df['Node'].nunique()}")
        print(f"  Window size: {window_size}, Stride: {stride}")

        # 윈도우 인덱스 생성
        self.windows = self._create_windows()

        print(f"  ✓ {len(self.windows):,} windows created")

        # 라벨 분포 확인
        self._print_label_distribution()

    def _create_windows(self) -> List[Dict]:
        """
        각 (Scenario, Node)에서 슬라이딩 윈도우 추출

        Returns:
            List of dict with keys: 'scenario', 'node', 'start_idx', 'end_idx'
        """
        windows = []

        for (scenario, node), group in self.df.groupby(['Scenario', 'Node'], sort=True):
            # 이 그룹의 인덱스 범위
            group_indices = group.index.tolist()
            n_timesteps = len(group_indices)

            # 윈도우가 생성되지 않는 경우 경고
            if n_timesteps < self.window_size:
                print(
                    f"  Warning: Scenario {scenario}, Node {node} has only {n_timesteps} timesteps "
                    f"(< window_size {self.window_size}). Skipping."
                )
                continue

            # 슬라이딩 윈도우
            for start in range(0, n_timesteps - self.window_size + 1, self.stride):
                end = start + self.window_size

                windows.append({
                    'scenario': scenario,
                    'node': node,
                    'start_idx': group_indices[start],
                    'end_idx': group_indices[end - 1]  # inclusive (마지막 인덱스)
                })

        return windows

    def _print_label_distribution(self):
        """라벨 분포 출력"""
        total_timesteps = len(self.df)
        normal_timesteps = (self.df['Label'] == 0).sum()
        leak_timesteps = (self.df['Label'] == 1).sum()

        print(f"\n  Label distribution (timesteps):")
        print(f"    Normal: {normal_timesteps:,} ({normal_timesteps/total_timesteps*100:.2f}%)")
        print(f"    Leak:   {leak_timesteps:,} ({leak_timesteps/total_timesteps*100:.2f}%)")

    def __len__(self) -> int:
        """데이터셋 크기 (윈도우 개수)"""
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        인덱스에 해당하는 윈도우 데이터 반환

        Args:
            idx: 윈도우 인덱스

        Returns:
            Dictionary with:
                - values: [window_size] 스케일링된 수요 값 (ValueScaled)
                - labels: [window_size] 바이너리 라벨 (0=정상, 1=누수)
                - scenario: 시나리오 번호 (scalar)
                - node: 노드 번호 (scalar)
        """
        window_info = self.windows[idx]

        # 윈도우 데이터 추출 (start_idx부터 end_idx까지 inclusive)
        window_data = self.df.loc[
            window_info['start_idx']:window_info['end_idx']
        ]

        # 윈도우 크기 검증
        if len(window_data) != self.window_size:
            raise RuntimeError(
                f"Window size mismatch: expected {self.window_size}, "
                f"got {len(window_data)} for window {idx}"
            )

        # Tensor 변환
        values = torch.FloatTensor(window_data['ValueScaled'].values)
        labels = torch.FloatTensor(window_data['Label'].values)

        return {
            'values': values,          # [window_size]
            'labels': labels,          # [window_size]
            'scenario': window_info['scenario'],  # scalar
            'node': window_info['node']          # scalar
        }

    def get_window_info(self, idx: int) -> Dict:
        """윈도우 메타데이터 반환 (디버깅용)"""
        return self.windows[idx]


# ===== 사용 예시 =====
if __name__ == '__main__':
    from torch.utils.data import DataLoader

    print("=" * 70)
    print("LeakDB Dataset Test")
    print("=" * 70)

    # Dataset 생성
    train_dataset = LeakDBDataset(
        csv_file='data/processed/net1/train_processed.csv',
        window_size=96,
        stride=1
    )

    # DataLoader
    print("-" * 70)
    print("Creating DataLoader...")
    print("-" * 70)
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 샘플 배치 확인
    print("-" * 70)
    print("Sample Batch:")
    print("-" * 70)
    batch = next(iter(train_loader))

    print(f"Batch values shape: {batch['values'].shape}")  # [256, 96]
    print(f"Batch labels shape: {batch['labels'].shape}")  # [256, 96]
    print(f"Batch scenario type: {type(batch['scenario'])}")
    print(f"Batch node type: {type(batch['node'])}")

    print(f"\nFirst sample in batch:")
    print(f"  Values: min={batch['values'][0].min():.4f}, max={batch['values'][0].max():.4f}, "
          f"mean={batch['values'][0].mean():.4f}, std={batch['values'][0].std():.4f}")
    print(f"  Labels: {batch['labels'][0][:10].tolist()} ... (first 10)")
    print(f"  Has leak: {(batch['labels'][0] > 0).any().item()}")
    print(f"  Scenario: {batch['scenario'][0].item()}")
    print(f"  Node: {batch['node'][0].item()}")

    # 전체 배치 통계
    print(f"\nBatch statistics:")
    print(f"  Unique scenarios: {len(torch.unique(batch['scenario']))}")
    print(f"  Unique nodes: {len(torch.unique(batch['node']))}")
    print(f"  Windows with leak: {sum((batch['labels'].sum(dim=1) > 0))}/{len(batch['labels'])}")

    print("\n" + "=" * 70)
    print("Dataset test complete!")
    print("=" * 70)