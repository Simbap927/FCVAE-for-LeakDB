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
        stride: int = 1,
        mode: str = "train",
        use_label: bool = False
    ):
        """
        Args:
            csv_file: 전처리된 CSV 경로
            window_size: 윈도우 크기 (기본값: 96 = 48시간, 30분 간격)
            stride: 슬라이딩 윈도우 스트라이드 (기본값: 1)
            mode: "train", "valid", "test"
            use_label: True면 실제 라벨 사용 (semi-supervised), False면 완전 비지도
        """
        self.csv_file = Path(csv_file)
        self.window_size = window_size
        self.stride = stride
        self.mode = mode
        self.use_label = use_label

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
        
        print(f"  Mode: {mode}")
        print(f"  Use real labels: {use_label}")

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
        
        원본 FCVAE 방식:
        - Dataset은 원본 데이터만 반환 (증강 없음)
        - 데이터 증강은 Training loop의 batch_data_augmentation()에서 처리
        - use_label에 따라 값 전처리만 수행

        Args:
            idx: 윈도우 인덱스

        Returns:
            Dictionary with:
                - values: [window_size] 시계열 값 (전처리됨)
                - labels: [window_size] 라벨 (y_all)
                - missing: [window_size] 결측 마스크 (z_all)
                - scenario, node: 메타데이터
        """
        window_info = self.windows[idx]

        # 윈도우 데이터 추출
        window_data = self.df.loc[
            window_info['start_idx']:window_info['end_idx']
        ]

        # 윈도우 크기 검증
        if len(window_data) != self.window_size:
            raise RuntimeError(
                f"Window size mismatch: expected {self.window_size}, "
                f"got {len(window_data)} for window {idx}"
            )

        # 기본 데이터
        values = torch.FloatTensor(window_data['ValueScaled'].values).clone()
        labels = torch.FloatTensor(window_data['Label'].values)
        missing = torch.zeros(self.window_size)  # LeakDB는 결측값 없음
        
        # === 원본 FCVAE 방식: use_label에 따른 전처리 ===
        if (self.mode == "train" or self.mode == "valid"):
            if self.use_label:
                # Semi-supervised: 실제 이상치 위치의 값을 0으로
                values[labels == 1] = 0.0
            else:
                # Unsupervised: 모든 라벨을 0으로 (라벨 무시)
                labels = torch.zeros_like(labels)
        # Test mode: 항상 원본 라벨 유지 (use_label 설정 무시)

        return {
            'values': values,           # [window_size] - 전처리된 값
            'labels': labels,           # [window_size] - y_all
            'missing': missing,         # [window_size] - z_all
            'scenario': window_info['scenario'],
            'node': window_info['node']
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