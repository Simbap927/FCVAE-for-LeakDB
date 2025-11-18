"""
LeakDB 데이터 전처리 모듈
- Timestamp 변환 (문자열 → datetime)
- 노드별 스케일링 (Z-score / Min-Max)
- FCVAE 전처리 로직을 LeakDB에 맞게 수정
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
from typing import Literal, Optional


class LeakDBPreprocessor:
    """LeakDB 데이터 전처리 클래스 (노드별 스케일링)"""

    def __init__(
        self,
        scaling_method: Literal['standard', 'minmax', 'none'] = 'standard'
    ):
        """
        Args:
            scaling_method: 스케일링 방법 (노드별로 적용)
                - 'standard': Z-score 표준화 (mean=0, std=1) [권장]
                - 'minmax': Min-Max 정규화 (range=[0, 1])
                - 'none': 스케일링 없음 (원본 값 사용)
        """
        self.scaling_method = scaling_method

        # 노드별 scaler 저장 (fit 시 생성)
        self.scalers = {}

    def timestamp_to_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        타임스탬프를 datetime 객체로 변환
        - 문자열 → datetime 객체

        Args:
            df: DataFrame with 'Timestamp' column

        Returns:
            DataFrame with 'Timestamp' as datetime type
        """
        df = df.copy()

        # 문자열 → datetime
        if df['Timestamp'].dtype == 'object':
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        return df

    def scale_data(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        노드별 데이터 스케일링
        - 각 노드는 서로 다른 수요 패턴과 규모를 가짐
        - Train 데이터로 scaler를 fit하고, Val/Test는 transform만 수행

        Args:
            df: DataFrame with 'Value', 'Node' columns
            fit: True면 scaler를 학습, False면 기존 scaler 사용

        Returns:
            DataFrame with added 'ValueScaled' column
        """
        df = df.copy()

        if self.scaling_method == 'none':
            df['ValueScaled'] = df['Value']
            return df

        # 노드별 스케일링
        scaled_values = []

        for node in sorted(df['Node'].unique()):
            mask = df['Node'] == node
            node_values = df.loc[mask, 'Value'].values.reshape(-1, 1)

            if fit:
                # Scaler 생성 및 학습
                if self.scaling_method == 'standard':
                    scaler = StandardScaler()  # mean=0, std=1
                elif self.scaling_method == 'minmax':
                    scaler = MinMaxScaler()    # range=[0, 1]
                
                scaled = scaler.fit_transform(node_values)
                self.scalers[node] = scaler
            else:
                # 기존 scaler 사용
                if node not in self.scalers:
                    raise ValueError(
                        f"Scaler for node {node} not found. "
                        f"Run with fit=True on training data first."
                    )
                scaled = self.scalers[node].transform(node_values)

            scaled_values.append((mask, scaled.flatten()))

        # 결과 병합
        df['ValueScaled'] = 0.0
        for mask, values in scaled_values:
            df.loc[mask, 'ValueScaled'] = values

        return df

    def process(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        전체 전처리 파이프라인 실행
        1. Timestamp → datetime
        2. 노드별 스케일링
        3. 컬럼 순서 정리

        Args:
            df: 원본 DataFrame
            fit: True면 scaler 학습 (train), False면 적용만 (val/test)

        Returns:
            전처리된 DataFrame
        """
        print(f"Preprocessing {'(fit)' if fit else '(transform only)'}...")
        print(f"  Input: {len(df):,} rows")
        print(f"  Scaling method: {self.scaling_method} (per-node)")

        # 1. Timestamp → datetime
        df = self.timestamp_to_datetime(df)

        # 2. 노드별 스케일링
        df = self.scale_data(df, fit=fit)

        # 3. 컬럼 순서 정리: [Scenario, Node, Timestamp, Value, ValueScaled, Label]
        column_order = ['Scenario', 'Node', 'Timestamp', 'Value', 'ValueScaled', 'Label']
        df = df[column_order]

        print(f"  ✓ Processed: {len(df):,} rows")
        print(f"  ✓ Columns: {list(df.columns)}")

        # 통계 출력
        if fit and self.scaling_method != 'none':
            print(f"\n  Statistics (after scaling):")
            print(f"    ValueScaled mean: {df['ValueScaled'].mean():.4f}")
            print(f"    ValueScaled std:  {df['ValueScaled'].std():.4f}")
            print(f"    ValueScaled min:  {df['ValueScaled'].min():.4f}")
            print(f"    ValueScaled max:  {df['ValueScaled'].max():.4f}")

        return df

    def save_scalers(self, filepath: str):
        """
        Scaler 저장 (재사용을 위해)

        Args:
            filepath: 저장할 파일 경로 (.pkl)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(self.scalers, f)

        print(f"  ✓ Scalers saved: {filepath}")

    def load_scalers(self, filepath: str):
        """
        저장된 Scaler 로드

        Args:
            filepath: 로드할 파일 경로 (.pkl)
        """
        with open(filepath, 'rb') as f:
            self.scalers = pickle.load(f)

        print(f"  ✓ Scalers loaded: {filepath}")
        print(f"    Number of scalers: {len(self.scalers)}")


def preprocess_all_splits(
    data_dir: str,
    output_dir: Optional[str] = None,
    scaling_method: str = 'standard',
    save_scalers: bool = True
):
    """
    Train/Val/Test 전체 분할에 대해 전처리 수행
    
    Args:
        data_dir: train.csv, val.csv, test.csv가 있는 디렉토리
        output_dir: 출력 디렉토리 (None이면 data_dir와 동일)
        scaling_method: 스케일링 방법 ('standard', 'minmax', 'none')
        save_scalers: Scaler 저장 여부
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir) if output_dir else data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LeakDB Data Preprocessing")
    print("=" * 70)
    print(f"Input directory:  {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Scaling method:   {scaling_method} (per-node)")
    print()

    # Preprocessor 초기화
    preprocessor = LeakDBPreprocessor(
        scaling_method=scaling_method
    )

    # 1. Train 데이터 전처리 (fit)
    print("-" * 70)
    print("Processing Train data...")
    print("-" * 70)
    train_df = pd.read_csv(data_dir / 'train.csv')
    train_processed = preprocessor.process(train_df, fit=True)
    train_output = output_dir / 'train_processed.csv'
    train_processed.to_csv(train_output, index=False)
    print(f"  ✓ Saved: {train_output}")
    print()

    # Scaler 저장
    if save_scalers:
        scaler_path = output_dir / 'scalers.pkl'
        preprocessor.save_scalers(scaler_path)
        print()

    # 2. Val 데이터 전처리 (transform only)
    print("-" * 70)
    print("Processing Val data...")
    print("-" * 70)
    val_df = pd.read_csv(data_dir / 'val.csv')
    val_processed = preprocessor.process(val_df, fit=False)
    val_output = output_dir / 'val_processed.csv'
    val_processed.to_csv(val_output, index=False)
    print(f"  ✓ Saved: {val_output}")
    print()

    # 3. Test 데이터 전처리 (transform only)
    print("-" * 70)
    print("Processing Test data...")
    print("-" * 70)
    test_df = pd.read_csv(data_dir / 'test.csv')
    test_processed = preprocessor.process(test_df, fit=False)
    test_output = output_dir / 'test_processed.csv'
    test_processed.to_csv(test_output, index=False)
    print(f"  ✓ Saved: {test_output}")
    print()

    # 파일 크기 요약
    print("=" * 70)
    print("Summary:")
    print("=" * 70)
    for path in [train_output, val_output, test_output]:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"{path.name:25s}: {size_mb:7.2f} MB")
    print()

    # 샘플 데이터 확인
    print("-" * 70)
    print("Sample data (Train, first 5 rows):")
    print("-" * 70)
    print(train_processed.head())
    print()

    print("=" * 70)
    print("Preprocessing complete!")
    print("=" * 70)


# ===== 사용 예시 =====
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Preprocess LeakDB data (Train/Val/Test)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing train.csv, val.csv, test.csv'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: same as data-dir)'
    )
    parser.add_argument(
        '--scaling-method',
        type=str,
        default='standard',
        choices=['standard', 'minmax', 'none'],
        help='Scaling method per node (default: standard=Z-score)'
    )
    parser.add_argument(
        '--no-save-scalers',
        action='store_true',
        help='Do not save scalers'
    )

    args = parser.parse_args()

    preprocess_all_splits(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        scaling_method=args.scaling_method,
        save_scalers=not args.no_save_scalers
    )

    # 예시 실행 커맨드:
    # python -m src.data.preprocessor \
    #     --data-dir data/processed/net1 \
    #     --scaling-method standard  # Z-score (권장)
