"""
LeakDB 데이터를 하나의 통합 CSV로 변환

각 (Scenario, Node) → 실제 수요(Demands + Leak) 계산
통합 CSV 형식: ['Scenario', 'Node', 'Timestamp', 'Value', 'Label']
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore')


class LeakDBLoader:
    """LeakDB 데이터를 하나의 통합 CSV로 변환"""

    def __init__(self, dataset_root: str):
        """
        Args:
            dataset_root: LeakDB 루트 (예: 'leakdb/Net1_CMH')
        """
        self.dataset_root = Path(dataset_root)
        self.network_name = self.dataset_root.name

    def get_leak_nodes(self, scenario_num: int) -> set:
        """
        시나리오의 누수 발생 노드 목록 추출

        Returns:
            set of leak node IDs (as strings)
        """
        scenario_dir = self.dataset_root / f"Scenario-{scenario_num}"
        leaks_dir = scenario_dir / "Leaks"

        leak_nodes = set()

        if leaks_dir.exists():
            info_files = list(leaks_dir.glob("Leak_*_info.csv"))

            for info_file in info_files:
                # 파일명에서 노드 번호 추출: Leak_13_info.csv -> '13'
                node_id = info_file.stem.split('_')[1]
                leak_nodes.add(node_id)

        return leak_nodes

    def load_scenario(self, scenario_num: int) -> pd.DataFrame:
        """
        전체 시나리오를 한 번에 로드

        Returns:
            DataFrame with [Scenario, Node, Timestamp, Value, Label]
        """
        scenario_dir = self.dataset_root / f"Scenario-{scenario_num}"

        # 공통 데이터 한 번만 로드
        timestamps = pd.read_csv(scenario_dir / "Timestamps.csv", engine='pyarrow')['Timestamp'].values
        global_labels = pd.read_csv(scenario_dir / "Labels.csv", engine='pyarrow')['Label'].values

        # 누수 노드 목록 추출
        leak_nodes = self.get_leak_nodes(scenario_num)

        # 노드 ID 리스트
        demands_dir = scenario_dir / "Demands"
        node_files = sorted(demands_dir.glob("Node_*.csv"))
        node_ids = [f.stem.split('_')[1] for f in node_files]

        # 모든 노드 데이터를 담을 리스트
        all_node_dfs = []

        for node_id in node_ids:
            # 정상 수요
            normal_demand = pd.read_csv(
                scenario_dir / "Demands" / f"Node_{node_id}.csv",
                usecols=['Value']
            )['Value'].values

            # 누수 여부 확인
            has_leak = node_id in leak_nodes

            if has_leak:
                # 누수가 있는 노드: 전역 라벨 사용 + 누수 수요 추가
                labels = global_labels

                # 누수 수요 로드
                leak_file = scenario_dir / "Leaks" / f"Leak_{node_id}_demand.csv"
                if leak_file.exists():
                    leak_demand = pd.read_csv(leak_file, usecols=['Value'], engine='pyarrow')['Value'].values
                else:
                    leak_demand = np.zeros(len(normal_demand))
            else:
                # 누수가 없는 노드: 모두 Label=0, 누수 수요 없음
                labels = np.zeros(len(timestamps))
                leak_demand = np.zeros(len(normal_demand))

            # 실제 총 수요
            total_demand = normal_demand + leak_demand

            # DataFrame 생성
            node_df = pd.DataFrame({
                'Scenario': scenario_num,
                'Node': int(node_id),
                'Timestamp': timestamps,
                'Value': total_demand,
                'Label': labels
            })

            all_node_dfs.append(node_df)

        # 모든 노드 데이터 병합
        scenario_df = pd.concat(all_node_dfs, ignore_index=True)

        return scenario_df

    def create_unified_csv(
        self,
        scenario_range: tuple,
        output_file: str,
        chunk_size: int = 50,
        parallel: bool = False,
        n_workers: int = None
    ):
        """
        모든 시나리오/노드를 하나의 CSV로 통합

        Args:
            scenario_range: (start, end) 예: (1, 1001) for scenarios 1-1000
            output_file: 출력 CSV 경로
            chunk_size: 메모리 절약을 위한 청크 크기
            parallel: 병렬 처리 사용 여부
            n_workers: 병렬 처리 워커 수 (None이면 CPU 코어 수)
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Network: {self.network_name}")
        print(f"Scenarios: {scenario_range[0]} to {scenario_range[1]-1}")
        print(f"Output: {output_file}")
        print(f"Parallel processing: {parallel}")
        if parallel:
            n_workers = n_workers or cpu_count()
            print(f"Workers: {n_workers}")
        print(f"{'='*60}\n")

        # CSV 파일 초기화 (헤더만 작성)
        first_df = pd.DataFrame(columns=['Scenario', 'Node', 'Timestamp', 'Value', 'Label'])
        first_df.to_csv(output_file, index=False, mode='w')

        total_rows = 0
        error_count = 0

        scenario_nums = list(range(scenario_range[0], scenario_range[1]))

        if parallel:
            # 병렬 처리
            with Pool(n_workers) as pool:
                # 청크 단위로 처리
                for chunk_start in tqdm(
                    range(0, len(scenario_nums), chunk_size),
                    desc="Processing chunks"
                ):
                    chunk_end = min(chunk_start + chunk_size, len(scenario_nums))
                    chunk_scenarios = scenario_nums[chunk_start:chunk_end]

                    try:
                        # 병렬로 시나리오 로드
                        chunk_dfs = pool.map(self.load_scenario, chunk_scenarios)

                        # 병합 및 저장
                        if chunk_dfs:
                            chunk_combined = pd.concat(chunk_dfs, ignore_index=True)
                            chunk_combined.to_csv(output_file, index=False, mode='a', header=False)

                            total_rows += len(chunk_combined)
                            del chunk_dfs, chunk_combined

                    except Exception as e:
                        error_count += 1
                        print(f"\n⚠ Error in chunk {chunk_start}-{chunk_end}: {e}")
        else:
            # 순차 처리
            for chunk_start in tqdm(
                range(0, len(scenario_nums), chunk_size),
                desc="Processing chunks"
            ):
                chunk_end = min(chunk_start + chunk_size, len(scenario_nums))
                chunk_scenarios = scenario_nums[chunk_start:chunk_end]

                chunk_dfs = []

                for scenario_num in chunk_scenarios:
                    try:
                        df = self.load_scenario(scenario_num)
                        chunk_dfs.append(df)

                    except Exception as e:
                        error_count += 1
                        print(f"\n⚠ Error in Scenario-{scenario_num}: {e}")
                        continue

                # 청크 병합 및 저장
                if chunk_dfs:
                    chunk_combined = pd.concat(chunk_dfs, ignore_index=True)
                    chunk_combined.to_csv(output_file, index=False, mode='a', header=False)

                    total_rows += len(chunk_combined)
                    del chunk_dfs, chunk_combined

        print(f"\n{'='*60}")
        print(f"✓ Unified CSV created: {output_file}")
        print(f"{'='*60}")

        # 파일 크기 확인
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"  Total rows: {total_rows:,}")
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"  Errors: {error_count}")

        # 샘플 데이터 확인
        print(f"\n{'='*60}")
        print(f"Sample data (first 10 rows):")
        print(f"{'='*60}")
        sample = pd.read_csv(output_file, nrows=10)
        print(sample)

        # 통계
        print(f"\n{'='*60}")
        print(f"Data statistics:")
        print(f"{'='*60}")
        full_df = pd.read_csv(output_file, engine='pyarrow')
        print(f"  Unique scenarios: {full_df['Scenario'].nunique()}")
        print(f"  Unique nodes: {full_df['Node'].nunique()}")
        print(f"  Date range: {full_df['Timestamp'].min()} to {full_df['Timestamp'].max()}")

        print(f"\n  Overall label distribution:")
        print(f"    - Normal (0): {(full_df['Label'] == 0).sum():,} ({(full_df['Label'] == 0).sum() / len(full_df) * 100:.2f}%)")
        print(f"    - Leak (1):   {(full_df['Label'] == 1).sum():,} ({(full_df['Label'] == 1).sum() / len(full_df) * 100:.2f}%)")

        # 노드별 누수 발생 통계
        leak_node_count = full_df[full_df['Label'] == 1].groupby('Node').size()
        if len(leak_node_count) > 0:
            print(f"\n  Nodes with leaks (leak timesteps):")
            for node, count in leak_node_count.items():
                print(f"    - Node {node}: {count:,} leak timesteps")
        else:
            print(f"\n  No leaks found in processed scenarios")

        print(f"{'='*60}\n")


def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='LeakDB 데이터 통합 CSV 생성')
    parser.add_argument('--dataset', type=str, default='Net1_CMH',
                        help='Dataset name (Net1_CMH or Hanoi_CMH)')
    parser.add_argument('--scenario_start', type=int, default=1,
                        help='시작 시나리오 번호')
    parser.add_argument('--scenario_end', type=int, default=11,
                        help='종료 시나리오 번호 (exclusive)')
    parser.add_argument('--chunk_size', type=int, default=50,
                        help='청크 크기 (한 번에 처리할 시나리오 수)')
    parser.add_argument('--parallel', action='store_true',
                        help='병렬 처리 사용')
    parser.add_argument('--workers', type=int, default=None,
                        help='병렬 처리 워커 수 (기본값: CPU 코어 수)')
    parser.add_argument('--output', type=str, default=None,
                        help='출력 파일 경로 (기본값: data/processed/{dataset}_full.csv)')

    args = parser.parse_args()

    # 출력 파일 경로 설정
    if args.output is None:
        output_file = f'data/processed/{args.dataset.lower()}_full.csv'
    else:
        output_file = args.output

    # 데이터 로더 생성
    loader = LeakDBLoader(f'leakdb/{args.dataset}')

    # 통합 CSV 생성
    loader.create_unified_csv(
        scenario_range=(args.scenario_start, args.scenario_end),
        output_file=output_file,
        chunk_size=args.chunk_size,
        parallel=args.parallel,
        n_workers=args.workers
    )


if __name__ == '__main__':
    main()
