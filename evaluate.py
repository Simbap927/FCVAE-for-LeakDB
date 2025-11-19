"""
FCVAE Evaluation Script for LeakDB
- 학습된 모델로 테스트 셋 평가
- LeakDB 공식 메트릭 계산 (F1, STPR, STNR, SED, MCC)
- MCMC inference로 비지도 이상 탐지

Optimization Notes (v2):
- Reduced threshold search from 1000 to 200 (5x speedup)
- Use percentile-based threshold range for robustness
- Vectorized scoring operations where possible
- Fixed test mode to preserve real labels (not zeroed)
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data.dataset import LeakDBDataset
from training.trainer import FCVAETrainer
from evaluation.leakdb_scorer import LeakDBScorer


def point_adjust(predict: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """
    Point Adjustment (원본 FCVAE 방식)
    이상 구간 내 일부만 탐지해도 전체 구간을 탐지한 것으로 간주
    
    Args:
        predict: Binary predictions [N]
        actual: Binary ground truth [N]
    
    Returns:
        Adjusted predictions [N]
    """
    predict = predict.copy()
    anomaly_state = False
    
    for i in range(len(predict)):
        if actual[i] and predict[i] and not anomaly_state:
            # 이상 구간 시작 지점 탐지 시
            anomaly_state = True
            # 역방향으로 이상 구간 시작점까지 True로 변경
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                predict[j] = True
        elif not actual[i]:
            anomaly_state = False
        
        if anomaly_state:
            predict[i] = True
    
    return predict


def find_optimal_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    metric: str = 'f1',
    n_thresholds: int = 200,  # Reduced from 1000 for speed
    use_point_adjust: bool = True
) -> tuple:
    """
    최적 임계값 찾기 (최적화 버전)
    
    Args:
        scores: Anomaly scores [N]
        labels: Ground truth labels [N]
        metric: Optimization metric ('f1', 'stpr', 'stnr')
        n_thresholds: Number of thresholds to try (reduced for speed)
        use_point_adjust: Apply point adjustment (FCVAE 원본 방식)
    
    Returns:
        (best_threshold, best_score, all_results)
    """
    # Use percentile range like reference FCVAE (more robust)
    max_th = np.percentile(scores, 99.5)
    min_th = float(scores.min())
    thresholds = np.linspace(min_th, max_th, n_thresholds)
    
    scorer = LeakDBScorer()
    labels_int = labels.astype(int)
    
    best_score = 0
    best_threshold = 0
    all_results = []
    
    # Vectorized computation where possible
    for threshold in tqdm(thresholds, desc=f"Finding optimal threshold ({metric})", 
                          ncols=80, leave=False):
        predictions = (scores > threshold).astype(int)
        
        # Point Adjustment 적용 (원본 FCVAE 방식)
        if use_point_adjust:
            predictions = point_adjust(predictions, labels_int)
        
        if metric == 'f1':
            score = scorer.compute_f1(labels_int, predictions)
        elif metric == 'stpr':
            score = scorer.compute_stpr(labels_int, predictions)
        elif metric == 'stnr':
            score = scorer.compute_stnr(labels_int, predictions)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        all_results.append({
            'threshold': threshold,
            'score': score
        })
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score, all_results


def evaluate(
    checkpoint_path: str,
    test_csv: str,
    batch_size: int = 128,
    num_workers: int = 4,
    device: str = 'cuda',
    save_dir: str = 'evaluation_results'
):
    """
    메인 평가 함수
    
    Args:
        checkpoint_path: 모델 체크포인트 경로
        test_csv: 테스트 데이터 CSV 경로
        batch_size: 배치 크기
        num_workers: DataLoader workers
        device: 'cuda' or 'cpu'
        save_dir: 결과 저장 디렉토리
    """
    print("\n" + "=" * 70)
    print("FCVAE Evaluation on LeakDB")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test data:  {test_csv}")
    print(f"Device:     {device}")
    print("=" * 70)
    
    # 디렉토리 생성
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 모델 로드
    print("\n[1/4] Loading model...")
    model = FCVAETrainer.load_from_checkpoint(
        checkpoint_path,
        map_location=device
    )
    model.eval()
    model.to(device)
    print(f"✓ Model loaded from {checkpoint_path}")
    
    # 2. 데이터셋 로드
    print("\n[2/4] Loading test dataset...")
    test_dataset = LeakDBDataset(
        csv_file=test_csv,
        window_size=model.hparams.window_size,
        stride=1,  # 평가 시에는 모든 윈도우 사용
        mode='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False
    )
    
    print(f"✓ Test dataset: {len(test_dataset):,} windows")
    print(f"✓ Test batches: {len(test_loader):,}")
    
    # 3. 추론 (MCMC)
    print("\n[3/4] Running MCMC inference...")
    
    # 원본 FCVAE 방식: 각 윈도우의 마지막 타임스텝만 예측
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            # 모든 텐서를 device로 이동
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            x = batch['values']  # [batch, window_size]
            y_all = batch['labels']  # [batch, window_size]
            
            # 마지막 타임스텝의 라벨 (윈도우가 예측하는 타깃)
            y_last = y_all[:, -1].cpu().numpy()  # [batch]
            
            # MCMC 추론으로 이상 스코어 계산
            result = model.test_step(batch, 0)
            
            # Anomaly scores (이미 -log prob로 계산됨)
            window_scores = result['scores'].cpu().numpy()  # [batch]
            
            all_scores.extend(window_scores)
            all_labels.extend(y_last)
    
    # NumPy 배열로 변환
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    print(f"✓ Inference complete")
    print(f"  Total windows:   {len(all_scores):,}")
    print(f"  Normal windows:  {(all_labels == 0).sum():,}")
    print(f"  Anomaly windows: {(all_labels == 1).sum():,}")
    
    # 4. 평가
    print("\n[4/4] Evaluating...")
    
    # 최적 임계값 찾기 (F1 최대화) - 200개로 축소하여 속도 개선
    best_threshold, best_f1, threshold_results = find_optimal_threshold(
        all_scores,
        all_labels,
        metric='f1',
        n_thresholds=200  # Reduced from 1000 for 5x speedup
    )
    
    print(f"\n✓ Optimal threshold: {best_threshold:.6f} (F1={best_f1:.2f}%)")
    
    # 최적 임계값으로 예측 (Point Adjustment 적용)
    final_predictions = (all_scores > best_threshold).astype(int)
    final_predictions = point_adjust(final_predictions, all_labels)
    
    # LeakDB 메트릭 계산
    scorer = LeakDBScorer(tw_ex=10, sed_thr=0.75)
    results = scorer.score_all(all_labels, final_predictions)
    
    # 결과 출력
    print("\n" + "=" * 70)
    print("LeakDB Metrics (with Point Adjustment)")
    print("=" * 70)
    scorer.print_results(results)
    
    # Confusion Matrix
    TP = np.sum((all_labels == 1) & (final_predictions == 1))
    TN = np.sum((all_labels == 0) & (final_predictions == 0))
    FP = np.sum((all_labels == 0) & (final_predictions == 1))
    FN = np.sum((all_labels == 1) & (final_predictions == 0))
    
    print("\nConfusion Matrix:")
    print(f"  True Positives:  {TP:,}")
    print(f"  True Negatives:  {TN:,}")
    print(f"  False Positives: {FP:,}")
    print(f"  False Negatives: {FN:,}")
    
    # 5. 결과 저장
    print(f"\n[5/5] Saving results to {save_dir}/...")
    
    # 결과 텍스트 저장
    with open(save_dir / 'evaluation_results.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("FCVAE Evaluation Results\n")
        f.write("=" * 70 + "\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Test data:  {test_csv}\n")
        f.write(f"\nOptimal threshold: {best_threshold:.6f}\n")
        f.write(f"\nLeakDB Metrics:\n")
        f.write(f"  F1 Score:  {results['F1']:6.2f}%\n")
        f.write(f"  STPR:      {results['STPR']:6.2f}%\n")
        f.write(f"  STNR:      {results['STNR']:6.2f}%\n")
        f.write(f"  SED:       {results['SED']:6.2f}%\n")
        f.write(f"  MCC:       {results['MCC']:7.4f}\n")
        f.write(f"\nConfusion Matrix:\n")
        f.write(f"  TP: {TP:,}, TN: {TN:,}\n")
        f.write(f"  FP: {FP:,}, FN: {FN:,}\n")
    
    # NumPy 배열 저장
    np.savez(
        save_dir / 'predictions.npz',
        scores=all_scores,
        predictions=final_predictions,
        labels=all_labels,
        threshold=best_threshold
    )
    
    # 시각화
    print("\nGenerating plots...")
    
    # 1. Score distribution (샘플링하여 메모리 효율화)
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    # 대용량 데이터의 경우 샘플링
    max_hist_samples = 100000
    normal_scores = all_scores[all_labels == 0]
    anomaly_scores = all_scores[all_labels == 1]
    
    if len(normal_scores) > max_hist_samples:
        normal_scores = np.random.choice(normal_scores, max_hist_samples, replace=False)
    if len(anomaly_scores) > max_hist_samples:
        anomaly_scores = np.random.choice(anomaly_scores, max_hist_samples, replace=False)
    
    plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='blue')
    plt.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', color='red')
    plt.axvline(best_threshold, color='green', linestyle='--', label=f'Threshold={best_threshold:.3f}')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Count')
    plt.title('Anomaly Score Distribution (Sampled)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Threshold vs F1
    plt.subplot(1, 3, 2)
    thresholds = [r['threshold'] for r in threshold_results]
    f1_scores = [r['score'] for r in threshold_results]
    plt.plot(thresholds, f1_scores, linewidth=2)
    plt.axvline(best_threshold, color='red', linestyle='--', label=f'Best: {best_threshold:.3f}')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score (%)')
    plt.title(f'Threshold Optimization (Best F1={best_f1:.2f}%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Confusion Matrix
    plt.subplot(1, 3, 3)
    cm = np.array([[TN, FP], [FN, TP]])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred Normal', 'Pred Anomaly'],
                yticklabels=['True Normal', 'True Anomaly'])
    plt.title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'evaluation_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Results saved to {save_dir}/")
    print(f"  - evaluation_results.txt")
    print(f"  - predictions.npz")
    print(f"  - evaluation_plots.png")
    
    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="FCVAE Evaluation Script for LeakDB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.ckpt file)'
    )
    
    parser.add_argument(
        '--test_csv',
        type=str,
        required=True,
        help='Path to test CSV file'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=512,
        help='Batch size for inference'
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of DataLoader workers'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )
    
    parser.add_argument(
        '--save_dir',
        type=str,
        default='evaluation_results',
        help='Directory to save results'
    )
    
    args = parser.parse_args()
    
    # 파일 존재 확인
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return 1
    
    if not Path(args.test_csv).exists():
        print(f"Error: Test CSV file not found: {args.test_csv}")
        return 1
    
    # 평가 실행
    try:
        evaluate(
            checkpoint_path=args.checkpoint,
            test_csv=args.test_csv,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            save_dir=args.save_dir
        )
        return 0
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
