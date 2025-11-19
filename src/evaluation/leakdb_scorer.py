"""
LeakDB Scoring Algorithm
- F1 Score
- STPR (Sensitivity / True Positive Rate)
- STNR (Specificity / True Negative Rate)
- SED (Early Detection Score)
- MCC (Matthews Correlation Coefficient)

Based on LeakDB official scoring_algorithm.m
"""

import numpy as np
from sklearn.metrics import matthews_corrcoef
from typing import Dict, Tuple


class LeakDBScorer:
    """
    LeakDB 공식 평가 메트릭 구현
    
    Reference: LeakDB/scoring_algorithm.m
    """
    
    def __init__(self, tw_ex: int = 10, sed_thr: float = 0.75):
        """
        Args:
            tw_ex: Early detection window 확장 (timesteps)
                   Default: 10 (5시간, 30분 간격)
            sed_thr: SED 계산 시 detection persistence 임계값
                     Default: 0.75 (75%)
        """
        self.tw_ex = tw_ex
        self.sed_thr = sed_thr
    
    def compute_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        F1 Score 계산
        
        F1 = 2 * TP / (2 * TP + FP + FN) * 100
        
        Args:
            y_true: Ground truth labels [N]
            y_pred: Predicted labels [N]
        
        Returns:
            F1 score (0-100)
        """
        # Flatten arrays to ensure 1D
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        
        denominator = 2 * TP + FP + FN
        if denominator == 0:
            return 0.0
        
        f1 = (2 * TP / denominator) * 100
        return f1
    
    def compute_stpr(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        STPR (Sensitivity / True Positive Rate) 계산
        
        STPR = TP / (TP + FN) * 100
        
        Args:
            y_true: Ground truth labels [N]
            y_pred: Predicted labels [N]
        
        Returns:
            STPR (0-100)
        """
        # Flatten arrays to ensure 1D
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        tp_plus_fn = np.sum(y_true == 1)
        
        if tp_plus_fn == 0:
            return 0.0
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        stpr = (tp / tp_plus_fn) * 100
        
        return stpr
    
    def compute_stnr(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        STNR (Specificity / True Negative Rate) 계산
        
        STNR = TN / (TN + FP) * 100
        
        Args:
            y_true: Ground truth labels [N]
            y_pred: Predicted labels [N]
        
        Returns:
            STNR (0-100)
        """
        # Flatten arrays to ensure 1D
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        tn_plus_fp = np.sum(y_true == 0)
        
        if tn_plus_fp == 0:
            return 0.0
        
        tn = np.sum((y_true == 0) & (y_pred == 0))
        stnr = (tn / tn_plus_fp) * 100
        
        return stnr
    
    def compute_sed(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        SED (Early Detection Score) 계산
        
        조기 탐지 성능을 평가하는 메트릭:
        1. 각 fault에 대해 detection window 설정
        2. 첫 탐지 시점까지의 delay 계산
        3. Detection persistence 확인
        4. 지수 감쇠 점수 부여
        
        Args:
            y_true: Ground truth labels [N]
            y_pred: Predicted labels [N]
        
        Returns:
            SED score (0-100)
        """
        # Flatten arrays to ensure 1D
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # Fault 시작/종료 지점 찾기
        padded_true = np.concatenate([[0], y_true, [0]])
        diff = np.diff(padded_true.astype(int))
        
        fault_starts = np.where(diff == 1)[0]
        fault_ends = np.where(diff == -1)[0] - 1
        
        if len(fault_starts) == 0:
            # No faults in ground truth
            return 0.0
        
        scores = []
        ideal_scores = []
        
        for start, end in zip(fault_starts, fault_ends):
            # Detection window: [start, end + tw_ex]
            tw_end = min(end + self.tw_ex, len(y_true))
            tw = tw_end - start
            
            if tw == 0:
                continue
            
            detection_window = y_pred[start:tw_end]
            
            # 첫 탐지 시점 찾기
            first_detect_idx = np.where(detection_window == 1)[0]
            
            if len(first_detect_idx) == 0:
                # No detection in window
                scores.append(0)
            else:
                dt = first_detect_idx[0]  # Detection delay
                
                # Detection persistence 확인
                # dt 이후 detection의 평균이 임계값 이상이어야 함
                if np.mean(detection_window[dt:]) > self.sed_thr:
                    # 지수 감쇠 점수: 빠를수록 높은 점수
                    score = 2 / (1 + np.exp((5 / tw) * dt))
                    scores.append(score)
                else:
                    # Persistence 부족
                    scores.append(0)
            
            ideal_scores.append(1)
        
        if sum(ideal_scores) == 0:
            return 0.0
        
        sed = (sum(scores) / sum(ideal_scores)) * 100
        return sed
    
    def compute_mcc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        MCC (Matthews Correlation Coefficient) 계산
        
        MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
        
        Args:
            y_true: Ground truth labels [N]
            y_pred: Predicted labels [N]
        
        Returns:
            MCC (-1 to 1)
        """
        return matthews_corrcoef(y_true, y_pred)
    
    def score_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        모든 메트릭 계산
        
        Args:
            y_true: Ground truth labels [N]
            y_pred: Predicted labels [N]
        
        Returns:
            Dictionary with all metrics
        """
        results = {
            'F1': self.compute_f1(y_true, y_pred),
            'STPR': self.compute_stpr(y_true, y_pred),
            'STNR': self.compute_stnr(y_true, y_pred),
            'SED': self.compute_sed(y_true, y_pred),
            'MCC': self.compute_mcc(y_true, y_pred)
        }
        
        return results
    
    def print_results(self, results: Dict[str, float]):
        """결과 출력"""
        print("\n" + "=" * 70)
        print("LeakDB Evaluation Results")
        print("=" * 70)
        print(f"F1 Score:  {results['F1']:6.2f}%")
        print(f"STPR:      {results['STPR']:6.2f}%  (Sensitivity / True Positive Rate)")
        print(f"STNR:      {results['STNR']:6.2f}%  (Specificity / True Negative Rate)")
        print(f"SED:       {results['SED']:6.2f}%  (Early Detection Score)")
        print(f"MCC:       {results['MCC']:7.4f}   (Matthews Correlation Coefficient)")
        print("=" * 70)


# ===== 사용 예시 =====
if __name__ == '__main__':
    # 테스트 데이터
    np.random.seed(42)
    
    # Simulated fault: timesteps 100-200
    y_true = np.zeros(500)
    y_true[100:200] = 1
    
    # Simulated detection with delay
    y_pred = np.zeros(500)
    y_pred[110:220] = 1  # Detection starts at 110 (delay=10)
    
    # Scorer
    scorer = LeakDBScorer(tw_ex=10, sed_thr=0.75)
    
    # 평가
    results = scorer.score_all(y_true, y_pred)
    scorer.print_results(results)
    
    print("\nTest metrics breakdown:")
    print(f"  True Positives:  {np.sum((y_true == 1) & (y_pred == 1))}")
    print(f"  True Negatives:  {np.sum((y_true == 0) & (y_pred == 0))}")
    print(f"  False Positives: {np.sum((y_true == 0) & (y_pred == 1))}")
    print(f"  False Negatives: {np.sum((y_true == 1) & (y_pred == 0))}")
