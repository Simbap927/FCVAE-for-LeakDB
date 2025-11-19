"""
데이터 증강 기법 모음
- 결측값 주입, 포인트 이상, 세그먼트 이상 생성
- 학습 데이터 다양성 증대를 통한 모델 일반화 성능 향상
- 원본 FCVAE 구현을 LeakDB에 맞게 조정
"""

import numpy as np
import torch


def missing_data_injection(x, y, z, rate):
    """
    결측값 주입: 무작위 위치에 결측값 추가
    
    Args:
        x: 입력 데이터 [batch, window_size]
        y: 이상 라벨 [batch, window_size]
        z: 결측 마스크 [batch, window_size]
        rate: 결측 비율 (예: 0.01 = 1%)
    
    Returns:
        x, y, z: 결측값이 주입된 데이터
    """
    batch_size, window_size = x.shape
    miss_size = int(rate * batch_size * window_size)
    
    if miss_size > 0:
        # 랜덤 위치 선택
        row = torch.randint(low=0, high=batch_size, size=(miss_size,))
        col = torch.randint(low=0, high=window_size, size=(miss_size,))
        
        # 결측값 주입
        x[row, col] = 0.0  # 결측값을 0으로 설정
        z[row, col] = 1.0  # 결측 마스크 업데이트
    
    return x, y, z


def point_anomaly(x, y, z, rate):
    """
    포인트 이상 생성: 마지막 시점에 큰 노이즈 추가
    - 급격한 값 변화 이상 패턴 시뮬레이션
    
    Args:
        x: 입력 데이터 [batch, window_size]
        y: 이상 라벨 [batch, window_size]
        z: 결측 마스크 [batch, window_size]
        rate: 증강 비율 (예: 0.05 = 5% 샘플 증강)
    
    Returns:
        x_aug, y_aug, z_aug: 포인트 이상이 추가된 증강 데이터
    """
    batch_size, window_size = x.shape
    aug_size = int(rate * batch_size)
    
    if aug_size == 0:
        # 증강할 샘플이 없으면 빈 텐서 반환
        return (
            torch.empty(0, window_size, device=x.device),
            torch.empty(0, window_size, device=x.device),
            torch.empty(0, window_size, device=x.device)
        )
    
    # 랜덤 샘플 선택
    idx = torch.randint(low=0, high=batch_size, size=(aug_size,))
    x_aug = x[idx].clone()
    y_aug = y[idx].clone()
    z_aug = z[idx].clone()
    
    # 양수와 음수 노이즈를 반반씩 생성
    n_positive = aug_size // 2
    n_negative = aug_size - n_positive
    
    ano_noise1 = torch.randint(low=1, high=20, size=(n_positive,), device=x.device)
    ano_noise2 = torch.randint(low=-20, high=-1, size=(n_negative,), device=x.device)
    ano_noise = torch.cat((ano_noise1, ano_noise2), dim=0).float() / 2.0
    
    # 마지막 시점에 노이즈 추가
    x_aug[:, -1] += ano_noise
    
    # 마지막 시점을 이상으로 표시
    y_aug[:, -1] = 1.0
    
    return x_aug, y_aug, z_aug


def segment_anomaly(x, y, z, rate, method="swap"):
    """
    세그먼트 이상 생성: 시계열 중간부터 다른 샘플로 교체
    - 지속적인 패턴 변화 이상 시뮬레이션
    
    Args:
        x: 입력 데이터 [batch, window_size]
        y: 이상 라벨 [batch, window_size]
        z: 결측 마스크 [batch, window_size]
        rate: 증강 비율 (예: 0.1 = 10% 샘플 증강)
        method: "swap" (다른 샘플로 교체)
    
    Returns:
        x_aug, y_aug, z_aug: 세그먼트 이상이 추가된 증강 데이터
    """
    batch_size, window_size = x.shape
    aug_size = int(rate * batch_size)
    
    if aug_size == 0:
        return (
            torch.empty(0, window_size, device=x.device),
            torch.empty(0, window_size, device=x.device),
            torch.empty(0, window_size, device=x.device)
        )
    
    # 서로 다른 샘플 쌍 선택
    idx_1 = torch.randint(low=0, high=batch_size, size=(aug_size,))
    idx_2 = torch.randint(low=0, high=batch_size, size=(aug_size,))
    
    # 같은 인덱스가 선택되지 않도록 보장
    for i in range(aug_size):
        while idx_1[i] == idx_2[i]:
            idx_2[i] = torch.randint(low=0, high=batch_size, size=(1,))[0]
    
    x_aug = x[idx_1].clone()
    y_aug = y[idx_1].clone()
    z_aug = z[idx_1].clone()
    
    if method == "swap":
        # 세그먼트 시작 지점 (최소 7 타임스텝 이후)
        time_start = torch.randint(low=7, high=window_size, size=(aug_size,))
        
        for i in range(aug_size):
            start = time_start[i].item()
            # start 이후 구간을 다른 샘플로 교체
            x_aug[i, start:] = x[idx_2[i], start:]
            # 교체된 구간을 이상으로 표시
            y_aug[i, start:] = 1.0
    
    return x_aug, y_aug, z_aug


def batch_data_augmentation(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    missing_rate: float = 0.01,
    point_rate: float = 0.05,
    segment_rate: float = 0.1
) -> tuple:
    """
    배치 단위 데이터 증강 (원본 FCVAE 방식)
    
    순서:
    1. Point Anomaly 생성 및 배치에 추가
    2. Segment Anomaly 생성 및 배치에 추가
    3. 전체 배치에 Missing Data 주입
    
    Args:
        x: [batch, window_size] - 입력 데이터
        y: [batch, window_size] - 이상 라벨 (y_all)
        z: [batch, window_size] - 결측 마스크 (z_all)
        missing_rate: 결측 주입 비율 (default: 0.01 = 1%)
        point_rate: 포인트 이상 증강 비율 (default: 0.05 = 5%)
        segment_rate: 세그먼트 이상 증강 비율 (default: 0.1 = 10%)
    
    Returns:
        x, y, z: 증강된 배치 데이터 (크기가 증가함)
        
    Example:
        >>> x = torch.randn(128, 96)  # 배치 크기 128
        >>> y = torch.zeros(128, 96)
        >>> z = torch.zeros(128, 96)
        >>> x_aug, y_aug, z_aug = batch_data_augmentation(x, y, z)
        >>> print(x_aug.shape)  # [147, 96] = 128 * (1 + 0.05 + 0.1)
    """
    # 1. Point Anomaly 증강
    if point_rate > 0:
        x_point, y_point, z_point = point_anomaly(x, y, z, point_rate)
        if x_point.size(0) > 0:
            x = torch.cat((x, x_point), dim=0)
            y = torch.cat((y, y_point), dim=0)
            z = torch.cat((z, z_point), dim=0)
    
    # 2. Segment Anomaly 증강
    if segment_rate > 0:
        x_seg, y_seg, z_seg = segment_anomaly(x, y, z, segment_rate, method="swap")
        if x_seg.size(0) > 0:
            x = torch.cat((x, x_seg), dim=0)
            y = torch.cat((y, y_seg), dim=0)
            z = torch.cat((z, z_seg), dim=0)
    
    # 3. Missing Data 주입 (전체 배치에 적용)
    if missing_rate > 0:
        x, y, z = missing_data_injection(x, y, z, missing_rate)
    
    return x, y, z


# ===== 테스트 =====
if __name__ == '__main__':
    print("=" * 70)
    print("Data Augmentation Test")
    print("=" * 70)
    
    # 샘플 데이터
    batch_size = 128
    window_size = 96
    
    x = torch.randn(batch_size, window_size)
    y = torch.zeros(batch_size, window_size)
    z = torch.zeros(batch_size, window_size)
    
    print(f"\n원본 배치 크기: {x.shape}")
    print(f"  정상 샘플: {(y.sum(dim=1) == 0).sum()}")
    print(f"  이상 샘플: {(y.sum(dim=1) > 0).sum()}")
    
    # 증강
    print("\n" + "-" * 70)
    print("데이터 증강 적용...")
    print("-" * 70)
    x_aug, y_aug, z_aug = batch_data_augmentation(
        x, y, z,
        missing_rate=0.01,
        point_rate=0.05,
        segment_rate=0.1
    )
    
    print(f"\n증강 후 배치 크기: {x_aug.shape}")
    print(f"  증가율: {x_aug.size(0) / batch_size:.2f}x")
    print(f"  정상 샘플: {(y_aug.sum(dim=1) == 0).sum()}")
    print(f"  이상 샘플: {(y_aug.sum(dim=1) > 0).sum()}")
    print(f"  결측 포함 샘플: {(z_aug.sum(dim=1) > 0).sum()}")
    
    # 개별 증강 테스트
    print("\n" + "-" * 70)
    print("개별 증강 테스트:")
    print("-" * 70)
    
    # Point Anomaly
    x_p, y_p, z_p = point_anomaly(x, y, z, rate=0.05)
    print(f"  Point Anomaly: {x_p.shape[0]} 샘플 생성")
    print(f"    마지막 타임스텝 이상: {(y_p[:, -1] == 1).sum()} / {x_p.shape[0]}")
    
    # Segment Anomaly
    x_s, y_s, z_s = segment_anomaly(x, y, z, rate=0.1)
    print(f"  Segment Anomaly: {x_s.shape[0]} 샘플 생성")
    print(f"    이상 구간 평균 길이: {y_s.sum(dim=1).mean().item():.1f} 타임스텝")
    
    # Missing Data
    x_m, y_m, z_m = missing_data_injection(x.clone(), y.clone(), z.clone(), rate=0.01)
    print(f"  Missing Data: {(z_m == 1).sum()} / {x_m.numel()} 포인트 ({(z_m == 1).sum() / x_m.numel() * 100:.2f}%)")
    
    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)
