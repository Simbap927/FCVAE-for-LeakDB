"""
FCVAE Training Module
- PyTorch Lightning 기반 학습 스크립트
- 배치 단위 데이터 증강 포함
"""

import torch
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Optional
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from models.fcvae import FCVAE
from data.data_augment import batch_data_augmentation


class FCVAETrainer(pl.LightningModule):
    """
    FCVAE PyTorch Lightning Trainer
    - 원본 FCVAE 방식: 배치 단위 데이터 증강
    """
    
    def __init__(
        self,
        window_size: int = 96,
        latent_dim: int = 8,
        condition_dim: int = 16,
        hidden_dim: int = 100,
        d_model: int = 256,
        d_ff: int = 512,
        n_head: int = 8,
        kernel_size: int = 16,
        stride: int = 8,
        dropout: float = 0.05,
        kl_weight: float = 0.005,
        learning_rate: float = 0.0005,
        # 데이터 증강 파라미터
        missing_rate: float = 0.01,
        point_rate: float = 0.05,
        segment_rate: float = 0.1
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # FCVAE 모델
        self.model = FCVAE(
            window_size=window_size,
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            d_model=d_model,
            d_ff=d_ff,
            n_head=n_head,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            kl_weight=kl_weight
        )
        
        self.learning_rate = learning_rate
        self.missing_rate = missing_rate
        self.point_rate = point_rate
        self.segment_rate = segment_rate
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor, mode: str = "train"):
        """Forward pass"""
        return self.model(x, mask=mask, mode=mode)
    
    def training_step(self, batch: Dict, batch_idx: int):
        """
        Training step with batch data augmentation
        
        원본 FCVAE 방식:
        1. 배치 로드
        2. 배치 단위 데이터 증강 (배치 크기 증가)
        3. 마스크 생성
        4. CM-ELBO 손실 계산
        """
        # 1. 배치 로드
        x = batch['values']      # [batch, window_size]
        y_all = batch['labels']  # [batch, window_size]
        z_all = batch['missing'] # [batch, window_size]
        
        # 2. 배치 단위 데이터 증강 (원본 방식)
        x, y_all, z_all = batch_data_augmentation(
            x, y_all, z_all,
            missing_rate=self.missing_rate,
            point_rate=self.point_rate,
            segment_rate=self.segment_rate
        )
        # x: [batch * 1.15, window_size] - 배치 크기 증가!
        
        # 3. CM-ELBO 마스크 생성
        # mask = NOT(y_all OR z_all)
        mask = (~(y_all.bool() | z_all.bool())).float()
        
        # 4. 손실 계산
        loss = self.model(x, mask=mask, mode="train")
        
        # 로깅
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('batch_size', float(x.size(0)), on_step=True)  # 증강 후 크기
        
        return loss
    
    def validation_step(self, batch: Dict, batch_idx: int):
        """
        Validation step without augmentation
        """
        x = batch['values']
        y_all = batch['labels']
        z_all = batch['missing']
        
        # 증강 없이 마스크만 생성
        mask = (~(y_all.bool() | z_all.bool())).float()
        
        # 손실 계산
        loss = self.model(x, mask=mask, mode="valid")
        
        # 로깅
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch: Dict, batch_idx: int):
        """
        Test step with MCMC inference
        """
        x = batch['values']
        y_true = batch['labels']  # 실제 라벨 (평가용)
        
        # MCMC 추론 (비지도)
        _, prob = self.model(x, mode="test")
        
        # 이상 스코어 계산 (-log probability)
        anomaly_scores = -prob.mean(dim=-1)  # [batch]
        
        # 실제 이상 여부 (윈도우에 하나라도 이상이 있으면 이상)
        has_anomaly = (y_true.sum(dim=1) > 0).float()
        
        return {
            'scores': anomaly_scores,
            'labels': has_anomaly
        }
    
    def configure_optimizers(self):
        """Optimizer and scheduler"""
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        
        # T_max는 trainer.max_epochs로 설정 (기본값 50)
        # Trainer 초기화 후 실제 값이 설정됨
        T_max = getattr(self.trainer, 'max_epochs', 50) if hasattr(self, 'trainer') else 50
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }


# ===== 사용 예시 =====
if __name__ == '__main__':
    print("=" * 70)
    print("FCVAE Trainer Test")
    print("=" * 70)
    
    # Trainer 생성
    trainer = FCVAETrainer(
        window_size=96,
        latent_dim=8,
        condition_dim=16,
        learning_rate=0.0005,
        missing_rate=0.01,
        point_rate=0.05,
        segment_rate=0.1
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in trainer.parameters()):,}")
    
    # 샘플 배치
    batch = {
        'values': torch.randn(128, 96),
        'labels': torch.zeros(128, 96),
        'missing': torch.zeros(128, 96)
    }
    
    print(f"\n원본 배치 크기: {batch['values'].shape}")
    
    # Training step 테스트
    trainer.train()
    loss = trainer.training_step(batch, 0)
    
    print(f"Training loss: {loss.item():.4f}")
    
    # Validation step 테스트
    trainer.eval()
    val_loss = trainer.validation_step(batch, 0)
    
    print(f"Validation loss: {val_loss.item():.4f}")
    
    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)
