"""
FCVAE Training Script for LeakDB
- PyTorch Lightning 기반 학습 스크립트
- 원본 FCVAE의 self-supervised learning 구현
- 배치 단위 데이터 증강 (Point Anomaly, Segment Anomaly, Missing Data)
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# Tensor Cores 최적화 (RTX GPU)
torch.set_float32_matmul_precision('high')  # 'medium' or 'high'
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor
)
from pytorch_lightning.loggers import TensorBoardLogger

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data.dataset import LeakDBDataset
from training.trainer import FCVAETrainer


def load_config(config_path: str = None) -> dict:
    """
    설정 파일 로드 또는 기본 설정 반환
    
    Args:
        config_path: YAML 설정 파일 경로 (optional)
    
    Returns:
        설정 딕셔너리
    """
    # 기본 설정
    default_config = {
        # Data
        'data': {
            'train_csv': 'data/processed/net1/train_processed.csv',
            'val_csv': 'data/processed/net1/val_processed.csv',
            'window_size': 96,
            'stride': 1,
            'batch_size': 128,
            'num_workers': 4,
            'use_label': False  # 완전 비지도 학습
        },
        
        # Model Architecture (FCVAE 원본 설정)
        'model': {
            'window_size': 96,
            'latent_dim': 8,
            'condition_dim': 16,
            'hidden_dim': 100,
            'd_model': 256,
            'd_ff': 512,
            'n_head': 8,
            'kernel_size': 16,
            'stride': 8,
            'dropout': 0.05,
            'kl_weight': 0.005
        },
        
        # Training
        'training': {
            'learning_rate': 0.0005,
            'max_epochs': 50,
            'gradient_clip_val': 2.0,
            'accumulate_grad_batches': 1,
            
            # Data Augmentation (원본 FCVAE 비율)
            'missing_rate': 0.01,   # 1% missing data injection
            'point_rate': 0.05,      # 5% point anomaly
            'segment_rate': 0.1      # 10% segment anomaly
        },
        
        # Callbacks
        'callbacks': {
            'early_stopping_patience': 10,
            'checkpoint_top_k': 3,
            'lr_monitor': True
        },
        
        # Logging
        'logging': {
            'log_dir': 'logs',
            'experiment_name': 'fcvae_net1',
            'save_dir': 'checkpoints'
        },
        
        # Hardware
        'hardware': {
            'accelerator': 'auto',
            'devices': 1,
            'precision': '32-true'
        }
    }
    
    # YAML 파일이 있으면 로드하여 병합
    if config_path and Path(config_path).exists():
        print(f"Loading config from {config_path}...")
        with open(config_path, 'r') as f:
            custom_config = yaml.safe_load(f)
        
        # Deep merge
        def merge_dict(base, custom):
            for key, value in custom.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dict(base[key], value)
                else:
                    base[key] = value
        
        merge_dict(default_config, custom_config)
    
    return default_config


def create_dataloaders(config: dict):
    """
    DataLoader 생성
    
    Args:
        config: 설정 딕셔너리
    
    Returns:
        train_loader, val_loader
    """
    data_config = config['data']
    
    print("=" * 70)
    print("Creating Datasets & DataLoaders")
    print("=" * 70)
    
    # Train Dataset
    train_dataset = LeakDBDataset(
        csv_file=data_config['train_csv'],
        window_size=data_config['window_size'],
        stride=data_config['stride'],
        mode='train',
        use_label=data_config['use_label']
    )
    
    # Validation Dataset
    val_dataset = LeakDBDataset(
        csv_file=data_config['val_csv'],
        window_size=data_config['window_size'],
        stride=data_config['stride'],
        mode='valid',
        use_label=data_config['use_label']
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
        pin_memory=True,
        persistent_workers=True if data_config['num_workers'] > 0 else False,
        drop_last=True  # 배치 증강을 위해 동일한 배치 크기 유지
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=True,
        persistent_workers=True if data_config['num_workers'] > 0 else False
    )
    
    print(f"\n✓ Train Dataset: {len(train_dataset):,} windows")
    print(f"✓ Val Dataset:   {len(val_dataset):,} windows")
    print(f"✓ Train Batches: {len(train_loader):,}")
    print(f"✓ Val Batches:   {len(val_loader):,}")
    print(f"✓ Batch Size:    {data_config['batch_size']}")
    
    return train_loader, val_loader


def create_model(config: dict) -> FCVAETrainer:
    """
    FCVAE 모델 생성
    
    Args:
        config: 설정 딕셔너리
    
    Returns:
        FCVAETrainer instance
    """
    model_config = config['model']
    train_config = config['training']
    
    print("\n" + "=" * 70)
    print("Creating FCVAE Model")
    print("=" * 70)
    
    model = FCVAETrainer(
        # Architecture
        window_size=model_config['window_size'],
        latent_dim=model_config['latent_dim'],
        condition_dim=model_config['condition_dim'],
        hidden_dim=model_config['hidden_dim'],
        d_model=model_config['d_model'],
        d_ff=model_config['d_ff'],
        n_head=model_config['n_head'],
        kernel_size=model_config['kernel_size'],
        stride=model_config['stride'],
        dropout=model_config['dropout'],
        kl_weight=model_config['kl_weight'],
        
        # Training
        learning_rate=train_config['learning_rate'],
        
        # Data Augmentation
        missing_rate=train_config['missing_rate'],
        point_rate=train_config['point_rate'],
        segment_rate=train_config['segment_rate']
    )
    
    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n✓ Model created successfully")
    print(f"✓ Total parameters:     {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    print(f"✓ Model size:           {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    return model


def create_callbacks(config: dict) -> list:
    """
    PyTorch Lightning Callbacks 생성
    
    Args:
        config: 설정 딕셔너리
    
    Returns:
        List of callbacks
    """
    callback_config = config['callbacks']
    logging_config = config['logging']
    
    callbacks = []
    
    # 1. Model Checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(logging_config['save_dir']) / logging_config['experiment_name'],
        filename='fcvae-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=callback_config['checkpoint_top_k'],
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # 2. Early Stopping
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=callback_config['early_stopping_patience'],
        mode='min',
        verbose=True,
        min_delta=1e-4
    )
    callbacks.append(early_stop_callback)
    
    # 3. Learning Rate Monitor
    if callback_config['lr_monitor']:
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        callbacks.append(lr_monitor)
    
    print("\n" + "=" * 70)
    print("Callbacks Configuration")
    print("=" * 70)
    print(f"✓ Model Checkpoint: Save top {callback_config['checkpoint_top_k']} models")
    print(f"✓ Early Stopping:   Patience = {callback_config['early_stopping_patience']} epochs")
    print(f"✓ LR Monitor:       {'Enabled' if callback_config['lr_monitor'] else 'Disabled'}")
    
    return callbacks


def create_logger(config: dict):
    """
    TensorBoard Logger 생성
    
    Args:
        config: 설정 딕셔너리
    
    Returns:
        TensorBoardLogger
    """
    logging_config = config['logging']
    
    logger = TensorBoardLogger(
        save_dir=logging_config['log_dir'],
        name=logging_config['experiment_name'],
        default_hp_metric=False
    )
    
    print(f"✓ TensorBoard logs: {logger.log_dir}")
    
    return logger


def train(config: dict):
    """
    메인 학습 함수
    
    Args:
        config: 설정 딕셔너리
    """
    print("\n" + "=" * 70)
    print("FCVAE Training for LeakDB")
    print("=" * 70)
    print(f"Experiment: {config['logging']['experiment_name']}")
    print("=" * 70)
    
    # 1. DataLoaders
    train_loader, val_loader = create_dataloaders(config)
    
    # 2. Model
    model = create_model(config)
    
    # 3. Callbacks
    callbacks = create_callbacks(config)
    
    # 4. Logger
    logger = create_logger(config)
    
    # 5. Trainer
    print("\n" + "=" * 70)
    print("PyTorch Lightning Trainer Configuration")
    print("=" * 70)
    
    train_config = config['training']
    hardware_config = config['hardware']
    
    trainer = pl.Trainer(
        max_epochs=train_config['max_epochs'],
        callbacks=callbacks,
        logger=logger,
        accelerator=hardware_config['accelerator'],
        devices=hardware_config['devices'],
        precision=hardware_config['precision'],
        gradient_clip_val=train_config['gradient_clip_val'],
        accumulate_grad_batches=train_config['accumulate_grad_batches'],
        log_every_n_steps=50,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False
    )
    
    print(f"✓ Max epochs:              {train_config['max_epochs']}")
    print(f"✓ Accelerator:             {hardware_config['accelerator']}")
    print(f"✓ Devices:                 {hardware_config['devices']}")
    print(f"✓ Precision:               {hardware_config['precision']}")
    print(f"✓ Gradient clipping:       {train_config['gradient_clip_val']}")
    print(f"✓ Accumulate grad batches: {train_config['accumulate_grad_batches']}")
    
    # 6. Start Training
    print("\n" + "=" * 70)
    print("Starting Training...")
    print("=" * 70)
    print("\n")
    
    trainer.fit(model, train_loader, val_loader)
    
    # 7. Training Complete
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"✓ Best model checkpoint: {trainer.checkpoint_callback.best_model_path}")
    print(f"✓ Best validation loss:  {trainer.checkpoint_callback.best_model_score:.4f}")
    print(f"✓ Total epochs trained:  {trainer.current_epoch + 1}")
    print("=" * 70)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="FCVAE Training Script for LeakDB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to YAML config file (optional)'
    )
    
    parser.add_argument(
        '--train_csv',
        type=str,
        default=None,
        help='Path to training CSV file (overrides config)'
    )
    
    parser.add_argument(
        '--val_csv',
        type=str,
        default=None,
        help='Path to validation CSV file (overrides config)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )
    
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=None,
        help='Maximum epochs (overrides config)'
    )
    
    parser.add_argument(
        '--experiment_name',
        type=str,
        default=None,
        help='Experiment name (overrides config)'
    )
    
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU if available'
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='Number of data loader workers (overrides config)'
    )
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    # CLI 인자로 설정 오버라이드
    if args.train_csv:
        config['data']['train_csv'] = args.train_csv
    
    if args.val_csv:
        config['data']['val_csv'] = args.val_csv
    
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    
    if args.max_epochs:
        config['training']['max_epochs'] = args.max_epochs
    
    if args.experiment_name:
        config['logging']['experiment_name'] = args.experiment_name
    
    if args.gpu:
        config['hardware']['accelerator'] = 'gpu'
    
    if args.num_workers is not None:
        config['data']['num_workers'] = args.num_workers
    
    # 디렉토리 생성
    Path(config['logging']['log_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['logging']['save_dir']).mkdir(parents=True, exist_ok=True)
    
    # 설정 출력
    print("\n" + "=" * 70)
    print("Configuration")
    print("=" * 70)
    print(f"Train CSV: {config['data']['train_csv']}")
    print(f"Val CSV:   {config['data']['val_csv']}")
    print(f"Batch Size: {config['data']['batch_size']}")
    print(f"Learning Rate: {config['training']['learning_rate']}")
    print(f"Max Epochs: {config['training']['max_epochs']}")
    print(f"Experiment: {config['logging']['experiment_name']}")
    print("=" * 70)
    
    # 학습 시작
    train(config)


if __name__ == '__main__':
    main()
