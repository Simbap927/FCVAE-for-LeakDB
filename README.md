# FCVAE-for-LeakDB
LeakDB 공개 데이터셋에 대해 FCVAE의 성능을 확인합니다.

### FCVAE
https://github.com/CSTCloudOps/FCVAE

### LeakDB
https://github.com/KIOS-Research/LeakDB

### Directory Structure - TODO
```
FCVAE-for-LeakDB/
├── leakdb/                    # 원본 LeakDB 데이터 (ignored)
│   ├── Net1_CMH/
│   └── Hanoi_CMH/
│
├── data/                      # 처리된 데이터
│   └── processed/
│       ├── net1_full.csv          # 통합 CSV
│       ├── hanoi_full.csv
│       └── net1/
│           ├── train.csv          # Split 데이터
│           ├── val.csv
│           ├── test.csv
│           ├── train_processed.csv  # 전처리 완료
│           ├── val_processed.csv
│           └── test_processed.csv
│
├── src/                       # 소스 코드
│   ├── data/
│   │   ├── leakdb_loader.py       # 데이터 로더
│   │   ├── split_data.py          # Train/Val/Test split
│   │   ├── preprocessor.py        # 전처리
│   │   └── dataset.py             # PyTorch Dataset
│   │
│   ├── models/
│   │   ├── cvae.py                # Conditional VAE
│   │   ├── attention.py           # Self-Attention
│   │   ├── fcvae.py               # FCVAE 통합
│   │   └── lightning_module.py    # Lightning wrapper
│   │
│   └── evaluation/
│       └── leakdb_scorer.py       # LeakDB 평가
│
├── train.py                   # 훈련 스크립트
├── evaluate.py                # 평가 스크립트
├── requirements.txt           
└── README.md
```
