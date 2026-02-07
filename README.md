# LG Aimers

`EXAONE-4.0-1.2B` 기반 QLoRA 학습 실험을 정리한 최소 레포입니다. 이 저장소는 `experiments/baseline_model/QLoRA_baseline_3000_val.ipynb` 한 개를 기준으로 재구성했습니다.

## 포함 내용

- `experiments/baseline_model/QLoRA_baseline_3000_val.ipynb`
  - `LGAI-EXAONE/EXAONE-4.0-1.2B` 모델 사용
  - `LGAI-EXAONE/MANTA-1M` 데이터셋 사용
  - `shuffle(seed=42)` 후 3000개 샘플 사용
  - 검증셋 150개 분리
  - QLoRA 학습 후 선택적으로 Merge + GPTQ + zip 수행

## 디렉토리 구조

```text
.
├── README.md
├── .gitignore
└── experiments
    └── baseline_model
        └── QLoRA_baseline_3000_val.ipynb
```

## 실험 개요

노트북은 아래 흐름으로 구성됩니다.

1. 공통 import 및 경로/하이퍼파라미터 설정
2. 데이터셋 로드 및 3000개 샘플 구성
3. SFT 입력 포맷 전처리
4. LoRA 적용 후 학습 및 검증 loss 모니터링
5. 선택적으로 Merge, GPTQ 양자화, 제출용 zip 생성

주요 설정값:

- `MODEL_ID`: `LGAI-EXAONE/EXAONE-4.0-1.2B`
- `DATASET_ID`: `LGAI-EXAONE/MANTA-1M`
- `NUM_TOTAL_SAMPLES`: `3000`
- `NUM_EVAL_SAMPLES`: `150`
- `MAX_TRAIN_SEQ_LEN`: `1024`
- `LEARNING_RATE`: `1e-5`
- `NUM_TRAIN_EPOCHS`: `1`
- `LORA_R`: `16`
- `LORA_ALPHA`: `32`
- `LORA_DROPOUT`: `0.05`
- `EVAL_STEPS`: `20`

## 실행 환경

노트북에서 사용하는 주요 라이브러리:

- `torch`
- `datasets`
- `transformers`
- `peft`
- `llmcompressor`

기본 작업 경로는 환경 변수 `WORKSPACE`를 사용하며, 미설정 시 `/workspace`를 사용합니다.

## 정리 원칙

GitHub 업로드를 위해 아래 항목은 모두 제거했습니다.

- 학습 결과물 및 체크포인트
- 양자화 모델 폴더와 `submit.zip` 계열 산출물
- 로그, 분석 리포트, 임시 캐시
- 실험 파생 스크립트와 중복 베이스라인 파일

## 주의사항

이 레포는 실행 결과물을 포함하지 않습니다. 모델 가중치, 캐시, 학습 산출물은 로컬 또는 별도 스토리지에서 관리하는 전제를 둡니다.
