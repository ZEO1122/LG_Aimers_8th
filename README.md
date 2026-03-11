# LG Aimers

`EXAONE-4.0-1.2B` 기반 QLoRA 학습과 선택적 GPTQ 양자화 실험을 정리한 레포입니다. 기준 실행 파일은 [qlora_baseline_3000_val_local.py](/home/ubuntu/Desktop/LG%20Aimers/qlora_baseline_3000_val_local.py) 입니다.

## 구성

- `LoRA_baseline.py`: 기존 베이스라인 스크립트
- `qlora_baseline_3000_val_local.py`: 로컬 실행용 스크립트
- `env/activate_exaone.sh`: `exaone` conda 환경 활성화 스크립트
- `env/exaone.env`: 실행용 환경변수 파일
- `env/local_environment.md`: 로컬 머신 환경 정리

## 실행 환경

현재 로컬 환경 기준:

- OS: `Ubuntu 24.04.3 LTS`
- Kernel: `Linux 6.8.0-101-generic`
- CPU: `Intel Core i7-14700K`
- Memory: `15 GiB`
- GPU: `NVIDIA GeForce RTX 4070 Ti SUPER`
- Conda env Python: `3.10.19`

GPU 확인 정보:

- PCI bus: `0000:01:00.0`
- Bus type: `PCIe`
- NVIDIA driver: `590.48.01`
- GPU firmware: `590.48.01`
- Video BIOS: `95.03.45.40.4b`
- Device node 확인: `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`

주요 패키지 버전:

- `torch==2.9.1`
- `transformers==4.57.3`
- `datasets==4.4.1`
- `peft==0.11.1`
- `bitsandbytes==0.49.1`
- `llmcompressor==0.9.0.1`
- `vllm==0.14.1`

참고:

- 현재 세션에서는 `nvidia-smi` 가 `Failed to initialize NVML: Unknown Error` 로 실패합니다.
- 대신 `/proc/driver/nvidia/gpus/*/information`, `/proc/driver/nvidia/version`, `lshw -C display` 기준으로 GPU 장치와 드라이버는 확인했습니다.
- GPU 장치 자체는 보이지만, 런타임 CUDA 상태는 별도 확인이 필요합니다.

세부 환경 정보는 [local_environment.md](/home/ubuntu/Desktop/LG%20Aimers/env/local_environment.md) 에 정리되어 있습니다.

## 환경변수

기본 실행은 [exaone.env](/home/ubuntu/Desktop/LG%20Aimers/env/exaone.env) 기준으로 동작합니다.

- `CONDA_ENV_NAME`: 사용할 conda 환경 이름
- `OPEN_DIR`: 로컬 모델/데이터셋 기준 폴더
- `WORKSPACE`: 출력 및 캐시 작업 폴더
- `MODEL_ID`: 로컬 모델 폴더 또는 Hugging Face 모델 ID
- `DATASET_ID`: 로컬 데이터셋 폴더 또는 Hugging Face 데이터셋 ID
- `HF_HOME`: Hugging Face 캐시 루트
- `HF_DATASETS_CACHE`: datasets 캐시 경로
- `TRANSFORMERS_CACHE`: transformers 캐시 경로

실행 예시:

```bash
source /home/ubuntu/Desktop/LG\ Aimers/env/activate_exaone.sh
python /home/ubuntu/Desktop/LG\ Aimers/qlora_baseline_3000_val_local.py
```

GPTQ까지 실행하려면:

```bash
source /home/ubuntu/Desktop/LG\ Aimers/env/activate_exaone.sh
python /home/ubuntu/Desktop/LG\ Aimers/qlora_baseline_3000_val_local.py --run-gptq
```

## 모델 메커니즘

스크립트의 전체 흐름은 아래와 같습니다.

1. 모델과 데이터셋 입력 소스 결정
2. 데이터 전처리
3. QLoRA 학습
4. LoRA adapter 저장
5. 선택적으로 Merge + GPTQ 양자화 + zip 저장

### 데이터 전처리

입력 데이터는 `conversations` 구조를 가정합니다.

- 마지막 turn이 `assistant` 인 샘플만 사용
- 마지막 `assistant` 응답만 answer로 사용
- 이전 대화는 prompt로 사용
- prompt 구간은 `labels=-100` 으로 마스킹
- answer 구간만 loss 계산

길이 처리 규칙:

- 전체 최대 길이: `1024`
- prompt는 뒤쪽 문맥 기준으로 유지
- 최소 prompt 길이: `128`
- answer가 잘려도 마지막 토큰은 항상 `eos_id` 유지

데이터셋 샘플링 규칙:

- `shuffle(seed=42)`
- 총 `3000`개 사용
- `150`개를 eval로 분리

전처리 결과 텐서 구조:

- `input_ids`: prompt + answer
- `attention_mask`: 실제 토큰 구간만 `1`
- `labels`: prompt는 `-100`, answer만 실제 token id

즉 학습 목적은 전체 문장을 복원하는 것이 아니라, prompt 뒤에 오는 assistant answer만 맞추도록 하는 SFT 구조입니다.

### QLoRA 학습

학습 단계에서는 4bit 양자화된 베이스 모델 위에 LoRA adapter만 학습합니다.

4bit 설정:

- `load_in_4bit=True`
- `bnb_4bit_quant_type="nf4"`
- `bnb_4bit_use_double_quant=True`
- compute dtype은 가능하면 `bf16`, 아니면 `fp16`, 그 외에는 `fp32`

LoRA 설정:

- `r=16`
- `alpha=32`
- `dropout=0.05`
- target은 모델 내 `Linear` 모듈을 스캔해 자동 선택

학습 메커니즘 요약:

- 베이스 모델 전체는 4bit 상태로 메모리에 유지
- 학습 가능한 파라미터는 LoRA adapter만 남김
- `prepare_model_for_kbit_training(...)` 으로 k-bit 학습 준비
- gradient checkpointing 활성화
- 학습 중 `use_cache=False`

주요 학습 설정:

- batch size: `1`
- gradient accumulation: `16`
- learning rate: `1e-5`
- epoch: `1`
- warmup ratio: `0.05`
- eval every `20` steps

학습 후에는 `trainer.state.log_history` 에서 train loss와 eval loss를 읽어 요약하고, 가능하면 loss plot도 저장합니다.

### GPTQ 양자화

`--run-gptq` 옵션을 주면 학습 후 LoRA를 베이스 모델에 병합하고 GPTQ를 수행합니다.

실행 순서:

1. 베이스 모델 다시 로드
2. 저장된 LoRA adapter 결합
3. `merge_and_unload()` 로 단일 모델 생성
4. prompt-only 캘리브레이션 데이터 구성
5. GPTQ one-shot 양자화 수행
6. 최종 모델과 zip 저장

캘리브레이션 방식:

- 데이터셋 앞 `1024`개 사용
- 마지막 `assistant` 응답 제거
- prompt-only 텍스트로 변환 후 사용

양자화 설정:

- scheme: `W8A8`
- target: `Linear`
- `actorder="static"`
- `dampening_frac=0.001`
- `block_size=128`

의미:

- `W8A8`: weight와 activation을 8bit 계열로 다루는 양자화 설정
- `actorder="static"` : activation ordering을 사용해 오차를 줄이는 설정
- `ignore` preset: 일부 민감 모듈은 양자화 제외

기본 ignore 대상:

- `embed_tokens`
- `lm_head`

### 저장 결과

학습 후 저장되는 주요 결과:

- LoRA adapter
- 체크포인트
- 선택적으로 병합 모델
- 선택적으로 GPTQ 양자화 모델
- 선택적으로 제출용 zip

정리하면 이 스크립트는 다음 두 단계를 한 파일에 묶은 구조입니다.

- 1단계: QLoRA로 adapter 학습
- 2단계: 필요 시 adapter를 병합한 뒤 GPTQ W8A8로 최종 양자화

자세한 메커니즘 설명은 [qlora_baseline_3000_val_local.md](/home/ubuntu/Desktop/LG%20Aimers/experiments/baseline_model/qlora_baseline_3000_val_local.md) 에 정리되어 있습니다.

## 주의사항

- 로컬 모델 폴더는 Hugging Face 포맷 전체 파일이 있어야 정상 실행됩니다.
- 로컬 데이터셋 폴더가 없으면 데이터셋은 Hugging Face ID로 fallback 됩니다.
- 이 레포는 실행 결과물과 대용량 모델 산출물을 포함하지 않습니다.
