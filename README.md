# LG Aimers Model Compression Portfolio

LG AI연구원 EXAONE-4.0-1.2B를 대상으로 진행한 LLM 경량화 프로젝트입니다.  
대회명은 `Aimers 8기 : 모델 경량화 온라인 해커톤`이며, DACON 환경에서 모델 성능과 추론 효율을 함께 최적화하는 것이 목표였습니다.

- 대회 개요: https://dacon.io/competitions/official/236673/overview/description
- 리더보드: https://dacon.io/competitions/official/236673/leaderboard
- 기준 코드: [qlora_baseline_3000_val_local.py](/home/ubuntu/Desktop/LG%20Aimers/qlora_baseline_3000_val_local.py)

## Project Summary

이 프로젝트는 단순히 모델 크기를 줄이는 것이 아니라, 실제 추론 환경에서 동작 가능한 수준의 속도와 성능 균형을 맞추는 데 초점을 둡니다.  
대회 공식 설명에 따르면 온라인 해커톤은 Hugging Face 표준 형식 모델만 제출 가능했고, 고정된 vLLM 추론 환경에서 성능과 효율을 함께 평가했습니다.

핵심 목표:

- EXAONE-4.0-1.2B 기반 경량화 실험
- 정확도 저하를 최소화하는 미세조정 전략 설계
- 양자화를 통한 추론 효율 개선
- 제출 가능한 HF 표준 포맷 결과물 생성

## Competition Context

DACON 대회 페이지 기준 핵심 조건은 다음과 같습니다.

- 기본 모델: `EXAONE-4.0-1.2B`
- 주제: `LLM 경량화`
- 평가 환경: `vLLM 기반 고정 추론 환경`
- 제출 형식: `submit.zip`
- 제출물 조건: Hugging Face 표준 형식 모델 가중치 및 설정 파일
- 실행 제약: 제한된 CPU, RAM, GPU 환경에서 추론 시간과 효율을 함께 평가

이 조건 때문에 단순 full fine-tuning 보다는, 메모리를 절약하면서 성능을 유지할 수 있는 `QLoRA + 후처리 양자화` 조합이 적합하다고 판단했습니다.

## My Approach

프로젝트는 두 단계로 구성했습니다.

1. `QLoRA` 기반 미세조정
2. 필요 시 `GPTQ W8A8` 기반 후처리 양자화

이 방식의 의도는 다음과 같습니다.

- 학습 단계에서는 LoRA adapter만 학습해 메모리 사용량을 줄이기
- 추론 단계에서는 병합된 모델을 다시 양자화해 배포 효율을 높이기
- 대회 제출 포맷과 평가 환경에 맞는 결과물로 정리하기

## Model Mechanism

### 1. Data Preprocessing

입력 데이터는 `conversations` 기반 멀티턴 대화 형식을 가정합니다.

전처리 규칙:

- 마지막 turn이 `assistant` 인 샘플만 사용
- 마지막 `assistant` 응답만 answer로 사용
- 이전 대화는 prompt로 사용
- prompt 구간은 `labels=-100` 으로 마스킹
- answer 토큰만 loss 계산

전처리 흐름:

1. `conversations[:-1]` 를 prompt로 사용
2. 마지막 `assistant` 응답을 answer로 사용
3. prompt는 `apply_chat_template(..., add_generation_prompt=True)` 적용
4. answer는 별도로 tokenize
5. answer 마지막에는 항상 `eos_id` 보장

길이 제한 처리:

- 최대 길이: `1024`
- prompt는 뒤쪽 문맥을 유지하는 방식으로 절단
- 최소 prompt 길이: `128`
- answer가 잘릴 경우에도 마지막 토큰은 `eos_id` 유지

샘플링 규칙:

- 전체 데이터셋을 `seed=42`로 셔플
- 앞 `3000`개만 사용
- 그 중 `150`개를 eval로 분리
- 최종적으로 train `2850`, eval `150`

즉 이 스크립트는 일반적인 next-token prediction이 아니라, prompt 뒤에 이어지는 assistant answer를 학습하는 SFT 구조입니다.

#### Sequence Length Strategy

대회 평가는 성능과 속도를 동시에 고려합니다.

- `PerfNorm`: 기본 모델 대비 성능 유지 또는 개선
- `Speed`: 토큰당 추론 시간이 짧을수록 유리

그래서 전처리 단계에서 `max_seq_len`은 가장 중요한 트레이드오프 포인트였습니다.  
최종적으로 학습 `max_seq_len`은 `1024`로 고정했습니다.

3k 샘플 길이 분석 기준:

- Prompt
  - mean `177.7`
  - median `98`
  - p95 `854.0`
  - max `1648`
- Assistant
  - mean `673.5`
  - median `705`
  - p95 `999.0`
  - max `1733`
- Total
  - mean `851.2`
  - median `833`
  - p90 `1218.6`
  - p95 `1585.1`
  - max `2927`

길이 커버리지:

| max_seq_len | prompt<= | answer<= | total<= |
|---:|---:|---:|---:|
| 512 | 91.13% | 22.23% | 13.23% |
| 768 | 93.87% | 62.93% | 38.73% |
| 1024 | 98.27% | 96.37% | 80.47% |
| 1536 | 99.90% | 99.87% | 94.57% |
| 2048 | 100.0% | 100.0% | 99.53% |

`1024`는 전체 샘플의 약 `80%`를 완전 보존하면서도 계산량과 실험 반복 비용을 통제할 수 있는 지점이라고 판단했습니다.

#### 왜 1536/2048로 올리지 않았는가

1. 계산량과 메모리 비용 증가
   - self-attention 비용이 길이에 대해 크게 증가하고, 고정 패딩이면 짧은 샘플도 불필요한 연산을 수행합니다.
2. 긴 컨텍스트의 잡음 증가
   - 유용한 정보뿐 아니라 잡담, 형식 문구, 중복 정보도 함께 늘어날 수 있습니다.
3. 소량 데이터에서의 학습 효율 저하
   - 같은 예산에서 실험 수가 줄고, 긴 샘플 스타일에 과적합될 위험이 있습니다.

#### 1024를 유지하면서 성능을 지키는 방법

핵심은 길이를 늘리는 대신, 잘리는 방식을 최적화하는 것이었습니다.

- prompt를 무제한으로 유지하지 않음
- 필요한 경우 prompt를 줄여서라도 answer 토큰을 최대한 남김
- prompt는 tail 유지
- answer는 head 유지 + EOS 보장

즉 최신 대화 문맥은 살리고, 실제 정답 구간도 최대한 보존하는 방향으로 전처리를 설계했습니다.

### 2. QLoRA Fine-tuning

학습 단계에서는 베이스 모델 전체를 4bit 상태로 유지하고, LoRA adapter만 학습합니다.

핵심 설정:

- `load_in_4bit=True`
- `bnb_4bit_quant_type="nf4"`
- `bnb_4bit_use_double_quant=True`
- compute dtype: `bf16 > fp16 > fp32`

LoRA 설정:

- `r=16`
- `alpha=32`
- `dropout=0.05`
- target modules는 모델 내 `Linear` 계층 suffix를 스캔해 자동 선택

학습 의도:

- 베이스 모델 전체를 직접 업데이트하지 않음
- 학습 가능한 파라미터를 LoRA adapter로 제한
- 메모리 사용량과 학습 비용을 줄이면서도 태스크 적응력 확보

학습 설정:

- batch size: `1`
- gradient accumulation: `16`
- learning rate: `1e-5`
- epoch: `1`
- warmup ratio: `0.05`
- eval every `20` steps

추가적으로:

- `prepare_model_for_kbit_training(...)` 사용
- gradient checkpointing 활성화
- 학습 중 `use_cache=False`

### 3. Quantization Strategy

추가 제출용 모델이 필요할 경우, 학습된 LoRA adapter를 베이스 모델에 병합한 뒤 `GPTQ`를 수행합니다.

양자화 단계 흐름:

1. 베이스 모델 다시 로드
2. 저장된 LoRA adapter 결합
3. `merge_and_unload()` 로 단일 모델 생성
4. calibration용 prompt-only 데이터 구성
5. `GPTQModifier` 기반 one-shot 양자화
6. HF 포맷 모델 및 zip 저장

양자화 설정:

- scheme: `W8A8`
- target: `Linear`
- `actorder="static"`
- `dampening_frac=0.001`
- `block_size=128`

ignore 대상:

- `embed_tokens`
- `lm_head`

의도:

- 학습 단계에서는 QLoRA로 메모리 효율 확보
- 배포 단계에서는 GPTQ로 추론 효율 강화
- 최종적으로 제출 가능한 HF 표준 포맷 결과물 생성

## Result

저희 팀 `쉬고싶으면쉬어도돼근데남들은안쉬긴해` 의 결과는 다음과 같습니다.

- 순위: `40위`
- 점수: `0.63166`
- 제출 수: `69`

## Technical Highlights

- EXAONE 계열 모델을 대상으로 한 실제 경량화 실험
- 대화형 SFT 데이터 전처리 직접 구현
- 4bit QLoRA 기반 파라미터 효율 미세조정
- LoRA adapter 병합 후 GPTQ W8A8 양자화 파이프라인 구성
- 제출 포맷까지 고려한 HF 모델 저장 및 zip 생성 구조 설계

## Execution

기본 실행 예시:

```bash
export OPEN_DIR="/path/to/open"
export WORKSPACE="/path/to/local_outputs"
export MODEL_ID="/path/to/open/base_model"
export DATASET_ID="LGAI-EXAONE/MANTA-1M"
python qlora_baseline_3000_val_local.py
```

GPTQ까지 실행하려면:

```bash
export OPEN_DIR="/path/to/open"
export WORKSPACE="/path/to/local_outputs"
export MODEL_ID="/path/to/open/base_model"
export DATASET_ID="LGAI-EXAONE/MANTA-1M"
python qlora_baseline_3000_val_local.py --run-gptq
```

## Note

이 저장소는 포트폴리오용 코드 공개를 목적으로 정리된 버전입니다.  
대회 데이터, 원본 모델 가중치, 제출 산출물은 포함하지 않습니다.
