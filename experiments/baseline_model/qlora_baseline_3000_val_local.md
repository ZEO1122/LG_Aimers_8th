# qlora_baseline_3000_val_local.py 정리

이 문서는 [qlora_baseline_3000_val_local.py](/home/ubuntu/Desktop/LG%20Aimers/experiments/baseline_model/qlora_baseline_3000_val_local.py) 의 동작 메커니즘을 코드 기준으로 정리한 것이다. 설명 대상은 로컬 실행용 스크립트이며, 전체 흐름은 다음과 같다.

1. 실행 경로와 캐시 경로 설정
2. 모델 및 데이터셋 로드
3. 대화형 데이터 전처리
4. QLoRA 학습
5. LoRA 어댑터 저장
6. 선택적으로 Merge + GPTQ 양자화 + zip 생성

## 1. 경로와 입력 소스

스크립트는 우선 로컬 경로를 기준으로 모델과 데이터셋을 찾고, 없으면 Hugging Face 식별자로 fallback 한다.

- 기본 작업 경로: `local_outputs/`
- 기본 모델 경로: `open/base_model`
- 기본 데이터셋 경로: `open/dataset`
- fallback 모델: `LGAI-EXAONE/EXAONE-4.0-1.2B`
- fallback 데이터셋: `LGAI-EXAONE/MANTA-1M`

환경변수나 CLI 인자로 다음 값을 덮어쓸 수 있다.

- `WORKSPACE`
- `OPEN_DIR`
- `MODEL_ID`
- `DATASET_ID`

캐시는 `WORKSPACE/.cache/huggingface` 아래에 고정된다.

## 2. 모델 로드 방식

학습 단계에서는 `BitsAndBytesConfig` 를 사용해 4bit QLoRA 방식으로 모델을 로드한다.

- `load_in_4bit=True`
- `bnb_4bit_quant_type="nf4"`
- `bnb_4bit_use_double_quant=True`
- 계산 dtype은 `bf16 > fp16 > fp32` 우선순위로 선택

이 단계의 목적은 베이스 모델 전체를 저정밀도로 유지하면서 LoRA adapter만 학습해 메모리를 줄이는 것이다.

모델 로드 후에는 다음 설정이 들어간다.

- `prepare_model_for_kbit_training(...)`
- gradient checkpointing 활성화
- 학습 중 `use_cache=False`

## 3. 데이터셋 로드 방식

데이터셋은 두 방식 중 하나로 로드된다.

1. 로컬 폴더가 존재하면
   - `csv`
   - `json/jsonl`
   - `parquet`
   형식을 자동 판별해 `datasets.load_dataset(...)` 로 읽음
2. 로컬 폴더가 없으면 Hugging Face dataset ID를 직접 사용

그 다음 전체 데이터셋에 대해 아래 과정을 거친다.

- `shuffle(seed=42)`
- 앞에서 3000개만 선택
- 그 중 150개를 eval로 분리

즉 최종 분할은 아래와 같다.

- train: 2850개
- eval: 150개

## 4. 데이터 전처리 메커니즘

입력 데이터는 `conversations` 형식의 멀티턴 대화를 가정한다.

전처리 핵심 규칙:

- 마지막 turn이 `assistant` 인 샘플만 사용
- `assistant` 마지막 응답만 정답(label)으로 사용
- 그 이전 대화는 prompt로 사용
- prompt 구간은 loss 계산에서 제외
- answer 구간만 loss 계산

구체적인 처리 순서는 다음과 같다.

1. `conversations[:-1]` 를 prompt로 사용
2. 마지막 `assistant`의 `content` 를 answer로 사용
3. prompt에는 `tokenizer.apply_chat_template(..., add_generation_prompt=True)` 적용
4. answer는 별도로 tokenize
5. answer 끝에는 항상 `eos_id` 보장

### 길이 제한 처리

최대 길이는 `1024` 토큰이다.

초과 시 정책:

- prompt는 뒤쪽 문맥을 남기기 위해 tail 기준으로 절단
- prompt는 최소 `MIN_PROMPT_TOKENS=128` 유지
- 남은 길이만큼 answer를 자름
- answer를 자를 때도 마지막 토큰은 항상 `eos_id` 로 강제

최종 출력 필드는 다음 3개다.

- `input_ids`
- `attention_mask`
- `labels`

`labels` 구성 방식:

- prompt 길이만큼 `-100`
- answer 토큰은 실제 token id

즉 causal LM 형식이지만, 실제 loss는 answer 영역에만 걸린다.

## 5. QLoRA 메커니즘

이 스크립트의 핵심 학습 방식은 QLoRA다.

구성은 다음과 같다.

- 베이스 모델: 4bit 양자화 상태로 메모리에 유지
- 학습 대상: LoRA adapter만 업데이트
- 베이스 가중치: 직접 full fine-tuning 하지 않음

LoRA 설정값:

- `r = 16`
- `alpha = 32`
- `dropout = 0.05`
- `bias = "none"`
- `task_type = "CAUSAL_LM"`

### LoRA target 선택

기본값은 `["auto"]` 이고, 실제로는 모델 내부 `Linear` 모듈 이름을 스캔해서 아래 suffix가 존재하는 것만 target으로 잡는다.

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

즉 모델 구조에 없는 target은 자동으로 제외된다.

## 6. 학습 설정

`Trainer` 기반으로 학습한다.

주요 하이퍼파라미터:

- batch size: `1`
- gradient accumulation: `16`
- learning rate: `1e-5`
- epoch: `1`
- warmup ratio: `0.05`
- scheduler: `cosine`
- max grad norm: `1.0`
- eval every `20` steps
- save every `500` steps

정밀도 설정:

- BF16 가능하면 `bf16=True`
- 아니면 CUDA 환경에서 `fp16=True`
- 둘 다 아니면 CPU/FP32

학습 후에는 `trainer.state.log_history` 에서:

- train loss
- eval loss

를 추출하고, 가능하면 `train_eval_loss.png` 로 저장한다.

## 7. 저장 결과

학습이 끝나면 LoRA adapter를 아래 경로에 저장한다.

- `WORKSPACE/qlora_adapter_3000_val`

체크포인트는 아래 경로를 사용한다.

- `WORKSPACE/qlora_ckpt_3000_val`

## 8. Merge 단계

`--run-gptq` 옵션을 주면 학습 후 추가 파이프라인이 실행된다.

순서는 다음과 같다.

1. 베이스 모델을 다시 로드
2. 저장된 LoRA adapter를 붙임
3. `merge_and_unload()` 로 LoRA를 베이스 모델에 병합
4. 병합된 단일 모델을 GPTQ 입력으로 사용

이 단계에서 `use_cache=True` 로 되돌리고, CUDA 가능 시 GPU에 올린다.

## 9. GPTQ 양자화 방식

양자화는 `llmcompressor` 의 `oneshot(...)` + `GPTQModifier(...)` 조합으로 수행한다.

### 목적

QLoRA 학습으로 성능을 맞춘 모델을, 최종 배포/제출용으로 다시 정수 양자화하는 단계다.

### 캘리브레이션 데이터

캘리브레이션은 원본 데이터셋에서 앞 `1024`개를 사용한다.

전처리 규칙:

- 마지막 `assistant` 응답은 제거
- prompt-only 텍스트만 남김
- `apply_chat_template(..., add_generation_prompt=True)` 적용

즉 GPTQ는 정답 응답이 아닌 실제 추론 프롬프트 구조를 기준으로 보정된다.

### GPTQ 설정

- scheme: `W8A8`
- target: `Linear`
- `actorder="static"`
- `dampening_frac=0.001`
- `block_size=128`
- `offload_hessians=False`

여기서 `W8A8` 은 가중치와 activation을 모두 8비트 계열로 다루는 설정이다.

### ignore 정책

양자화에서 일부 모듈은 제외할 수 있다.

기본 preset:

- `embed_tokens`
- `lm_head`

코드에는 `plus_o`, `plus_down`, `plus_o_down`, `attn_all` 같은 preset도 들어 있다. 또한 `DEEP_PROTECT_RATIO` 로 마지막 레이어 일부를 보호할 수 있게 되어 있지만, 현재 기본값은 `0.0` 이다.

## 10. GPTQ 결과 저장

양자화 후 결과는 아래에 저장된다.

- 모델 폴더: `WORKSPACE/out/model_w8a8_3000_val`
- 제출 zip: `WORKSPACE/submit_3000_val.zip`

zip 내부 구조는 다음 형태다.

```text
model/
  config...
  tokenizer...
  weights...
```

즉 외부 평가 시스템이나 제출 포맷에서 바로 읽기 쉬운 구조를 만든다.

## 11. 이 스크립트의 핵심 특징

이 파일의 메커니즘을 한 줄씩 요약하면 다음과 같다.

- 학습은 QLoRA: 4bit 베이스 + LoRA adapter 학습
- 전처리는 SFT 방식: prompt는 마스킹, answer만 loss
- 학습 중 eval loss를 주기적으로 확인
- 학습 후 adapter만 저장
- 필요하면 LoRA를 병합한 뒤 GPTQ W8A8로 최종 양자화
- 최종 결과는 HF 모델 폴더와 zip으로 저장

## 12. 용어 정리

### QLoRA

저정밀도 양자화된 베이스 모델 위에 LoRA adapter만 학습하는 방식이다. 적은 메모리로 대형 모델을 미세조정할 때 주로 사용한다.

### LoRA

원래 가중치 전체를 바꾸는 대신, 저랭크 행렬만 추가로 학습하는 파라미터 효율적 미세조정 기법이다.

### GPTQ

후처리 양자화 방식 중 하나로, 캘리브레이션 데이터를 사용해 가중치 양자화 오차를 줄이는 데 초점을 둔다.

### W8A8

가중치와 activation 모두를 8비트 수준으로 다루는 양자화 설정을 의미한다.
