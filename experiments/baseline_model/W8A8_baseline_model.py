import os
import shutil
from pathlib import Path  # (참고) 현재 코드에서는 Path를 직접 사용하지 않음

import torch

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# -----------------------------
# 1) 사용자 설정(경로/데이터/하이퍼파라미터)
# -----------------------------

# 로드할 베이스 모델 경로(로컬 디렉토리). 예: ../open/base_model
# 양자화(GPTQ)하기 전의 원본 LLM, 이 원본을 불러와서 W8A8로 양자화한 뒤, 결과를 OUT_DIR에 저장
MODEL_ID = "../open/base_model"

# 양자화된 모델 및 토크나이저를 저장할 출력 폴더
OUT_DIR = "./model"

# 캘리브레이션(quantization calibration)에 사용할 데이터셋
# LGAI-EXAONE/MANTA-1: 파인튜닝용 대화 데이터셋. 
# Question Answering 구조, Computer Science & Coding 24.82%, Natural Sciences 22.39%, Social Sciences 21.21%, Mathematics 17.37%
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

# calibration: 데이터셋 일부를 모델에 흘려보내서 중간 활성화 값과 레이어 입력/출력 값들의 분포 측정하고, 그걸 바탕으로 양자화에 필요한 기준(스케일/클리핑/양자화 파라미터)을 정하는 과정이야.
# Scale: 실수 값과 정수 값 사이의 환산 비율
# Clipping: 8bit 정수 범위를 넘는 값은 잘라버리는 것
# 양자화 파라미터(scale, zero-point, qmin, qmax...)
# zero-point: 정수 범위를 0 중심이 아니라 어떤 값에 맞춰 이동시키는 오프셋
NUM_CALIBRATION_SAMPLES = 256

# 한 샘플에 대해 사용할 최대 시퀀스 길이(토큰 길이). 길수록 메모리/시간 증가
MAX_SEQUENCE_LENGTH = 512

# -----------------------------
# 2) Quantization(GPTQ) 설정
# -----------------------------

# W8A8: Weight 8-bit, Activation 8-bit
SCHEME = "W8A8"

# 양자화를 적용할 레이어 타입(보통 Linear에 적용)
TARGETS = ["Linear"]

# 민감도가 높은 모듈(임베딩, lm_head)은 제외하는 경우가 많음
IGNORE = ["embed_tokens", "lm_head"]

# -----------------------------
# 3) 모델 / 토크나이저 로드
# -----------------------------

print("[INFO] 모델 로드 중...")

# 모델에 맞는 토크나이저 로드
# trust_remote_code=True:
#   - 모델/토크나이저가 커스텀 코드를 포함할 때 해당 코드를 실행하도록 허용
#   - 외부 모델을 받을 때는 보안상 주의 필요
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

# 생성용(CausalLM) 모델 로드
# torch_dtype=torch.bfloat16:
#   - bfloat16으로 로드하여 메모리 사용량을 줄이고(환경에 따라) 속도 이점을 기대
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
)

print("[INFO] 모델/토크나이저 로드 완료")

# -----------------------------
# 4) 캘리브레이션 데이터 로드 및 전처리
# -----------------------------

print("[INFO] 캘리브레이션 데이터 로드 중...")

# split=f"train[:256]" 형태로 앞에서부터 NUM_CALIBRATION_SAMPLES개만 로드
# 전체 1M를 다 쓰지 않고 일부만 써서 빠르게 캘리브레이션 진행
ds = load_dataset(
    DATASET_ID,
    split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]",
)


def preprocess(example):
    """데이터셋 샘플을 모델 입력용 텍스트로 변환.

    MANTA-1M은 대화형 데이터로 conversations 필드를 가지고 있음.
    tokenizer.apply_chat_template을 사용하면 모델이 기대하는 채팅 포맷으로
    system/user/assistant 메시지를 합쳐 하나의 텍스트 프롬프트로 만들 수 있음.

    - add_generation_prompt=True:
        모델이 다음 답변을 생성해야 하는 위치(assistant turn)까지 프롬프트를 구성
    - tokenize=False:
        여기서는 토큰 ids가 아니라 '문자열 텍스트'로 반환
    """
    return {
        "text": tokenizer.apply_chat_template(
            example["conversations"],
            add_generation_prompt=True,
            tokenize=False,
        )
    }


# 각 샘플에 preprocess를 적용해서 "text" 컬럼을 생성
ds = ds.map(preprocess)

print("[INFO] 데이터 전처리 완료")

# -----------------------------
# 5) GPTQ(oneshot) 실행
# -----------------------------

print(
    f"[INFO] GPTQ 시작 (scheme={SCHEME}, samples={NUM_CALIBRATION_SAMPLES}, max_len={MAX_SEQUENCE_LENGTH})..."
)

# llmcompressor는 'recipe'라는 리스트에 modifier들을 넣어 실행하는 구조
# 여기서는 GPTQModifier 하나만 사용
recipe = [
    GPTQModifier(
        scheme=SCHEME,  # W8A8 등 양자화 스킴
        targets=TARGETS,  # Linear 레이어 대상
        ignore=IGNORE,  # embed_tokens/lm_head 제외
    )
]

# oneshot:
#   - dataset을 이용해 캘리브레이션을 수행하고
#   - recipe에 정의된 방식(GPTQ)으로 모델을 양자화
#   - 결과는 model 객체에 반영(in-place)되는 방식이 일반적
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("[INFO] GPTQ 완료")

# -----------------------------
# 6) 저장 및 압축(zip) 패키징
# -----------------------------

# 출력 폴더 생성(이미 있으면 무시)
os.makedirs(OUT_DIR, exist_ok=True)

# save_compressed=True:
#   - llmcompressor가 생성한 압축/양자화 가중치를 저장할 때 사용
model.save_pretrained(OUT_DIR, save_compressed=True)

# 토크나이저도 함께 저장(추론/배포 시 필수)
tokenizer.save_pretrained(OUT_DIR)

print(f"[INFO] 모델 저장 완료: {OUT_DIR}")

# 결과 폴더를 zip으로 묶어 제출/배포하기 쉽게 만듦
zip_name = "baseline_W8A8"
print(f"[INFO] {zip_name}.zip 생성 중...")

# root_dir="." 기준으로 base_dir=OUT_DIR 폴더를 zip 압축
shutil.make_archive(
    base_name=zip_name,
    format="zip",
    root_dir=".",
    base_dir=OUT_DIR,
)

print(f"[INFO] 생성 완료: {zip_name}.zip")