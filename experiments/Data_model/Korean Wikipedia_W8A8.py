import os
import random
import shutil

import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# =========================
# Config (필요한 것만 수정)
# =========================
MODEL_ID = "../open/base_model"
OUT_DIR = "./model"

# Korean Wikipedia (cleaned articles)
WIKI_DATASET_ID = "wikimedia/wikipedia"
WIKI_CONFIG = "20231101.ko"   # 날짜/언어 config (ko: 한국어)
WIKI_SPLIT = "train"

NUM_CALIBRATION_SAMPLES = 216
MAX_SEQUENCE_LENGTH = 512

# Quantization
SCHEME = "W8A8"
TARGETS = ["Linear"]
IGNORE = ["embed_tokens", "lm_head"]

# Reproducibility
SEED = 42

# Chat-style wrapping (평가 프롬프트 분포에 맞추고 싶으면 True 권장)
USE_CHAT_TEMPLATE = True

# 위키 본문이 너무 길면 잘라서 OOM/느려짐 방지 (프롬프트 토큰 여유분 감안)
TRUNCATE_TOKENS = MAX_SEQUENCE_LENGTH - 64

# CUDA 메모리 파편화 완화(선택)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")


def truncate_to_tokens(tokenizer, text: str, max_tokens: int) -> str:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    ids = ids[:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def wrap_as_chat(tokenizer, article_text: str) -> str:
    """
    위키 텍스트를 '일반적인 생성 상황'처럼 만들기 위한 래핑.
    - 계속쓰기(continuation) 형태로 두면 LM 성질을 잘 자극하는 편.
    """
    if USE_CHAT_TEMPLATE and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": (
                    "다음은 한국어 위키백과 문서 일부야.\n"
                    "문체를 유지하면서 자연스럽게 이어서 계속 써줘.\n\n"
                    f"{article_text}\n"
                ),
            }
        ]
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    # chat template이 없거나 끄면 raw text 그대로
    return article_text


def build_calib_dataset(tokenizer) -> Dataset:
    """
    wikimedia/wikipedia(ko)에서 필요한 샘플 수만 streaming으로 뽑아 Dataset으로 만든다.
    """
    print(f"[INFO] 위키 데이터 로드: {WIKI_DATASET_ID} / {WIKI_CONFIG} (streaming=True)")
    stream = load_dataset(
        WIKI_DATASET_ID,
        WIKI_CONFIG,
        split=WIKI_SPLIT,
        streaming=True,
    )

    # streaming shuffle: buffer_size를 너무 키우면 RAM 사용 증가
    stream = stream.shuffle(seed=SEED, buffer_size=10_000)

    examples = []
    taken = 0

    # 여유있게 더 뽑아두고(필터링/빈문서 스킵 대비) 필요한 만큼만 채움
    for ex in stream.take(NUM_CALIBRATION_SAMPLES * 5):
        text = (ex.get("text") or "").strip()
        title = (ex.get("title") or "").strip()

        if not text:
            continue

        # 제목+본문
        article = f"{title}\n\n{text}" if title else text

        # 토큰 기준 truncate
        article = truncate_to_tokens(tokenizer, article, TRUNCATE_TOKENS)

        # chat template로 감싸기(선택)
        final_text = wrap_as_chat(tokenizer, article)

        examples.append({"text": final_text})
        taken += 1
        if taken >= NUM_CALIBRATION_SAMPLES:
            break

    if taken < NUM_CALIBRATION_SAMPLES:
        raise RuntimeError(
            f"샘플을 충분히 못 뽑았어: {taken}/{NUM_CALIBRATION_SAMPLES} "
            f"(config={WIKI_CONFIG} 확인 필요)"
        )

    print(f"[INFO] 캘리브레이션 샘플 생성 완료: {taken}개")
    return Dataset.from_list(examples)


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    print("[INFO] 모델/토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # 캘리브레이션에서는 KV cache 불필요(메모리 절약)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    print("[INFO] 모델/토크나이저 로드 완료")

    print("[INFO] 캘리브레이션 데이터셋 구성 중...")
    ds = build_calib_dataset(tokenizer)

    print(
        f"[INFO] GPTQ 시작 (scheme={SCHEME}, samples={NUM_CALIBRATION_SAMPLES}, max_len={MAX_SEQUENCE_LENGTH})..."
    )

    recipe = [
        GPTQModifier(
            scheme=SCHEME,
            targets=TARGETS,
            ignore=IGNORE,
        )
    ]

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )

    print("[INFO] GPTQ 완료")

    os.makedirs(OUT_DIR, exist_ok=True)

    model.save_pretrained(OUT_DIR, save_compressed=True)
    tokenizer.save_pretrained(OUT_DIR)

    print(f"[INFO] 모델 저장 완료: {OUT_DIR}")

    zip_name = f"baseline_{SCHEME}_wiki_ko"
    print(f"[INFO] {zip_name}.zip 생성 중...")

    shutil.make_archive(
        base_name=zip_name,
        format="zip",
        root_dir=".",
        base_dir=OUT_DIR,
    )

    print(f"[INFO] 생성 완료: {zip_name}.zip")


if __name__ == "__main__":
    main()
