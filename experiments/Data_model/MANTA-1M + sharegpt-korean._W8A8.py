import os
import random
import shutil

import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# =========================
# Config
# =========================
MODEL_ID = "../open/base_model"
OUT_DIR = "./model"

# 총 캘리브레이션 샘플 수
NUM_CALIBRATION_SAMPLES = 216
MAX_SEQUENCE_LENGTH = 512

# Quantization
SCHEME = "W8A8"
TARGETS = ["Linear"]
IGNORE = ["embed_tokens", "lm_head"]  # 점수 목적이면 lm_head 포함도 실험 가치 있음

# Reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# ---- Dataset mix (추천) ----
# MANTA-1M (원래 쓰던 데이터)
MANTA_ID = "LGAI-EXAONE/MANTA-1M"
MANTA_SPLIT = "train"
# Korean chat (추천: sharegpt-korean)
KCHAT_ID = "FreedomIntelligence/sharegpt-korean"
KCHAT_SPLIT = "train"

# 혼합 비율 (총 216개 중 MANTA 70%, KCHAT 30%)
MANTA_RATIO = 0.7

# streaming 설정 (큰 데이터일 때 다운로드/메모리 절약)
USE_STREAMING_FOR_MANTA = True
STREAM_BUFFER = 10_000


# =========================
# Helpers
# =========================
def truncate_text_to_max_tokens(tokenizer, text: str, max_tokens: int) -> str:
    """문자열을 토큰 기준으로 잘라 다시 텍스트로 복원(512 제한에서 OOM/비효율 방지용)."""
    ids = tokenizer(text, add_special_tokens=False).input_ids
    ids = ids[:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=False)


def sharegpt_to_messages(convs):
    """
    sharegpt-korean: [{"from":"human","value":"..."}, {"from":"gpt","value":"..."}]
    -> [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]
    """
    msgs = []
    for t in convs:
        frm = (t.get("from") or "").lower().strip()
        val = (t.get("value") or "").strip()
        if not val:
            continue
        if frm in ("human", "user"):
            msgs.append({"role": "user", "content": val})
        elif frm in ("gpt", "assistant"):
            msgs.append({"role": "assistant", "content": val})
        else:
            # 알 수 없는 값은 user로 취급
            msgs.append({"role": "user", "content": val})
    return msgs


def manta_to_text(tokenizer, ex):
    """
    MANTA는 기본적으로 conversations 컬럼을 네가 쓰던 방식처럼 그대로 chat_template에 넣을 수 있는 경우가 많음.
    혹시 다른 포맷이면 최소한의 방어 로직으로 처리.
    """
    convs = ex.get("conversations")

    # case 1) 원래 코드처럼 바로 넣기
    if isinstance(convs, list) and len(convs) > 0 and isinstance(convs[0], dict):
        # 이미 {"role","content"} 구조인 경우가 많음
        if "role" in convs[0] and "content" in convs[0]:
            text = tokenizer.apply_chat_template(convs, add_generation_prompt=True, tokenize=False)
            return truncate_text_to_max_tokens(tokenizer, text, MAX_SEQUENCE_LENGTH)

        # 혹시 sharegpt형이면 변환
        if "from" in convs[0] and "value" in convs[0]:
            msgs = sharegpt_to_messages(convs)
            text = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            return truncate_text_to_max_tokens(tokenizer, text, MAX_SEQUENCE_LENGTH)

    # fallback: content 필드만 뽑아서 단일 user로
    raw = ""
    if isinstance(convs, list):
        parts = []
        for t in convs:
            if isinstance(t, dict):
                parts.append(t.get("content") or t.get("value") or "")
        raw = "\n\n".join([p for p in parts if p.strip()])
    else:
        raw = str(convs) if convs is not None else ""

    if not raw.strip():
        raw = str(ex)

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": raw.strip()}],
        add_generation_prompt=True,
        tokenize=False,
    )
    return truncate_text_to_max_tokens(tokenizer, text, MAX_SEQUENCE_LENGTH)


def kchat_to_text(tokenizer, ex):
    """FreedomIntelligence/sharegpt-korean -> chat_template text"""
    msgs = sharegpt_to_messages(ex["conversations"])
    # 캘리브레이션은 '답변을 생성할 상황'이 중요해서 generation prompt를 붙이는 편이 유리
    text = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return truncate_text_to_max_tokens(tokenizer, text, MAX_SEQUENCE_LENGTH)


def load_manta_subset(tokenizer, n: int) -> Dataset:
    if USE_STREAMING_FOR_MANTA:
        ds_stream = load_dataset(MANTA_ID, split=MANTA_SPLIT, streaming=True)
        ds_stream = ds_stream.shuffle(seed=SEED, buffer_size=STREAM_BUFFER)

        rows = []
        for ex in ds_stream.take(n * 3):  # 스킵 대비 여유
            try:
                text = manta_to_text(tokenizer, ex)
                if text.strip():
                    rows.append({"text": text})
            except Exception:
                continue
            if len(rows) >= n:
                break

        if len(rows) < n:
            raise RuntimeError(f"MANTA 샘플 부족: {len(rows)}/{n}")
        return Dataset.from_list(rows)

    # non-streaming
    ds = load_dataset(MANTA_ID, split=MANTA_SPLIT).shuffle(seed=SEED).select(range(n))
    rows = [{"text": manta_to_text(tokenizer, ex)} for ex in ds]
    return Dataset.from_list(rows)


def load_kchat_subset(tokenizer, n: int) -> Dataset:
    # sharegpt-korean은 6k 수준이라 non-streaming이 편함
    ds = load_dataset(KCHAT_ID, split=KCHAT_SPLIT).shuffle(seed=SEED).select(range(n))
    rows = [{"text": kchat_to_text(tokenizer, ex)} for ex in ds]
    return Dataset.from_list(rows)


# =========================
# Main
# =========================
def main():
    print("[INFO] 모델 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval()

    # 캘리브레이션 시 캐시 끄면 메모리 안정화에 도움
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    print("[INFO] 모델/토크나이저 로드 완료")

    manta_n = int(NUM_CALIBRATION_SAMPLES * MANTA_RATIO)
    kchat_n = NUM_CALIBRATION_SAMPLES - manta_n

    print(f"[INFO] 캘리브레이션 샘플 구성: MANTA={manta_n}, KCHAT={kchat_n}")

    print("[INFO] MANTA 샘플 로드/전처리 중...")
    ds_manta = load_manta_subset(tokenizer, manta_n)

    print("[INFO] 한국어 대화 샘플 로드/전처리 중...")
    ds_kchat = load_kchat_subset(tokenizer, kchat_n)

    ds = concatenate_datasets([ds_manta, ds_kchat]).shuffle(seed=SEED)

    print(f"[INFO] GPTQ 시작 (scheme={SCHEME}, samples={NUM_CALIBRATION_SAMPLES}, max_len={MAX_SEQUENCE_LENGTH})...")

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

    zip_name = f"{SCHEME}_manta_sharegptko_{NUM_CALIBRATION_SAMPLES}s_{MAX_SEQUENCE_LENGTH}len"
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
