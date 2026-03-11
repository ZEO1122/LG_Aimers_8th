"""
LoRA 학습 → LoRA 병합(merge) → GPTQ(W8A8) 양자화 → 결과 zip 패키징까지 한 번에 수행하는 스크립트.

[이번 패치(필수 2개)]
✅ PATCH #1) keep_prompt < 0 (answer가 max_len보다 길어 prompt를 0으로 만들던 케이스) 제거
   - prompt_len=0 금지: 최소 prompt 토큰(MIN_PROMPT_TOKENS)을 항상 유지
   - 그만큼 answer를 줄임

✅ PATCH #2) answer_trimmed(=answer를 자르는 케이스)에서 EOS(endofturn) 무조건 보존
   - answer_ids를 budget에 맞춰 자르되, 마지막 토큰은 항상 eos_id로 강제

추가로:
- default_data_collator 유지(labels 덮어쓰기 방지)
- device_map="auto" 제거 + gradient checkpointing
- GPTQ 캘리브는 add_generation_prompt=True 유지

실행:
python lora_merge_gptq_patched_no_smoothquant.py
"""

import os
import gc
import shutil

import torch
from datasets import load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from peft import LoraConfig, get_peft_model

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier


# -----------------------------
# 0) (권장) CUDA 메모리 파편화 완화
# -----------------------------
os.environ.setdefault(
    "PYTORCH_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:128"
)


# -----------------------------
# 1) 사용자 설정
# -----------------------------
MODEL_ID = "../open/base_model"

DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

# LoRA 학습
NUM_TRAIN_SAMPLES = 2000
MAX_TRAIN_SEQ_LEN = 1024

# ✅ PATCH #1: prompt_len=0 금지를 위한 최소 프롬프트 토큰
# - prompt 평균이 ~115였으니 96~128 추천 (너무 키우면 answer가 더 잘림)
MIN_PROMPT_TOKENS = 128

# GPTQ 캘리브
NUM_CALIBRATION_SAMPLES = 1024
MAX_CALIB_SEQ_LEN = 1024

OUT_DIR = "./model"           # ✅ vLLM 비교 시 이 경로를 --model로 사용
LORA_DIR = "./lora_adapter"
MERGED_DIR = "./merged_model"

SCHEME = "W8A8"
TARGETS = ["Linear"]

IGNORE_PRESETS = {
    "base": ["embed_tokens", "lm_head"],
    "plus_o": ["embed_tokens", "lm_head", "o_proj"],
    "plus_down": ["embed_tokens", "lm_head", "down_proj"],
    "plus_o_down": ["embed_tokens", "lm_head", "o_proj", "down_proj"],
    "attn_all": ["embed_tokens", "lm_head", "q_proj", "k_proj", "v_proj", "o_proj"],
}
IGNORE_MODE = "base"
IGNORE = IGNORE_PRESETS[IGNORE_MODE]

# LoRA 하이퍼파라미터
PREFERRED_LORA_TARGETS = ["q_proj", "v_proj", "o_proj", "up_proj", "down_proj"]

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


# -----------------------------
# 2) 유틸: LoRA 타겟 자동 추정
# -----------------------------
def infer_lora_targets(model, preferred):
    names = set()
    for n, _ in model.named_modules():
        tail = n.split(".")[-1]
        if tail in preferred:
            names.add(tail)
    return sorted(names) if names else preferred


# -----------------------------
# 3) 모델/토크나이저 로드
# -----------------------------
print("[INFO] 모델/토크나이저 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
eos_id = tokenizer.eos_token_id  # [|endofturn|]

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model.config.use_cache = False
model.gradient_checkpointing_enable()

if torch.cuda.is_available():
    model.to("cuda")

print("[INFO] 로드 완료")


# -----------------------------
# 4) LoRA 학습 데이터 로드/전처리 (필수 2개 패치 반영)
# -----------------------------
print("[INFO] LoRA 학습 데이터 로드 중...")
train_raw = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_TRAIN_SAMPLES}]")


def preprocess_train(example):
    """
    목표:
    - labels는 answer 토큰만 학습 (-100 마스킹)
    - (PATCH #1) prompt_len=0 금지: prompt는 최소 MIN_PROMPT_TOKENS 유지
    - (PATCH #2) answer가 잘릴 때도 마지막 토큰은 eos_id 강제
    """
    convs = example["conversations"]
    if not convs or convs[-1].get("role") != "assistant":
        return {"input_ids": [], "attention_mask": [], "labels": []}

    prompt_convs = convs[:-1]
    answer_text = convs[-1].get("content", "") or ""

    # prompt: assistant 시작점까지 포함
    prompt_text = tokenizer.apply_chat_template(
        prompt_convs,
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    # answer: 정답 토큰 + eos 강제
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
    if not answer_ids or answer_ids[-1] != eos_id:
        answer_ids = answer_ids + [eos_id]

    max_len = MAX_TRAIN_SEQ_LEN

    # -------------------------
    # (1) 길이 초과 처리: prompt를 줄이되, 최소 MIN_PROMPT_TOKENS는 유지
    # -------------------------
    total = len(prompt_ids) + len(answer_ids)
    if total > max_len:
        # 먼저 prompt는 뒤쪽을 유지(최근 문맥)
        keep_prompt = min(len(prompt_ids), max_len)  # 일단 상한
        keep_prompt = max(keep_prompt, MIN_PROMPT_TOKENS)  # ✅ PATCH #1 (하한)
        keep_prompt = min(keep_prompt, len(prompt_ids))    # 실제 prompt 길이보다 클 수 없음

        # keep_prompt로 prompt 자르기
        prompt_ids = prompt_ids[-keep_prompt:]

        # 남은 예산을 answer에 배정
        budget_for_answer = max_len - len(prompt_ids)

        if budget_for_answer <= 0:
            # 이론상 거의 없음. 안전 처리: prompt를 더 줄이고 eos만 남김
            prompt_ids = prompt_ids[-(max_len - 1):]
            answer_ids = [eos_id]
        else:
            # -------------------------
            # (2) answer 잘림 처리: EOS 무조건 보존
            # -------------------------
            if len(answer_ids) > budget_for_answer:
                # ✅ PATCH #2: 마지막은 eos_id
                if budget_for_answer == 1:
                    answer_ids = [eos_id]
                else:
                    answer_ids = answer_ids[:budget_for_answer - 1] + [eos_id]

    # 최종
    input_ids = prompt_ids + answer_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + answer_ids

    # (안전) 혹시라도 길이 초과가 남아있으면 최종 컷 (EOS 보존)
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        attention_mask = attention_mask[:max_len]
        labels = labels[:max_len]
        # 마지막 라벨 토큰 eos 강제(정답 구간이 있다면)
        if labels and labels[-1] != -100:
            labels[-1] = eos_id
        # input도 마지막을 eos로(일관성)
        input_ids[-1] = eos_id

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


train_ds = train_raw.map(preprocess_train, remove_columns=train_raw.column_names)

# (검증 로그)
ex0 = train_ds[0]
unmasked = sum(1 for v in ex0["labels"] if v != -100)
print(f"[DEBUG] unmasked labels tokens = {unmasked} / total = {len(ex0['labels'])}")
print(f"[DEBUG] last label token = {ex0['labels'][-1]} (expect eos_id={eos_id} if unmasked)")

print("[INFO] 학습 데이터 전처리 완료")


# -----------------------------
# 5) LoRA 적용 + 학습
# -----------------------------
lora_targets = infer_lora_targets(model, PREFERRED_LORA_TARGETS)
print(f"[INFO] LoRA target_modules = {lora_targets}")

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=lora_targets,
)

model = get_peft_model(model, lora_config)

# LoRA 적용 실패 방지
trainable = [n for n, p in model.named_parameters() if p.requires_grad]
if len(trainable) == 0:
    raise RuntimeError("LoRA 적용 실패: 학습 가능한 파라미터가 0개입니다. target_modules를 확인하세요.")

model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="./lora_ckpt",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=1e-5,
    num_train_epochs=1,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    weight_decay=0.0,
    max_grad_norm=1.0,
    logging_steps=4,
    save_steps=500,
    save_total_limit=2,
    bf16=True,
    fp16=False,
    report_to=[],
    optim="adamw_torch",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    data_collator=default_data_collator,
)

print("[INFO] LoRA 학습 시작...")
trainer.train()
print("[INFO] LoRA 학습 완료")

os.makedirs(LORA_DIR, exist_ok=True)
model.save_pretrained(LORA_DIR)
tokenizer.save_pretrained(LORA_DIR)
print(f"[INFO] LoRA 어댑터 저장 완료: {LORA_DIR}")


# -----------------------------
# 6) LoRA merge
# -----------------------------
print("[INFO] LoRA merge 중...")
merged_model = model.merge_and_unload()
merged_model.eval()
merged_model.config.use_cache = True

os.makedirs(MERGED_DIR, exist_ok=True)
merged_model.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)
print(f"[INFO] merge된 모델 저장 완료: {MERGED_DIR}")

# 메모리 정리
del trainer
del model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# -----------------------------
# 7) GPTQ(W8A8) 양자화 (SmoothQuant 제거 버전)
# -----------------------------
print("[INFO] GPTQ 캘리브레이션 데이터 로드 중...")
calib_raw = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")


def to_calib_text(example):
    convs = example["conversations"]
    # 마지막 assistant 제거 (prompt-only)
    if convs and convs[-1].get("role") == "assistant":
        convs = convs[:-1]

    return {
        "text": tokenizer.apply_chat_template(
            convs,
            add_generation_prompt=True,
            tokenize=False,
        )
    }

calib_ds = calib_raw.map(to_calib_text, remove_columns=calib_raw.column_names)


print(f"[INFO] GPTQ 시작 (scheme={SCHEME}, ignore_mode={IGNORE_MODE})...")
print(f"[INFO] GPTQ ignore list = {IGNORE}")

recipe = [
    GPTQModifier(
        scheme="W8A8",
        targets=["Linear"],
        ignore=IGNORE,
        actorder="static",      # 정확도 회복 best, 런타임 cost 없음(문서)
        dampening_frac=0.001,   # 기본값에 가까움(불안정하면 0.01~0.05 테스트)
        block_size=128,
        offload_hessians=False,
    ),
]

if torch.cuda.is_available():
    merged_model.to("cuda")

oneshot(
    model=merged_model,
    dataset=calib_ds,
    recipe=recipe,
    max_seq_length=MAX_CALIB_SEQ_LEN,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("[INFO] GPTQ 완료")

if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR, ignore_errors=True)
os.makedirs(OUT_DIR, exist_ok=True)

merged_model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)
print(f"[INFO] 양자화 모델 저장 완료: {OUT_DIR}")


# -----------------------------
# 8) zip 패키징
# -----------------------------
zip_name = f"lora_then_gptq_{SCHEME}_{IGNORE_MODE}"
print(f"[INFO] {zip_name}.zip 생성 중...")

shutil.make_archive(
    base_name=zip_name,
    format="zip",
    root_dir=".",
    base_dir=OUT_DIR,
)

print(f"[INFO] 생성 완료: {zip_name}.zip")
