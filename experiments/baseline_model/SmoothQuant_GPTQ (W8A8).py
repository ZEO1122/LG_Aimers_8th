import os
import re
import torch
import shutil
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# SmoothQuant import (버전 차이 대비)
try:
    from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
except ImportError:
    from llmcompressor.modifiers.smoothquant.base import SmoothQuantModifier


MODEL_ID = "../open/base_model"
OUT_DIR  = "./model"

DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

NUM_CALIBRATION_SAMPLES = 216
MAX_SEQUENCE_LENGTH = 512

# Quantization
SCHEME = "W8A8"
TARGETS = ["Linear"]
IGNORE  = ["embed_tokens", "lm_head"]

# SmoothQuant
SQ_ALPHA = 0.8
SQ_APPLY_ATTN = False  # ✅ 안전을 위해 False: (q/k/v) SmoothQuant 미적용, MLP(gate/up)만 적용


def build_smoothquant_mappings(model, apply_attn: bool = False):
    """
    EXAONE은 self_attn 내부에 q_norm 같은 구조가 있어서,
    SmoothQuant를 attention(q/k/v)에 잘못 적용하면 품질이 크게 무너질 수 있음.

    여기서는 안전하게:
      - 기본: MLP(gate/up)만 SmoothQuant
      - 옵션: apply_attn=True로 바꾸면 q/k/v도 시도 가능(권장 X)
    """
    layer_pat = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)\.")
    per_layer = defaultdict(list)

    for name, _ in model.named_modules():
        m = layer_pat.search(name)
        if m:
            per_layer[int(m.group(1))].append(name)

    mappings = []
    attn_cnt, mlp_cnt = 0, 0
    skip_attn, skip_mlp = 0, 0

    for i in sorted(per_layer.keys()):
        names = per_layer[i]

        # proj들
        q = next((n for n in names if n.endswith(".q_proj") or n.endswith("q_proj")), None)
        k = next((n for n in names if n.endswith(".k_proj") or n.endswith("k_proj")), None)
        v = next((n for n in names if n.endswith(".v_proj") or n.endswith("v_proj")), None)

        gate = next((n for n in names if n.endswith(".gate_proj") or n.endswith("gate_proj")), None)
        up   = next((n for n in names if n.endswith(".up_proj")   or n.endswith("up_proj")), None)

        # smooth layer 후보: layernorm/rmsnorm 계열만
        norm_like = []
        for n in names:
            ln = n.lower()
            if ("layernorm" in ln) or ("rmsnorm" in ln):
                # self_attn 내부 q_norm/k_norm/v_norm 같은 것은 제외
                if (".self_attn." in ln) and (ln.endswith("q_norm") or ln.endswith("k_norm") or ln.endswith("v_norm")):
                    continue
                norm_like.append(n)

        # (옵션) attention smooth: 기본은 꺼둠
        if apply_attn and q and k and v:
            # attention쪽에 맞을 법한 norm 우선
            cand = next((n for n in norm_like if "post_attention" not in n.lower() and "feedforward" not in n.lower()), None)
            if cand is None and norm_like:
                cand = norm_like[0]
            if cand is None:
                skip_attn += 1
            else:
                mappings.append([[q, k, v], cand])
                attn_cnt += 1

        # MLP smooth: 가능한 경우 적용
        if gate and up:
            # EXAONE에서 보이는 이름을 우선 사용 (예: post_feedforward_layernorm)
            cand = next((n for n in norm_like if "post_feedforward" in n.lower()), None)
            if cand is None:
                # 일반적인 이름 후보
                cand = next((n for n in norm_like if "post_attention" in n.lower() or "ln_2" in n.lower() or "norm2" in n.lower()), None)
            if cand is None and norm_like:
                cand = norm_like[-1]

            if cand is None:
                skip_mlp += 1
            else:
                mappings.append([[gate, up], cand])
                mlp_cnt += 1

    print(f"[INFO] SmoothQuant mappings: attn={attn_cnt} (skip {skip_attn}), mlp={mlp_cnt} (skip {skip_mlp})")

    if not mappings:
        raise RuntimeError("SmoothQuant mappings 생성 실패: 사용할 norm layer를 찾지 못했습니다.")

    return mappings


print("[INFO] 모델 로드 중...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
).eval()

# (메모리 안정화)
if hasattr(model.config, "use_cache"):
    model.config.use_cache = False

print("[INFO] 모델/토크나이저 로드 완료")

# ✅ SmoothQuant mappings 생성 (베이스라인에서 최소 추가)
print("[INFO] SmoothQuant mappings 생성 중...")
sq_mappings = build_smoothquant_mappings(model, apply_attn=SQ_APPLY_ATTN)
print(f"[INFO] SmoothQuant mappings 개수: {len(sq_mappings)} (예시 2개)")
for m in sq_mappings[:2]:
    print("  ", m)

print("[INFO] 캘리브레이션 데이터 로드 중...")

ds = load_dataset(
    DATASET_ID,
    split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]",
)

def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["conversations"],
            add_generation_prompt=True,
            tokenize=False)
    }

ds = ds.map(preprocess, remove_columns=ds.column_names)

# ✅ SmoothQuant / GPTQ 파이프라인 안정화를 위해 토크나이즈해서 넘김
def tokenize_fn(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        add_special_tokens=False,
    )

ds = ds.map(tokenize_fn, remove_columns=ds.column_names)

print("[INFO] 데이터 전처리 완료")

print(f"[INFO] SmoothQuant + GPTQ 시작 (alpha={SQ_ALPHA}, scheme={SCHEME}, samples={NUM_CALIBRATION_SAMPLES}, max_len={MAX_SEQUENCE_LENGTH})...")

recipe = [
    SmoothQuantModifier(
        smoothing_strength=SQ_ALPHA,
        mappings=sq_mappings,   # ✅ EXAONE에서 자동추론 실패하므로 직접 제공
        ignore=None,
    ),
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

print("[INFO] SmoothQuant + GPTQ 완료")

os.makedirs(OUT_DIR, exist_ok=True)

# ✅ (중요) EOS/PAD & generation_config 저장: “정답 끝맺음” 꼬임 방지
# tokenizer에 pad가 없으면 eos로 대체(모델에 따라 필요)
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token

# 모델 config에도 반영
if getattr(model.config, "eos_token_id", None) is None and tokenizer.eos_token_id is not None:
    model.config.eos_token_id = tokenizer.eos_token_id
if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
    model.config.pad_token_id = tokenizer.pad_token_id

# base 모델 generation_config를 가져와 저장(없으면 현재 모델 config 기반으로 생성)
try:
    gen_cfg = GenerationConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
except Exception:
    gen_cfg = GenerationConfig.from_model_config(model.config)

# eos/pad를 generation_config에도 명시
if gen_cfg.eos_token_id is None and tokenizer.eos_token_id is not None:
    gen_cfg.eos_token_id = tokenizer.eos_token_id
if gen_cfg.pad_token_id is None and tokenizer.pad_token_id is not None:
    gen_cfg.pad_token_id = tokenizer.pad_token_id

model.generation_config = gen_cfg

model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)
gen_cfg.save_pretrained(OUT_DIR)

print(f"[INFO] 모델 저장 완료: {OUT_DIR}")

zip_name = "SQ_GPTQ_W8A8_MLPONLY"
print(f"[INFO] {zip_name}.zip 생성 중...")

shutil.make_archive(
    base_name=zip_name,
    format="zip",
    root_dir=".",
    base_dir=OUT_DIR,
)

print(f"[INFO] 생성 완료: {zip_name}.zip")
