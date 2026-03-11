import argparse
import gc
import os
import zipfile
from pathlib import Path
from typing import List

# ============================================================
# 0) 공통 import + 실험 설정값 모음
# ------------------------------------------------------------
# 이 스크립트는 QLoRA_baseline_3000_val.ipynb 흐름을
# 로컬 머신 실행용으로 옮긴 버전입니다.
# 이 구간에서 실험의 핵심 하이퍼파라미터를 한 번에 관리합니다.
# (학습/평가/양자화/저장 경로)
# ============================================================
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

try:
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
except ImportError:
    oneshot = None
    GPTQModifier = None


SEED = 42
NUM_TOTAL_SAMPLES = 3000
NUM_EVAL_SAMPLES = 150
MAX_TRAIN_SEQ_LEN = 1024
MIN_PROMPT_TOKENS = 128
TRAIN_BATCH_SIZE = 1
GRAD_ACC = 16
LEARNING_RATE = 1e-5
NUM_TRAIN_EPOCHS = 1
WARMUP_RATIO = 0.05
EVAL_STEPS = 20
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["auto"]
NUM_CALIBRATION_SAMPLES = 1024
MAX_CALIB_SEQ_LEN = 1024
IGNORE_MODE = "base"
DEEP_PROTECT_RATIO = 0.0
GPTQ_DAMPENING_FRAC = 0.001
GPTQ_BLOCK_SIZE = 128
SAVE_COMPRESSED = False
MERGE_DTYPE = "fp16"
DATASET_SPLIT = "train"


# ---------------------------
# 로컬/환경 변수 기반 기본 경로
# ---------------------------
def parse_args():
    repo_root = Path(__file__).resolve().parent
    local_open_dir = repo_root / "open"
    local_model_dir = local_open_dir / "base_model"
    local_dataset_dir = local_open_dir / "dataset"
    default_workspace = repo_root / "local_outputs"
    default_model = str(local_model_dir) if local_model_dir.exists() else "LGAI-EXAONE/EXAONE-4.0-1.2B"
    default_dataset = str(local_dataset_dir) if local_dataset_dir.exists() else "LGAI-EXAONE/MANTA-1M"
    parser = argparse.ArgumentParser(description="Run local QLoRA baseline training.")
    parser.add_argument("--workspace", default=os.environ.get("WORKSPACE", str(default_workspace)))
    parser.add_argument("--open-dir", default=os.environ.get("OPEN_DIR", str(local_open_dir)))
    parser.add_argument("--model-id", default=os.environ.get("MODEL_ID", default_model))
    parser.add_argument("--dataset-id", default=os.environ.get("DATASET_ID", default_dataset))
    parser.add_argument("--run-gptq", action="store_true", help="Run merge + GPTQ + zip after LoRA training.")
    parser.add_argument("--save-compressed", action="store_true", help="Save compressed GPTQ weights if supported.")
    return parser.parse_args()


# ============================================================
# 1) 유틸 함수
# ------------------------------------------------------------
# - 경로 생성
# - GPU/BF16 지원 여부
# - HF 캐시 경로 고정(재실행 시 다운로드 재사용)
# - LoRA target 자동 탐지
# - GPTQ 저장 전 quantization format 정리(필요 시)
# ============================================================
def ensure_dir(path) -> str:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def cuda_available() -> bool:
    """CUDA 사용 가능 여부."""
    available = torch.cuda.is_available()
    if not available:
        print("[WARN] CUDA is not available. Falling back to CPU.")
    return available


def bf16_supported() -> bool:
    """현재 GPU에서 BF16을 안정적으로 쓸 수 있는지 확인."""
    return cuda_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()


def pick_compute_dtype():
    """4bit 학습 시 내부 계산 dtype 선택 (BF16 > FP16 > FP32)."""
    if bf16_supported():
        return torch.bfloat16
    return torch.float16 if cuda_available() else torch.float32


def setup_hf_cache(workspace: str) -> None:
    """HF 캐시를 로컬 workspace 아래로 고정해 재실행/재학습 시 재사용."""
    cache_root = Path(workspace) / ".cache" / "huggingface"
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "transformers"))
    ensure_dir(cache_root)
    ensure_dir(cache_root / "datasets")
    ensure_dir(cache_root / "transformers")


def resolve_existing_path(path_or_id: str):
    candidate = Path(path_or_id).expanduser()
    return str(candidate.resolve()) if candidate.exists() else path_or_id


def load_training_dataset(dataset_source: str):
    source_path = Path(dataset_source).expanduser()
    if source_path.exists():
        csv_files = sorted(source_path.glob("*.csv"))
        jsonl_files = sorted(source_path.glob("*.jsonl"))
        json_files = sorted(source_path.glob("*.json"))
        parquet_files = sorted(source_path.glob("*.parquet"))

        if csv_files:
            data_files = {f.stem: str(f) for f in csv_files}
            split = "train" if "train" in data_files else next(iter(data_files))
            ds = load_dataset("csv", data_files=data_files, split=split)
            print(f"[INFO] loaded local CSV dataset from {source_path}")
            return ds
        if jsonl_files or json_files:
            files = jsonl_files or json_files
            data_files = {f.stem: str(f) for f in files}
            split = "train" if "train" in data_files else next(iter(data_files))
            ds = load_dataset("json", data_files=data_files, split=split)
            print(f"[INFO] loaded local JSON dataset from {source_path}")
            return ds
        if parquet_files:
            data_files = {f.stem: str(f) for f in parquet_files}
            split = "train" if "train" in data_files else next(iter(data_files))
            ds = load_dataset("parquet", data_files=data_files, split=split)
            print(f"[INFO] loaded local Parquet dataset from {source_path}")
            return ds

    return load_dataset(dataset_source, split=DATASET_SPLIT)


def infer_lora_targets_from_model(model, preferred_suffixes):
    """
    모델 내부 Linear 모듈 이름을 스캔해서,
    preferred suffix(q_proj, k_proj ...) 중 실제 존재하는 suffix만 반환.
    """
    candidates = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        suffix = name.split(".")[-1]
        if suffix in preferred_suffixes:
            candidates.append(name)

    suffix_hits = []
    for suffix in preferred_suffixes:
        if any(name.split(".")[-1] == suffix for name in candidates):
            suffix_hits.append(suffix)

    if not suffix_hits:
        raise RuntimeError("LoRA target auto-detection failed.")
    return suffix_hits


def get_deep_ignore_patterns(model: torch.nn.Module, protect_ratio: float = 0.0) -> List[str]:
    """
    GPTQ에서 마지막 N% 레이어를 보호(양자화 제외)할 때 사용할 ignore prefix 생성.
    protect_ratio=0.0이면 보호 안 함.
    """
    if protect_ratio <= 0:
        return []

    total_layers = 0
    layer_prefix = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        total_layers, layer_prefix = len(model.model.layers), "model.layers"
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        total_layers, layer_prefix = len(model.transformer.h), "transformer.h"

    if total_layers == 0 or layer_prefix is None:
        return []

    num_protect = max(1, int(total_layers * protect_ratio))
    start_idx = total_layers - num_protect
    return [f"{layer_prefix}.{i}." for i in range(start_idx, total_layers)]


def fix_quantization_format_warning(model: torch.nn.Module, save_compressed: bool) -> None:
    """
    save_compressed=True일 때만 quantization format 메타를 재정리.
    False면 baseline.py와 동일하게 즉시 return(no-op).
    """
    if not save_compressed:
        return

    for module in model.modules():
        qs = getattr(module, "quantization_scheme", None)
        if qs is None:
            continue
        if hasattr(qs, "format") and getattr(qs, "format") is not None:
            setattr(qs, "format", None)

    try:
        from llmcompressor.transformers.compression.quantization_format import (
            infer_and_set_per_module_quantization_format,
        )

        infer_and_set_per_module_quantization_format(
            model,
            quantization_format=None,
            save_compressed=True,
            sparsity_structure=None,
        )
    except Exception as exc:
        print(f"[WARN] quantization format reset skipped ({type(exc).__name__}: {exc})")


# ============================================================
# 2) 모델/토크나이저 로드 + 데이터 3000 샘플 구성
# ------------------------------------------------------------
# 순서:
# 1) 토크나이저 준비
# 2) 4bit QLoRA 학습용 모델 로드
# 3) 로컬 폴더 또는 HF 데이터셋 로드 -> shuffle(seed=42) -> 3000개 절단
# 4) 랜덤 150개를 eval로 분리(5%)
# ============================================================
def preprocess_train(example, tokenizer, eos_id):
    # ============================================================
    # 3) 전처리: SFT 학습 입력/라벨 생성
    # ------------------------------------------------------------
    # 규칙:
    # - conversations 마지막 turn이 assistant인 샘플만 사용
    # - prompt는 loss 제외(-100), answer 토큰만 loss 계산
    # - max_seq_len(1024) 초과 시 baseline.py 방식으로 절단
    # ============================================================
    convs = example.get("conversations", None)
    if not convs or not isinstance(convs, list):
        return {"input_ids": [], "attention_mask": [], "labels": []}

    # SFT 포맷: 마지막이 assistant 응답이어야 정답(label) 생성 가능
    if convs[-1].get("role") != "assistant":
        return {"input_ids": [], "attention_mask": [], "labels": []}

    # prompt: assistant 마지막 턴 전까지, answer: 마지막 assistant 내용
    prompt_convs = convs[:-1]
    answer_text = convs[-1].get("content", "") or ""

    # prompt는 chat_template 적용, answer는 raw tokenize
    prompt_text = tokenizer.apply_chat_template(
        prompt_convs,
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]

    # answer 끝에 eos 보장
    if not answer_ids or answer_ids[-1] != eos_id:
        answer_ids = answer_ids + [eos_id]

    # 길이 제한: baseline.py와 동일한 절단 정책
    total = len(prompt_ids) + len(answer_ids)
    max_len = MAX_TRAIN_SEQ_LEN
    if total > max_len:
        # 프롬프트는 뒤쪽 정보가 더 중요하다고 보고 tail 유지
        keep_prompt = min(len(prompt_ids), max_len)
        keep_prompt = max(keep_prompt, MIN_PROMPT_TOKENS)
        keep_prompt = min(keep_prompt, len(prompt_ids))
        prompt_ids = prompt_ids[-keep_prompt:]

        # 남은 budget 내에서 answer 유지
        budget = max_len - len(prompt_ids)
        if budget <= 0:
            prompt_ids = prompt_ids[-(max_len - 1):]
            answer_ids = [eos_id]
        elif len(answer_ids) > budget:
            answer_ids = answer_ids[: max(1, budget - 1)] + [eos_id]

    # 최종 입력
    input_ids = prompt_ids + answer_ids
    attention_mask = [1] * len(input_ids)

    # prompt 구간은 loss 제외, answer 구간만 학습
    labels = [-100] * len(prompt_ids) + answer_ids

    # 고정 길이 패딩(배치 텐서화 안정화)
    pad_id = tokenizer.pad_token_id or eos_id
    pad_len = max_len - len(input_ids)
    if pad_len > 0:
        input_ids += [pad_id] * pad_len
        attention_mask += [0] * pad_len
        labels += [-100] * pad_len

    # 안전 장치: 마지막 토큰 eos 보정
    input_ids[-1] = eos_id
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def plot_losses(history, workspace: str) -> None:
    train_logs = [(x.get("step"), x.get("loss")) for x in history if "loss" in x and "eval_loss" not in x]
    eval_logs = [(x.get("step"), x.get("eval_loss")) for x in history if "eval_loss" in x]

    if train_logs:
        last_step, last_train_loss = train_logs[-1]
        print(f"[INFO] last train loss: step={last_step}, loss={last_train_loss:.6f}")
    else:
        print("[WARN] no train loss logs found.")

    if eval_logs:
        last_eval_step, last_eval_loss = eval_logs[-1]
        print(f"[INFO] last eval loss: step={last_eval_step}, eval_loss={last_eval_loss:.6f}")
    else:
        print("[WARN] no eval loss logs found.")

    try:
        import matplotlib.pyplot as plt

        if not train_logs and not eval_logs:
            return

        plt.figure(figsize=(8, 4.5))
        if train_logs:
            ts, tl = zip(*train_logs)
            plt.plot(ts, tl, label="train loss", marker="o", linewidth=1)
        if eval_logs:
            es, el = zip(*eval_logs)
            plt.plot(es, el, label="eval loss", marker="s", linewidth=1)
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("Training vs Validation Loss")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plot_path = Path(workspace) / "train_eval_loss.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[INFO] loss plot saved: {plot_path}")
    except Exception as exc:
        print(f"[WARN] loss plot skipped: {exc}")


# ============================================================
# 5) (선택) Merge + GPTQ + zip
# ------------------------------------------------------------
# - 로컬에서 제출용 모델이 필요할 때 실행
# - 학습만 확인하려면 이 단계는 건너뛰어도 됩니다.
# ============================================================
def run_gptq_pipeline(model_id, dataset_id, tokenizer, adapter_dir, out_dir, out_zip, save_compressed):
    if oneshot is None or GPTQModifier is None:
        raise ImportError("llmcompressor is required for --run-gptq.")

    # Merge 단계 dtype 결정
    merge_dtype = torch.float16 if MERGE_DTYPE == "fp16" else torch.bfloat16
    if not cuda_available():
        merge_dtype = torch.float32

    # 베이스 모델 로드 -> LoRA adapter 결합 -> 단일 모델로 merge
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=merge_dtype,
        device_map=None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    base_model = PeftModel.from_pretrained(base_model, adapter_dir)
    base_model = base_model.merge_and_unload()
    base_model.eval()
    base_model.config.use_cache = True
    if cuda_available():
        base_model.to("cuda")

    # GPTQ에서 보호할 모듈 preset
    ignore_presets = {
        "base": ["embed_tokens", "lm_head"],
        "plus_o": ["embed_tokens", "lm_head", "o_proj"],
        "plus_down": ["embed_tokens", "lm_head", "down_proj"],
        "plus_o_down": ["embed_tokens", "lm_head", "o_proj", "down_proj"],
        "attn_all": ["embed_tokens", "lm_head", "q_proj", "k_proj", "v_proj", "o_proj"],
    }
    base_ignore = ignore_presets.get(IGNORE_MODE, ignore_presets["base"])
    deep_patterns = get_deep_ignore_patterns(base_model, protect_ratio=DEEP_PROTECT_RATIO)
    final_ignore = list(base_ignore) + list(deep_patterns)

    # 캘리브 데이터 준비 (assistant 응답 제거한 prompt만 사용)
    calib_raw = load_dataset(
        dataset_id,
        split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]",
    )

    def to_calib_text(example):
        convs = example.get("conversations", None)
        if not convs:
            return {"text": ""}
        if convs and convs[-1].get("role") == "assistant":
            convs = convs[:-1]
        if not convs:
            return {"text": ""}
        return {
            "text": tokenizer.apply_chat_template(
                convs,
                add_generation_prompt=True,
                tokenize=False,
            )
        }

    calib_ds = calib_raw.map(to_calib_text, remove_columns=calib_raw.column_names)
    calib_ds = calib_ds.filter(lambda x: len(x.get("text", "").strip()) > 0)

    # GPTQ 양자화 (W8A8)
    print("[INFO] GPTQ W8A8 starting")
    recipe = [
        GPTQModifier(
            scheme="W8A8",
            targets=["Linear"],
            ignore=final_ignore,
            actorder="static",
            dampening_frac=GPTQ_DAMPENING_FRAC,
            block_size=GPTQ_BLOCK_SIZE,
            offload_hessians=False,
        )
    ]
    oneshot(
        model=base_model,
        dataset=calib_ds,
        recipe=recipe,
        max_seq_length=MAX_CALIB_SEQ_LEN,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )
    print("[INFO] GPTQ complete")

    # save_compressed=True일 때만 format 재정리 로직 동작
    fix_quantization_format_warning(base_model, save_compressed=save_compressed)

    # 모델 저장
    save_kwargs = {}
    if save_compressed:
        save_kwargs["save_compressed"] = True
    base_model.save_pretrained(out_dir, safe_serialization=True, **save_kwargs)
    tokenizer.save_pretrained(out_dir)
    print(f"[INFO] quantized model saved: {out_dir}")

    # 제출 형식 zip 생성 (model/ 하위 구조)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in Path(out_dir).rglob("*"):
            if file_path.is_file():
                zf.write(file_path, "model/" + str(file_path.relative_to(out_dir)).replace("\\", "/"))
    print(f"[INFO] submission zip saved: {out_zip}")

    del base_model
    gc.collect()
    if cuda_available():
        torch.cuda.empty_cache()


def main():
    # 로컬 캐시/출력 경로 준비
    args = parse_args()
    workspace = str(Path(args.workspace).resolve())
    open_dir = str(Path(args.open_dir).resolve())
    adapter_dir = str(Path(workspace) / "qlora_adapter_3000_val")
    out_dir = str(Path(workspace) / "out" / "model_w8a8_3000_val")
    out_zip = str(Path(workspace) / "submit_3000_val.zip")
    ckpt_dir = ensure_dir(Path(workspace) / "qlora_ckpt_3000_val")
    model_source = resolve_existing_path(args.model_id)
    dataset_source = resolve_existing_path(args.dataset_id)

    ensure_dir(workspace)
    ensure_dir(open_dir)
    ensure_dir(adapter_dir)
    ensure_dir(out_dir)
    setup_hf_cache(workspace)

    print(f"[INFO] WORKSPACE={workspace}")
    print(f"[INFO] OPEN_DIR={open_dir}")
    print(f"[INFO] MODEL_ID={model_source}")
    print(f"[INFO] DATASET_ID={dataset_source}")

    compute_dtype = pick_compute_dtype()
    # tokenizer 준비 (pad token 없으면 eos로 대체)
    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_id = tokenizer.eos_token_id

    # QLoRA 4bit 설정 (NF4 + double quant)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    print("[INFO] loading model (4bit NF4 + DQ)")
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        quantization_config=bnb_config,
        device_map="auto" if cuda_available() else None,
        trust_remote_code=True,
    )
    # k-bit 학습 준비 + gradient checkpointing
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # 학습 중에는 use_cache 비활성화가 일반적
    model.config.use_cache = False

    print(f"[INFO] loading dataset: {dataset_source}")
    raw = load_training_dataset(dataset_source)
    # 요청사항: 반드시 seed=42로 셔플 후 샘플 추출
    raw = raw.shuffle(seed=SEED)
    raw3000 = raw.select(range(min(NUM_TOTAL_SAMPLES, len(raw))))
    # 요청사항: 3000 중 랜덤 150개를 eval로 분리
    splits = raw3000.train_test_split(test_size=NUM_EVAL_SAMPLES, seed=SEED, shuffle=True)
    train_raw = splits["train"]
    eval_raw = splits["test"]
    print(f"[INFO] raw train={len(train_raw)}, raw eval={len(eval_raw)}")

    # train/eval 각각 동일 전처리 적용
    train_ds = train_raw.map(
        lambda x: preprocess_train(x, tokenizer, eos_id),
        remove_columns=train_raw.column_names,
    )
    train_ds = train_ds.filter(lambda x: len(x["input_ids"]) > 0)

    eval_ds = eval_raw.map(
        lambda x: preprocess_train(x, tokenizer, eos_id),
        remove_columns=eval_raw.column_names,
    )
    eval_ds = eval_ds.filter(lambda x: len(x["input_ids"]) > 0)
    print(f"[INFO] final train={len(train_ds)}, eval={len(eval_ds)}")

    # ============================================================
    # 4) LoRA 적용 + Trainer 학습
    # ------------------------------------------------------------
    # 요청사항 반영:
    # - warmup_ratio=0.05
    # - eval_dataset 사용으로 학습 중 val loss 확인
    # - 검증 주기 eval_steps=20
    # ============================================================

    # auto 모드면 모델에 실제 존재하는 target suffix만 사용
    preferred = (
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        if LORA_TARGETS == ["auto"]
        else LORA_TARGETS
    )
    lora_targets = infer_lora_targets_from_model(model, preferred)
    print(f"[INFO] LoRA target_modules={lora_targets}")

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_targets,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    use_bf16 = bf16_supported()
    use_fp16 = cuda_available() and not use_bf16

    # 체크포인트 저장 경로
    training_args = TrainingArguments(
        output_dir=ckpt_dir,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        logging_steps=20,

        # validation loss를 학습 중 주기적으로 확인
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=500,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to=[],
        optim="adamw_torch",
        remove_unused_columns=False,

        # 가장 좋은 eval_loss 체크포인트 로드
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
    )

    print("[INFO] starting QLoRA training")
    trainer.train()
    print("[INFO] QLoRA training complete")

    # ------------------------------------------------------------
    # 학습 로그 요약: train loss / validation(eval) loss 모두 확인
    # ------------------------------------------------------------
    plot_losses(trainer.state.log_history, workspace)

    # 학습된 LoRA adapter 저장
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"[INFO] LoRA adapter saved: {adapter_dir}")

    if args.run_gptq:
        run_gptq_pipeline(
            model_id=model_source,
            dataset_id=dataset_source,
            tokenizer=tokenizer,
            adapter_dir=adapter_dir,
            out_dir=out_dir,
            out_zip=out_zip,
            save_compressed=args.save_compressed or SAVE_COMPRESSED,
        )


if __name__ == "__main__":
    main()
