import time
import math
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from neural_compressor import GPTQModifier, oneshot

# === 사용자 설정 ===
MODEL_ID = "./base_model"
OUT_DIR = "./model"
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "validation"  # 평가용 데이터 (예시)
MAX_EVAL_SAMPLES = 100  # 평가에 사용할 샘플 수
MAX_SEQUENCE_LENGTH = 512
SCHEME = "W4A16"
TARGETS = ["Linear"]
IGNORE = ["embed_tokens", "lm_head"]

# 기준 모델의 성능/속도 (예시 값; 실제 기준 모델 측정값으로 교체해야 함)
BASE_PERPLEXITY = 15.0        # 예: 기본 모델의 perplexity
BASE_TOKENS_PER_SEC = 800.0   # 예: 기본 모델의 처리 속도 (tokens/sec)

def load_eval_dataset(tokenizer, split=DATASET_SPLIT, max_samples=MAX_EVAL_SAMPLES):
    dataset = load_dataset(DATASET_ID, split=split)
    subset = dataset.select(range(min(max_samples, len(dataset))))
    def tokenize_fn(examples):
        return tokenizer(examples["text"],
                         truncation=True,
                         max_length=MAX_SEQUENCE_LENGTH,
                         return_tensors="pt")
    return subset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

def calc_perplexity(model, eval_dataset):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in eval_dataset:
            input_ids = batch["input_ids"].squeeze(0).to(model.device)
            outputs = model(input_ids, labels=input_ids)
            # outputs.loss 는 배치 평균 loss
            loss = outputs.loss.item() * input_ids.numel()
            total_loss += loss
            total_tokens += input_ids.numel()
    avg_neg_log_likelihood = total_loss / total_tokens
    return math.exp(avg_neg_log_likelihood)

def measure_speed(model, tokenizer, eval_texts):
    model.eval()
    start_time = time.time()
    total_tokens = 0
    with torch.no_grad():
        for text in eval_texts:
            inputs = tokenizer(text,
                               truncation=True,
                               max_length=MAX_SEQUENCE_LENGTH,
                               return_tensors="pt").to(model.device)
            total_tokens += inputs["input_ids"].numel()
            model.generate(**inputs, max_length=MAX_SEQUENCE_LENGTH)
    elapsed = time.time() - start_time
    return total_tokens / elapsed  # tokens per second

def compute_score(perf, speed, base_perf=BASE_PERPLEXITY, base_speed=BASE_TOKENS_PER_SEC):
    """
    perf: 낮을수록 좋은 perplexity (perplexity가 작을수록 성능↑)
    speed: tokens per second (값이 클수록 속도↑)
    base_perf: 기본 모델 perplexity
    base_speed: 기본 모델 속도
    """
    # 성능 비율(높을수록 좋게 재조정; perplexity는 낮을수록 좋으므로 역수 처리)
    perf_ratio = (base_perf / perf)
    # 속도 비율 (제출 모델 속도가 빠를수록 높음)
    speed_ratio = (speed / base_speed)
    score = 0.5 * perf_ratio + 0.5 * speed_ratio
    return max(score, 0.0)

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )

    # 양자화 수행 (앞서 제시한 방법과 동일, 생략 가능)
    # ... oneshot() 호출 후 양자화 모델을 OUT_DIR에 저장했다고 가정 ...

    quant_model = AutoModelForCausalLM.from_pretrained(
        OUT_DIR, device_map="auto", trust_remote_code=True
    )

    # 평가 데이터 로드
    eval_dataset = load_eval_dataset(tokenizer)
    eval_texts = [ex["text"] for ex in load_dataset(DATASET_ID, split=DATASET_SPLIT).select(range(10))]

    # 기준 모델 평가
    base_ppl = calc_perplexity(base_model, eval_dataset)
    base_speed = measure_speed(base_model, tokenizer, eval_texts)

    # 양자화 모델 평가
    quant_ppl = calc_perplexity(quant_model, eval_dataset)
    quant_speed = measure_speed(quant_model, tokenizer, eval_texts)

    print(f"Base model perplexity: {base_ppl:.4f}, speed: {base_speed:.2f} tokens/sec")
    print(f"Quant model perplexity: {quant_ppl:.4f}, speed: {quant_speed:.2f} tokens/sec")

    # 기준값과 비교하여 점수 계산
    score = compute_score(quant_ppl, quant_speed,
                          base_perf=base_ppl,
                          base_speed=base_speed)
    print(f"Normalized score: {score:.4f}")

if __name__ == "__main__":
    main()
