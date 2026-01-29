import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"  # PDF에 안내된 HF 체크포인트 예시:contentReference[oaicite:1]{index=1}

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    # GPU 있으면 자동으로 GPU에 올리고(가능하면 bf16), 없으면 CPU로 동작
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    ).eval()

    prompt = "다음 문장을 한 줄로 요약해줘: EXAONE은 LG AI연구원의 생성형 AI 모델이다."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    print(tokenizer.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
