# LG Aimers Hackathon – EXAONE Inference Baseline

본 레포지토리는 **LG Aimers 해커톤**을 대비하여  
LG AI Research의 **EXAONE 모델**을 **로컬 환경에서 안정적으로 추론(inference)** 하기 위한
기본 실행 환경과 서버 구성을 제공합니다.

본 프로젝트는 **vLLM 기반 OpenAI-compatible API 서버**를 중심으로 구성되어 있으며,  
해커톤 문제 공개 이후 **평가 코드(eval)** 를 즉시 추가할 수 있도록 설계되었습니다.

---

## 📌 대회 개요 (LG Aimers Hackathon)

LG Aimers 해커톤은  
**대규모 언어 모델(LLM)의 경량화, 추론 효율화, 응용 및 평가**를 목표로 하는 실전형 AI 대회입니다.

본 해커톤에서는:
- 사전 제공된 강의(PDF)를 기반으로
- 실제 LLM을 직접 실행 및 최적화하고
- 문제 요구사항에 맞는 추론 및 평가 파이프라인을 구성하는 것이 핵심입니다.

---

## 🧠 사용 모델: EXAONE

- **Model**: `LGAI-EXAONE/EXAONE-4.0-1.2B`
- **Provider**: LG AI Research
- **특징**
  - 경량화된 대규모 언어 모델
  - 로컬 환경(GPU/CPU)에서 실행 가능
  - Chat / Instruction 기반 추론에 적합

본 레포지토리는 **EXAONE 모델을 로컬에서 직접 실행**하며,  
클라우드 API(OpenAI 등)를 사용하지 않습니다.

---

## ⚙️ 시스템 구성 개요

```text
[Client / Eval Script]
        ↓ (HTTP 요청)
[vLLM OpenAI-compatible Server]
        ↓
[EXAONE Model (Local GPU/CPU)]
```
- vLLM을 사용하여 모델을 메모리에 상주시킨 상태로 서버 실행
- /v1/chat/completions API를 통해 추론 요청 처리
- 해커톤 평가 스크립트와 쉽게 연동 가능

---

## 🚀 서버 실행 방법 (vLLM)
1️⃣ 환경 준비
pip install -r requirements.txt

2️⃣ vLLM 서버 실행
bash server/run_vllm.sh
서버 실행 후, 기본 포트는 8000입니다.

---

## 🧪 동작 확인 (간단 테스트)
curl http://localhost:8000/v1/models 또는 Python 클라이언트: python client/client_test.py

---

## 📊 평가(Evaluation) 확장 계획

해커톤 문제 공개 이후 다음과 같은 구조로 평가를 확장할 예정입니다:
- eval/
- 문제 데이터 로드
- vLLM API 호출
- 결과 저장 (JSON / CSV)
- 정확도, 규칙 기반 점수 또는 LLM-as-a-Judge 방식 평가

본 레포지토리는 문제 유형과 무관하게 평가 코드만 추가하면 바로 실험이 가능하도록
기본 추론 환경을 완성해 둔 상태입니다.

---

## 📂 프로젝트 구조
```text
LG_Aimers/
├─ README.md
├─ requirements.txt
├─ server/
│  └─ run_vllm.sh        # vLLM 서버 실행 스크립트
├─ client/
│  └─ client_test.py     # 간단한 API 호출 테스트
├─ experiments/
│  └─ run_exaone.py      # (선택) Transformers 기반 로컬 실험
└─ eval/                 # (추후 추가) 평가 코드
```
---

## 🎯 목표
- EXAONE 모델을 안정적으로 로컬에서 실행
- 해커톤 문제 공개 즉시 평가 파이프라인 연결
- 재현 가능하고 깔끔한 실험 환경 유지

---

## 📝 참고
- 본 레포지토리는 해커톤 학습 및 실험 목적으로 사용됩니다.
- 모델 가중치는 Hugging Face를 통해 최초 1회 다운로드되며, 이후 추론은 로컬에서 수행됩니다.

---
