# LG Aimers 8기: 모델 경량화 온라인 해커톤 실험 레포
EXAONE-4.0-1.2B 기반 모델 경량화(GPTQ/AWQ/SmoothQuant 등) 실험, 제출용 `submit.zip` 생성, 로컬 평가 스크립트를 모아둔 레포입니다.

대회 페이지: `https://dacon.io/competitions/official/236673/overview/description` ([dacon.io](https://dacon.io/competitions/official/236673/overview/description))

---

## 대회 요약
- 배경: On-Device 환경에서의 응답 지연, 메모리, 비용 제약을 고려한 경량화 필요성이 강조됩니다. ([dacon.io](https://dacon.io/competitions/official/236673/overview/description))
- 주제: LLM 경량화 (Large Language Model Compression). ([dacon.io](https://dacon.io/competitions/official/236673/overview/description))
- 기본 모델: EXAONE-4.0-1.2B. ([dacon.io](https://dacon.io/competitions/official/236673/overview/description))
- 평가 환경: vLLM 기반 고정 추론 환경에서 성능·효율 비교. ([dacon.io](https://dacon.io/competitions/official/236673/overview/description))
- 온라인 해커톤(Phase 2): vLLM 라이브러리 수정 불가, HF 표준 형식 모델 가중치/설정 파일만 제출. ([dacon.io](https://dacon.io/competitions/official/236673/overview/description))
- 오프라인 해커톤(Phase 3): 경량화 모델 + vLLM 커스터마이징 제출 가능. ([dacon.io](https://dacon.io/competitions/official/236673/overview/description))
- 제출 형식: `submit.zip` 업로드, HF 표준 형식의 모델 가중치/설정만 포함, 운영진 고정 스크립트로 평가. ([dacon.io](https://dacon.io/competitions/official/236673/overview/description))
- 제한: 전체 추론 시간 ≤ 20분, 제출 파일 ≤ 10GB(압축 해제 후 32GB), 6 vCPU/28GB RAM/L4 22.4GiB 환경. ([dacon.io](https://dacon.io/competitions/official/236673/overview/description))


---

## 레포 구성
- `README.md`: 프로젝트 설명 및 대회 요약.
- `Setup.md`: 로컬 환경 세팅 가이드.
- `open/base_model`: EXAONE 기본 모델(HF 포맷) 디렉토리.
- `experiments/baseline_model`: GPTQ/AWQ/SmoothQuant 베이스라인 실험.
- `experiments/Data_model`: 캘리브레이션 데이터 변형 실험.
- `experiments/pine-tuning_model`: 캘리브레이션 세부 튜닝 실험.
- `experiments/*/score_zip.py`: `submit.zip` 평가 스크립트.
- `experiments/base_cache.json`: 베이스 모델 성능/속도 캐시.

---

## 주요 실험 스크립트
- GPTQ W8A8: `experiments/baseline_model/GPTQ/W8A8_baseline_model.py`
- AWQ: `experiments/baseline_model/AWQ/AWQ (W4A16).py`
- SmoothQuant + GPTQ: `experiments/baseline_model/SmoothQuant_GPTQ (W8A8).py`
- 데이터 믹스: `experiments/Data_model/MANTA-1M + sharegpt-korean._W8A8.py`
- 한국어 위키 캘리브레이션: `experiments/Data_model/Korean Wikipedia_W8A8.py`
- 파라미터 튜닝: `experiments/pine-tuning_model/NOT_IGNORE_lm_head_W8A8.py`, `experiments/pine-tuning_model/MAX1024_W8A8.py`

---

## 빠른 시작
- 실행 위치: 스크립트 내부 기본 경로가 `../open/base_model`이므로 `experiments/`에서 실행하는 것을 전제로 합니다.
- 환경 준비: `Setup.md` 참고.
- 예시 실행:
- `cd experiments`
- `python baseline_model/GPTQ/W8A8_baseline_model.py`

---

## 로컬 평가 예시
- 실행 코드

VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HUB_ENABLE_HF_TRANSFER=0 \
python score_zip.py \
  --zip ./YOUR_W8A8.zip \
  --base_model ../open/base_model \
  --tasks gsm8k \
  --lm_eval_backend vllm \
  --batch_size auto \
  --vllm_tp 1 \
  --vllm_gpu_mem_util 0.85 \
  --vllm_apply_chat_template \
  --perf_metric_mode flexible

- 평가 예시
========== RESULT ==========
Perf_model          : 0.225171
Perf_base           : 0.236543
PerfNorm_model      : 0.951923
TimePerToken_model  : 0.000257   (time=2.725s, tokens=10608)
TimePerToken_base   : 0.000334   (time=3.551s, tokens=10630)
SpeedNorm_model     : 0.231116
Score               : 0.591520
============================


---

## 제출 체크리스트
- `submit.zip` 안에 HF 표준 형식 모델 폴더 전체가 포함되어 있는지 확인. ([dacon.io](https://dacon.io/competitions/official/236673/overview/description))
- 제출 파일 용량 및 실행 환경 제한을 충족하는지 점검. ([dacon.io](https://dacon.io/competitions/official/236673/overview/description))

---
