# Exaone Environment

이 폴더는 로컬 실행용 `exaone` conda 환경 정보를 정리한 것입니다.

## 파일

- `exaone.env`: 실험 실행에 필요한 환경변수
- `activate_exaone.sh`: conda 활성화 + 환경변수 로드 스크립트
- `requirements-exaone.txt`: 현재 `exaone` 환경에서 실험에 직접 관련된 주요 패키지 목록
- `../open`: 로컬 모델/데이터셋을 두는 기본 경로

## 사용 순서

```bash
source /home/ubuntu/Desktop/LG\ Aimers/env/activate_exaone.sh
python /home/ubuntu/Desktop/LG\ Aimers/qlora_baseline_3000_val_local.py
```

GPTQ까지 실행하려면:

```bash
source /home/ubuntu/Desktop/LG\ Aimers/env/activate_exaone.sh
python /home/ubuntu/Desktop/LG\ Aimers/qlora_baseline_3000_val_local.py --run-gptq
```

## 메모

- 기본 출력 경로는 `/home/ubuntu/Desktop/LG Aimers/local_outputs` 입니다.
- Hugging Face 캐시는 `local_outputs/.cache/huggingface` 아래에 저장됩니다.
- 기본 모델 경로는 `/home/ubuntu/Desktop/LG Aimers/open/base_model` 입니다.
- 기본 데이터셋 경로는 `/home/ubuntu/Desktop/LG Aimers/open/dataset` 입니다.
- `conda env list` 조회는 현재 시스템의 conda 플러그인 오류 때문에 실패했지만, 실제 환경 경로 `/home/ubuntu/anaconda3/envs/exaone` 는 확인했습니다.
