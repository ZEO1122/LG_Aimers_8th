# Local Environment Summary

기준 수집 시각: `2026-03-10 23:01:28 KST`

이 문서는 현재 `/home/ubuntu/Desktop/LG Aimers` 작업 기준 로컬 환경 정보를 정리한 것이다.

## OS

- 배포판: Ubuntu 24.04.3 LTS
- 코드네임: Noble Numbat
- 커널: `Linux 6.8.0-101-generic`
- 아키텍처: `x86_64`

## CPU

- 모델: `Intel(R) Core(TM) i7-14700K`
- 논리 CPU: `28`
- 소켓: `1`
- 코어 구성:
  - `20` cores
  - `2` threads per core
- 최대 클럭: `5600 MHz`
- 가상화: `VT-x`

## Memory

- 총 메모리: `15 GiB`
- 사용 중: `4.6 GiB`
- free: `1.8 GiB`
- available: `10 GiB`
- swap: `31 GiB`

## Disk

- 마운트: `/`
- 디스크: `/dev/nvme1n1p3`
- 전체 용량: `1.8T`
- 사용량: `224G`
- 여유 공간: `1.5T`
- 사용률: `13%`

## GPU

`nvidia-smi` 대신 커널/드라이버 레벨 정보로 GPU를 다시 확인했다.

- GPU: `NVIDIA Corporation AD103 [GeForce RTX 4070 Ti SUPER]`
- PCI bus: `0000:01:00.0`
- Bus type: `PCIe`
- IRQ: `216`
- Driver: `nvidia`
- NVIDIA kernel module version: `590.48.01`
- GPU firmware: `590.48.01`
- Video BIOS: `95.03.45.40.4b`
- GPU UUID 확인됨

현재 세션에서의 런타임 상태:

- `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm` 등 장치 파일 존재
- `nvidia-smi` 실행 결과: `Failed to initialize NVML: Unknown Error`

즉 물리 GPU와 드라이버 적재는 확인되지만, 현재 세션에서는 NVML 초기화 문제가 있어 `nvidia-smi` 기반 상태 조회가 정상 동작하지 않는다.

## Python / Conda

- 시스템 기본 Python: `3.13.5`
- Anaconda base Python: `3.13.5`
- `exaone` conda 환경 Python: `3.10.19`

## 주요 ML 라이브러리 (`exaone` 환경)

- `torch==2.9.1`
- `transformers==4.57.3`
- `datasets==4.4.1`
- `peft==0.11.1`
- `bitsandbytes==0.49.1`
- `llmcompressor==0.9.0.1`
- `vllm==0.14.1`

## 실행 관련 메모

- 현재 `exaone` 환경은 존재하고 Python 및 주요 패키지 설치도 확인됨
- 다만 GPU 조회가 `nvidia-smi` 수준에서 실패하고 있어, 실제 CUDA 사용 가능 여부는 별도 런타임 테스트가 필요함
- QLoRA / GPTQ 실행 전에 `open/base_model` 내부의 Hugging Face 모델 파일 구성이 완전한지 확인해야 함
