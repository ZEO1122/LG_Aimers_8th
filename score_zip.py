#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
score_zip.py
- ZIP 제출물(압축/양자화 모델)을 입력으로 받아
  1) lm_eval로 Perf_model, Perf_base 평가
  2) vLLM로 TimePerToken_model, TimePerToken_base 측정
  3) (Perf/Speed) 정규화 후 Score 산출

⭐ 중요(이번 에러의 핵심 수정)
- apply_chat_template는 vLLM EngineArgs 인자가 아님!
  -> lm_eval에서는 model_args에 넣지 말고, CLI 옵션 --apply_chat_template 로 켜야 함.

서버 스펙(사용자가 공유한 것):
- Inference Engine: vLLM 0.14.1
- tensor_parallel_size = 1
- gpu_memory_utilization = 0.85
- batch_size = auto
- max_gen_toks = 16384
- apply_chat_template = true
"""

# --- MUST be at top (before torch/vllm touches CUDA) ---
import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import argparse
import contextlib
import json
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


# -------------------------
# Utils: unzip / locate model
# -------------------------
def unzip_to_temp(zip_path: Path) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="submission_"))
    shutil.unpack_archive(str(zip_path), str(tmp))
    return tmp


def find_hf_model_dir(root: Path) -> Path:
    """
    ZIP 안에서 HuggingFace 모델 폴더를 찾는다.
    기준: config.json이 있는 디렉토리
    """
    candidates = [p.parent for p in root.rglob("config.json")]
    if not candidates:
        raise FileNotFoundError(
            "ZIP 안에서 config.json을 찾지 못했습니다.\n"
            "- save_pretrained()로 저장된 폴더 전체를 zip로 묶었는지 확인하세요.\n"
        )
    # 가장 얕은(루트와 가까운) 폴더를 우선
    candidates.sort(key=lambda p: len(p.parts))
    return candidates[0]


def find_latest_results_json(output_root: Path) -> Path:
    """
    lm_eval --output_path 가 directory 일 때, results_*.json 등이 생성됨.
    가장 최신 파일을 찾아서 반환.
    """
    candidates = list(output_root.rglob("results_*.json"))
    if not candidates:
        # 혹시 다른 이름인 경우도 대비
        candidates = list(output_root.rglob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"lm_eval 결과 json을 {output_root} 아래에서 찾지 못했습니다.")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


# -------------------------
# lm_eval runner (hf/vllm)
# -------------------------
def run_lm_eval(
    backend: str,
    model_dir: Path,
    tasks: str,
    device: str,
    batch_size: str,
    out_dir: Path,
    limit: Optional[int] = None,
    label: str = "lm_eval",
    log_path: Optional[Path] = None,
    heartbeat_sec: int = 15,
    # vLLM server-matching options
    vllm_gpu_mem_util: float = 0.85,
    vllm_tp: int = 1,
    vllm_dtype: str = "auto",
    vllm_max_gen_toks: int = 16384,
    apply_chat_template: bool = True,
) -> Dict[str, Any]:
    """
    lm_eval을 subprocess로 실행하고, stdout을 실시간으로 출력/로그 저장.
    완료 후 결과 json을 읽어서 dict로 반환.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if backend not in ("hf", "vllm"):
        raise ValueError("backend must be 'hf' or 'vllm'")

    if backend == "hf":
        # hf backend (느리지만 비교용 유지)
        model_args = f"pretrained={model_dir},trust_remote_code=True"
        # hf backend batch_size는 int만 안정적으로 동작
        try:
            bs_int = str(int(batch_size))
        except Exception:
            raise ValueError("hf backend는 --batch_size에 정수만 지원합니다. 예: --batch_size 4")

        cmd = [
            "python", "-m", "lm_eval",
            "--model", "hf",
            "--model_args", model_args,
            "--tasks", tasks,
            "--device", device,
            "--batch_size", bs_int,
            "--output_path", str(out_dir),
            "--gen_kwargs", f"max_gen_toks={vllm_max_gen_toks}",  # 최대한 조건 맞추기(가능하면)
        ]
        if apply_chat_template:
            cmd += ["--apply_chat_template"]

    else:
        # vllm backend
        # ✅ IMPORTANT: apply_chat_template는 vLLM EngineArgs 인자가 아님.
        #    -> model_args에 넣으면 TypeError 발생.
        #    -> 반드시 lm_eval CLI 플래그 --apply_chat_template 로 켜야 함.
        model_args = (
            f"pretrained={model_dir},"
            f"dtype={vllm_dtype},"
            f"gpu_memory_utilization={vllm_gpu_mem_util},"
            f"tensor_parallel_size={vllm_tp},"
            f"trust_remote_code=True"
        )
        cmd = [
            "python", "-m", "lm_eval",
            "--model", "vllm",
            "--model_args", model_args,
            "--tasks", tasks,
            "--device", device,
            "--batch_size", batch_size,     # vllm은 auto 가능
            "--output_path", str(out_dir),
            "--gen_kwargs", f"max_gen_toks={vllm_max_gen_toks}",
        ]
        if apply_chat_template:
            cmd += ["--apply_chat_template"]

    if limit is not None:
        cmd += ["--limit", str(limit)]

    env = os.environ.copy()

    print(f"\n[INFO] ({label}) running:\n  {' '.join(cmd)}\n")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )

    q: "queue.Queue[str]" = queue.Queue()
    collected: List[str] = []

    def _reader():
        assert proc.stdout is not None
        for line in proc.stdout:
            q.put(line)
        try:
            proc.stdout.close()
        except Exception:
            pass

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    start = time.time()
    log_fh_ctx = open(log_path, "w", encoding="utf-8") if log_path else contextlib.nullcontext()
    with log_fh_ctx as log_fh:
        while True:
            try:
                line = q.get(timeout=heartbeat_sec)
                if not line.endswith("\n"):
                    line += "\n"
                print(f"[{label}] {line}", end="")
                if log_fh:
                    log_fh.write(line)
                    log_fh.flush()
                collected.append(line.rstrip("\n"))
            except queue.Empty:
                if proc.poll() is not None:
                    # 남은 출력 flush
                    while not q.empty():
                        try:
                            line = q.get_nowait()
                        except queue.Empty:
                            break
                        if not line.endswith("\n"):
                            line += "\n"
                        print(f"[{label}] {line}", end="")
                        if log_fh:
                            log_fh.write(line)
                            log_fh.flush()
                        collected.append(line.rstrip("\n"))
                    break
                elapsed = int(time.time() - start)
                hb = f"[{label}] ...still running ({elapsed}s elapsed, no new output)\n"
                print(hb, end="")
                if log_fh:
                    log_fh.write(hb)
                    log_fh.flush()

    rc = proc.wait()
    t.join(timeout=1)

    if rc != 0:
        tail = "\n".join(collected[-250:])
        raise RuntimeError(
            f"{label} 실행 실패 (returncode={rc})\n--- last logs ---\n{tail}\n"
        )

    result_json = find_latest_results_json(out_dir)
    return json.loads(result_json.read_text(encoding="utf-8"))


# -------------------------
# Metric extraction
# -------------------------
def extract_perf_from_lm_eval(lm_eval_json: Dict[str, Any], tasks: str) -> float:
    """
    lm_eval output에서 task별 score를 뽑아 평균.
    gsm8k는 보통 exact_match가 flexible/strict 두 개 존재.
    서버가 어떤 걸 쓰는지 확실치 않으면 flexible 우선.
    """
    wanted = [t.strip() for t in tasks.split(",") if t.strip()]
    results = lm_eval_json.get("results", lm_eval_json.get("result", {}))
    if not isinstance(results, dict):
        raise KeyError("lm_eval json에서 results 딕셔너리를 찾지 못했습니다.")

    metric_priority = [
        "exact_match,flexible-extract",
        "exact_match,strict-match",
        "exact_match",
        "acc",
        "accuracy",
    ]

    scores: List[float] = []
    for t in wanted:
        t_res = results.get(t)
        if not isinstance(t_res, dict):
            raise KeyError(f"task result missing: {t}")

        val: Optional[float] = None
        for k in metric_priority:
            if k in t_res and isinstance(t_res[k], (int, float)):
                val = float(t_res[k])
                break

        if val is None:
            # fallback: 첫 numeric metric
            numeric = [(k, v) for k, v in t_res.items() if isinstance(v, (int, float))]
            if numeric:
                val = float(numeric[0][1])

        if val is None:
            raise KeyError(f"metric not found for task={t}: keys={list(t_res.keys())}")

        scores.append(val)

    return float(sum(scores) / len(scores))


# -------------------------
# Base cache (skip_base)
# -------------------------
def load_json_if_exists(p: Path) -> Optional[Dict[str, Any]]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_json(p: Path, obj: Dict[str, Any]) -> None:
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def base_cache_signature(args: argparse.Namespace, base_model_dir: Path) -> Dict[str, Any]:
    return {
        "base_model_dir": str(base_model_dir),
        "tasks": args.tasks,
        "device": args.device,
        "lm_eval_backend": args.lm_eval_backend,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "vllm_tp": args.vllm_tp,
        "vllm_gpu_mem_util": args.vllm_gpu_mem_util,
        "vllm_dtype": args.vllm_dtype,
        "vllm_max_gen_toks": args.vllm_max_gen_toks,
        "apply_chat_template": bool(args.vllm_apply_chat_template),
        "speed_num_prompts": args.speed_num_prompts,
        "speed_max_tokens": args.speed_max_tokens,
    }


# -------------------------
# Scoring
# -------------------------
def compute_score(perf_model: float, perf_base: float, tpt_model: float, tpt_base: float) -> Tuple[float, float, float]:
    perf_norm = (perf_model / perf_base) if perf_base > 0 else 0.0
    speed_norm = 1.0 - (tpt_model / tpt_base) if tpt_base > 0 else 0.0
    score = max(0.5 * perf_norm + 0.5 * speed_norm, 0.0)
    return perf_norm, speed_norm, score


# -------------------------
# vLLM speed benchmark
# -------------------------
def build_speed_prompts_default(n: int) -> List[str]:
    base = [
        "GSM8K 스타일로 풀이 과정을 보여주고 마지막에 정답만 숫자로 말해줘: 12개의 사과가 있고 5개를 먹었다. 남은 사과는?",
        "다음 수학 문제를 풀어줘: 어떤 수의 3배에서 7을 빼면 20이다. 그 수는?",
        "간단한 덧셈 문제: 389 + 512 = ?",
        "10개의 쿠키를 4명이 똑같이 나눌 때 한 명이 받는 쿠키 수는?",
    ]
    return (base * ((n + len(base) - 1) // len(base)))[:n]


def load_speed_prompts_jsonl(path: Path, n: int) -> List[str]:
    """
    jsonl 각 줄: {"prompt": "..."} 형태 기대
    """
    prompts: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            p = obj.get("prompt")
            if isinstance(p, str) and p.strip():
                prompts.append(p.strip())
            if len(prompts) >= n:
                break
    if not prompts:
        raise ValueError(f"speed_prompts_jsonl이 비어있거나 prompt 필드가 없습니다: {path}")
    return prompts


def apply_chat_template_to_prompts(model_dir: Path, prompts: List[str]) -> List[str]:
    """
    서버 스펙 apply_chat_template=true를 최대한 맞추기 위해
    tokenizer.apply_chat_template로 "user" 메시지로 래핑해 prompt 생성.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
    )

    formatted: List[str] = []
    for p in prompts:
        try:
            formatted.append(
                tok.apply_chat_template(
                    [{"role": "user", "content": p}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            )
        except Exception:
            # 혹시 템플릿이 없거나 실패하면 원문 그대로
            formatted.append(p)
    return formatted


def speed_benchmark_vllm_time_per_token(
    model_dir: Path,
    prompts: List[str],
    max_tokens: int,
    vllm_tp: int,
    vllm_gpu_mem_util: float,
    vllm_dtype: str,
    apply_chat_template: bool,
) -> Tuple[float, float, int]:
    """
    vLLM로 prompts를 한 번에 generate해서
    total_time / generated_tokens = time_per_token 반환.
    """
    from vllm import LLM, SamplingParams

    if apply_chat_template:
        prompts = apply_chat_template_to_prompts(model_dir, prompts)

    llm = LLM(
        model=str(model_dir),
        trust_remote_code=True,
        dtype=vllm_dtype,
        tensor_parallel_size=vllm_tp,
        gpu_memory_utilization=vllm_gpu_mem_util,
        disable_log_stats=True,
    )

    params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        # gsm8k에서 쓰는 stop과 유사
        stop=["Question:", "</s>", "<|im_end|>"],
    )

    # warmup 1회(초기 커널/그래프 준비)
    _ = llm.generate([prompts[0]], params)

    t0 = time.perf_counter()
    outs = llm.generate(prompts, params)
    t1 = time.perf_counter()

    total_time = t1 - t0
    gen_tokens = 0
    for o in outs:
        if not o.outputs:
            continue
        token_ids = getattr(o.outputs[0], "token_ids", None)
        if token_ids is not None:
            gen_tokens += len(token_ids)

    tpt = (total_time / gen_tokens) if gen_tokens > 0 else float("inf")
    return tpt, total_time, gen_tokens


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, type=str)
    ap.add_argument("--base_model", required=True, type=str)
    ap.add_argument("--tasks", default="gsm8k", type=str)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", type=str)

    # lm_eval settings
    ap.add_argument("--lm_eval_backend", default="vllm", choices=["hf", "vllm"])
    ap.add_argument("--batch_size", default="auto", type=str)  # vllm은 auto 가능
    ap.add_argument("--limit", default=None, type=int)
    ap.add_argument("--heartbeat_sec", default=15, type=int)

    # server-matching vLLM options
    ap.add_argument("--vllm_tp", default=1, type=int)
    ap.add_argument("--vllm_gpu_mem_util", default=0.85, type=float)
    ap.add_argument("--vllm_dtype", default="auto", type=str)
    ap.add_argument("--vllm_max_gen_toks", default=16384, type=int)
    ap.add_argument("--vllm_apply_chat_template", default=True, action=argparse.BooleanOptionalAction)

    # speed benchmark settings
    ap.add_argument("--speed_num_prompts", default=32, type=int)
    ap.add_argument("--speed_max_tokens", default=4096, type=int)  # 토큰 수는 서버와 동일하게 하고 싶으면 16384로도 가능
    ap.add_argument("--speed_prompts_jsonl", default=None, type=str)

    # base cache / skip base
    ap.add_argument("--skip_base", action="store_true")
    ap.add_argument("--base_cache", default="./base_cache.json", type=str)

    args = ap.parse_args()

    zip_path = Path(args.zip).resolve()
    base_dir = Path(args.base_model).resolve()
    base_cache_path = Path(args.base_cache).resolve()

    tmp_root = unzip_to_temp(zip_path)
    try:
        sub_model_dir = find_hf_model_dir(tmp_root)

        print(f"[INFO] submission model dir: {sub_model_dir}")
        print(f"[INFO] base model dir      : {base_dir}")

        # -------------------------
        # PERF (lm_eval) submission
        # -------------------------
        with tempfile.TemporaryDirectory(prefix="lm_eval_out_") as outd:
            outd = Path(outd)

            print("\n[INFO] lm_eval: submission...")
            sub_eval = run_lm_eval(
                backend=args.lm_eval_backend,
                model_dir=sub_model_dir,
                tasks=args.tasks,
                device=args.device,
                batch_size=args.batch_size,
                out_dir=outd / "sub",
                limit=args.limit,
                label="lm_eval:submission",
                log_path=Path("lm_eval_submission.log"),
                heartbeat_sec=args.heartbeat_sec,
                vllm_gpu_mem_util=args.vllm_gpu_mem_util,
                vllm_tp=args.vllm_tp,
                vllm_dtype=args.vllm_dtype,
                vllm_max_gen_toks=args.vllm_max_gen_toks,
                apply_chat_template=bool(args.vllm_apply_chat_template),
            )
            perf_model = extract_perf_from_lm_eval(sub_eval, args.tasks)

            # -------------------------
            # PERF (lm_eval) base (cache)
            # -------------------------
            sig = base_cache_signature(args, base_dir)
            cache = load_json_if_exists(base_cache_path)

            if args.skip_base and cache and cache.get("signature") == sig and "perf_base" in cache:
                perf_base = float(cache["perf_base"])
                print(f"\n[INFO] lm_eval: base (cached) perf_base={perf_base}")
            else:
                print("\n[INFO] lm_eval: base...")
                base_eval = run_lm_eval(
                    backend=args.lm_eval_backend,
                    model_dir=base_dir,
                    tasks=args.tasks,
                    device=args.device,
                    batch_size=args.batch_size,
                    out_dir=outd / "base",
                    limit=args.limit,
                    label="lm_eval:base",
                    log_path=Path("lm_eval_base.log"),
                    heartbeat_sec=args.heartbeat_sec,
                    vllm_gpu_mem_util=args.vllm_gpu_mem_util,
                    vllm_tp=args.vllm_tp,
                    vllm_dtype=args.vllm_dtype,
                    vllm_max_gen_toks=args.vllm_max_gen_toks,
                    apply_chat_template=bool(args.vllm_apply_chat_template),
                )
                perf_base = extract_perf_from_lm_eval(base_eval, args.tasks)

        # -------------------------
        # SPEED (vLLM time per token)
        # -------------------------
        if args.speed_prompts_jsonl:
            prompts = load_speed_prompts_jsonl(Path(args.speed_prompts_jsonl), args.speed_num_prompts)
        else:
            prompts = build_speed_prompts_default(args.speed_num_prompts)

        print("\n[INFO] speed benchmark (vllm): submission...")
        tpt_model, time_model, tok_model = speed_benchmark_vllm_time_per_token(
            model_dir=sub_model_dir,
            prompts=prompts,
            max_tokens=args.speed_max_tokens,
            vllm_tp=args.vllm_tp,
            vllm_gpu_mem_util=args.vllm_gpu_mem_util,
            vllm_dtype=args.vllm_dtype,
            apply_chat_template=bool(args.vllm_apply_chat_template),
        )

        cache = load_json_if_exists(base_cache_path)
        if args.skip_base and cache and cache.get("signature") == sig and "tpt_base" in cache:
            tpt_base = float(cache["tpt_base"])
            print(f"\n[INFO] speed benchmark: base (cached) tpt_base={tpt_base}")
        else:
            print("\n[INFO] speed benchmark (vllm): base...")
            tpt_base, time_base, tok_base = speed_benchmark_vllm_time_per_token(
                model_dir=base_dir,
                prompts=prompts,
                max_tokens=args.speed_max_tokens,
                vllm_tp=args.vllm_tp,
                vllm_gpu_mem_util=args.vllm_gpu_mem_util,
                vllm_dtype=args.vllm_dtype,
                apply_chat_template=bool(args.vllm_apply_chat_template),
            )

        # cache update (base만 캐시)
        if args.skip_base:
            new_cache = {
                "signature": sig,
                "perf_base": float(perf_base),
                "tpt_base": float(tpt_base),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            save_json(base_cache_path, new_cache)
            print(f"\n[INFO] base cache updated: {base_cache_path}")

        # -------------------------
        # SCORE
        # -------------------------
        perf_norm, speed_norm, score = compute_score(perf_model, perf_base, tpt_model, tpt_base)

        print("\n========== RESULT ==========")
        print(f"Perf_model          : {perf_model:.6f}")
        print(f"Perf_base           : {perf_base:.6f}")
        print(f"PerfNorm_model      : {perf_norm:.6f}")
        print(f"TimePerToken_model  : {tpt_model:.6f}   (time={time_model:.3f}s, tokens={tok_model})")
        print(f"TimePerToken_base   : {tpt_base:.6f}")
        print(f"SpeedNorm_model     : {speed_norm:.6f}")
        print(f"Score               : {score:.6f}")
        print("============================\n")

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    # spawn 강제(특히 vLLM + CUDA에서 fork 문제 방지)
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
