#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# MUST be at top (before torch/vllm touches CUDA)
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
# unzip / locate model
# -------------------------
def unzip_to_temp(zip_path: Path) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="submission_"))
    shutil.unpack_archive(str(zip_path), str(tmp))
    return tmp


def find_hf_model_dir(root: Path) -> Path:
    candidates = [p.parent for p in root.rglob("config.json")]
    if not candidates:
        raise FileNotFoundError(
            "ZIP 안에서 config.json을 찾지 못했습니다.\n"
            "- save_pretrained()로 저장된 폴더 전체를 zip로 묶었는지 확인하세요.\n"
        )
    candidates.sort(key=lambda p: len(p.parts))
    return candidates[0]


def find_latest_results_json(output_root: Path) -> Path:
    candidates = list(output_root.rglob("results_*.json"))
    if not candidates:
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
    # vLLM options
    vllm_gpu_mem_util: float = 0.85,
    vllm_tp: int = 1,
    vllm_dtype: str = "auto",
    apply_chat_template: bool = True,
    # IMPORTANT: lm_eval에 max_gen_toks 강제 주지 않는 것이 기본
    lm_eval_max_gen_toks: Optional[int] = None,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    if backend not in ("hf", "vllm"):
        raise ValueError("backend must be 'hf' or 'vllm'")

    if backend == "hf":
        model_args = f"pretrained={model_dir},trust_remote_code=True"
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
        ]
        if apply_chat_template:
            cmd += ["--apply_chat_template"]

    else:
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
            "--batch_size", batch_size,   # vllm은 auto 가능
            "--output_path", str(out_dir),
        ]
        if apply_chat_template:
            cmd += ["--apply_chat_template"]

    if limit is not None:
        cmd += ["--limit", str(limit)]

    # ✅ (선택) 정말 필요할 때만 강제
    if lm_eval_max_gen_toks is not None:
        cmd += ["--gen_kwargs", f"max_gen_toks={lm_eval_max_gen_toks}"]

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
# Perf metric extraction
# -------------------------
def extract_perf_from_lm_eval(
    lm_eval_json: Dict[str, Any],
    tasks: str,
    metric_mode: str = "strict",
) -> float:
    wanted = [t.strip() for t in tasks.split(",") if t.strip()]
    results = lm_eval_json.get("results", lm_eval_json.get("result", {}))
    if not isinstance(results, dict):
        raise KeyError("lm_eval json에서 results 딕셔너리를 찾지 못했습니다.")

    if metric_mode not in ("strict", "flexible"):
        raise ValueError("metric_mode must be one of: strict, flexible")

    if metric_mode == "strict":
        metric_priority = [
            "exact_match,strict-match",
            "exact_match,flexible-extract",
            "exact_match",
            "acc",
            "accuracy",
        ]
    else:
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
            numeric = [(k, v) for k, v in t_res.items() if isinstance(v, (int, float))]
            if numeric:
                val = float(numeric[0][1])

        if val is None:
            raise KeyError(f"metric not found for task={t}: keys={list(t_res.keys())}")

        scores.append(val)

    return float(sum(scores) / len(scores))


# -------------------------
# Score
# -------------------------
def compute_score(perf_model: float, perf_base: float, tpt_model: float, tpt_base: float) -> Tuple[float, float, float]:
    perf_norm = (perf_model / perf_base) if perf_base > 0 else 0.0
    speed_norm = 1.0 - (tpt_model / tpt_base) if tpt_base > 0 else 0.0
    score = max(0.5 * perf_norm + 0.5 * speed_norm, 0.0)
    return perf_norm, speed_norm, score


# -------------------------
# Speed prompts (no server prompts available)
# -------------------------
def build_fallback_prompts(n: int) -> List[str]:
    base = [
        "다음 수학 문제를 풀고 마지막 줄에 정답만 숫자로 적어줘: 37 + 58 = ?",
        "다음 문제를 풀어줘. 마지막 줄에 정답만 숫자로: 어떤 수의 2배에 9를 더하면 31이다. 그 수는?",
        "다음 문제: 120을 8로 나누면? 마지막 줄에 정답만 숫자로.",
        "다음 문제: 15개의 사탕을 4명이 똑같이 나눌 때 한 명이 받는 사탕 개수는? (나머지는 버림) 마지막 줄에 정답만 숫자로.",
    ]
    return (base * ((n + len(base) - 1) // len(base)))[:n]


def load_gsm8k_questions(n: int) -> List[str]:
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    questions = []
    for ex in ds:
        q = ex.get("question")
        if isinstance(q, str) and q.strip():
            questions.append(q.strip())
        if len(questions) >= n:
            break
    if not questions:
        raise RuntimeError("gsm8k 질문을 불러오지 못했습니다.")
    return questions


def get_speed_prompts(n: int, source: str) -> List[str]:
    if source == "gsm8k":
        try:
            return load_gsm8k_questions(n)
        except Exception as e:
            print(f"[WARN] gsm8k 질문 로드 실패, fallback 사용: {e}")
            return build_fallback_prompts(n)
    elif source == "fallback":
        return build_fallback_prompts(n)
    else:
        raise ValueError("speed_prompt_source must be one of: gsm8k, fallback")


def apply_chat_template_to_prompts(model_dir: Path, prompts: List[str]) -> List[str]:
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
    include_init_time: bool,
) -> Tuple[float, float, int]:
    from vllm import LLM, SamplingParams

    if apply_chat_template:
        prompts = apply_chat_template_to_prompts(model_dir, prompts)

    params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        stop=["Question:", "</s>", "<|im_end|>"],
    )

    llm = LLM(
        model=str(model_dir),
        trust_remote_code=True,
        dtype=vllm_dtype,
        tensor_parallel_size=vllm_tp,
        gpu_memory_utilization=vllm_gpu_mem_util,
        disable_log_stats=True,
    )

    if include_init_time:
        t0 = time.perf_counter()
        outs = llm.generate(prompts, params)
        t1 = time.perf_counter()
    else:
        # warmup 후 측정(서버와 유사하게 "순수 추론"에 가까움)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, type=str)
    ap.add_argument("--base_model", required=True, type=str)
    ap.add_argument("--tasks", default="gsm8k", type=str)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", type=str)

    ap.add_argument("--lm_eval_backend", default="vllm", choices=["hf", "vllm"])
    ap.add_argument("--batch_size", default="auto", type=str)
    ap.add_argument("--limit", default=None, type=int)
    ap.add_argument("--heartbeat_sec", default=15, type=int)

    ap.add_argument("--perf_metric_mode", default="strict", choices=["strict", "flexible"])

    ap.add_argument("--vllm_tp", default=1, type=int)
    ap.add_argument("--vllm_gpu_mem_util", default=0.85, type=float)
    ap.add_argument("--vllm_dtype", default="auto", type=str)
    ap.add_argument("--vllm_apply_chat_template", default=True, action=argparse.BooleanOptionalAction)

    # ✅ lm_eval에 max_gen_toks 강제는 기본 OFF
    ap.add_argument("--lm_eval_max_gen_toks", default=None, type=int)

    ap.add_argument("--speed_prompt_source", default="gsm8k", choices=["gsm8k", "fallback"])
    ap.add_argument("--speed_num_prompts", default=32, type=int)
    ap.add_argument("--speed_max_tokens", default=2048, type=int)

    # ✅ 기본은 init 제외(서버 점수에 더 가까운 경향)
    ap.add_argument("--speed_include_init_time", default=False, action=argparse.BooleanOptionalAction)

    ap.add_argument("--skip_base", action="store_true")
    args = ap.parse_args()

    zip_path = Path(args.zip).resolve()
    base_dir = Path(args.base_model).resolve()

    tmp_root = unzip_to_temp(zip_path)
    try:
        sub_model_dir = find_hf_model_dir(tmp_root)

        print(f"[INFO] submission model dir: {sub_model_dir}")
        print(f"[INFO] base model dir      : {base_dir}")

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
                apply_chat_template=bool(args.vllm_apply_chat_template),
                lm_eval_max_gen_toks=args.lm_eval_max_gen_toks,
            )
            perf_model = extract_perf_from_lm_eval(sub_eval, args.tasks, metric_mode=args.perf_metric_mode)

            if args.skip_base:
                print("\n[INFO] lm_eval: base skipped by --skip_base")
                perf_base = float("nan")
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
                    apply_chat_template=bool(args.vllm_apply_chat_template),
                    lm_eval_max_gen_toks=args.lm_eval_max_gen_toks,
                )
                perf_base = extract_perf_from_lm_eval(base_eval, args.tasks, metric_mode=args.perf_metric_mode)

        prompts = get_speed_prompts(args.speed_num_prompts, args.speed_prompt_source)

        print("\n[INFO] speed benchmark (vllm): submission...")
        tpt_model, time_model, tok_model = speed_benchmark_vllm_time_per_token(
            model_dir=sub_model_dir,
            prompts=prompts,
            max_tokens=args.speed_max_tokens,
            vllm_tp=args.vllm_tp,
            vllm_gpu_mem_util=args.vllm_gpu_mem_util,
            vllm_dtype=args.vllm_dtype,
            apply_chat_template=bool(args.vllm_apply_chat_template),
            include_init_time=bool(args.speed_include_init_time),
        )

        if args.skip_base:
            print("\n[INFO] speed benchmark: base skipped by --skip_base")
            print(f"TimePerToken_model  : {tpt_model:.6f}   (time={time_model:.3f}s, tokens={tok_model})")
            return

        print("\n[INFO] speed benchmark (vllm): base...")
        tpt_base, time_base, tok_base = speed_benchmark_vllm_time_per_token(
            model_dir=base_dir,
            prompts=prompts,
            max_tokens=args.speed_max_tokens,
            vllm_tp=args.vllm_tp,
            vllm_gpu_mem_util=args.vllm_gpu_mem_util,
            vllm_dtype=args.vllm_dtype,
            apply_chat_template=bool(args.vllm_apply_chat_template),
            include_init_time=bool(args.speed_include_init_time),
        )

        perf_norm, speed_norm, score = compute_score(perf_model, perf_base, tpt_model, tpt_base)

        print("\n========== RESULT ==========")
        print(f"Perf_model          : {perf_model:.6f}")
        print(f"Perf_base           : {perf_base:.6f}")
        print(f"PerfNorm_model      : {perf_norm:.6f}")
        print(f"TimePerToken_model  : {tpt_model:.6f}   (time={time_model:.3f}s, tokens={tok_model})")
        print(f"TimePerToken_base   : {tpt_base:.6f}   (time={time_base:.3f}s, tokens={tok_base})")
        print(f"SpeedNorm_model     : {speed_norm:.6f}")
        print(f"Score               : {score:.6f}")
        print("============================\n")

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
