"""Offline vLLM latency benchmark — FA2 vs FlashInfer vs TurboQuant.

Separates prefill from decode by running twice:
  T_short = prefill + 1 decode
  T_long  = prefill + N decodes
  decode_per_step = (T_long - T_short) / (N - 1)
  prefill_s       = T_short - decode_per_step
"""
import argparse
import json
import os
import statistics
import time


def time_generate(llm, prompts, output_len, trials: int):
    from vllm import SamplingParams
    import torch
    sp = SamplingParams(
        max_tokens=output_len,
        min_tokens=output_len,
        temperature=0.0,
        ignore_eos=True,
    )
    per_trial = []
    for _ in range(trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        llm.generate(prompts, sampling_params=sp, use_tqdm=False)
        torch.cuda.synchronize()
        per_trial.append(time.perf_counter() - t0)
    return statistics.median(per_trial), per_trial


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--input-len", type=int, required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--backend", choices=["baseline", "tq", "flashinfer"], required=True)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--out", required=True)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    args = p.parse_args()

    if args.backend == "tq":
        os.environ["VLLM_ATTENTION_BACKEND"] = "CUSTOM"

    from vllm import LLM, TokensPrompt
    import torch

    if args.backend == "tq":
        import turboquant.vllm_plugin
        turboquant.vllm_plugin.register()
        print(f"[bench] registered TurboQuant; "
              f"VLLM_ATTENTION_BACKEND={os.environ.get('VLLM_ATTENTION_BACKEND')}")

    max_len = args.input_len + args.output_len + 16
    kwargs = dict(
        model=args.model,
        dtype="float16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=max_len,
        enforce_eager=True,  # A100 can't compile fp8e4nv — same constraint for all
        disable_log_stats=True,
    )
    if args.backend == "tq":
        kwargs["kv_cache_dtype"] = "fp8"
        kwargs["attention_backend"] = "CUSTOM"
    elif args.backend == "flashinfer":
        kwargs["attention_backend"] = "FLASHINFER"

    llm = LLM(**kwargs)
    prompts = [TokensPrompt(prompt_token_ids=[1] * args.input_len)
               for _ in range(args.batch)]

    # Warmup both shapes so the caching allocator and any JIT compile settles
    # before we time anything.
    print("[bench] warmup output_len=1 …", flush=True)
    time_generate(llm, prompts, 1, trials=1)
    print(f"[bench] warmup output_len={args.output_len} …", flush=True)
    time_generate(llm, prompts, args.output_len, trials=1)

    print(f"[bench] timed: output_len=1 (prefill + 1 decode) × {args.trials} trials", flush=True)
    t_short, short_trials = time_generate(llm, prompts, 1, args.trials)

    print(f"[bench] timed: output_len={args.output_len} × {args.trials} trials", flush=True)
    t_long, long_trials = time_generate(llm, prompts, args.output_len, args.trials)

    n_decode_steps = args.output_len - 1  # output_len steps minus the 1 counted in T_short
    decode_per_step_s = (t_long - t_short) / n_decode_steps
    prefill_s = t_short - decode_per_step_s  # one decode step included in T_short
    decode_tokens_per_s = args.batch / decode_per_step_s
    total_tokens_per_s = (args.batch * args.output_len) / t_long

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    try:
        import subprocess
        line = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"]
        ).decode().strip().splitlines()[0]
        nvsmi_mem_gb = float(line) / 1024.0
    except Exception:
        nvsmi_mem_gb = None

    result = {
        "backend": args.backend,
        "model": args.model,
        "input_len": args.input_len,
        "batch": args.batch,
        "output_len": args.output_len,
        "trials": args.trials,
        "t_short_s": t_short,
        "t_long_s": t_long,
        "short_trials_s": short_trials,
        "long_trials_s": long_trials,
        "prefill_s": prefill_s,
        "decode_per_step_s": decode_per_step_s,
        "decode_tokens_per_s": decode_tokens_per_s,       # clean decode-only
        "total_tokens_per_s": total_tokens_per_s,         # prefill-included (legacy)
        "peak_gpu_mem_gb": peak_mem,
        "nvsmi_gpu_mem_gb": nvsmi_mem_gb,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
