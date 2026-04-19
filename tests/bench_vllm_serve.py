"""Offline vLLM latency benchmark — baseline vs TurboQuant.

Runs `LLM.generate` on a batch of prompts with fixed input/output length,
reports per-request latency and decode-token throughput.
"""
import argparse
import json
import os
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--input-len", type=int, required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--backend", choices=["baseline", "tq", "flashinfer"], required=True)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--out", required=True)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    args = p.parse_args()

    if args.backend == "tq":
        os.environ["VLLM_ATTENTION_BACKEND"] = "CUSTOM"

    from vllm import LLM, SamplingParams
    import torch

    if args.backend == "tq":
        import turboquant.vllm_plugin
        turboquant.vllm_plugin.register()
        print(f"[bench] registered TurboQuant backend; "
              f"VLLM_ATTENTION_BACKEND={os.environ.get('VLLM_ATTENTION_BACKEND')}")

    max_len = args.input_len + args.output_len + 16
    kwargs = dict(
        model=args.model,
        dtype="float16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=max_len,
        enforce_eager=True,  # fair, and A100 can't compile fp8e4nv
        disable_log_stats=True,
    )
    if args.backend == "tq":
        kwargs["kv_cache_dtype"] = "fp8"  # TurboQuant uses own quantizer; vllm treats as 1-byte alloc
        kwargs["attention_backend"] = "CUSTOM"
    elif args.backend == "flashinfer":
        kwargs["attention_backend"] = "FLASHINFER"

    llm = LLM(**kwargs)

    from vllm import TokensPrompt
    prompts = [TokensPrompt(prompt_token_ids=[1] * args.input_len) for _ in range(args.batch)]
    sp = SamplingParams(
        max_tokens=args.output_len,
        min_tokens=args.output_len,
        temperature=0.0,
        ignore_eos=True,
    )

    for _ in range(args.warmup):
        llm.generate(prompts, sampling_params=sp, use_tqdm=False)

    per_trial = []
    for _ in range(args.trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = llm.generate(prompts, sampling_params=sp, use_tqdm=False)
        torch.cuda.synchronize()
        per_trial.append(time.perf_counter() - t0)

    import statistics
    elapsed = statistics.median(per_trial)
    decode_tokens = args.batch * args.output_len

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    # Also capture nvidia-smi main GPU used memory (engine runs in subprocess)
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        ).decode().strip().splitlines()[0]
        nvsmi_mem_gb = float(out) / 1024.0
    except Exception:
        nvsmi_mem_gb = None

    result = {
        "backend": args.backend,
        "model": args.model,
        "input_len": args.input_len,
        "batch": args.batch,
        "output_len": args.output_len,
        "trials": args.trials,
        "per_trial_s": per_trial,
        "median_s": elapsed,
        "decode_tokens_per_s": decode_tokens / elapsed,
        "per_request_s": elapsed,
        "peak_gpu_mem_gb": peak_mem,
        "nvsmi_gpu_mem_gb": nvsmi_mem_gb,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
