"""HYP-042: profile a single decode step under nsys/ncu, with NVTX phase ranges.

Run *under* `nsys profile -t cuda,nvtx,osrt` or
`ncu --kernel-name regex:"flash|turbo|TurboQuant|quantize" ...`.

Two passes:
  python profile_decode_step.py --backend baseline --out /tmp/baseline.nsys-rep
  python profile_decode_step.py --backend tq       --out /tmp/tq.nsys-rep
"""
import argparse
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--input-len", type=int, default=8192)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--output-len", type=int, default=2)  # HYP-042b uses 128
    p.add_argument("--backend", choices=["baseline", "tq"], required=True)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    args = p.parse_args()

    if args.backend == "tq":
        os.environ["VLLM_ATTENTION_BACKEND"] = "CUSTOM"

    import torch
    from vllm import LLM, SamplingParams, TokensPrompt

    if args.backend == "tq":
        import turboquant.vllm_plugin
        turboquant.vllm_plugin.register()

    trace_dir = os.environ.get(
        "VLLM_TORCH_PROFILER_DIR", f"/workspace/shared/hyp042/{args.backend}"
    )
    os.makedirs(trace_dir, exist_ok=True)
    kwargs = dict(
        model=args.model,
        dtype="float16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.input_len + args.output_len + 16,
        enforce_eager=True,
        disable_log_stats=True,
        profiler_config={"profiler": "torch", "torch_profiler_dir": trace_dir},
    )
    if args.backend == "tq":
        kwargs["kv_cache_dtype"] = "fp8"
        kwargs["attention_backend"] = "CUSTOM"

    llm = LLM(**kwargs)

    prompts = [
        TokensPrompt(prompt_token_ids=[1] * args.input_len) for _ in range(args.batch)
    ]
    sp = SamplingParams(
        max_tokens=args.output_len,
        min_tokens=args.output_len,
        temperature=0.0,
        ignore_eos=True,
    )

    # Warmup outside the profiler region (also triggers TurboQuant kernel JIT).
    print(f"[profile] {args.backend}: warmup …")
    llm.generate(prompts, sampling_params=sp, use_tqdm=False)
    torch.cuda.synchronize()

    # vLLM v1's engine runs in a subprocess; LLM.start_profile() propagates
    # torch.profiler into the engine over RPC and saves traces to
    # VLLM_TORCH_PROFILER_DIR.
    print(f"[profile] {args.backend}: start_profile …")
    llm.start_profile()
    out = llm.generate(prompts, sampling_params=sp, use_tqdm=False)
    torch.cuda.synchronize()
    llm.stop_profile()

    print(f"[profile] {args.backend}: produced "
          f"{sum(len(o.outputs[0].token_ids) for o in out)} decode tokens; "
          f"trace under VLLM_TORCH_PROFILER_DIR={os.environ.get('VLLM_TORCH_PROFILER_DIR')}")


if __name__ == "__main__":
    main()
