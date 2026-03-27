"""TTFT/TPOT benchmark: vLLM eager baseline vs TurboQuant eager.

Measures:
- TTFT: Time to First Token (prefill latency)
- TPOT: Time Per Output Token (decode latency)
"""

import time
import torch
import gc
from vllm import LLM, SamplingParams


def bench(llm, prompts, params, warmup=1, runs=3):
    """Measure TTFT and TPOT."""
    # Warmup
    for _ in range(warmup):
        llm.generate(prompts[:1], params)

    ttfts = []
    tpots = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, params)
        torch.cuda.synchronize()
        total = time.perf_counter() - t0

        for out in outputs:
            n_input = len(out.prompt_token_ids)
            n_output = len(out.outputs[0].token_ids)
            # Approximate: TTFT ~ total * (n_input / (n_input + n_output))
            # TPOT ~ (total - ttft) / n_output
            # Better: use per-request metrics if available
            ttft_approx = total / (1 + n_output / max(n_input, 1))
            tpot_approx = (total - ttft_approx) / max(n_output, 1)
            ttfts.append(ttft_approx)
            tpots.append(tpot_approx)

    return {
        "ttft_ms": sum(ttfts) / len(ttfts) * 1000,
        "tpot_ms": sum(tpots) / len(tpots) * 1000,
        "total_ms": total * 1000,
    }


prompts = [
    "The capital of France is",
    "Machine learning is a subset of",
    "Water boils at",
    "Albert Einstein developed the theory of",
]
params = SamplingParams(temperature=0.0, max_tokens=50)

# Baseline
print("=== BASELINE (vLLM eager, FlashAttention) ===")
llm = LLM(model="Qwen/Qwen3-1.7B", dtype="float16", max_model_len=256,
          gpu_memory_utilization=0.4, enforce_eager=True)
base = bench(llm, prompts, params)
print(f"  TTFT: {base['ttft_ms']:.1f} ms")
print(f"  TPOT: {base['tpot_ms']:.1f} ms")
print(f"  Total: {base['total_ms']:.1f} ms")

# Check output quality
base_out = llm.generate(prompts, params)
for i, out in enumerate(base_out):
    print(f"  [{i}] {out.outputs[0].text[:60]}")
del llm; gc.collect(); torch.cuda.empty_cache()

# TurboQuant
print()
print("=== TURBOQUANT (vLLM eager, 4-bit quantized KV) ===")
llm = LLM(model="Qwen/Qwen3-1.7B", dtype="float16", max_model_len=256,
          gpu_memory_utilization=0.4, enforce_eager=True,
          attention_backend="TURBOQUANT")
tq = bench(llm, prompts, params)
print(f"  TTFT: {tq['ttft_ms']:.1f} ms")
print(f"  TPOT: {tq['tpot_ms']:.1f} ms")
print(f"  Total: {tq['total_ms']:.1f} ms")

tq_out = llm.generate(prompts, params)
for i, out in enumerate(tq_out):
    print(f"  [{i}] {out.outputs[0].text[:60]}")

# Summary
print()
print("=== SUMMARY ===")
print(f"  TTFT: baseline={base['ttft_ms']:.1f}ms  TQ={tq['ttft_ms']:.1f}ms  overhead={tq['ttft_ms']/base['ttft_ms']:.2f}x")
print(f"  TPOT: baseline={base['tpot_ms']:.1f}ms  TQ={tq['tpot_ms']:.1f}ms  overhead={tq['tpot_ms']/base['tpot_ms']:.2f}x")
print(f"  Total: baseline={base['total_ms']:.1f}ms  TQ={tq['total_ms']:.1f}ms  overhead={tq['total_ms']/base['total_ms']:.2f}x")
