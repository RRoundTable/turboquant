# HYP-041: v5_paged vs baseline vLLM at end-to-end serving (Qwen3-8B, A100-40GB)

## Hypothesis

At commit `5db1d80` ("Batch × seq sweep: long-context latency gap is NOT
serving-amortizable"), the v5_paged TurboQuant kernel — wired through vLLM
as `attention_backend="CUSTOM" + kv_cache_dtype="fp8"` — should at least
*match* baseline vLLM (FlashInfer, fp16 KV cache) for decode throughput at
small batch and short context, and trade some throughput for the claimed
3.76× KV-cache memory savings at large batch and long context.

## Prediction

For Qwen/Qwen3-8B on a single A100-40GB, output_len=128, eager mode
(both backends; A100 SM80 cannot torch.compile fp8e4nv):

- **At seq=1024, batch=1**: TQ within 0.95× of baseline (decode is launch-
  overhead bound; quant cost negligible).
- **At seq=4096–16384, batch=8–32**: TQ within 0.8× of baseline on tok/s
  but using ≤ baseline GPU memory (KV cache compressed 3.76×).
- **OOM**: baseline OOMs first at the largest configs; TQ runs further
  thanks to compressed cache.

## Method

Sweep grid: `seq ∈ {1024, 4096, 8192, 16384} × batch ∈ {1, 8, 32}` (12
configs). For each, one Forge job (1 A100-40GB, image `tq-hyp029:pr` with
current `docker/vllm_patches/` overlaid at runtime) runs:

1. **baseline**: `LLM(model='Qwen/Qwen3-8B', dtype=fp16, kv_cache_dtype=auto, enforce_eager=True)`
2. **tq**: same but `kv_cache_dtype='fp8', attention_backend='CUSTOM'`,
   `turboquant.vllm_plugin.register()` called.

Same inputs (token_id `[1] * input_len` × batch). 1 warmup, 3 trials,
median of 3. Decode tokens/s = `batch * output_len / median_s`. GPU memory
captured from `nvidia-smi` after the engine is up.

Bench script: `tests/bench_vllm_serve.py`. Entrypoint: `tests/bench_entry.sh`.
Raw JSON + aggregate in `results/v5_vs_baseline/`.

## Status: rejected

## Result

| seq×batch | base tok/s | tq tok/s | tq/base | base mem (GB) | tq mem (GB) |
|----------:|-----------:|---------:|--------:|--------------:|------------:|
|   1024×1  |       48.4 |     38.4 |   0.79× |         34.26 |       34.31 |
|   1024×8  |      376.8 |    291.6 |   0.77× |         34.26 |       34.62 |
|  1024×32  |     1415.6 |    961.2 |   0.68× |         34.26 |       37.88 |
|   4096×1  |       47.2 |     38.1 |   0.81× |         34.45 |       34.49 |
|   4096×8  |      373.9 |    291.3 |   0.78× |         34.45 |       37.00 |
|  4096×32  |     1268.9 |      OOM |       — |         34.45 |         OOM |
|   8192×1  |       47.6 |     37.3 |   0.78× |         34.83 |       34.92 |
|   8192×8  |      380.3 |    215.1 |   0.57× |         34.83 |       38.51 |
|  8192×32  |      922.0 |      OOM |       — |         34.83 |         OOM |
|  16384×1  |       48.2 |     38.2 |   0.79× |         34.83 |       34.94 |
|  16384×8  |      351.3 |      OOM |       — |         34.83 |         OOM |
| 16384×32  |      600.5 |      OOM |       — |         34.83 |         OOM |

Three things were predicted; all three were wrong:

1. **TQ is uniformly slower**, not at-parity at small batch. Even at
   seq=1024, batch=1 (decode-bound, launch-dominated), TQ is 0.79× of
   baseline. The quant/dequant overhead per decode step is non-trivial
   even when only ~1024 KV tokens exist per request.

2. **TQ uses *more* GPU memory, not less**, in every config where both
   ran. At seq=8192×8, TQ took 38.5 GB vs baseline's 34.8 GB. The
   cache-only 3.76× savings are dwarfed by the per-batch workspace
   (`(B, num_kv_heads, max_len, qbytes)` k_quant/v_quant tensors
   allocated in `vllm_backend_fused._get_v5_ws`).

3. **TQ OOMs first**, not baseline. Baseline ran every config; TQ failed
   at 4 of 12 (s4096×32, s8192×32, s16384×8, s16384×32). The OOM trace
   points at the workspace allocation, not the cache itself.

## Analysis

Three distinct issues, in order of severity:

### Issue A — workspace allocation dominates and prevents long-context use

`turboquant/vllm_backend_fused.py:_get_v5_ws` allocates a per-batch
workspace whose size is `O(batch × num_kv_heads × max_len × qbytes)`.
For Qwen3-8B with 8 KV heads, qbytes ≈ 64, this hits ~268 MB per tensor
at (32, 16384) — and there are several. The workspace allocation runs
*every decode step* (it's not pre-allocated to its peak). This:

- Defeats the cache-side memory savings. The workspace is sized like an
  un-quantized expansion buffer, so total resident memory is *higher*
  than baseline for the same workload.
- Causes OOM at exactly the configs where TQ should have shone (long
  context × large batch — the decode-bound regime where memory savings
  matter most).

### Issue B — eager-mode tax is structural, not a fluke

We had to set `enforce_eager=True` because A100 SM80 can't torch.compile
the fp8e4nv ops vLLM lowers to when `kv_cache_dtype='fp8'`. This
penalizes both backends, but baseline (FLASHINFER + fp16 cache) loses
less from eager mode than TQ does (TQ's dequant + attention path has
more launches per layer). On H100 this constraint disappears, but until
HYP-041 is re-run there, A100-40GB serving numbers will look pessimistic
for TQ.

### Issue C — the 0.57–0.81× throughput gap is not explained by quant cost alone

At seq=8192×8 the gap is 0.57× (TQ runs at 215 tok/s vs baseline 380).
That's a 1.77× decode-step latency increase, far larger than what the
HYP-030/HYP-035 micro-benchmarks predicted for the kernel itself. The
delta points at integration overhead (Python dispatch, workspace alloc
per step, copy-in/copy-out) rather than the kernel hot path.

## Next-step candidates (to be filed as their own HYPs)

1. **Pre-allocate the workspace to (max_batch × max_len)** at engine-init
   time and slice per step. Should fix Issue A (OOM) and recover some of
   the gap from Issue C.
2. **Profile a single TQ decode step with nsys/ncu** at seq=8192×8 to
   attribute the 1.77× slowdown across {workspace alloc, dispatch,
   kernel, copy}.
3. **Re-run HYP-041 on H100** so torch.compile + CUDA graphs are
   available — needed before quoting any serving numbers externally.
4. **Audit `_get_v5_ws` against the v5 kernel signature** — confirm
   `qbytes` and `max_len` are actually needed at the dimensions the
   allocator uses, or whether they can shrink to per-page rather than
   per-request.
