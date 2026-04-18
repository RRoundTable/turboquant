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

### Code under test

- Repo commit: `5db1d80` (`main`).
- TurboQuant package installed via `pip install -e .` at job runtime.
- vLLM source-overlaid at job runtime by copying current
  `docker/vllm_patches/{v1/attention/backend.py, v1/kv_cache_interface.py,
  v1/worker/gpu_model_runner.py, model_executor/layers/attention/attention.py}`
  on top of vLLM's site-packages — same overlay used by `Dockerfile`.
- **The overlay carries the customizations from
  [vllm-project/vllm#39868](https://github.com/vllm-project/vllm/pull/39868)**
  ("[v1] Allow attention backends to declare custom KV cache page size"):
  `AttentionBackend.get_kv_cache_page_size`, `AttentionSpec.custom_page_size`,
  and the `_reshape_kv_cache_tensors` view-prefix path. `TurboQuantBackend`
  implements `get_kv_cache_page_size` to return its packed-byte size when
  `cache_dtype == "fp8"`. **This is the customized-vLLM setting, not stock
  vLLM** — confirmation that the page-size override fired at runtime is
  in §Result below.
- TurboQuant backend explicitly registered: `turboquant.vllm_plugin.register()`
  before `LLM(...)` construction.

### Environment (per job container)

- Forge image: `tq-hyp029:pr` (id `3e953d1f`, base `mlops-notebook`).
- vLLM `v0.19.0`, FlashInfer (image-pinned), torch+cuda from base image.
- 1× A100-SXM4-40GB per job, no security profile, `--shared-nfs`.
- Code & cache mounted from `/workspace/shared/turboquant-bench/` and
  `HF_HOME=/workspace/shared/hf-cache` (model weights pre-warmed there).

### Workload

- Model: `Qwen/Qwen3-8B`, `dtype=float16`. (8 KV heads, 36 layers, 128 head dim.)
- Inputs: `prompts = [TokensPrompt(prompt_token_ids=[1]*input_len)] * batch`
  — synthetic, no tokenizer call, identical across configs.
- `SamplingParams(max_tokens=128, min_tokens=128, temperature=0.0, ignore_eos=True)`.
- `output_len=128` for every config (decode-heavy: 1 prefill, 128 decode).
- 1 warmup `LLM.generate(...)` + 3 timed trials; report **median** wall time.
- Decode throughput = `batch * 128 / median_seconds` (counts only generated tokens).
- GPU memory: `nvidia-smi --query-gpu=memory.used` snapshot from the bench
  python process after generate (engine subprocess shares the device).

### Per-config sweep

`seq ∈ {1024, 4096, 8192, 16384} × batch ∈ {1, 8, 32}` ⇒ 12 jobs total,
fanned out in parallel on Forge. Each job runs **baseline first, then TQ
back-to-back** in fresh `LLM` instances. If a partial result file already
exists on the shared NFS, that backend is skipped (lets us re-run only
the missing TQ legs without re-running baseline).

Per-engine kwargs:

```python
common = dict(
    model="Qwen/Qwen3-8B", dtype="float16",
    gpu_memory_utilization=0.85,
    max_model_len=input_len + output_len + 16,
    enforce_eager=True,                # see "constraints" below
    disable_log_stats=True,
)
baseline = LLM(**common)               # kv_cache_dtype=auto, FLASHINFER backend
tq       = LLM(**common,
               kv_cache_dtype="fp8",
               attention_backend="CUSTOM")
```

vLLM defaults left on for both: `enable_prefix_caching=True`,
`enable_chunked_prefill=True`, `tensor_parallel_size=1`, `seed=0`.

### Constraints (forced by the platform, not by the experiment)

- `enforce_eager=True` for **both** backends. With `enforce_eager=False`
  (cuda graphs + torch.compile), the TQ leg crashes during inductor
  autotune with `ValueError("type fp8e4nv not supported in this
  architecture. The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')")` on
  A100 (SM80). Setting `kv_cache_dtype="fp8_e5m2"` to dodge fp8e4nv was
  rejected by `TurboQuantBackend.supports_kv_cache_dtype` (only `"fp8"`
  / `"fp8_e4m3"` are accepted). Choosing eager keeps the comparison
  apples-to-apples; H100/H200 should lift this constraint.
- `attention_backend="CUSTOM"` is required for the TQ leg — without it,
  vLLM auto-selects FLASHINFER even with the plugin registered, so the
  measured backend was confirmed via the engine log line
  `Using AttentionBackendEnum.CUSTOM backend.`

### Forge jobs (for traceability)

| seq×batch | job id     | result    |
|----------:|------------|-----------|
| 1024×1    | `c93a6950` | SUCCEEDED |
| 1024×8    | `eb5c3215` | SUCCEEDED |
| 1024×32   | `1ade2ce4` | SUCCEEDED |
| 4096×1    | `e3e50205` | SUCCEEDED |
| 4096×8    | `bab852d9` | SUCCEEDED |
| 4096×32   | `534afd03` | TQ OOM    |
| 8192×1    | `2748639a` | SUCCEEDED |
| 8192×8    | `df6dd4fb` | SUCCEEDED |
| 8192×32   | `07bae58a` | TQ OOM    |
| 16384×1   | `946882f5` | SUCCEEDED |
| 16384×8   | `bb8ef5c8` | TQ OOM    |
| 16384×32  | `b8aa64c9` | TQ OOM    |

### Files

- Bench: `tests/bench_vllm_serve.py`
- Per-job entrypoint: `tests/bench_entry.sh` (consumes `SEQ`, `BATCH` env vars)
- Raw per-config JSON + `aggregate.py` + `REPORT.md`: `results/v5_vs_baseline/`

## Status: rejected

## Result

### PR #39868 cache-compression check (sanity)

Engine logs report `GPU KV cache size` after profiling — same value across
every job (sweep used the same `max_model_len` ≈ 1168 only at seq=1024;
the long-seq jobs use a per-config `max_model_len = input_len + 128 + 16`,
but the cache budget is set by `gpu_memory_utilization=0.85` so the
per-token math is comparable):

|        | KV tokens | × baseline |
|--------|----------:|-----------:|
| baseline (fp16, FLASHINFER)   | 126,416 |  1.00× |
| TQ (fp8, CUSTOM, PR #39868)   | 404,544 |  3.20× |

The **3.20× cache compression from PR #39868 is fully realized** — the
per-page byte budget really did shrink. So the remaining issues below
are *not* "the PR didn't take effect"; they are downstream of a
correctly-compressed cache.

### Throughput / memory / OOM

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

- **Defeats the cache-side memory savings even though they are real.**
  PR #39868 successfully shrank the cache (3.20× more KV tokens per
  byte, confirmed above), but the workspace is sized like an
  un-quantized expansion buffer, so total resident memory at runtime is
  *higher* than baseline for the same workload.
- Causes OOM at exactly the configs where TQ should have shone (long
  context × large batch — the decode-bound regime where memory savings
  matter most). The OOM tracebacks all land at
  `_get_v5_ws` allocating `k_quant`/`v_quant` tensors, not at
  `_reshape_kv_cache_tensors` allocating the cache.

So the picture is: PR #39868 does its job (smaller cache pages), but the
backend's per-step scratch buffers eat the headroom and then some.

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
