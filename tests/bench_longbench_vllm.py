"""LongBench eval via upstream vLLM's native TurboQuant KV-cache quantization.

Verifies whether upstream vLLM (v0.20.0+) `--kv-cache-dtype turboquant_*`
reproduces the TurboQuant paper's claim of fp16-parity at 3.5-bit on
Llama-3.1-8B-Instruct. IMPORTANT: upstream's `turboquant/` module is NOT
the paper's Algorithm 2 — it's the HIGGS (Malinovskii 2025) / Cache-Me-
If-You-Must (Shutova 2025) scalar-Lloyd-Max recipe, rebadged. See
`docs/reference/turboquant-paper-methodology.md` §6 for the algorithm delta.

Reuses the task prompts, truncation, chat-template gating, and scorers
from `bench_longbench_kv_quant.py` to guarantee this shares the HF-path
eval pipeline exactly.

Environment variables / CLI flags:
  --model        HF model id                     default meta-llama/Llama-3.1-8B-Instruct
  --backend      vLLM kv_cache_dtype             required
                 one of: auto (fp16), turboquant_k8v4, turboquant_4bit_nc,
                 turboquant_k3v4_nc, turboquant_3bit_nc, fp8, fp8_e4m3, fp8_e5m2
  --preset       preset name from tests/longbench_presets.json
  --tasks        comma-separated task list (override preset)
  --samples-per-task  N
  --max-model-len N                              default 32768
  --out          single JSON output path
  --gpu-mem      GPU memory utilisation          default 0.85
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

# Reuse the paper-faithful task prompts, scorers, truncation, chat gating.
from bench_longbench_kv_quant import (
    TASK_PROMPTS,
    TASK_MAXGEN,
    _NO_CHAT_TEMPLATE_TASKS,
    load_task,
    score_prediction,
    truncate_prompt,
)


_ALLOWED_BACKENDS = {
    "auto",
    "fp8", "fp8_e4m3", "fp8_e5m2",
    "turboquant_k8v4",
    "turboquant_4bit_nc",
    "turboquant_k3v4_nc",
    "turboquant_3bit_nc",
    # Our TurboQuant plugin (docker/vllm_patches + turboquant.vllm_plugin).
    # Requires `--attention-backend CUSTOM` which we set automatically.
    "ours_fp8",
}


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--backend", required=True,
                   help=f"vLLM kv_cache_dtype; one of {sorted(_ALLOWED_BACKENDS)}")
    p.add_argument("--preset", default=None)
    p.add_argument("--tasks", default=None,
                   help="comma-separated task list; overrides preset")
    p.add_argument("--samples-per-task", type=int, default=None)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--out", required=True, help="single JSON output path")
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--data-dir", default=None,
                   help="LongBench JSONL dir (default /mnt/models/longbench/data)")
    return p.parse_args()


def _load_preset(name: str) -> dict:
    path = THIS_DIR / "longbench_presets.json"
    with open(path) as f:
        presets = json.load(f)
    if name not in presets:
        raise KeyError(f"preset {name!r} not in {list(presets)}")
    return presets[name]


def _resolve_tasks(args: argparse.Namespace) -> dict[str, int]:
    if args.tasks is not None:
        n = args.samples_per_task or 10
        return {t.strip(): n for t in args.tasks.split(",") if t.strip()}
    if args.preset:
        preset = _load_preset(args.preset)
        spec = dict(preset["samples_per_task"])
        if args.samples_per_task is not None:
            spec = {t: args.samples_per_task for t in spec}
        return spec
    raise SystemExit("need --tasks or --preset")


def _render_prompt(tokenizer, task: str, record: dict, max_input_len: int) -> str:
    """Build the final prompt string: task template → middle-truncate → chat-template."""
    raw = TASK_PROMPTS[task].format(context=record["context"], input=record["input"])
    # LongBench canonical: truncate raw task prompt first, chat-template after.
    truncated = truncate_prompt(tokenizer, raw, max_input_len)
    if task in _NO_CHAT_TEMPLATE_TASKS:
        return truncated
    messages = [{"role": "user", "content": truncated}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def main() -> int:
    args = _parse_cli()
    if args.backend not in _ALLOWED_BACKENDS:
        raise SystemExit(
            f"--backend {args.backend!r} not in allowed set {sorted(_ALLOWED_BACKENDS)}")

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data_dir = args.data_dir or os.environ.get(
        "HYP055_LONGBENCH_DIR", "/mnt/models/longbench/data")

    tasks = _resolve_tasks(args)
    print(f"[vllm-lb] model={args.model}  backend={args.backend}  "
          f"tasks={list(tasks)}  out={out_path}", flush=True)

    # Reserve room for the longest per-task gen.
    max_new = max(TASK_MAXGEN[t] for t in tasks)
    # Leave ~16 tokens of chat-template overhead headroom on top of the
    # middle-truncate target.
    max_input_len = args.max_model_len - max_new - 16

    # Load tokenizer separately for truncation + chat template.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # ----- initialise vLLM ------------------------------------------------
    from vllm import LLM, SamplingParams

    llm_kwargs = dict(
        model=args.model,
        dtype="float16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        enforce_eager=True,   # avoid CUDA-graph warmup for deterministic eval
        disable_log_stats=True,
        trust_remote_code=True,
    )
    if args.backend == "ours_fp8":
        llm_kwargs["kv_cache_dtype"] = "fp8"
        llm_kwargs["attention_backend"] = "CUSTOM"
    elif args.backend != "auto":
        llm_kwargs["kv_cache_dtype"] = args.backend

    # Sanity: confirm plugin is registered before we touch vLLM.
    if args.backend == "ours_fp8":
        try:
            import turboquant.vllm_plugin as _tq_plugin  # noqa: F401
            _tq_plugin.register()
            print("[vllm-lb] TurboQuant plugin register() called", flush=True)
        except Exception as e:
            print(f"[vllm-lb] plugin import/register failed: {e}", flush=True)
            raise

    t_load = time.perf_counter()
    print(f"[vllm-lb] loading vLLM LLM with kwargs={llm_kwargs}", flush=True)
    llm = LLM(**llm_kwargs)
    print(f"[vllm-lb] loaded in {time.perf_counter() - t_load:.1f}s", flush=True)

    # ----- eval loop ------------------------------------------------------
    results_by_task: dict[str, dict] = {}

    for task, n in tasks.items():
        print(f"[vllm-lb] === task={task} n={n} ===", flush=True)
        records = load_task(task, n, data_dir=data_dir)

        # Build all prompts for this task; we can batch the whole task in one
        # llm.generate call so vLLM's continuous batching amortises.
        prompts = [_render_prompt(tokenizer, task, ex, max_input_len)
                   for ex in records]
        sp = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=TASK_MAXGEN[task],
            min_tokens=1,
        )

        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params=sp, use_tqdm=False)
        dt_total = time.perf_counter() - t0

        # vLLM preserves input order when passing a list.
        preds: list[dict] = []
        for i, (ex, out) in enumerate(zip(records, outputs)):
            gen = out.outputs[0].text if out.outputs else ""
            refs = ex.get("answers", ex.get("answer", []))
            all_classes = ex.get("all_classes")
            s = score_prediction(task, gen, refs, all_classes=all_classes)
            preds.append({
                "id": i,
                "pred": gen,
                "refs": refs,
                "score": float(s),
                "input_len_tok": len(out.prompt_token_ids),
                "gen_len_tok": len(out.outputs[0].token_ids) if out.outputs else 0,
            })
            if i < 3 or (i + 1) % 20 == 0:
                print(f"[vllm-lb][{task}] {i+1}/{len(records)} "
                      f"ilen={preds[-1]['input_len_tok']} "
                      f"olen={preds[-1]['gen_len_tok']} "
                      f"score={s:.3f}", flush=True)

        mean = sum(p["score"] for p in preds) / max(1, len(preds))
        results_by_task[task] = {
            "task": task,
            "n": len(preds),
            "mean_score": mean,
            "total_time_s": dt_total,
            "mean_time_s": dt_total / max(1, len(preds)),
            "predictions": preds,
        }
        print(f"[vllm-lb] task={task} mean_score={mean:.3f} "
              f"total_t={dt_total:.1f}s", flush=True)

    # ----- aggregate ------------------------------------------------------
    summary = {
        "model": args.model,
        "backend": args.backend,
        "preset": args.preset,
        "tasks": {k: {kk: vv for kk, vv in v.items() if kk != "predictions"}
                  for k, v in results_by_task.items()},
        "predictions_by_task": {k: v["predictions"] for k, v in results_by_task.items()},
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"[vllm-lb] wrote {out_path}", flush=True)
    try:
        import torch
        print(f"[vllm-lb] peak GPU mem = "
              f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
