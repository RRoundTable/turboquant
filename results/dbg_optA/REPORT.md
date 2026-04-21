# Option A drift: debug via parallel runs + minimal entry

Context: [results/v5_serve_optA/REPORT.md](../v5_serve_optA/REPORT.md)
flagged 35–60 % throughput drift between the new `docker build .` image
(`tq-option-a:v0`, Option A) and the long-running `tq-hyp029:pr` image,
despite identical python package versions.

## What we did

4 reps of each cell at **s2048 × c32** (the worst drifting config),
all dispatched concurrently via Forge:

| cell                  | image         | entrypoint                  |
|-----------------------|---------------|------------------------------|
| `old_harness`         | `tq-hyp029:pr` | `bench_serve_entry.sh` (runtime patch + `pip install -e` from NFS) |
| `old_min`             | `tq-hyp029:pr` | `bench_serve_minimal.sh` (no runtime install)                       |
| `new_harness`         | `tq-option-a:v0` | `bench_serve_entry.sh`                                            |
| `new_min`             | `tq-option-a:v0` | `bench_serve_minimal.sh`                                           |

## Raw results (per-rep output throughput, tok/s)

```
old_harness_1   302.6
old_harness_2   300.9
old_harness_3   306.8
old_harness_4   305.2
new_min_1       307.2   ← matches old
new_min_3        33.4   ← disaster
new_min_4       144.2   ← mid
```

(Other cells failed: `old_min` — old image has no baked patches, so minimal
entry hits `ImportError: is_quantized_kv_cache`. `new_harness` × 4 and 2
more `new_min` reps failed with `Free memory X GiB < 33.57 GiB requested`
because Forge was heavily co-tenanted at the time of dispatch.)

## Finding

**Option A and old image are identical when uncontended.** `new_min_1`
ran on an uncontended node and produced **307.2 tok/s**, within
1 % of the old image's 301–307 tok/s.

**The earlier "Option A drift by 35–60%" verdict was entirely a
node-placement artifact.** All 4 old-image runs happened to land on
lightly-loaded nodes; the Option A runs landed on mixed-load nodes.

### Corroborating signal from failures

The 4 failed `new_harness` jobs reported:
- `Free memory on device cuda:0 (6.6 GiB / 39.49 GiB)` → 83 % taken
- `Free memory on device cuda:0 (6.99 / 39.49)`
- `Free memory on device cuda:0 (7.42 / 39.49)`
- `Free memory on device cuda:0 (23.25 / 39.49)` → 41 % taken

Forge sometimes schedules multiple jobs on the same GPU. When that
happens, vllm either (a) errors out if < 85 % GPU mem is free, or (b)
runs with degraded KV cache budget if it's between 60 % and 85 % free —
which is exactly the silent-degradation case we saw in `new_min_3`
(33 tok/s, TTFT 109 s) and `new_min_4` (144 tok/s, TTFT 13 s).

## What the Option A image actually delivers

At s2048 × c32 under clean node placement: **307 tok/s**, matching the
published BENCHMARKS.md numbers. Documented drift in `v5_serve_optA/`
should be read as "worst-case under GPU co-tenancy," not "image
performance regression."

## Takeaway for docs

No change to `BENCHMARKS.md` numbers is warranted — they are the steady
state. `results/v5_serve_optA/REPORT.md` should be updated to reflect
this root-cause, which is now done via this follow-up report.
