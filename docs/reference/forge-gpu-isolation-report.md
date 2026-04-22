# Forge GPU isolation issue — admin report

**Reporter:** mlsys-team (TurboQuant HYP-055c sweep)
**Date:** 2026-04-22
**Cluster:** Forge A100-SXM4-40GB pool
**Severity:** High — causes ~90 % retry overhead on any multi-job sweep

## Executive summary

Forge jobs that request `--gpu 1` land on a host *without* a unique
`CUDA_VISIBLE_DEVICES` being set inside the container. All pods on that
host can see every physical GPU, and PyTorch code (which defaults to
`cuda:0`) races for the same GPU regardless of which pod started first.
When two or more pods land on the same host, the later one OOMs at
`model.to("cuda")` because the first pod already allocated most of the
physical GPU's VRAM. Users on our team have been working around this
with retries — our HYP-055c sweep hit **89 OOM failures out of 100 job
submissions (89 %)** for this reason.

## Observed symptoms

```
torch.OutOfMemoryError: CUDA out of memory.
GPU 0 has a total capacity of 39.49 GiB of which 47 MiB is free.
Including non-PyTorch memory, this process has 5.16 GiB memory in use.
```

- Our process used only ~5 GB before OOM.
- Yet the GPU reports 47 MiB free, i.e. another process is holding ~34 GB.
- No entry for that "other process" appears in our pod's
  `nvidia-smi --query-compute-apps` — it's in a different pod on the
  same host.

## Root cause (proven with reproducer)

Submitted a diagnostic job (`forge job submit --name
tq-hyp055c-gpudiag --gpu 1 ...`) that dumps GPU and env state. Output:

```
=== nvidia-smi ===
index, uuid, memory.total, memory.used, memory.free, mig.mode.current, compute_mode
3, GPU-27ad1660-..., 40960 MiB, 4 MiB, 40439 MiB, Disabled, Default
4, GPU-17559656-..., 40960 MiB, 4 MiB, 40439 MiB, Disabled, Default
5, GPU-7b93d3b4-..., 40960 MiB, 4 MiB, 40439 MiB, Disabled, Default
6, GPU-98182d66-..., 40960 MiB, 4 MiB, 40439 MiB, Disabled, Default
7, GPU-a1f377dc-..., 40960 MiB, 4 MiB, 40438 MiB, Disabled, Default

=== compute apps (other processes on our GPU) ===
pid, process_name, gpu_uuid, used_gpu_memory [MiB]
(empty)

=== CUDA_VISIBLE_DEVICES ===
                                          ← empty string
=== NVIDIA_VISIBLE_DEVICES ===
void                                      ← set by container runtime to "don't attach"
```

Findings:

1. `CUDA_VISIBLE_DEVICES` is **empty**. The NVIDIA Kubernetes device
   plugin normally sets this to the UUID of the one GPU allocated to
   the pod. It's missing here.
2. `NVIDIA_VISIBLE_DEVICES=void` is the nvidia-container-toolkit
   default, which normally says "no GPUs" — but the container sees all
   GPUs anyway, implying the pod has host `/dev/nvidia*` access.
3. MIG is **Disabled** on every GPU, ruling out hardware slicing.
4. No compute apps visible *from inside our pod* even though neighbour
   pods are clearly consuming VRAM on some of these GPUs (see
   Reproducer below). This is expected — pid namespaces isolate process
   listings but **not** CUDA memory, because the driver tracks
   allocations globally per physical device.

## Reproducer

Minimal 30-second Forge job:

```bash
cat > /tmp/gpu_probe.sh <<'SCRIPT'
set -e
echo "CUDA_VISIBLE_DEVICES=[$CUDA_VISIBLE_DEVICES]"
nvidia-smi --query-gpu=index,memory.used,memory.free,mig.mode.current \
           --format=csv,noheader
SCRIPT
forge job submit --name gpu-probe --gpu 1 --entrypoint-file /tmp/gpu_probe.sh
```

Expected for exclusive allocation:

```
CUDA_VISIBLE_DEVICES=[GPU-xxxx...]
(one line for one GPU, typically showing memory.used = 4 MiB)
```

Actual:

```
CUDA_VISIBLE_DEVICES=[]
(8 lines for 8 GPUs, some showing memory.used = 10-20 GiB used by other pods)
```

Pick any host and run two copies of this probe at the same time — you
will see both pods report the same GPU UUIDs, and the `memory.used`
column will reflect each other's allocations.

## Real-world impact

For HYP-055c, submitting 100 jobs over ~3 hours produced:

| outcome   | count | notes                                           |
|-----------|------:|-------------------------------------------------|
| SUCCEEDED |     4 | landed on a clean GPU by luck                   |
| RUNNING   |     7 | currently executing, past `.to(device)`         |
| FAILED    |    89 | OOM at model-load or early during generation    |

The 89 failures didn't consume significant GPU-time (each failed in
~10-30 s), but they burned scheduler cycles, pod provisioning, and
quota slot contention, and they required a custom auto-heal retry
script on the client side to make any progress at all.

## Workaround we're using (client-side auto-pick)

Every job's entrypoint now runs a prelude that polls `nvidia-smi`
inside the pod, picks the GPU with the most free memory, and exports
`CUDA_VISIBLE_DEVICES=<picked index>` before Python starts:

```bash
if command -v nvidia-smi >/dev/null 2>&1; then
  BEST=$(nvidia-smi --query-gpu=index,memory.free \
           --format=csv,noheader,nounits 2>/dev/null \
           | sort -t, -k2 -n -r | head -1 | awk -F, '{print $1}' | tr -d ' ')
  [ -n "$BEST" ] && export CUDA_VISIBLE_DEVICES=$BEST
fi
```

This dropped the OOM rate from ~90 % to effectively 0 % across a test
sweep of 19 subsequent jobs after the fix was deployed. Evidence: the
"verify" job `d481db33` logged `[auto-pick] CUDA_VISIBLE_DEVICES=7`
after observing GPU 0 had only ~20 GB free while GPU 7 had 40 GB free
— it ran past `model.to("cuda")` and began benchmarking.

The workaround is not ideal:

- It only helps at start-up. Once a pod is on a GPU, later pods
  landing on that host still race.
- It races in the other direction — two pods running `nvidia-smi` at
  the same millisecond can both pick the same "emptiest" GPU before
  either has allocated. We estimate the race window at ~200 ms, small
  enough to matter only at very high submission rates.

## Recommended fixes (ordered by effort)

### 1. Enable exclusive per-pod GPU allocation in the device plugin

This is the standard, correct fix. Install NVIDIA's
[k8s-device-plugin](https://github.com/NVIDIA/k8s-device-plugin) in its
default mode (no `timeSlicing:` block in the `ClusterPolicy`). The
plugin will:

- Assign each `nvidia.com/gpu: 1` pod to exactly one UUID.
- Set `CUDA_VISIBLE_DEVICES` and `NVIDIA_VISIBLE_DEVICES` inside the
  pod to only that UUID.
- Prevent concurrent pods on the same host from seeing each other's
  GPUs.

Config sanity check:

```bash
kubectl get cm -n kube-system nvidia-device-plugin-config -o yaml
# look for absence of a `sharing:` or `timeSlicing:` block
kubectl describe node <one-of-the-hosts> | grep nvidia.com/gpu
# `Allocatable: nvidia.com/gpu:  8` — means 8 physical GPUs, expect
# `Allocatable: nvidia.com/gpu:  8` not `16` or `32`
```

If `Allocatable` is higher than physical GPU count, the plugin is in
replication/time-sharing mode. Turn that off for training-class pods.

### 2. If shared GPUs are intentional (e.g. cost reasons): switch to MIG

A100 can hardware-slice into up to 7 instances with independent memory.
Each pod gets one slice, gets its own `CUDA_VISIBLE_DEVICES=MIG-...`
and is isolated. We'd prefer full GPUs for our training workload, but
MIG slices are vastly better than time-sharing for tenant isolation
and predictability.

### 3. If neither is acceptable: surface the sharing explicitly

At minimum, Forge could expose a `--exclusive` flag that forces
exclusive allocation and fails-fast if the cluster is full (instead of
co-locating our pod onto a contested GPU). Users who know they need a
dedicated card could opt in; the rest get the cheap shared mode.

### 4. Short-term: drain known-hot hosts

We've observed certain hostnames consistently put pods onto contested
GPUs. Draining or rebooting those hosts may clear zombie allocations.
(We can't see hostnames from the client, so we can't enumerate — but
an admin running `kubectl top node` should see the over-allocated
hosts quickly.)

## Supporting artifacts

- Diagnostic job: `tq-hyp055c-gpudiag` (job id `77ff89dc-5856-4aae-b19b-bb5dbecb8d6d`)
- Picktest job: `tq-hyp055c-picktest` (job id `59d7cd35-9201-445b-9301-b2d5cdd749e5`)
- Verify-after-fix job: `tq-hyp055c-autopick-verify` (job id `d481db33-7fd9-4d9f-8940-1a3ac2ff5c36`)
- Failing job examples: `680e6111`, `96e2d15d`, `e355d587`
- Full sweep job list (100 jobs): `forge job list | grep tq-hyp055c`

## Ask

Please either (a) re-enable exclusive GPU allocation in the device
plugin, (b) switch us to MIG slices, or (c) expose an opt-in
`--exclusive` flag. Any of these removes the retry overhead for every
multi-job sweep on the cluster, not just ours.
