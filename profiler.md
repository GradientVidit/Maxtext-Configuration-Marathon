## 1. Why does `profiler` exist?

Optimizing Model FLOPs Utilization (MFU) on modern TPUs and GPUs requires deep visibility into kernel execution, memory transfers, HBM bandwidth, and collective communication bubbles (all-reduce, all-to-all).

Different hardware architectures use different profiling ecosystems:
- **TPUs / Google Cloud:** TensorBoard Profiler / XPlane traces capture hardware performance counters, matrix units (MXUs), and SparseCore ops.
- **NVIDIA GPUs:** NVIDIA Nsight Systems (`nsys`) captures CUDA kernel launches, cuDNN execution, NCCL collectives, and SM occupancy.

```text
                     profiler Selection
                             │
     ┌───────────────────────┼───────────────────────┐
     ▼                       ▼                       ▼
    ""                    "xplane"                "nsys"
(Profiling Off)      (TPU/TensorBoard)        (NVIDIA GPUs)
                     Captures XPlane trace    Traces CUDA & NCCL via
                     into output GCS bucket   nsys command-line wrapper
```

`profiler` selects the profiling backend engine to capture hardware execution traces.

---

## 2. Fundamentals & Mechanics

- **`""` (Default):** Profiling is disabled; minimal host-accelerator tracing overhead.
- **`"xplane"`:** Triggers `jax.profiler.start_trace()` capturing XPlane profile archives (`.xplane.pb`) uploaded to TensorBoard or Cloud Storage.
- **`"nsys"`:** MaxText marks profiling ranges for NVIDIA Nsight Systems (`cudaProfilerStart/Stop`).

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `""` | Profiling disabled. |
| TPU Profiling | `"xplane"` | Captures Google TensorBoard XPlane traces for TPU analysis. |
| GPU Profiling | `"nsys"` | Traces GPU kernel execution and NCCL via NVIDIA Nsight Systems. |

---

## 4. Interactions & Dependencies

```text
profiler: "xplane"
     │
     ├──> skip_first_n_steps_for_profiler (Skip warmup & JIT compile)
     ├──> profiler_steps (Number of steps to trace)
     ├──> upload_all_profiler_results (Host 0 vs all hosts)
     └──> profile_cleanly (Inserts step alignment barriers)
```

---

## 5. Practical Scenarios & Failure Modes

- **Capturing a TPU Trace:** Set `profiler: "xplane"`, `skip_first_n_steps_for_profiler: 5`, `profiler_steps: 2`. Open the resulting trace directory in TensorBoard to inspect MXU utilization.
- **GPU Nsys Command:** When using `"nsys"`, wrap the python run: `nsys profile -s none --force-overwrite true --capture-range=cudaProfilerApi ... python3 -m maxtext.trainers.pre_train.train ... profiler=nsys`.

---

### One-line intuition

> **`profiler` selects the hardware trace profiling engine (`"xplane"` for TPUs or `"nsys"` for GPUs) to capture granular execution timelines.**
