## 1. Why it exists: statistical stability in hardware benchmarking

Benchmarking a single execution step on hardware accelerators produces noisy, misleading numbers:

```text
Single Iteration Benchmark (High Variance):
[Launch Kernel] ──> [OS jitter / Cache misses / Host-device synchronization] ──> Misleading Latency

Multi-Iteration Benchmark (Statistical Steady-State):
┌───────────┐   ┌───────────┐       ┌───────────┐
│ Iter 1    │ + │ Iter 2    │ + ... │ Iter N    │ ──> Discard warmup ──> Accurate Mean / P50 / P99
└───────────┘   └───────────┘       └───────────┘
```

On TPUs and GPUs, initial iterations incur host-to-device launch overhead, XLA graph caching, and dynamic power/frequency ramp-up. Furthermore, in the autoregressive decode phase, measuring only 1 or 2 generated tokens fails to measure steady-state memory bandwidth utilization.

`inference_microbenchmark_loop_iters` sets the number of repetition cycles executed for each benchmark configuration, ensuring measurements reach statistical convergence and represent steady-state production throughput.

---

## 2. Mechanics: measurement timing loop

Inside the benchmark harness, after an unmetered warmup pass to trigger JAX/XLA compilation, the execution loop repeats for `inference_microbenchmark_loop_iters` times:

```text
               ┌──────────────────────────────────────────────┐
               │          Execute Warmup Pass (Untimed)        │
               │   - Forces JIT compilation & HBM allocation  │
               └──────────────────────┬───────────────────────┘
                                      │
                                      ▼
               ┌──────────────────────────────────────────────┐
               │              Start High-Res Timer            │
               │          t_start = time.perf_counter()       │
               └──────────────────────┬───────────────────────┘
                                      │
               ┌──────────────────────┴───────────────────────┐
               ▼                                              ▼
       Iteration 1 (Timed)                           Iteration N (Timed)
       - Run model forward pass                       - Run model forward pass
       - jax.block_until_ready()                      - jax.block_until_ready()
               │                                              │
               └──────────────────────┬───────────────────────┘
                                      │
                                      ▼
               ┌──────────────────────────────────────────────┐
               │               Stop High-Res Timer            │
               │           t_end = time.perf_counter()        │
               │  mean_latency = (t_end - t_start) / loop_iters│
               └──────────────────────────────────────────────┘
```

Key aspects of the loop:
- **`jax.block_until_ready()`**: JAX executes asynchronously by default. MaxText explicitly synchronizes at the boundary of the benchmark loop to measure true accelerator device execution time rather than Python dispatch time.
- **Averaging**: Total elapsed time across all `loop_iters` is divided by `loop_iters` to calculate average step latency, tokens/sec, and FLOPS.

---

## 3. Options & Configuration

Default in `base.yml`:

```yaml
inference_microbenchmark_loop_iters: 10
```

| Setting | Value | Characteristics | Best Used For |
|---|---|---|---|
| Quick validation | `3`–`5` | Rapid CI smoke tests, verifying kernel compilation without long waits. | Automated integration pipelines and sanity checks. |
| Standard benchmark (default) | `10` | Balances benchmark runtime with statistical stability (<2% variance). | General development profiling and hardware comparisons. |
| Production SLA calibration | `50`–`100` | High-precision tail-latency profiling (captures P90/P99 latency variance). | Final hardware qualification before production cluster rollout. |

---

## 4. Parameter Interactions

```text
┌───────────────────────────────────────────────────────────┐
│            inference_microbenchmark_loop_iters            │
└─────────────┬───────────────────────────────┬─────────────┘
              │                               │
              ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│Total Benchmark Duration   │   │inference_microbenchmark_  │
│Time = (Num Lengths)       │   │stages                     │
│     * (Num Samples)       │   │In "generate" stage, sets  │
│     * (loop_iters)        │   │number of sequential       │
│     * (Step Time)         │   │autoregressive steps.      │
└───────────────────────────┘   └───────────────────────────┘
```

- **Combinatorial run time**: Total benchmark time scales multiplicatively:
  $$\text{Total Runs} = |\text{prefill\_lengths}| \times |\text{num\_samples}| \times \text{loop\_iters}$$
  If testing 6 lengths and 5 batch sizes with 100 iterations, the runner executes 3,000 passes.
- **`inference_microbenchmark_log_file_path`**: The final averaged metrics per iteration configuration are written to this destination.

---

## 5. Practical Scenarios & Failure Modes

### Benchmarking Trade-offs
- **Too low (`loop_iters: 1`)**: Results will be corrupted by Python garbage collection spikes, residual async dispatch delays, and clock frequency fluctuations.
- **Too high (`loop_iters: 1000`)**: Benchmarking can take hours, especially for large models (e.g. 70B+) where a single forward pass takes tens of milliseconds.

---

### One-line intuition

> **`inference_microbenchmark_loop_iters` sets the number of repeated timing passes per benchmark configuration, ensuring reported latency and throughput reflect true steady-state accelerator performance rather than transient jitter.**
