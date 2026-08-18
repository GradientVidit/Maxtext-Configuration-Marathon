## 1. Why it exists: decoupling prefill and decode profiling

An LLM inference lifecycle consists of two radically different computational phases:

```text
Prompt Ingestion (Prefill)          Autoregressive Generation (Decode)
┌──────────────────────────┐        ┌───────────────────────────────────┐
│ Input: [x_1, ..., x_N]   │        │ Input: [x_{N+k}] + Cache          │
│ Matrix-Matrix (GEMM)     │        │ Matrix-Vector (GEMV)              │
│ Compute Bound (FLOPs)    │ ────>  │ Memory Bandwidth Bound (HBM I/O)  │
│ Sets Time-To-First-Token │        │ Sets Inter-Token Latency (ITL)    │
│ (TTFT)                   │        │ & Tokens Per Second (TPS)         │
└──────────────────────────┘        └───────────────────────────────────┘
```

When evaluating model efficiency, running full end-to-end inference conflates bottlenecks:
- A slow prefill might be masked if the generation phase is fast.
- A memory bandwidth bottleneck in decode might obscure an underperforming attention kernel in prefill.
- In **disaggregated serving**, prefill and decode execute on separate TPU slices; benchmarking them requires running each stage independently.

`inference_microbenchmark_stages` exists to select which specific inference phases are benchmarked, allowing engineers to isolate prefill latency (TTFT) from autoregressive step latency (ITL/throughput).

---

## 2. Mechanics: execution branch control

The microbenchmark runner parses `inference_microbenchmark_stages` into a set of active test phases:

```text
Config: inference_microbenchmark_stages: "prefill,generate"
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
   Stage: "prefill"                      Stage: "generate"
┌───────────────────────────┐         ┌───────────────────────────┐
│ Run synthetic prompt GEMMs│         │ Load/mock KV cache        │
│ Measure TTFT              │         │ Step-by-step token decode │
│ Output: Prefill ms, TFLOPs│         │ Measure step latency, TPS │
└───────────────────────────┘         └───────────────────────────┘
```

- If `"prefill"` is included:
  - Iterates over all lengths in `inference_microbenchmark_prefill_lengths`.
  - Runs full-prompt forward passes, measuring time to populate KV cache.
- If `"generate"` is included:
  - Executes autoregressive decode steps (token-by-token loop).
  - Measures per-token generation latency and token generation throughput across `inference_microbenchmark_loop_iters`.

---

## 3. Options & Configuration

Default in `base.yml`:

```yaml
inference_microbenchmark_stages: "prefill,generate"
```

| Value | Tested Stages | Primary Metrics Captured | When to Use |
|---|---|---|---|
| `"prefill,generate"` (default) | Both stages sequentially | TTFT, prefill TFLOPs, Inter-Token Latency (ITL), aggregate TPS | Comprehensive baseline profiling across both regimes. |
| `"prefill"` | Prefill stage only | TTFT, prompt ingestion TFLOP/s, MXU utilization | Optimizing prompt ingestion, FlashAttention kernels, or sizing prefill TPU slices in disaggregated setups. |
| `"generate"` | Decode stage only | Per-token step latency (ms/tok), generation TPS, HBM memory bandwidth utilization | Optimizing autoregressive decode kernels, KV cache memory layouts, or sizing decode TPU slices. |

---

## 4. Parameter Interactions

```text
┌───────────────────────────────────────────────────────────┐
│              inference_microbenchmark_stages              │
└─────────────┬───────────────────────────────┬─────────────┘
              │                               │
    Contains "prefill"               Contains "generate"
              │                               │
              ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│inference_microbenchmark_  │   │inference_microbenchmark_  │
│prefill_lengths            │   │loop_iters                 │
│Active: sweeps over prompt │   │Controls number of decode  │
│token lengths.             │   │tokens/steps evaluated.    │
└───────────────────────────┘   └───────────────────────────┘
```

- **`inference_microbenchmark_prefill_lengths`**: Only evaluated when `"prefill"` is present in stages.
- **`inference_microbenchmark_loop_iters`**: Sets the number of repetitions for prefill averaging, and controls the number of sequential decode steps executed when `"generate"` is active.
- **`prefill_slice` / `generate_slice`**: When testing disaggregated configurations, you isolate the benchmark to `"prefill"` on the prefill slice and `"generate"` on the generate slice.

---

## 5. Practical Scenarios & Failure Modes

### Targeting Specific Bottlenecks
- **Kernel Optimization**: When optimizing fused attention kernels (e.g. Tokamax or Splash Attention), setting `inference_microbenchmark_stages: "prefill"` allows rapid iterative compilation and timing without waiting for lengthy decode loops.
- **Serving SLA Analysis**: When diagnosing whether an SLA breach is caused by slow TTFT or low token generation speed, running separate benchmarks for `"prefill"` and `"generate"` pinpoints the failing subsystem.

### Common Pitfalls
- **Empty stage string**: Setting `inference_microbenchmark_stages: ""` causes the microbenchmark runner to skip all test loops and exit immediately with zero recorded metrics.
- **Invalid stage names**: Supplying unparsed strings like `"decode"` or `"eval"` instead of `"generate"` will cause MaxText's stage dispatcher to silently ignore the stage or throw an unrecognized stage error.

---

### One-line intuition

> **`inference_microbenchmark_stages` selects whether to benchmark prompt ingestion (`"prefill"`), token generation (`"generate"`), or both, isolating compute-bound TTFT from memory-bandwidth-bound decode latency.**
