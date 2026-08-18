## 1. Why it exists: isolating prefill latency across prompt regimes

In LLM inference, the **prefill stage** (processing the input prompt) has fundamentally different computational characteristics than the **decode stage** (generating tokens one-by-one):

```text
Prefill Phase (Compute-bound):
[Prompt Tokens: T_0, T_1, ..., T_L] ──> Large GEMMs (Matrix-Matrix) ──> Fills KV Cache
FLOPs scale as O(L^2) in attention; compute density is high.

Decode Phase (Memory-bandwidth bound):
[Single Token T_next] + [KV Cache: L tokens] ──> GEMV (Matrix-Vector) ──> Next Token
FLOPs scale as O(L); memory bandwidth to load KV cache is the primary bottleneck.
```

When optimizing inference kernels, evaluating overall request latency hides whether bottlenecks stem from prompt ingestion or autoregressive step iteration. Furthermore, prefill performance does not scale linearly: short prompts (e.g. 64 tokens) are under-utilized on massive TPU matrix units (MXUs), while long prompts (e.g. 8192 tokens) saturate MXUs and can encounter activation memory or attention tiling bottlenecks.

`inference_microbenchmark_prefill_lengths` exists to run an automated sweep over discrete prompt lengths in isolation, profiling Time-To-First-Token (TTFT), MXU utilization, and memory throughput across realistic context window sizes.

---

## 2. Mechanics: the benchmarking sweep loop

During microbenchmarking execution (`maxtext/inference_microbenchmark.py`), MaxText parses `inference_microbenchmark_prefill_lengths` into a list of integers and iterates over them:

```text
Config: inference_microbenchmark_prefill_lengths: "64,128,256,512,1024"
                               │
                               ▼
            Parse to: [64, 128, 256, 512, 1024]
                               │
       ┌───────────────────────┴───────────────────────┐
       ▼                                               ▼
Length = 64                                    Length = 1024
- Allocate dummy prompt [B, 64]                - Allocate dummy prompt [B, 1024]
- Warmup JIT compile                           - Warmup JIT compile
- Measure N iters (TTFT, TFLOP/s)              - Measure N iters (TTFT, TFLOP/s)
- Record latency & KV cache footprint          - Record latency & KV cache footprint
```

For each length $L$:
1. Synthetic input token IDs of shape `[batch_size, L]` are constructed.
2. The prefill computation is triggered (evaluating the model forward pass up to KV cache generation).
3. Timers record execution durations across `inference_microbenchmark_loop_iters`.
4. Latency (ms), throughput (tokens/sec), and MFU/TFLOPs are calculated and logged.

---

## 3. Options & Configuration

Default in `base.yml`:

```yaml
inference_microbenchmark_prefill_lengths: "64,128,256,512,1024"
```

| Value Format | Example | Use Case |
|---|---|---|
| Comma-separated string (powers of 2) | `"64,128,256,512,1024,2048,4096"` | Standard throughput curve profiling across typical prompt lengths. |
| Single length | `"2048"` | Targeted profiling for a specific workload SLA (e.g., fixed document summarization). |
| Extreme context lengths | `"4096,8192,16384,32768"` | Long-context stress testing; identifying out-of-memory (OOM) boundaries and attention kernel scaling. |

---

## 4. Parameter Interactions

```text
┌───────────────────────────────────────────────────────────┐
│        inference_microbenchmark_prefill_lengths           │
└─────────────┬───────────────────────────────┬─────────────┘
              │                               │
              ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│inference_microbenchmark_  │   │inference_microbenchmark_  │
│stages                     │   │num_samples                │
│Must include "prefill" to  │   │Cartesian product: sweeps  │
│execute these lengths.     │   │(Length x Batch Size).     │
└───────────────────────────┘   └───────────────────────────┘
              │
              ▼
┌───────────────────────────┐
│max_target_length          │
│All prefill lengths MUST   │
│be <= max_target_length.   │
└───────────────────────────┘
```

- **`inference_microbenchmark_stages`**: If set to `"generate"` only, `inference_microbenchmark_prefill_lengths` is ignored.
- **`inference_microbenchmark_num_samples`**: The benchmark evaluates the Cartesian product of `prefill_lengths × num_samples`. If you provide 5 lengths and 5 sample counts, 25 distinct benchmark configurations will execute.
- **`max_target_length` / `max_prefill_predict_length`**: Any prefill length exceeding the compiled model sequence bounds will trigger array shape assertion errors.
- **`stack_prefill_result_cache` & `prefill_cache_axis_order`**: Dictates the memory layout of the resulting KV cache generated during the benchmarked prefill step.

---

## 5. Practical Scenarios & Failure Modes

### Tuning for Production Sizing
- **Finding the MXU saturation knee**: Small prefill lengths (<128 tokens) on large TPU slices (e.g., v5p / v6e) show low TFLOPs because tensor cores are starved. Running `"32,64,128,256,512,1024,2048"` reveals the exact prompt length where tensor utilization plateaus.
- **Validating Disaggregated Serving**: In disaggregated setups, the prefill TPU cluster must complete prompt processing within strict SLAs (e.g., TTFT < 50ms). This parameter allows direct benchmarking against target customer prompt distributions.

### What breaks if misconfigured:
- **Exceeding compiled buffer bounds**: If a length in the list exceeds `max_target_length`, JAX throws a shape mismatch during positional embedding lookup or attention mask construction.
- **HBM Out-of-Memory (OOM)**: Testing very large prefill lengths (e.g. 64k) combined with large sample counts will OOM the TPU if activation memory exceeds physical HBM.

---

### One-line intuition

> **`inference_microbenchmark_prefill_lengths` defines the sequence length sweep for prompt-ingestion benchmarking, exposing how Time-To-First-Token and MXU utilization scale from short queries to long documents.**
