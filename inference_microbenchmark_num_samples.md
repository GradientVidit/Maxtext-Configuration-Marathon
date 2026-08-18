## 1. Why it exists: mapping the batch-size vs. throughput frontier

In inference serving, batch size ($B$) determines hardware utilization and determines the trade-off between **latency** (time per request) and **throughput** (total tokens served across all users):

```text
Low Batch Size (e.g. B=1):
┌─────────────────────────────────┐
│ TPU Matrix Units Under-utilized │ ──> Lowest latency per token, poor cost efficiency
└─────────────────────────────────┘

Optimal Saturation Batch Size (e.g. B=16 or B=64):
┌─────────────────────────────────┐
│ High MXU & Memory Bandwidth Eff │ ──> Linear throughput scaling, acceptable latency SLA
└─────────────────────────────────┘

Excessive Batch Size (e.g. B=512):
┌─────────────────────────────────┐
│ KV Cache OOM / Latency Degrad.  │ ──> HBM exhausted or latency explodes past SLA
└─────────────────────────────────┘
```

Finding the operational sweet spot for maximum concurrency without blowing past latency SLAs or exhausting TPU High Bandwidth Memory (HBM) requires profiling across a range of batch sizes.

`inference_microbenchmark_num_samples` defines the list of concurrent sample counts (batch sizes) evaluated during inference microbenchmarking.

---

## 2. Mechanics: the sample/batch-size sweep

During microbenchmarking, MaxText evaluates each sample count $B \in \text{num\_samples}$:

```text
Config: inference_microbenchmark_num_samples: [1, 2, 4, 8, 16]
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
       Batch = 1          Batch = 4          Batch = 16
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │ Prompt: [1] │    │ Prompt: [4] │    │ Prompt:[16] │
    │ Measure     │    │ Measure     │    │ Measure     │
    │ Latency/TPS │    │ Latency/TPS │    │ Latency/TPS │
    └─────────────┘    └─────────────┘    └─────────────┘
```

For each sample count:
1. Input tensors are shaped with batch dimension $B$.
2. The KV cache is allocated with batch dimension $B$.
3. JIT compilation is invoked for that batch shape (or padded shape if dynamic batching is tested).
4. Both prefill and decode phases are timed over `inference_microbenchmark_loop_iters`.
5. Aggregate system throughput is computed as:
   $$\text{Throughput} = \frac{B \times \text{Tokens Generated}}{\text{Elapsed Time (s)}}$$

---

## 3. Options & Configuration

Default in `base.yml`:

```yaml
inference_microbenchmark_num_samples: [1, 2, 3, 4, 5]
```

| Value Format | Example | Use Case |
|---|---|---|
| Sequential small list (default) | `[1, 2, 3, 4, 5]` | Smoke testing and quick concurrency scaling checks. |
| Powers-of-2 scaling sweep | `[1, 2, 4, 8, 16, 32, 64, 128]` | Production capacity planning: identifies the exact batch size where tokens/sec saturates before HBM runs out. |
| Fixed target concurrency | `[32]` | Target benchmarking for a known serving deployment shape (e.g. JetStream server configured for max batch size 32). |

---

## 4. Parameter Interactions

```text
┌───────────────────────────────────────────────────────────┐
│           inference_microbenchmark_num_samples            │
└─────────────┬───────────────────────────────┬─────────────┘
              │                               │
              ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│KV Cache Memory Footprint  │   │inference_microbenchmark_  │
│Size = B * L * Layers *    │   │prefill_lengths            │
│       Heads * D_KV * bytes│   │Executed for every         │
│Must fit within HBM.       │   │(sample_count x length).   │
└───────────────────────────┘   └───────────────────────────┘
```

- **`inference_microbenchmark_prefill_lengths`**: The benchmark evaluates the Cartesian product of `num_samples × prefill_lengths`.
- **KV Cache Memory Capacity**: The total size of the KV cache is directly proportional to sample count $B$. If $B$ is too high, the run will fail with an out-of-memory (OOM) error during cache tensor allocation.
- **`ici_tensor_parallelism` / `ici_fsdp_parallelism`**: Sharding determines how many batch elements exist per physical TPU chip. High TP splits the heads across chips, leaving batch dimension intact, while DP/FSDP splits the batch across chips.

---

## 5. Practical Scenarios & Failure Modes

### Identifying the Throughput Plateau
When running a sweep of `[1, 2, 4, 8, 16, 32, 64, 128]`:
- **$B=1 \to 8$**: Latency stays nearly flat while aggregate throughput scales linearly (compute utilization increases).
- **$B=8 \to 32$**: Throughput reaches maximum saturation (memory bandwidth and compute fully utilized).
- **$B=64 \to 128$**: Latency increases significantly per token due to memory traffic; risk of KV cache OOM.

### What breaks if misconfigured:
- **HBM Exhaustion (OOM)**: Specifying sample sizes that exceed TPU device memory for the model's KV cache dimensions results in an immediate XLA allocation crash.
- **Excessive compilation time**: In JAX, every distinct batch size triggers a separate compilation pass unless dynamic batching or bucketed padding is configured. Testing too many discrete batch sizes increases total benchmark setup time.

---

### One-line intuition

> **`inference_microbenchmark_num_samples` defines the batch size sweep for inference profiling, allowing you to discover the optimal concurrency level where token throughput maximizes before latency degrades or memory exhausts.**
