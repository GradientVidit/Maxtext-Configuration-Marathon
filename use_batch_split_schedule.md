
## 1. The all-to-all communication bottleneck

Expert parallelism with all-to-all dispatch has a fundamental problem: all-to-all communication can dominate the forward pass latency at large EP degrees, leaving accelerators idle while waiting for tokens to arrive from remote devices.

One solution: instead of waiting for all tokens to arrive before starting expert compute, split the batch into micro-batches. While micro-batch k is in flight (communication), run expert computation on micro-batch k-1. This is the batch-split schedule.

`use_batch_split_schedule` enables this communication-compute overlap strategy.

---

## 2. The mechanic

```text
Without batch-split:
[all-to-all for full batch] → [expert compute on full batch] → [all-to-all return]

With batch-split (batch_split_factor=2):
micro-batch 0:  [all-to-all dispatch] → [expert compute]
micro-batch 1:              [all-to-all dispatch] → [expert compute]
                ↑ overlap: mb-1 dispatch ↔ mb-0 compute
```

The communication for the next micro-batch overlaps with the computation of the current one.

---

## 3. Current limitation: DeepSeek sparse layers only

```yaml
use_batch_split_schedule: false  # (default)
```

As of current MaxText, this optimization is **only implemented for DeepSeek sparse layers**. It won't do anything useful for generic MoE models.

---

## 4. The `batch_split_factor` companion

```yaml
batch_split_factor: 1  # (default) no splitting
batch_split_factor: 2  # split batch into 2 micro-batches
batch_split_factor: 4  # split into 4
```

Only effective when `use_batch_split_schedule=True`. Controls how many micro-batches the batch is divided into. Higher = more pipeline stages = more overlap opportunity, but also more overhead per micro-batch.

---

## 5. Options

| `use_batch_split_schedule` | `batch_split_factor` | Behavior |
|---|---|---|
| `false` (default) | Any | Disabled — no batch splitting |
| `true` | `1` | Enabled but degenerate (no splitting, no overlap) |
| `true` | `2` | Split into 2 micro-batches, overlap enabled |
| `true` | `4` | Split into 4, more overlap stages |

---

## 6. Interaction with related parameters

| Related param | Interaction |
|---|---|
| `batch_split_factor` | Direct companion — controls number of micro-batches |
| `use_manual_quantization` | Also used in batch-split context for quantized dispatch |
| `use_ring_of_experts` | Different overlap strategy — `use_batch_split_schedule` is the all-to-all alternative |
| `num_moe_token_chunks` | Token-chunk overlap for ring path; batch-split overlap for all-to-all path |

---

### One-line intuition

> **`use_batch_split_schedule` splits the token batch into micro-batches to overlap expert-parallel all-to-all communication with compute — currently only implemented for DeepSeek sparse layers; leave `false` for other architectures.**
