## 1. Why does `elastic_min_slice_count` exist?

As TPU slices fail in an elastic training job, the coordinator re-shards the model across the remaining slices:

```text
Original Cluster: 16 Slices (Global Batch Size = 2048) ──> High Throughput, Target Batch Dynamics
       │ (Repeated failures over time)
       ▼
Degraded State:    2 Slices (Global Batch Size = 256)   ──> 8x Slower Throughput, Altered Training Dynamics!
```

Allowing a job to shrink indefinitely creates two serious risks:
1. **Altered Training Dynamics**: A radically smaller global batch size (or higher gradient accumulation) shifts the effective learning rate and gradient noise scale, potentially compromising model convergence.
2. **Economic Inefficiency**: Running an expensive multi-billion-parameter model on a fraction of the intended cluster results in severely degraded step times and poor resource utilization.

`elastic_min_slice_count` defines a strict minimum threshold of healthy slices required to continue training. If the active slice count drops below this floor, the coordinator immediately stops the run rather than continuing under unacceptable degradation.

Setting `elastic_min_slice_count: -1` disables this floor (no minimum enforced).

---

## 2. Mechanics & Termination Condition

At each cluster health evaluation:
- Let $S_{\text{active}}$ be the number of currently healthy, responsive TPU slices.
- If $\text{elastic\_min\_slice\_count} > 0$ and $S_{\text{active}} < \text{elastic\_min\_slice\_count}$:
  1. The elastic coordinator stops the training loop.
  2. A checkpoint is flushed to persistent storage.
  3. The job exits with an informative message alerting operators to cluster degradation.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `elastic_min_slice_count` | `int` | `-1` | Positive integer (e.g. `4`, `8`), or `-1` (no minimum enforced) |

---

## 4. Interactions with Related Parameters

| Related Parameter | Interaction |
| :--- | :--- |
| `elastic_enabled` | `elastic_min_slice_count` is active only when `elastic_enabled: true`. |
| `num_slices` | Target initial slice count configured for the run. |
| `per_device_batch_size` | Effective global batch size equals $S_{\text{active}} \times \text{devices\_per\_slice} \times \text{per\_device\_batch\_size}$. |

---

## 5. Practical Scenarios & Best Practices

| Setting | Behavior | Recommendation |
| :--- | :--- | :--- |
| `elastic_min_slice_count: -1` (Default) | No floor enforced; job continues as long as $\ge 1$ slice survives. | Acceptable for exploratory runs and fault-tolerance testing. |
| `elastic_min_slice_count: 6` (on 8 slices) | Tolerates losing up to 2 slices (75% capacity retained); halts if 3+ slices drop. | **Strongly recommended for expensive pretraining runs** to protect training dynamics and MFU. |

---

### One-line intuition

> `elastic_min_slice_count` sets the minimum number of active TPU slices required for elastic training to continue, preventing runs from silently degrading to inefficiently small cluster sizes.
