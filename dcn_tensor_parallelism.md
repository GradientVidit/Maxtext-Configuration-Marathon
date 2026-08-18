## 1. Why does it exist?

Tensor Parallelism (TP) partitions individual matrix multiplications (such as Multi-Head Attention QKV projections or MLP up/down projections) across devices. Every single transformer layer requires multiple **blocking All-Reduce collectives** in both the forward and backward passes.

Because Data Center Network (DCN) connections between slices have much higher latency and lower bandwidth than intra-slice optical interconnects, running Tensor Parallelism across DCN results in catastrophic communication stalls where TPU matrix units sit idle waiting on network packets.

```text
Intra-Slice TP (Fast ICI):
  Core 0 ──[ High-speed ICI All-Reduce (sub-microsecond) ]── Core 1  <-- High MFU

Cross-Slice TP (Slow DCN):
  Slice 0 ──[ Slower DCN Ethernet All-Reduce (~milliseconds) ]── Slice 1 <-- TPU Stalled!
```

MaxText provides `dcn_tensor_parallelism` as a physical mesh axis size configuration, but explicitly notes in `base.yml` that setting this $> 1$ is **never recommended**.

---

## 2. Options & Configuration

| Value | Status | Meaning |
|---|---|---|
| `1` (default) | **Recommended** | Confines Tensor Parallelism strictly within slices; no TP all-reduces cross the DCN. |
| $> 1$ | **Never Recommended** | Forces Megatron TP all-reduces over the datacenter network, degrading compute utilization (MFU). |

Default in `base.yml`:
```yaml
dcn_tensor_parallelism: 1 # never recommended
```

---

## 3. Practical Guidance

- Keep `dcn_tensor_parallelism: 1` at all times.
- If model layers require tensor parallelism, set `ici_tensor_parallelism` within the slice (over fast ICI links) instead.

---

### One-line intuition

> **`dcn_tensor_parallelism` sets the tensor parallel degree over the datacenter network — defaults to `1` and is strictly never recommended due to severe cross-slice all-reduce latency.**
