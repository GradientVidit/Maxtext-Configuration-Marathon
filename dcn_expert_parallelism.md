## 1. Why does it exist?

In Mixture-of-Experts (MoE) architectures, Expert Parallelism (EP) places different subsets of experts on different accelerator devices. Tokens are dynamically dispatched to their selected experts via an **All-to-All** communication collective, processed through the expert MLPs, and combined back via a reverse All-to-All.

When training huge MoE models (e.g. 64 to 256+ experts like DeepSeek-V3 or Mixtral), the number of experts may exceed the device count of a single slice, requiring experts to be distributed across multiple slices.

```text
Cross-Slice MoE Token Dispatch:
  Slice 0 (Experts 0..31)  ───[ DCN All-to-All ]─── Slice 1 (Experts 32..63)
```

`dcn_expert_parallelism` sets the degree of Expert Parallelism distributed across separate TPU slices over the Data Center Network.

---

## 2. Options & Configuration

| Value | Behavior |
|---|---|
| `1` (default) | All expert all-to-all communications stay within individual TPU slices. |
| Integer $> 1$ | Distributes experts across `N` slices; token all-to-all collectives cross DCN links. |

Default in `base.yml`:
```yaml
dcn_expert_parallelism: 1
```

---

## 3. Performance & Optimization Interactions

- **`num_moe_token_chunks`**: When running cross-slice EP, chunking tokens allows overlapping DCN All-to-All communication with Grouped Matrix Multiply (GMM) compute.
- **`use_batch_split_schedule`**: Further mitigates DCN all-to-all latency by splitting batches into micro-batches for communication pipelining.
- **`ici_expert_parallelism`**: Controls expert partitioning within each slice.

---

### One-line intuition

> **`dcn_expert_parallelism` distributes MoE experts across multiple TPU slices, enabling models with hundreds of experts to span multi-slice clusters.**
