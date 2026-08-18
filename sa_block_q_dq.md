## 1. Why does it exist?

Computing the gradient with respect to Queries ($dQ$) requires multiplying the incoming output gradient $dO$ and attention scores against Key tensors ($dQ = dS \times K$).

In Flash Attention algorithms, the $dQ$ computation holds a Query tile stationary in Vector Memory (VMEM) while Key and Value chunks are streamed past.

`sa_block_q_dq` specifies the Query block size held in VMEM while computing the Query gradient $dQ$.

```text
Stationary in VMEM: Query Block of size `sa_block_q_dq` (512)
  KV Stream ──→ Accumulates dQ gradients in local registers
```

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `512` (default) | Standard 512-token query tile size for $dQ$ backward computation. |
| Positive integer (e.g. `256`, `512`) | Custom query block size. |

Default in `base.yml`:
```yaml
sa_block_q_dq: 512
```

---

### One-line intuition

> **`sa_block_q_dq` sets the stationary Query block size (default 512) held in on-chip memory during the backward accumulation of Query gradients ($dQ$).**
