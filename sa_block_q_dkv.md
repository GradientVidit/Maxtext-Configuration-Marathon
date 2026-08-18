## 1. Why does it exist?

In the backward pass of Flash Attention, computing gradients with respect to Key ($dK$) and Value ($dV$) tensors requires iterating over Query tokens and backpropagating through the attention softmax weights.

Because the memory access patterns and live variable lifetimes during gradient computation differ substantially from the forward pass, Pallas uses dedicated backward tiling parameters.

```text
Backward Pass (Gradients w.r.t. Key and Value):
  Outer Loop over KV Tiles (sa_block_kv_dkv)
    Inner Loop over Query Tiles (sa_block_q_dkv = 512)
      ──→ Accumulate dK and dV gradients
```

`sa_block_q_dkv` specifies the Query block size used when evaluating Key and Value gradients ($dK, dV$) in Splash Attention.

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `512` (default) | Standard 512-token query tile size for $dK/dV$ backward computations. |
| Positive integer (e.g. `256`, `512`) | Custom query block size. |

Default in `base.yml`:
```yaml
sa_block_q_dkv: 512
```

---

### One-line intuition

> **`sa_block_q_dkv` sets the query tile size (default 512) streamed during the backward pass to calculate gradients with respect to Keys and Values ($dK, dV$).**
