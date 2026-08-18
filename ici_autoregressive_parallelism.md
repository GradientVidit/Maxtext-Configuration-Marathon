## 1. Why does it exist?

During low-latency autoregressive token generation (e.g. LLM inference serving), single tokens are generated iteratively. The `autoregressive` mesh axis provides a dedicated communication channel for speculative decoding, KV cache sharing, or sequence generation across chips.

`ici_autoregressive_parallelism` sets the degree of this autoregressive generation dimension within a single TPU slice over the fast Inter-Chip Interconnect.

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `1` (default) | Standard execution (dedicated autoregressive axis disabled). |
| Integer $> 1$ | Allocates `N` chips to autoregressive decoding parallelism within the slice. |

Default in `base.yml`:
```yaml
ici_autoregressive_parallelism: 1
```

---

### One-line intuition

> **`ici_autoregressive_parallelism` allocates chips within a TPU slice to a dedicated autoregressive generation axis for optimized low-latency token generation.**
