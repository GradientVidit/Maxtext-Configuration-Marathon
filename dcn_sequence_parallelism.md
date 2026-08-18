## 1. Why does it exist?

Sequence Parallelism shards activations along the token sequence length dimension within transformer sub-blocks to prevent activation memory from growing linearly with context length.

Because sequence parallelism requires tight collective communication (All-Gather / Reduce-Scatter) around every single attention and MLP layer, running sequence parallelism across separate TPU slices over the Data Center Network (DCN) incurs severe latency overhead.

MaxText provides `dcn_sequence_parallelism` as a structural mesh dimension, but explicitly marks it in `base.yml` as **never recommended** for values $> 1$.

```text
Sequence Parallelism:
  Requires sub-millisecond layer-by-layer collective synchronization.
  Over DCN ──→ Severe network stalls and low accelerator utilization.
```

---

## 2. Options & Configuration

| Value | Status | Meaning |
|---|---|---|
| `1` (default) | **Recommended** | Confines sequence parallelism strictly within slices over ICI. |
| $> 1$ | **Never Recommended** | Forces sequence-parallel collectives across slices over DCN. |

Default in `base.yml`:
```yaml
dcn_sequence_parallelism: 1  # never recommended
```

---

### One-line intuition

> **`dcn_sequence_parallelism` sets sequence parallel degree over the datacenter network — defaults to `1` and is never recommended due to high cross-slice collective latency.**
