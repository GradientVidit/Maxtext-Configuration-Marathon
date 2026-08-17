
## 1. Why does it exist?

The M in N:M sparsity — see `weight_sparsity_n` for the full context. M defines the block size over which sparsity is computed.

```text
For 2:4 sparsity:
  weight_sparsity_n: 2  → at most 2 non-zeros...
  weight_sparsity_m: 4  → ...in every consecutive block of 4 values
```

M must be set together with N. Setting M without N (or vice versa) is invalid.

Default in base.yml:
```yaml
weight_sparsity_m: null  # null = sparsity disabled
```

See [[weight_sparsity_n]] for the complete deep-dive on N:M structured sparsity, hardware support, and training mechanics.

---

### One-line intuition

> **`weight_sparsity_m` is the block size M in N:M sparsity — it determines how many consecutive weights form a block within which at most N are kept non-zero; always set in tandem with `weight_sparsity_n`.**
