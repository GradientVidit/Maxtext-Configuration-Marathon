## 1. Why does it exist?

Sequence Parallelism partitions activations along the sequence length dimension across accelerator chips in regions where weights are replicated, avoiding redundant activation memory during forward operations.

`ici_sequence_parallelism` sets the degree of sequence parallelism within a single TPU slice across the high-speed Inter-Chip Interconnect (ICI).

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `1` (default) | Pure FSDP or standard Tensor Parallelism. |
| Integer $> 1$ | Splits sequence activations across `N` chips in the slice. |

Default in `base.yml`:
```yaml
ici_sequence_parallelism: 1
```

---

### One-line intuition

> **`ici_sequence_parallelism` splits activation memory along the token sequence length across chips in a single TPU slice over the fast interconnect.**
