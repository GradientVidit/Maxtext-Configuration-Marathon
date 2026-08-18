## 1. Why does it exist?

When `use_ragged_attention` is enabled, the kernel chunks the ragged sequence into discrete blocks to parallelize matrix multiplication tiles across hardware vector execution units (VMUs / Tensor Cores).

If the block size is too large, short documents suffer from internal block fragmentation. If the block size is too small, kernel invocation overhead and vector register under-utilization reduce throughput.

```text
Sequence Chunks:
  [ Block 0 (256 tok) ][ Block 1 (256 tok) ][ Block 2 (256 tok) ]
```

`ragged_block_size` sets the token block tiling size for the ragged attention kernel.

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `256` (default) | Standard 256-token tile size; balances vector unit utilization with document boundary granularity. |
| Positive integer (e.g. `128`, `512`) | Custom block tile size. |

Default in `base.yml`:
```yaml
ragged_block_size: 256
```

---

### One-line intuition

> **`ragged_block_size` sets the chunk tiling granularity (default 256) for hardware execution units in the ragged attention kernel.**
