## 1. Why does it exist?

During the backward pass of Splash Attention, computing gradients for a block of Key and Value tokens ($dK, dV$) requires accumulating contributions across all attending Query tokens.

`sa_block_kv_dkv` sets the outer Key/Value block size held stationary in on-chip Vector Memory (VMEM) while Query chunks are streamed past to compute $dK$ and $dV$.

```text
Held in VMEM: KV Block of size `sa_block_kv_dkv` (512)
  Query Stream ──→ Accumulates gradients directly into local VMEM accumulators
```

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `512` (default) | Standard 512-token stationary KV block size for $dK/dV$ backward passes. |
| Positive integer (e.g. `256`, `512`, `1024`) | Custom KV block size. |

Default in `base.yml`:
```yaml
sa_block_kv_dkv: 512
```

---

### One-line intuition

> **`sa_block_kv_dkv` defines the stationary Key/Value block size (default 512) held in on-chip memory during the backward gradient accumulation for $dK$ and $dV$.**
