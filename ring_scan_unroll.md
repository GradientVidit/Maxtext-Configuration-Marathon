## 1. Why does it exist?

When computing Ring Attention in Tokamax / Splash Attention, devices iterate through circular scans of sequence chunks using a `jax.lax.scan` loop.

Loop unrolling in hardware compilers duplicates loop bodies to expose instruction-level parallelism and allow the compiler to overlap independent Direct Memory Access (DMA) transfers with Matrix Multiply Unit (MXU) compute.

`ring_scan_unroll` sets the unroll factor for Tokamax ring-attention scans.

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `1` (default) | Standard single-step scan (no unrolling). |
| `0` or $\ge \text{ring size}$ | **Fully unrolls** the ring attention scan loop across all devices. |
| Integer $> 1$ (e.g. `2`, `4`) | Unrolls loop iterations by a factor of $N$. |

Default in `base.yml`:
```yaml
ring_scan_unroll: 1 # unroll factor for Tokamax ring attention scans; 0 or >= ring size fully unrolls.
```

---

### One-line intuition

> **`ring_scan_unroll` controls the loop unrolling factor for Tokamax ring-attention scans (setting `0` fully unrolls), exposing instruction-level parallelism for communication overlap.**
