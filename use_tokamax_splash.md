## 1. Why does `use_tokamax_splash` exist?

Splash Attention is Google's high-performance, tiled attention algorithm designed specifically to optimize FlashAttention patterns on Google TPUs.

In the MaxText ecosystem, Splash Attention can be executed through multiple backend implementations:
1. **Pallas-based native kernels** (direct TPU assembly emission via JAX Pallas).
2. **Tokamax library kernels** (Google DeepMind's optimized high-performance TPU kernel suite).
3. **Pure JAX fallback implementations** (for functional validation without custom hardware kernels).

```text
Attention Computation Path:
          │
          ├─── use_tokamax_splash: true  ──> Tokamax Library Splash Kernel (Highly Optimized TPU Schedules)
          │
          ├─── use_jax_splash: true      ──> Pure JAX Reference Splash Kernel (No Pallas / Portable)
          │
          └─── default                   ──> Standard MaxText Pallas Splash Kernel
```

`use_tokamax_splash` switches the Splash Attention engine to use the specialized kernel implementation from the Tokamax library, unlocking tuned memory access patterns, custom register allocation, and fused ring-scan scheduling on supported TPU hardware.

---

## 2. Mechanics & Execution

When `use_tokamax_splash: true`:
- MaxText routes dot-product attention queries, keys, and values to `tokamax.splash_attention` instead of the internal Pallas flash kernel.
- Tokamax provides fine-grained control over tile sizes (`sa_block_q`, `sa_block_kv`), register spilling mitigations, and collective communication overlap (e.g. Ring Attention for Context Parallelism).
- It enables advanced features such as `sa_fuse_reciprocal` and custom exponent base scheduling.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `use_tokamax_splash` | `bool` | `false` | `true` (use Tokamax Splash Attention), `false` (use standard MaxText Pallas kernel) |

---

## 4. Interactions with Related Parameters

| Related Parameter | Interaction |
| :--- | :--- |
| `use_jax_splash` | Mutually exclusive backend choice. If both are enabled, configuration validation raises an error. |
| `attention` | Must be set to a Splash-compatible mode (e.g., `autoselected` or `flash`). |
| `sa_block_q` / `sa_block_kv` | Tile block sizes passed directly to the Tokamax kernel compiler. |
| `ring_scan_unroll` | Tokamax ring-attention unroll factor for context-parallel distributed scans. |

---

## 5. Practical Guidance & Failure Modes

| Scenario | Recommendation | Notes |
| :--- | :--- | :--- |
| **High-Throughput TPU v5e/v5p/v6e Training** | Test `use_tokamax_splash: true` | Tokamax provides bleeding-edge TPU micro-optimizations that can improve MFU over generic kernels. |
| **Unsupported Platform / GPU** | Keep `false` | Tokamax splash kernels are TPU-specific. |
| **Kernel Compilation Failure** | Fallback to `false` or `use_jax_splash: true` | If Tokamax encounters unsupported head dimension or sequence tiling combinations. |

---

### One-line intuition

> `use_tokamax_splash` routes attention operations to DeepMind's Tokamax library, leveraging TPU-tuned Splash Attention kernels for maximum hardware throughput.
