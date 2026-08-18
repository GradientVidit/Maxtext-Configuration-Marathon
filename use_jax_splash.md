## 1. Why does `use_jax_splash` exist?

Hardware-specific custom kernels (like Pallas Splash Attention or Tokamax) are compiled down to low-level accelerator assembly. While fast, they present key operational challenges:
1. **Debugging & Tracing**: Hardware-level kernel bugs, NaN generation, or illegal memory access are difficult to trace inside opaque Pallas blocks.
2. **Portability**: Pallas/Tokamax TPU kernels cannot run directly on platforms lacking Pallas backend support (e.g., pure CPU emulation or certain non-standard environments).
3. **Numerical Verification**: Verifying whether an accuracy regression is caused by algorithmic logic or custom kernel precision quirks requires a pure, golden-reference implementation.

```text
Standard Path:   Attention Projections ──> Pallas / Tokamax TPU Custom Kernel (Low-level assembly)
                                                 ▲
                                                 │ Debugging / Portability Check
                                                 ▼
JAX Splash Path: Attention Projections ──> Pure JAX Functional Primitives (jax.lax.scan / jax.numpy)
```

`use_jax_splash` instructs MaxText to execute Splash Attention using a pure, non-Pallas JAX-native implementation composed of high-level JAX primitives.

---

## 2. Mechanics & Execution

When `use_jax_splash: true`:
- MaxText completely bypasses the Pallas TPU kernel invocation.
- The tiled Splash Attention algorithm is executed via native `jax.lax.scan` and `jax.lax.dot_general` operations.
- The intermediate softmax reductions and online normalization follow Splash Attention's block-tiled mathematics, but rely entirely on the standard XLA compiler for optimization and memory layout.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `use_jax_splash` | `bool` | `false` | `true` (use pure JAX non-Pallas implementation), `false` (use Pallas/Tokamax hardware kernel) |

---

## 4. Interactions with Related Parameters

| Related Parameter | Interaction |
| :--- | :--- |
| `force_q_layout` | When `use_jax_splash: true`, `force_q_layout` can be set to `true` to force Query tensors into `[..., NUM_HEADS, HEAD_DIM, SEQ_LENGTH]` for optimized JAX loop layout. |
| `use_tokamax_splash` | Mutually exclusive; only one Splash Attention backend can be active at a time. |
| `attention` | Applicable when attention is configured for Splash Attention. |

---

## 5. Practical Scenarios & When to Use

| Scenario | Recommended Setting | Rationale |
| :--- | :--- | :--- |
| **Production Pretraining / High-Throughput Run** | `use_jax_splash: false` | Pallas/Tokamax kernels are substantially faster and avoid XLA compilation overhead. |
| **Debugging NaNs or Divergence in Attention** | `use_jax_splash: true` | Allows full inspection via `jax.debug.print` or JAX checkify inside attention loops. |
| **CPU Unit Testing & CI Environments** | `use_jax_splash: true` | Eliminates TPU Pallas dependency in CI runners without physical accelerators. |

---

### One-line intuition

> `use_jax_splash` forces MaxText to run Splash Attention through a pure, non-Pallas JAX implementation for debugging, testing, and numerical golden-rule verification.
