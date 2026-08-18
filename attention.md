## 1. Why does `attention` exist?

Computing multi-head attention scores ($\text{softmax}(QK^T / \sqrt{d})V$) is both compute-intensive ($O(N^2)$ FLOPs) and memory-bandwidth-heavy. Depending on the target accelerator (TPU vs. NVIDIA GPU), compiler (XLA vs. cuDNN), and execution mode (prefill vs. decode vs. training), different kernel implementations yield vastly different memory consumption and hardware utilization.

MaxText separates the **attention kernel implementation** (`attention`) from the **attention mathematical pattern** (`attention_type`).

```text
Attention equation: Attention(Q, K, V) = softmax(QKᵀ / √d) · V

                  ┌── 'autoselected'  --> Dispatches optimal kernel per accelerator
                  │
                  ├── 'dot_product'   --> Pure unfused JAX/XLA matmuls (O(N²) HBM memory)
attention (knob) ─┼── 'flash'         --> Pallas / Splash tiled kernel (O(N) memory, TPU)
                  │
                  └── 'cudnn_flash_te'--> NVIDIA cuDNN FlashAttention via TransformerEngine (GPU)
```

---

## 2. Options & Defaults

| Value | Underlying Kernel / Engine | Target Accelerator | Memory Complexity | Notes |
|---|---|---|---|---|
| `'autoselected'` | Environment-aware dispatch | Any (TPU / GPU) | Depends on backend | **Default**. Automatically routes to the highest-performance backend kernel. |
| `'dot_product'` | Pure JAX `lax.dot_general` + `jnp.einsum` | CPU / TPU / GPU | $O(N^2)$ (materializes full score matrix) | Baseline reference. Slowest at long sequences, but 100% transparent and inspectable for debugging. |
| `'flash'` | Pallas Splash / FlashAttention kernel | TPU (v4, v5e, v5p, v6e) | $O(N)$ (tiled block computation) | High-performance tiled attention on TPU without materializing $N 	imes N$ logits in HBM. |
| `'cudnn_flash_te'` | NVIDIA cuDNN / TransformerEngine | NVIDIA GPU (Ampere, Hopper, Blackwell) | $O(N)$ (fused GPU kernel) | Leverages NVIDIA hardware tensor cores and FP8/BF16 optimized GEMMs. |

Default in `base.yml`: `'autoselected'`

---

## 3. How `attention` executes in the pipeline

```text
Input Activations (X)
         │
         ▼
[ Q, K, V Projections ]
         │
         ▼
[ attention Selection ]
   ├── 'dot_product'   ──> Materialize [B, H, Q_len, KV_len] ──> Softmax ──> Matmul V
   ├── 'flash'         ──> Pallas Block Tiling (SRAM / VMEM) ──> Online Softmax ──> Out
   └── 'cudnn_flash_te'──> Transformer Engine Fused C++ cuDNN Kernel ───────────> Out
```

`dot_product` materializes the entire intermediate score matrix in High Bandwidth Memory (HBM). At sequence length $N = 32{,}768$ with 32 heads in `bfloat16`, a single layer's attention matrix consumes $pprox 64\text{ GB}$ of memory. `flash` and `cudnn_flash_te` tile the computation in on-chip SRAM/VMEM using online softmax normalization, dropping intermediate memory to $O(N)$.

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[attention_type]] | Orthogonal dimension. `attention` sets *how* attention is computed; `attention_type` sets *which tokens attend to which* (e.g. `'mla'`, `'local_sliding'`, `'global'`). |
| [[float32_qk_product]] | Applies specifically to `'dot_product'` attention, upcasting inputs to FP32 for numerical stability. |
| [[float32_logits]] | Applies to `'dot_product'` attention, computing softmax in FP32. |
| [[use_qk_clip]] | Supported with `'dot_product'` or Tokamax Splash implementations. |
| [[sliding_window_size]] | Requires kernel-level mask support (Splash attention or custom masked dot product). |

---

## 5. Practical Scenarios & Failure Modes

- **Debugging NaN losses or gradient issues:** Switch `attention: 'dot_product'`. Pallas and cuDNN custom fused kernels can sometimes obscure numerical underflow/overflow or mask misalignments.
- **Out of Memory (OOM) on TPU during long context:** Ensure you are using `attention: 'flash'` or `'autoselected'`. Leaving it on `'dot_product'` with sequences $>4\text{K}$ will quickly exhaust TPU HBM.
- **Running on GPU clusters (A100 / H100):** Set `attention: 'cudnn_flash_te'` (and install `transformer_engine`) to achieve full Tensor Core FLOP utilization.
- **Unsupported Hardware / Simulators:** Use `attention: 'dot_product'` when testing on CPU or lightweight test runners where custom TPU Pallas kernels or NVIDIA CUDA drivers are unavailable.

---

### One-line intuition

> **`attention` dictates the physical compute kernel (unfused JAX, TPU Pallas Flash, or GPU cuDNN) used to evaluate attention scores, trading debugging transparency for hardware-tiled memory and speed.**
