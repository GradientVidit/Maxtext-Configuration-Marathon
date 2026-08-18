## 1. Why does `fused_qkv` exist?

In standard Multi-Head Attention, computing Query ($Q$), Key ($K$), and Value ($V$) representations requires three independent matrix multiplications against the input activation tensor $X \in \mathbb{R}^{B 	imes S 	imes d_{model}}$:

$$\text{Unfused:}\quad Q = X W_q, \quad K = X W_k, \quad V = X W_v$$

Because each matrix multiplication runs as a distinct kernel, the large input tensor $X$ must be **loaded from High Bandwidth Memory (HBM) into accelerator on-chip SRAM/registers three separate times**.

**Fused QKV Projection** concatenates $W_q, W_k, W_v$ along the output feature dimension into a single joint projection matrix:

$$W_{qkv} = [W_q, W_k, W_v] \in \mathbb{R}^{d_{model} 	imes (d_q + d_k + d_v)}$$

$$QKV = X W_{qkv} \quad (\text{Single GEMM kernel})$$

$$Q, K, V = \text{split}(QKV, \text{indices})$$

```text
Unfused Projections (fused_qkv = False):
  X in HBM ──┬──> GEMM(W_q) ──> Q in HBM   (Read X #1)
             ├──> GEMM(W_k) ──> K in HBM   (Read X #2)
             └──> GEMM(W_v) ──> V in HBM   (Read X #3)

Fused Projection (fused_qkv = True):
  X in HBM ──> Single Fused GEMM(W_qkv) ──> [Q, K, V] in HBM   (Read X ONCE!)
```

---

## 2. Options & Defaults

| Value | Projection Layout | HBM Memory Reads of $X$ | Notes |
|---|---|---|---|
| `false` | Three separate GEMM calls ($W_q, W_k, W_v$). | $3	imes$ reads per layer. | **Default**. Standard modular layout. |
| `true` | Single fused GEMM call ($W_{qkv}$). | $1	imes$ read per layer. | Higher compute-to-memory operational intensity. |

Default in `base.yml`: `false`

---

## 3. Hardware Bandwidth Savings

For a 70B model with $d_{model} = 8192$, batch size $B=16$, and sequence length $S=8192$ in `bfloat16`:
- **Size of Activation Tensor $X$:** $16 	imes 8192 	imes 8192 	imes 2\text{ bytes} pprox 2.15\text{ GB}$.
- **Unfused Reads per Layer:** $3 	imes 2.15\text{ GB} = 6.45\text{ GB}$.
- **Fused Reads per Layer:** $1 	imes 2.15\text{ GB} = 2.15\text{ GB}$.
- **HBM Bandwidth Saved across 80 Layers:** **$pprox 344\text{ GB}$ of memory traffic per training step.**

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[fused_mlp]] | Companion fusion parameter for Feed-Forward Network projections. |
| [[share_kv_projections]] | If `share_kv_projections: true`, the fused layout packs $[W_q, W_{kv}]$ ($2	imes$ projection width) instead of $[W_q, W_k, W_v]$. |
| [[attention_bias]] | When biases are enabled, $b_q, b_k, b_v$ are concatenated into a single $[b_{qkv}]$ 1D tensor. |

---

## 5. Practical Scenarios

- **Memory-Bandwidth-Bound Workloads:** Enable `fused_qkv: true` on memory-constrained GPUs/TPUs to increase Model FLOPs Utilization (MFU).
- **Exporting / Fine-Tuning HF Checkpoints:** When loading checkpoints that store separate `q_proj`, `k_proj`, `v_proj` weights without fusion, keep `fused_qkv: false` to avoid tensor reshaping steps.

---

### One-line intuition

> **`fused_qkv=true` merges Q, K, and V projections into a single joint GEMM, loading input activations from HBM once instead of three times to maximize accelerator memory bandwidth efficiency.**
