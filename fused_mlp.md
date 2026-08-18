## 1. Why does `fused_mlp` exist?

In modern Gated Feed-Forward Networks (such as **SwiGLU** in LLaMA, Gemma, and Mistral), the intermediate projection requires two simultaneous linear projections of input activations $X \in \mathbb{R}^{B 	imes S 	imes d_{model}}$:

$$\text{Gate Projection:}\quad g = X W_{\text{gate}} \quad (W_{\text{gate}} \in \mathbb{R}^{d_{model} 	imes d_{mlp}})$$

$$\text{Up Projection:}\quad u = X W_{\text{up}} \quad (W_{\text{up}} \in \mathbb{R}^{d_{model} 	imes d_{mlp}})$$

$$\text{SwiGLU Activation:}\quad h = \text{SiLU}(g) \odot u$$

In an unfused implementation, $X$ is read from HBM twice to execute two independent matrix multiplications.

**Fused MLP** packs $W_{gate}$ and $W_{up}$ into a single contiguous weight tensor:

$$W_{\text{gate\_up}} = [W_{\text{gate}}, W_{\text{up}}] \in \mathbb{R}^{d_{model} 	imes (2 \cdot d_{mlp})}$$

$$[g, u] = X W_{\text{gate\_up}} \quad (\text{Single GEMM Kernel})$$

```text
Unfused MLP (fused_mlp = False):
  X in HBM ──┬──> GEMM(W_gate) ──> g [d_mlp]   (Read X #1)
             └──> GEMM(W_up)   ──> u [d_mlp]   (Read X #2)

Fused MLP (fused_mlp = True):
  X in HBM ──> Single GEMM(W_gate_up) ──> [g, u] [2 * d_mlp]   (Read X ONCE!)
```

---

## 2. Options & Defaults

| Value | Projection Layout | HBM Memory Reads of $X$ | Notes |
|---|---|---|---|
| `false` | Two separate GEMMs for Gate and Up projections. | $2	imes$ reads per MLP. | **Default**. Matches standard HuggingFace parameter naming. |
| `true` | Single combined GEMM ($W_{gate\_up}$). | $1	imes$ read per MLP. | Eliminates redundant activation memory traffic. |

Default in `base.yml`: `false`

---

## 3. Performance Benefits in Large Models

Because $d_{mlp}$ is typically $4	imes$ to $8/3	imes d_{model}$ (e.g. $d_{mlp} = 28{,}672$ in LLaMA 3 70B), the MLP intermediate GEMMs are the single largest matrix operations in the network.

Combining Gate and Up projections doubles the contraction dimension for the compiler, improving systolic array tile utilization on TPUs and Tensor Cores on GPUs.

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[fused_qkv]] | Companion fusion flag for attention projections. Typically enabled together for maximum MFU. |
| [[mlp_activations]] | Fused MLP is specifically designed for gated two-input activations like `['silu', 'linear']` (SwiGLU) or `['gelu', 'linear']` (GeGLU). |
| [[mlp_bias]] | If `mlp_bias: true`, Gate and Up biases are packed into a single 1D vector. |

---

## 5. Practical Scenarios

- **Pretraining Large Dense LLMs:** Set `fused_mlp: true` and `fused_qkv: true` to minimize HBM round-trips and achieve peak hardware MFU.
- **Serving with vLLM / TensorRT-LLM:** Modern serving runtimes store weights in fused `gate_up_proj` format natively.

---

### One-line intuition

> **`fused_mlp=true` combines SwiGLU Gate and Up projections into a single joint GEMM, halving HBM activation memory reads and boosting accelerator throughput.**
