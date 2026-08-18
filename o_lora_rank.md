## 1. Why does `o_lora_rank` exist?

In standard transformer attention layers, the multi-head attention output projection matrix ($W_o$) is a full-rank linear transformation that maps the concatenated heads back to the residual stream:

$$W_o \in \mathbb{R}^{(N_{\text{heads}} \cdot d_{\text{head}}) 	imes d_{model}}$$

For wide architectures with large head dimensions, $W_o$ represents a massive parameter and memory footprint (e.g. $d_{model}^2$ parameters).

**Compressed Attention** (`attention_type: 'compressed'`) factorizes the output projection into a low-rank down-projection and up-projection bottleneck using LoRA decomposition:

$$O = \text{Concat}(\text{heads}) \cdot W_{o,\text{down}} \cdot W_{o,\text{up}}$$

$$\text{Where:}\quad W_{o,\text{down}} \in \mathbb{R}^{(N_h \cdot d_h) 	imes R}, \quad W_{o,\text{up}} \in \mathbb{R}^{R 	imes d_{model}}, \quad R = \text{o\_lora\_rank}$$

```text
Standard Output Projection (o_lora_rank = 0):
  Concat(Heads) [B, S, N_heads * d_head] ──> Matmul(W_o) ──> Out [B, S, d_model]

Compressed Output Projection (o_lora_rank = R):
  Concat(Heads) ──> Down-Proj(W_o_down) ──> Latent [B, S, R] ──> Up-Proj(W_o_up) ──> Out [B, S, d_model]
```

When `o_lora_rank=0`, low-rank output compression is disabled and standard full-rank $W_o$ is instantiated.

---

## 2. Options & Defaults

| Value | Behavior | Notes |
|---|---|---|
| `0` | Disabled. Standard full-rank output projection. | **Default**. |
| Any integer $> 0$ (e.g. `256`, `512`) | Compresses output projection through rank $R$. | Parameter count drops from $(N_h d_h) d_{model}$ to $R 	imes (N_h d_h + d_{model})$. |

Default in `base.yml`: `0`

---

## 3. Parameter Savings Analysis

For a model with $d_{model} = 8192$ and $N_h d_h = 8192$:
- **Full Rank ($R=0$):** $8192 	imes 8192 = 67{,}108{,}864$ parameters per layer ($pprox 67.1\text{M}$).
- **Compressed Rank ($R=512$):** $512 	imes (8192 + 8192) = 8{,}388{,}608$ parameters per layer ($pprox 8.4\text{M}$).

This achieves an **$87.5\%$ parameter reduction** on the output projection.

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[o_groups]] | Companion parameter that partitions the output projection into grouped linear sub-matrices. |
| [[compress_ratios]] | Configures per-layer attention compression strides across the model depth. |
| [[attention_type]] | Active when `attention_type: 'compressed'`. |

---

## 5. Practical Scenarios

- **Compressed Architecture Research:** Set `o_lora_rank: 512` alongside `attention_type: 'compressed'` when designing sub-quadratic or compact memory-bound LLMs.
- **Standard Pretraining:** Leave `o_lora_rank: 0` (default).

---

### One-line intuition

> **`o_lora_rank` sets the low-rank bottleneck dimension $R$ for Compressed Attention's output projection, dramatically shrinking output matrix parameter count.**
