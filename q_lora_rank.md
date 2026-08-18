## 1. Why does `q_lora_rank` exist?

In standard Multi-Head Attention, Query projections project the full hidden dimension $d_{model}$ directly into all attention heads:

$$Q = X W_q \quad (W_q \in \mathbb{R}^{d_{model} 	imes (N_{\text{heads}} \cdot d_{\text{head}})})$$

For wide models with large head counts (e.g. DeepSeek-V2 with 128 heads and $d_{head}=192$), $W_q$ requires enormous parameter storage and activation memory during training.

**Multi-Head Latent Attention (MLA)** introduces an optional low-rank Query bottleneck ($c_q$). When `q_lora_rank > 0`, Query projections are factorized into down-projection and up-projection matrices:

$$c_q = X W_{DQ} \quad (W_{DQ} \in \mathbb{R}^{d_{model} 	imes d_q'})$$

$$Q_{\text{nope}} = c_q W_{UQ}, \quad Q_{\text{rope}} = \text{RoPE}(c_q W_{QR})$$

```text
Standard Query Projection (q_lora_rank = 0):
  X [B, S, d_model] ──> Matmul(W_q) ──> Q [B, S, N_heads, d_head]

MLA Query Compression (q_lora_rank = d_q'):
  X [B, S, d_model] ──> Down-Proj(W_DQ) ──> c_q [B, S, d_q'] ──┬──> Up-Proj(W_UQ) ──> Q_nope
                                                               └──> RoPE-Proj(W_QR) ──> Q_rope
```

When `q_lora_rank=0`, Query LoRA compression is disabled, and queries are projected using standard uncompressed projections.

---

## 2. Options & Defaults

| Value | Behavior | Notes |
|---|---|---|
| `0` | Query LoRA compression disabled (full-rank query projection). | **Default**. Simplifies query projection while keeping KV compression active. |
| Any integer $> 0$ (e.g. `1536`) | Compresses Query into a latent vector $c_q \in \mathbb{R}^{d_q'}$. | `1536` is the exact DeepSeek-V2 / V3 architecture setting. |

Default in `base.yml`: `0`

---

## 3. Parameter Savings in Wide Architectures

For DeepSeek-V2 ($d_{model}=5120$, $N_{heads}=128$, $d_{head}=192$):
- **Full Rank $W_q$:** $5120 	imes (128 	imes 192) = 125{,}829{,}120$ parameters ($pprox 125.8\text{M}$ params per layer).
- **LoRA Rank $d_q'=1536$:** $(5120 	imes 1536) + (1536 	imes 24{,}576) = 45{,}613{,}056$ parameters ($pprox 45.6\text{M}$ params per layer).

This saves **$63.7\%$ of query projection weights** across all attention layers.

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[kv_lora_rank]] | Companion parameter compressing Key and Value states. `kv_lora_rank` compresses the KV cache; `q_lora_rank` compresses query parameters. |
| [[qk_nope_head_dim]], [[qk_rope_head_dim]] | Determine the split target dimensions ($d_{nope}$ and $d_{rope}$) for query up-projections. |
| [[attention_type]] | Active only when `attention_type: 'mla'`. |

---

## 5. Practical Scenarios

- **Reproducing DeepSeek-V2 / DeepSeek-V3:** Set `q_lora_rank: 1536`.
- **Hybrid MLA Training:** Leave `q_lora_rank: 0` (default) if you only want KV-cache inference memory compression (`kv_lora_rank: 512`) without compressing query representations.

---

### One-line intuition

> **`q_lora_rank` sets the low-rank bottleneck dimension $d_q'$ for MLA queries, cutting Query projection parameter count by over 60% in wide multi-head models.**
