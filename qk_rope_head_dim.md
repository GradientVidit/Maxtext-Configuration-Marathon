## 1. Why does `qk_rope_head_dim` exist?

Because Multi-Head Latent Attention (MLA) compresses Keys into a position-invariant latent $c_{kv}$, standard Rotary Position Embeddings (RoPE) cannot be applied to the latent without destroying compression.

To retain precise positional awareness, MLA introduces **Decoupled Rotary Position Embeddings**:
1. A separate, dedicated key projection generates a compact position-aware key $K_{rope} \in \mathbb{R}^{d_{rope}}$:

$$K_{\text{rope}} = \text{RoPE}(X W_{KR}) \quad (W_{KR} \in \mathbb{R}^{d_{model} 	imes d_{\text{rope}}})$$

2. The query projection similarly generates a matching rotary query $Q_{rope} \in \mathbb{R}^{N_h 	imes d_{rope}}$:

$$Q_{\text{rope}} = \text{RoPE}(c_q W_{QR}) \quad (\text{or } \text{RoPE}(X W_{QR}))$$

3. During attention, the positional score is computed directly between $Q_{rope}$ and $K_{rope}$ and added to the content score:

$$\text{Score}(q, k) = rac{q_{\text{nope}} k_{\text{nope}}^T + q_{\text{rope}} k_{\text{rope}}^T}{\sqrt{d_{\text{nope}} + d_{\text{rope}}}}$$

`qk_rope_head_dim` sets the dimensionality $d_{rope}$ of these decoupled rotary vectors.

---

## 2. Options & Defaults

| Value | Behavior | Notes |
|---|---|---|
| `64` | Decoupled RoPE dimension of $d_{rope}=64$. | **Default**. Exact DeepSeek-V2 / DeepSeek-V3 specification. |
| Any integer $> 0$ | Custom RoPE head dimension. | Must match RoPE rotary embedding frequency configurations. |

Default in `base.yml`: `64`

---

## 3. Shared vs. Multi-Head RoPE Keys

In DeepSeek-V2 and V3, while Queries have independent $Q_{rope}$ vectors for each of the $N_h$ heads, the Key vector $K_{rope}$ is **shared across all attention heads** within a layer ($N_{kv\_rope\_heads} = 1$):

```text
Query Heads (128 heads):
  Head 0: [ Q_nope_0 (128) | Q_rope_0 (64) ]
  Head 1: [ Q_nope_1 (128) | Q_rope_1 (64) ]
  ...
  Head 127: [ Q_nope_127 (128) | Q_rope_127 (64) ]

Key Heads (Shared RoPE Key):
  Head 0..127: [ K_nope_i (from c_kv) | K_rope (Shared 64-dim vector) ]
```

This ensures the positional key adds only **64 floats per token** to the entire KV cache, regardless of head count.

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[qk_nope_head_dim]] | Companion semantic dimension ($d_{nope}=128$). Effective head dim is $128 + 64 = 192$. |
| [[kv_lora_rank]] | Total cached elements per token is $\text{kv\_lora\_rank } (512) + \text{qk\_rope\_head\_dim } (64) = 576$. |
| [[attention_type]] | Active when `attention_type: 'mla'`. |

---

## 5. Practical Scenarios

- **DeepSeek-V2 / V3 Architectures:** Keep `qk_rope_head_dim: 64`.
- **Long-Context RoPE Scaling:** YaRN or extended base frequency scaling in MLA applies strictly to the `qk_rope_head_dim` slice.

---

### One-line intuition

> **`qk_rope_head_dim` sets the dimension ($d_{rope}=64$) of MLA's decoupled positional embedding, allowing position awareness to be evaluated without polluting the compressed KV latent.**
