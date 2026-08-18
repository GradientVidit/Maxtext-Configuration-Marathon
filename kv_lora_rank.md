## 1. Why does `kv_lora_rank` exist?

The primary bottleneck in serving large language models is the **Key-Value (KV) cache memory footprint**. In standard Multi-Head Attention (MHA) or Grouped-Query Attention (GQA), every token in context stores separate Key and Value vectors for each head:

$$\text{MHA KV Cache per Token} = 2 	imes N_{\text{kv\_heads}} 	imes d_{\text{head}} 	imes \text{bytes\_per\_element}$$

For DeepSeek-V2 (128 heads, $d_{head}=128$, `bfloat16`), standard MHA consumes:

$$2 	imes 128 	imes 128 	imes 2\text{ bytes} = 65{,}536\text{ bytes per token}$$

At a 128K context window, a single user session requires **8.5 GB of HBM just for KV cache**.

**Multi-Head Latent Attention (MLA)** compresses Keys and Values jointly into a single low-rank latent vector $c_{kv} \in \mathbb{R}^{d_c}$ where $d_c = \text{kv\_lora\_rank}$:

$$c_{kv} = X W_{DKV} \quad (W_{DKV} \in \mathbb{R}^{d_{model} 	imes d_c})$$

```text
Standard MHA / GQA (Caches 2 * N_heads * d_head elements):
  Token ──> [Key Head 0, Key Head 1, ..., Key Head 127] + [Val Head 0, Val Head 1, ..., Val Head 127]
  Cache: 32,768 floats / token  (= 65,536 bytes in bfloat16)

MLA (Caches single latent vector c_kv + decoupled RoPE key):
  Token ──> [ Compressed Latent c_kv (kv_lora_rank = 512) ] + [ Decoupled RoPE Key (qk_rope_head_dim = 64) ]
  Cache: 576 floats / token  (= 1,152 bytes) ── 98.2% KV Cache Reduction!
```

---

## 2. Options & Defaults

| Value | Behavior | KV Cache Footprint |
|---|---|---|
| `512` | Compresses KV representation into a 512-dimensional latent vector. | **Default**. Exact DeepSeek-V2 / DeepSeek-V3 setting. |
| Any integer $> 0$ | Custom latent bottleneck dimension $d_c$. | Lower = smaller cache, higher compression; Higher = greater representational capacity. |

Default in `base.yml`: `512`

---

## 3. How Keys and Values are reconstructed from $c_{kv}$

During computation, the compressed latent $c_{kv}$ is expanded into per-head non-positional Keys ($K_{nope}$) and Values ($V$):

$$K_{\text{nope}} = c_{kv} W_{UK} \quad (W_{UK} \in \mathbb{R}^{d_c 	imes (N_h \cdot d_{\text{nope}})})$$

$$V = c_{kv} W_{UV} \quad (W_{UV} \in \mathbb{R}^{d_c 	imes (N_h \cdot d_v)})$$

Because positional information cannot be compressed into a position-invariant latent, a separate decoupled Key with RoPE ($K_{rope} \in \mathbb{R}^{d_{rope}}$) is computed directly from $X$ and appended to $K_{nope}$:

$$K = [K_{\text{nope}}, K_{\text{rope}}]$$

$$\text{Total Cache per Token} = \text{kv\_lora\_rank } (512) + \text{qk\_rope\_head\_dim } (64) = 576\text{ elements}$$

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[attention_type]] | Must be set to `attention_type: 'mla'`. |
| [[mla_naive_kvcache]] | If `false`, caches $c_{kv} + K_{rope}$ directly (full memory savings). If `true`, caches uncompressed $K, V$ tensors. |
| [[qk_nope_head_dim]] | Width of the non-positional key slice reconstructed from $c_{kv}$. |
| [[v_head_dim]] | Width of the value slice reconstructed from $c_{kv}$. |
| [[use_sliced_mla_proj]] | Slices the $W_{UK}, W_{UV}$ up-projection kernel for faster contraction. |

---

## 5. Practical Scenarios

- **DeepSeek-V2 / V3 / V3.2 Training & Inference:** Keep `kv_lora_rank: 512`.
- **Serving 128K+ Context Windows:** MLA reduces KV cache memory to $< 2\%$ of standard MHA, allowing $50	imes$ higher serving concurrency on the same accelerator cluster.

---

### One-line intuition

> **`kv_lora_rank` is the bottleneck dimension ($d_c=512$) of MLA's compressed KV cache, reducing inference memory consumption by up to 93% compared to standard multi-head attention.**
