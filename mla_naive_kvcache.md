## 1. Why does `mla_naive_kvcache` exist?

Multi-Head Latent Attention (MLA) was invented by DeepSeek to compress the inference KV cache by storing only the compressed latent $c_{kv} \in \mathbb{R}^{512}$ and decoupled key $K_{rope} \in \mathbb{R}^{64}$ ($576$ floats/token), rather than uncompressed per-head Keys and Values ($32{,}768$ floats/token).

However, during autoregressive decoding, computing attention directly against compressed latents requires **matrix absorption**:

$$Q_{\text{absorbed}} = Q_{\text{nope}} W_{UK}^T \quad (Q_{\text{absorbed}} \in \mathbb{R}^{N_h 	imes d_c})$$

$$\text{Attention Logits} = Q_{\text{absorbed}} c_{kv}^T + Q_{\text{rope}} K_{\text{rope}}^T$$

While mathematically elegant, implementing matrix absorption requires specialized inference attention kernels.

The **Naive KV Cache** approach materializes and stores the full, uncompressed per-head Keys ($K \in \mathbb{R}^{N_h 	imes 192}$) and Values ($V \in \mathbb{R}^{N_h 	imes 128}$) in the KV cache, exactly like standard MHA:

```text
mla_naive_kvcache=True (Default in MaxText):
  Cache Shape per Token: [N_heads, d_head] for K and V (32,768 floats)
  Decode Execution: Standard MHA FlashAttention / PagedAttention kernel
  Memory Savings: 0% (Standard MHA memory footprint)

mla_naive_kvcache=False (True MLA Cache):
  Cache Shape per Token: c_kv (512) + K_rope (64) = 576 floats
  Decode Execution: Matrix-absorbed latent attention kernel
  Memory Savings: ~93% to 98% KV cache reduction
```

---

## 2. Options & Defaults

| Value | KV Cache Storage Format | Decode Mechanism | Inference KV Memory |
|---|---|---|---|
| `true` | Full uncompressed $K, V$ per head. | Standard MHA attention kernel. | Standard ($O(N_h \cdot d)$). **Default in MaxText**. |
| `false` | Compressed latent $c_{kv} + K_{rope}$. | Matrix-absorbed MLA decode kernel. | Compressed ($pprox 576$ elements/token, up to $98\%$ savings). |

Default in `base.yml`: `true`

---

## 3. Why is `mla_naive_kvcache: true` the Default?

MaxText defaults to `mla_naive_kvcache: true` for **broad compatibility**:
1. It allows MLA models to run immediately on any standard attention backend (unfused JAX `dot_product`, Pallas `flash`, vLLM, or cuDNN) without requiring specialized MLA-absorbed decode kernels.
2. During **training**, KV caching is not used at all (all activations are computed in parallel across the sequence), so `mla_naive_kvcache` has **zero impact on training speed or training memory**.

Switching to `mla_naive_kvcache: false` is intended for production inference serving engines where KV cache HBM capacity is the primary throughput bottleneck.

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[kv_lora_rank]] | Sets the compressed latent dimension ($512$) stored when `mla_naive_kvcache: false`. |
| [[qk_rope_head_dim]] | Sets the decoupled RoPE key dimension ($64$) stored alongside $c_{kv}$. |
| [[attention_type]] | Active only when `attention_type: 'mla'`. |

---

## 5. Practical Scenarios

- **Pretraining DeepSeek Models:** `mla_naive_kvcache` has no effect on training (leave at default `true`).
- **High-Throughput Inference Serving:** Set `mla_naive_kvcache: false` and pair with an MLA-aware serving engine (e.g. vLLM / SGLang / MaxEngine MLA kernel) to unlock massive batch sizes and 128K context windows.

---

### One-line intuition

> **`mla_naive_kvcache=true` caches uncompressed per-head K/V tensors for universal kernel compatibility, while setting it to `false` caches only the 576-dim latent vector to unlock MLA's 93%+ inference memory reduction.**
