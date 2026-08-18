## 1. Why does `attention_type` exist?

Standard causal self-attention allows every token to attend to every prior token in an autoregressive sequence. While universal, different architectural paradigms and sequence scaling goals require fundamentally different **visibility topologies** (attention masks and parameter compression schemes):

```text
Sequence of 8 Tokens (t1 ... t8):

1. 'global' (Full Causal)          2. 'local_sliding' (Window W=3)       3. 'block_diffusion' (Block B=4)
   t1 [X . . . . . . .]               t1 [X . . . . . . .]                  t1 [X X X X . . . .]
   t2 [X X . . . . . .]               t2 [X X . . . . . .]                  t2 [X X X X . . . .]
   t3 [X X X . . . . .]               t3 [X X X . . . . .]                  t3 [X X X X . . . .]
   t4 [X X X X . . . .]               t4 [. X X X . . . .]                  t4 [X X X X . . . .]
   t5 [X X X X X . . .]               t5 [. . X X X . . .]                  t5 [X X X X X X X X]
   t6 [X X X X X X . .]               t6 [. . . X X X . .]                  t6 [X X X X X X X X]
   t7 [X X X X X X X .]               t7 [. . . . X X X .]                  t7 [X X X X X X X X]
   t8 [X X X X X X X X]               t8 [. . . . . X X X]                  t8 [X X X X X X X X]
```

`attention_type` specifies this structural connectivity pattern independently of the compute backend kernel (`attention`).

---

## 2. Options & Variants

| Value | Attention Topology / Architecture | Masking / Compression Logic | Primary Use Case |
|---|---|---|---|
| `'global'` | Standard Causal Attention | Token $i$ attends to all $j \le i$. | Default standard autoregressive LLMs (LLaMA, Mistral, GPT). |
| `'local_sliding'` | Sliding Window Attention | Token $i$ attends only to $j \in [\max(0, i - W + 1), i]$. | Long-context efficiency ($O(N \cdot W)$); pairs with `sliding_window_size`. |
| `'chunk'` | Chunked Block Attention | Tokens attend within discrete chunk blocks + causally across preceding chunks. | Chunked document pretraining and structured caching. |
| `'mla'` | Multi-Head Latent Attention | Compresses $K, V$ into low-rank latent $c_{kv}$; splits $Q, K$ into NoPE and RoPE. | DeepSeek-V2 / V3 / V3.2 architectures for 64x KV cache reduction. |
| `'full'` | Bidirectional Attention | Unmasked $N 	imes N$ attention (every token sees all tokens). | Encoders (BERT, T5 encoder), prefix conditioning, masked LMs. |
| `'compressed'` | Compressed Sparse / Heavy Attention | Multi-group output LoRA + per-layer compression ratios. | Heavy-hitter / compressed attention research architectures. |
| `'block_diffusion'` | Block-Causal Diffusion Attention | Bidirectional within token blocks of size $B$, causal across blocks. | Discrete diffusion models (Block Diffusion, arXiv:2503.09573). |

Default in `base.yml`: `'global'`

---

## 3. How `attention_type` routes in the model

```text
                  attention_type
                        │
       ┌────────────────┼────────────────┬────────────────┐
       ▼                ▼                ▼                ▼
   'global'      'local_sliding'       'mla'      'block_diffusion'
  Triangular      Band-diagonal       Low-rank       Block-lower
  Causal Mask      Window Mask       KV Latents      Triangular
```

The resolved attention type dictates:
1. **Projection geometry:** Standard MHA/GQA projections vs. MLA low-rank projections (`kv_lora_rank`, `q_lora_rank`).
2. **Attention mask generation:** Generating causal, band-limited, or block-diagonal masks.
3. **KV cache allocation:** Full per-head tensors vs. compressed latent vectors.

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[sliding_window_size]] | Active only when `attention_type='local_sliding'`. |
| [[chunk_attn_window_size]] | Active only when `attention_type='chunk'`. |
| [[causal_block_size]] | Active only when `attention_type='block_diffusion'`. |
| [[kv_lora_rank]], [[q_lora_rank]] | Configure the latent bottleneck when `attention_type='mla'`. |
| [[share_kv_projections]] | **Incompatible** with `attention_type='mla'` (MLA has distinct NoPE/RoPE projections). |
| [[num_kv_shared_layers]] | Trailing layer KV sharing only occurs between layers of the *same* `attention_type`. |

---

## 5. Practical Scenarios

- **Pretraining standard dense/MoE autoregressive models:** Keep `attention_type: 'global'`.
- **Training DeepSeek-V2 / V3 style architectures:** Set `attention_type: 'mla'` along with `kv_lora_rank: 512`, `qk_nope_head_dim: 128`, `qk_rope_head_dim: 64`.
- **Sliding-window hybrid models (e.g. Gemma 2):** Alternating layers use `local_sliding` and `global` attention to bound KV memory during long generation.
- **Discrete diffusion language modeling:** Set `attention_type: 'block_diffusion'` with `causal_block_size: 32` to allow parallel bidirectional denoising within blocks while maintaining causal sequence generation across blocks.

---

### One-line intuition

> **`attention_type` defines the mathematical connectivity mask and projection topology of attention (global causal, sliding window, block diffusion, or low-rank MLA), controlling who can attend to whom.**
