## 1. Why does `causal_block_size` exist?

Discrete diffusion models (such as **Block Diffusion**, arXiv:2503.09573) generate language non-autoregressively within local token blocks while maintaining causal autoregression across successive blocks.

Standard attention masks cannot represent this paradigm:
- **Pure Causal Mask:** Forces strict token-by-token left-to-right generation, preventing parallel intra-block diffusion denoising.
- **Full Bidirectional Mask:** Allows future tokens to leak into past tokens, breaking autoregressive sequence generation.

**Block-Causal Attention** solves this by establishing bidirectional attention *inside* each block of size $B$, and causal attention *across* blocks:

```text
Sequence of 8 Tokens with causal_block_size B = 4:

Block 0: [t0, t1, t2, t3]    (Tokens attend bidirectionally to each other)
Block 1: [t4, t5, t6, t7]    (Tokens attend bidirectionally to each other AND causally to Block 0)

Attention Mask Matrix:
      t0 t1 t2 t3 | t4 t5 t6 t7
  t0 [ 1  1  1  1 |  0  0  0  0 ]   <── Block 0 (fully bidirectional within block)
  t1 [ 1  1  1  1 |  0  0  0  0 ]
  t2 [ 1  1  1  1 |  0  0  0  0 ]
  t3 [ 1  1  1  1 |  0  0  0  0 ]
  ─────────────────────────────
  t4 [ 1  1  1  1 |  1  1  1  1 ]   <── Block 1 sees all of Block 0 + full Block 1
  t5 [ 1  1  1  1 |  1  1  1  1 ]
  t6 [ 1  1  1  1 |  1  1  1  1 ]
  t7 [ 1  1  1  1 |  1  1  1  1 ]
```

---

## 2. Options & Defaults

| Value | Behavior | Notes |
|---|---|---|
| `32` | Block size $B=32$. Tokens within each 32-token window attend bidirectionally; blocks attend causally. | **Default**. Matches Block Diffusion baseline. |
| Any integer $> 0$ | Custom block size $B$. Must evenly divide the training sequence length. | $B=1$ degenerates to pure causal; $B=N$ degenerates to full bidirectional. |

Default in `base.yml`: `32`

---

## 3. Boundary Cases of the Block Size Continuum

$$egin{aligned}
B = 1 &\implies \text{Pure Autoregressive Causal Attention (LLaMA / GPT)} \
1 < B < N &\implies \text{Block-Causal Diffusion Attention (Block Diffusion)} \
B = N &\implies \text{Full Bidirectional Attention (BERT / T5 Encoder)}
\end{aligned}$$

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[attention_type]] | Must be set to `attention_type: 'block_diffusion'`. (Note: MaxText's `_resolve_attention_type` automatically enforces block-causal handling when configured). |
| [[attention]] | Requires an attention kernel capable of executing block-diagonal causal masks (e.g. DotProduct or Splash Attention). |
| [[model_call_mode]] | During diffusion sampling, parallel iterative denoising runs inside active blocks. |

---

## 5. Practical Scenarios

- **Training Block Diffusion Models (arXiv:2503.09573):** Set `attention_type: 'block_diffusion'` with `causal_block_size: 32` (or `64`) to denoise $B$ tokens simultaneously per diffusion step.
- **Speculative Decoding / Semi-Autoregressive Generation:** Use block-causal structures to verify multi-token draft predictions in parallel.

---

### One-line intuition

> **`causal_block_size` sets the token block width $B$ for Block Diffusion models, enabling parallel bidirectional attention within blocks while preserving causal autoregression across blocks.**
