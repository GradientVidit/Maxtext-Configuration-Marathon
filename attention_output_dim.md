
## 1. Why does `attention_output_dim` exist?

Standard attention has:

```text
output projection: [num_query_heads × head_dim] → [emb_dim]
```

When the canonical constraint `emb_dim = num_q_heads × head_dim` holds, this projection is square — no information is injected or destroyed. But some architectures break this constraint intentionally:

- MLA (Multi-head Latent Attention) uses a different inner head dimension for key/value (latent dim) vs. the query head dimension
- Custom architecture research may want `attention_output_dim ≠ emb_dim`
- Models with different Q and output dimensions

`attention_output_dim` explicitly decouples "what dimension does the attention output land in" from the implicit `num_q_heads × head_dim`.

---

## 2. Default behavior

```yaml
attention_output_dim: -1
```

`-1` means **auto-infer**: the output dimension is computed as `num_query_heads × head_dim`. For standard architectures where the canonical constraint holds, this is always correct and you never need to set this explicitly.

---

## 3. When to set it explicitly

**Case 1 — Mismatched emb_dim:**
You want `base_emb_dim=4096`, `base_num_query_heads=32`, `head_dim=256` (would give 32×256=8192, not 4096). Set `attention_output_dim: 4096` to project back correctly.

**Case 2 — MLA-style architectures:**
In MLA, the query head dimension and the output projection dimension differ from the standard relationship. `attention_output_dim` is how MaxText handles this without hardcoding the projection in every layer.

**Case 3 — Experimenting with attention bottlenecking:**
Project attention output to a smaller dimension, then expand in MLP. Uncommon but architecturally valid.

---

## 4. What it controls exactly

The output (`O`) projection in attention:

```text
Attention output: [batch, seq, num_q_heads, head_dim]
                          │
                     concatenate heads
                          │
              [batch, seq, num_q_heads × head_dim]
                          │
                   W_o: [num_q_heads × head_dim, attention_output_dim]
                          │
              [batch, seq, attention_output_dim]
                          │
         added to residual stream (must match emb_dim)
```

If `attention_output_dim ≠ emb_dim`, the residual addition will fail unless the model architecture handles the mismatch explicitly.

---

## 5. Interaction with residual stream

The output of the attention block feeds directly into the residual addition:

```text
residual = residual + attention_output
```

For this to work: `attention_output_dim == base_emb_dim`. Setting these to mismatch is only valid if the decoder block implementation explicitly handles the size difference (e.g., with an additional projection).

---

## 6. When to leave it alone

For all standard model families (LLaMA, Gemma, Mistral, GPT-style), leave `attention_output_dim: -1`. The auto-inference is correct. Only touch it if you're implementing a custom architecture that genuinely requires a non-standard output projection.

---

### One-line intuition

> **`attention_output_dim` explicitly sets the output dimension of the attention block's output projection — leave it at `-1` (auto-infer `num_q_heads × head_dim`) unless your architecture has a non-standard relationship between attention and residual stream dimensions.**
