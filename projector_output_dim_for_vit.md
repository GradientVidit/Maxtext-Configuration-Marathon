## 1. Why does `projector_output_dim_for_vit` exist?

For the language model decoder to attend to visual tokens seamlessly, visual token embeddings must reside in the exact same vector space dimension as the text token embeddings (`base_emb_dim`).

```text
Visual Features ──► [ Projector MLP ] ──► Projected Tokens [projector_output_dim_for_vit = 4096]
                                                        │
                                                        ▼
                                       Matches LLM base_emb_dim (4096)
                                                        │
                                                        ▼
                                       Concatenated with Text Token Embeddings
```

`projector_output_dim_for_vit` sets the output feature dimension emitted by the projector, aligning with the LLM embedding space.

---

## 2. Options and Defaults

| Value | Alignment |
|---|---|
| `4096` (Default) | Aligns with 4096-dim LLM embeddings (e.g. Llama 3 8B, Llama 4) |
| Matching `base_emb_dim` | Set equal to the host LLM's `base_emb_dim` |

---

## 3. Failure Modes

- If `projector_output_dim_for_vit` does not exactly match `base_emb_dim`, tensor concatenation in the decoder input layer will throw shape mismatch exceptions.

---

### One-line intuition
> **`projector_output_dim_for_vit` specifies the output feature dimension of the vision projector, matching the host LLM's embedding dimension (`base_emb_dim`).**
