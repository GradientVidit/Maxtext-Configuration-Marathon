
## 1. Why does `decoder_block` exist?

A transformer's "decoder layer" is not a single fixed algorithm — it's a family of design decisions:

```text
What norm type?       (RMSNorm vs LayerNorm)
Where does norm go?   (pre-norm vs post-norm vs sandwich)
What attention?       (standard MHA, GQA, MLA, sliding-window...)
What MLP activation?  (SwiGLU, GELU, ReLU...)
Any extra sublayers?  (post-attn norm, QK norm, per-layer embeddings...)
```

Different model families (LLaMA, Gemma, DeepSeek, Mixtral...) make different choices on all of these. Rather than expose 15 fine-grained toggles for every possible combination, MaxText uses `decoder_block` as a **family selector** — a single string that routes to the correct Python class implementing that family's layer stack.

---

## 2. What it actually does

```text
decoder_block = "llama2"
        │
        ▼
  MaxText dispatcher
        │
        ├── LlamaDecoderLayer  (pre-RMSNorm, SwiGLU MLP, GQA support)
        ├── Gemma2DecoderLayer (post-norm after attn/FFN, logit soft-cap)
        ├── DeepSeekDecoderLayer (MoE routing, MLA, latent KV)
        └── ...
```

The string selects the Python class that defines:
- Layer ordering (norm before or after sublayer)
- Which sublayers exist (e.g., Gemma4 adds per-layer embeddings)
- How attention and MLP communicate with each other
- Any architecture-specific quirks (e.g., soft-capping logits)

---

## 3. How it's set in practice

In practice, you almost never set `decoder_block` directly. You set `model_name`:

```yaml
model_name: "llama2-7b"
```

...and the preset wires up `decoder_block: "llama2"` (plus all the `base_*` dims) automatically.

You'd set `decoder_block` directly only when:
- Prototyping a custom architecture that reuses an existing layer implementation
- Running ablations where you want a specific layer style with non-standard dimensions
- Debugging — isolating the layer implementation from the dim presets

---

## 4. Options

| Value | Architecture family | Notes |
|---|---|---|
| `"llama2"` | LLaMA / LLaMA-2 style | Default. Pre-RMSNorm, SwiGLU, GQA-compatible |
| `"gemma2"` | Gemma 2 style | Post-norm after attention and FFN, soft-cap logits |
| `"gemma4"` | Gemma 4 (small variants) | Adds per-layer embeddings (`hidden_size_per_layer_input`) |
| `"deepseek"` | DeepSeek V2/V3 style | MLA attention, MoE layers, latent KV compression |
| `"mixtral"` | Mixtral style | Sparse MoE with sliding window attention |
| (others) | Various | Check MaxText model registry for full list |

Default:
```yaml
decoder_block: "llama2"
```

---

## 5. Interaction with `model_name`

```text
model_name="llama2-7b"
    │
    ▼
MaxText model registry
    │
    ├── decoder_block = "llama2"
    ├── base_emb_dim = 4096
    ├── base_num_query_heads = 32
    └── ... (all dims set)
```

`model_name` is the high-level preset; `decoder_block` is one of the things it sets. If `override_model_config: true`, you can override `decoder_block` after `model_name` has been applied.

---

## 6. What breaks if wrong

Setting `decoder_block` to a value that doesn't match the expected layer architecture for your `base_*` dims can cause:
- Shape mismatches (the layer expects specific dim relationships)
- Missing required sub-parameters (e.g., setting `decoder_block: "gemma4"` without setting `hidden_size_per_layer_input`)
- Silently wrong math (if the layer's forward pass makes assumptions about norm placement)

Don't mix `decoder_block` from one family with dims from another without understanding both.

---

### One-line intuition

> **`decoder_block` selects the transformer layer *family* (LLaMA, Gemma, DeepSeek, etc.) — it's the single string that picks which Python class implements the forward pass of every decoder layer in the model.**
