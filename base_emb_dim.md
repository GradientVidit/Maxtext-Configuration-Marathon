
## 1. Why does `base_emb_dim` exist?

The embedding dimension (also called hidden dimension or model width) is **the single most central hyperparameter** in a transformer. Every computation flows through it:

```text
tokens
  ↓
[vocab → emb_dim]           token embedding lookup
  ↓
[emb_dim → emb_dim]         attention (Q/K/V projections)
  ↓
[emb_dim → mlp_dim → emb_dim]  MLP block
  ↓
[emb_dim → vocab]           output unembedding (or tied weights)
```

`base_emb_dim` is the width of this main highway. Everything else in the model — head sizes, MLP widths, parameter counts — is typically expressed as a ratio to it.

---

## 2. What it controls

`base_emb_dim` sets the dimension of:
- Token embedding vectors
- Residual stream (the vector that flows through all layers)
- Attention block input/output
- Layer norm input/output
- Output logit projection input

It directly determines the size of most weight matrices in the model.

---

## 3. Parameter count impact

For a rough sense of scale, most weight matrices are `emb_dim × emb_dim` or `emb_dim × mlp_dim`:

```text
Attention Q, K, V projections: each emb_dim × head_dim × num_heads
Attention output projection:   (num_heads × head_dim) × emb_dim
MLP up/gate projection:        emb_dim × mlp_dim
MLP down projection:           mlp_dim × emb_dim
```

Doubling `base_emb_dim` roughly **4× the parameter count** (since weight matrices scale as `emb_dim²`).

---

## 4. Default

```yaml
base_emb_dim: 2048
```

2048 is the default — a ~117M parameter model when combined with other defaults. Common values in production:

| Model size | base_emb_dim |
|---|---|
| ~117M (default) | 2048 |
| ~1B | 2048–4096 |
| ~7B | 4096 |
| ~13B | 5120 |
| ~70B | 8192 |

---

## 5. Interaction with `global_parameter_scale`

When `global_parameter_scale > 1`, the effective embedding dim is:

```text
effective_emb_dim = base_emb_dim × global_parameter_scale
```

For explicit control of model shape, set `base_emb_dim` directly and leave `global_parameter_scale: 1`.

---

## 6. Interaction with `head_dim` and `base_num_query_heads`

A common constraint is:

```text
emb_dim = num_query_heads × head_dim
```

For example: `4096 = 32 heads × 128 head_dim`. Violating this forces a projection mismatch that MaxText resolves via `attention_output_dim`, but deviating from it is unusual and can affect sharding efficiency.

---

## 7. Sharding implications

`base_emb_dim` must be divisible by the tensor parallelism degree. A 4096-dim model with 8-way tensor parallelism requires each head group sees 512 dimensions — always check divisibility when setting this.

---

### One-line intuition

> **`base_emb_dim` is the width of the transformer's residual stream — the primary knob for model capacity and parameter count, since most weight matrices scale as `emb_dim²`.**
