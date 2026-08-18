
## 1. Why does `mlp_activations` exist as a list?

Classical transformers used a single activation function in the MLP:

```text
output = W_down( activation( W_up × input ) )
```

But gated activation functions (SwiGLU, GeGLU, etc.) changed this — they require **two parallel paths** through the MLP that are multiplied together:

```text
output = W_down( activation(W_gate × input) ⊙ W_up × input )
                 ─────────────────────────    ──────────────
                 "gate" branch                "value" branch
```

`mlp_activations` is a list because it defines **the activation function for each branch**. With SwiGLU:

```text
mlp_activations: ["silu", "linear"]
    ↓
branch 0: SiLU(W_gate × x)       — the nonlinear gate
branch 1: linear(W_up × x) = W_up × x  — the linear value
output = branch_0 ⊙ branch_1      — element-wise multiply
```

A single string couldn't express this. A list lets MaxText generalize to any combination of gated activations.

---

## 2. Default

```yaml
mlp_activations: ["silu", "linear"]
```

This is **SwiGLU** (Swish-Gated Linear Unit) — the standard activation for LLaMA 2+, Mistral, Gemma, Qwen, and most modern open-weight LLMs. It requires **two** input projection matrices (Wi_0 and Wi_1), both mapping `emb_dim → mlp_dim`.

---

## 3. Common configurations

| `mlp_activations` | Architecture style | Notes |
|---|---|---|
| `["silu", "linear"]` | SwiGLU | Default. LLaMA, Mistral, Gemma, Qwen family |
| `["gelu"]` | Standard GELU | GPT-2/GPT-3 style; single projection |
| `["gelu", "linear"]` | GeGLU | Gated variant of GELU |
| `["relu"]` | Standard ReLU | Oldest style; rarely used in modern LLMs |
| `["relu", "linear"]` | ReGLU | Gated ReLU |

---

## 4. Impact on MLP parameter count

With a **gated** activation (list of 2), the MLP has 3 weight matrices (Wi_0, Wi_1, Wo) instead of 2:

```text
Non-gated ["gelu"]:   W_up (E×M) + W_down (M×E)     = 2EM params
Gated ["silu","linear"]: W_gate (E×M) + W_up (E×M) + W_down (M×E) = 3EM params
```

This is why LLaMA-style `base_mlp_dim` is ~2/3 of what you'd expect at 4× ratio — to keep total MLP params equal, you shrink `mlp_dim` by 2/3 to compensate for the extra matrix.

---

## 5. How MaxText implements it

MaxText's MLP forward pass:

```python
# For each activation in mlp_activations, apply it to the corresponding projection
branches = [act_fn(Wi @ x) for act_fn, Wi in zip(mlp_activations, Wi_list)]
gate = elementwise_product(branches)  # multiply all branches together
output = Wo @ gate
```

If `mlp_activations` has 1 element → single projection, no gating.
If 2 elements → gated: two projections multiplied element-wise.

---

## 6. Interaction with `mlp_activations_limit`

`mlp_activations_limit` clips activation values to `[-limit, +limit]`. When using SwiGLU, the SiLU branch can produce large values at initialization; setting a limit can help if you see early training instability (though most runs don't need it). See [[mlp_activations_limit]].

---

### One-line intuition

> **`mlp_activations` is a list because gated activations like SwiGLU require two parallel MLP branches multiplied together — `["silu", "linear"]` is SwiGLU, the default for all modern LLaMA-family models.**
