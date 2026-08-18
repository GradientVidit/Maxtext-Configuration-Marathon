
## 1. Why does `base_mlp_dim` exist?

The MLP block is the other half of a transformer layer (besides attention). It's a two-layer feed-forward network that processes each token position independently:

```text
Input: [batch, seq, emb_dim]
    ↓
[emb_dim → base_mlp_dim]   "expand" projection (Wi or Wi_0/Wi_1 for gated)
    ↓
nonlinearity (SiLU, GELU, etc.)
    ↓
[base_mlp_dim → emb_dim]   "contract" projection (Wo)
    ↓
Output: [batch, seq, emb_dim]
```

The intermediate dimension (`base_mlp_dim`) controls how wide this expansion gets. A wider MLP can store more "factual knowledge" — empirically, MLP layers act as key-value memories for factual associations.

---

## 2. Default

```yaml
base_mlp_dim: 7168
```

7168 ≈ 3.5 × 2048 (`base_emb_dim`). This ~3.5× ratio is common (LLaMA, Mistral style). Some architectures use 4× (classic transformer), and gated variants (SwiGLU) typically use 2/3 × 4× ≈ 2.67× to compensate for having two input projections.

---

## 3. Parameter count impact

For a single MLP block with `emb_dim=E` and `mlp_dim=M`:
- Without gating: 2 matrices → `2 × E × M` parameters
- With gating (SwiGLU): 3 matrices (Wi_0, Wi_1, Wo) → `3 × E × M` parameters

For a 32-layer model with `E=4096, M=11008`:
```text
MLP params = 32 × 3 × 4096 × 11008 ≈ 4.3B parameters
```
MLP typically accounts for ~2/3 of total parameters in dense transformers.

---

## 4. The gated activation multiplier

With `mlp_activations: ["silu", "linear"]` (SwiGLU), the MLP has **two** input projections (Wi_0 and Wi_1), both mapping `emb_dim → mlp_dim`. This doubles the input parameter count vs. a plain MLP.

To keep total MLP parameter count equal to a 4× plain MLP:
```text
gated mlp_dim ≈ (2/3) × 4 × emb_dim
```
So for `emb_dim=4096`: `mlp_dim ≈ (2/3) × 4 × 4096 ≈ 10,922 → rounded to 11,008`.

MaxText's default of 7168 with emb_dim=2048 follows this: `7168 ≈ 10,922 × (2048/4096)`.

---

## 5. Interaction with MoE

For Mixture of Experts models, `base_mlp_dim` sets the MLP width for **dense** MLP layers. Expert MLP layers may use `base_moe_mlp_dim` instead. See [[base_moe_mlp_dim]].

---

## 6. Common values

| Model | base_emb_dim | base_mlp_dim | Ratio |
|---|---|---|---|
| Default | 2048 | 7168 | 3.5× |
| LLaMA 2 7B | 4096 | 11008 | 2.69× (SwiGLU) |
| LLaMA 2 13B | 5120 | 13824 | 2.7× |
| LLaMA 2 70B | 8192 | 28672 | 3.5× |
| GPT-style (4× ratio) | 768 | 3072 | 4× |

---

## 7. Sharding

`base_mlp_dim` must be divisible by the tensor parallelism degree (it's typically sharded along this dimension). Powers of 2 or multiples of 128/1024 are safest.

---

### One-line intuition

> **`base_mlp_dim` is the width of the transformer's feed-forward expansion — wider means more "memory" for facts and patterns, and it typically accounts for 2/3 of total model parameters.**
