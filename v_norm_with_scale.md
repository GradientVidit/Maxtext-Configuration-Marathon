## 1. Why does `v_norm_with_scale` exist?

In architectures that apply normalization to Value vectors ($V$) prior to the attention weighted sum (such as QK-Norm and Gemma 2 variants), Value vectors are projected to unit norm:

$$\hat{V} = rac{V}{\|V\|_2 + \epsilon}$$

While unit-norm normalization strictly bounds value magnitudes and prevents activation explosion, it strips the model of the ability to learn **channel-wise magnitude importance** (some features need to be larger than others to carry higher informational weight).

`v_norm_with_scale` determines whether a learnable per-channel scale parameter $\gamma_v$ is applied after Value normalization:

$$\text{v\_norm\_with\_scale=True:}\quad V_{\text{norm}} = \gamma_v \odot rac{V}{\|V\|_2 + \epsilon}$$

$$\text{v\_norm\_with\_scale=False:}\quad V_{\text{norm}} = rac{V}{\|V\|_2 + \epsilon}$$

```text
Without Learnable Scale (False):
  V ──> RMSNorm / L2Norm ──> V_norm (Strictly unit sphere, fixed variance)

With Learnable Scale (True, Default):
  V ──> RMSNorm / L2Norm ──> Multiply by learnable γ_v ──> V_norm (Dynamic range restored)
```

---

## 2. Options & Defaults

| Value | Behavior | Notes |
|---|---|---|
| `true` | Includes a learnable scale parameter $\gamma_v \in \mathbb{R}^{d_{head}}$ for Value normalization. | **Default**. Standard for models utilizing Value normalization. |
| `false` | Normalization without learnable scale (fixed unit variance). | Constrained expressivity. |

Default in `base.yml`: `true`

---

## 3. Parameter Impact

When Value normalization is active, `v_norm_with_scale=true` adds:

$$\text{Parameters} = N_{kv\_heads} 	imes d_{head} \quad (\text{or } 1 	imes d_{head} \text{ if shared across heads})$$

This adds a negligible parameter footprint ($<0.001\%$) while restoring the expressive dynamic range of the Value subspace.

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[qk_norm_with_scale]] | Sister parameter controlling learnable scale on Query and Key normalizations. |
| [[decoder_block]] | Only active when the configured decoder block architecture implements Value normalization. |
| [[normalization_layer_epsilon]] | Numerical stability constant used in the denominator during Value normalization. |

---

## 5. Practical Scenarios

- **Pretraining with Value Normalization:** Keep `v_norm_with_scale: true` (default) to ensure attention value aggregation retains feature-scaling flexibility.
- **Ablating Pure Geometric Attention:** Set to `false` only in specialized research studying scale-free spherical attention spaces.

---

### One-line intuition

> **`v_norm_with_scale=true` provides a learnable per-channel scale multiplier $\gamma_v$ on normalized Value vectors, restoring feature dynamic range while retaining bounded activation stability.**
