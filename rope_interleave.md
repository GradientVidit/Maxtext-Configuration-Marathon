## 1. Why does `rope_interleave` exist?

RoPE treats consecutive pairs of embedding coordinates as 2D complex numbers $(x_{2i}, x_{2i+1})$ and applies a 2D Givens rotation:

$$\begin{pmatrix} x_{2i}' \\ x_{2i+1}' \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$$

There are two distinct tensor layout conventions in deep learning for organizing these paired coordinates:

```text
1. Interleaved Layout (rope_interleave: true):
   Coordinates: [ x_0, x_1,   x_2, x_3,   ..., x_{d-2}, x_{d-1} ]
   Pairing:     (x_0, x_1),   (x_2, x_3), ..., (x_{d-2}, x_{d-1})  ──> Adjacent elements

2. Half-Split / Chunked Layout (rope_interleave: false):
   Coordinates: [ x_0, x_1, ..., x_{d/2-1}  |  x_{d/2}, ..., x_{d-1} ]
   Pairing:     (x_0, x_{d/2}), (x_1, x_{d/2+1}), ...             ──> Split across tensor halves
```

Different model families use different conventions (e.g. HuggingFace Llama uses half-split, while Jax/Flax, Megatron-LM, and original RoFormer implementations often use interleaved).

`rope_interleave` exists to select between adjacent-element interleaving and split-half concatenation.

---

## 2. What it actually controls

```yaml
rope_interleave: true
```

- When `true` (default in base.yml): Rotary pairs are formed by adjacent indices `[..., 2k]` and `[..., 2k+1]`.
- When `false`: Rotary pairs are formed by splitting the head dimension in half `[..., :d/2]` and `[..., d/2:]`.

```text
Tensor Layout Transformation:
Vector: [v0, v1, v2, v3]
  rope_interleave: true  ──> Rotate (v0, v1) with theta_0, Rotate (v2, v3) with theta_1
  rope_interleave: false ──> Rotate (v0, v2) with theta_0, Rotate (v1, v3) with theta_1
```

---

## 3. Options and Defaults

| Value | Pairing Scheme | Framework Alignment |
|---|---|---|
| `true` (default) | Adjacent pairs `(2i, 2i+1)` | Megatron-LM, Jax/Flax native RoPE, YaRN |
| `false` | Half-split `(i, i + d/2)` | HuggingFace Transformers `LlamaRotaryEmbedding` |

---

## 4. Interactions and Checkpoint Conversion

- **Weight / Checkpoint Compatibility**: When porting weights from HuggingFace to MaxText (or vice versa), mismatching `rope_interleave` will corrupt attention score calculations.
- **`rope_pairwise`**: Works in conjunction with `rope_pairwise` for rank-5 pair representation.

---

## 5. Practical Scenarios

- **MaxText Native Training**: Keep `rope_interleave: true`.
- **HuggingFace Weight Ingestion**: Ensure weight conversion tools or checkpoint loaders permute rotary weight matrices if bridging between half-split and interleaved representations.

---

### One-line intuition

> **`rope_interleave` selects whether RoPE coordinates are paired as adjacent elements `(2i, 2i+1)` or split across tensor halves `(i, i + d/2)`.**
