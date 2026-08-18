## 1. Why does `partial_rotary_factor` exist?

Rotary Position Embedding (RoPE) injects relative position information by rotating query and key vectors in 2D complex planes.

In full RoPE (`partial_rotary_factor = 1.0`), every coordinate in head dimension $d$ is rotated:

```text
Full RoPE (factor = 1.0):
Head Dim [0 .................................................... d-1]
         └───────────── All d dimensions rotated ──────────────┘

Partial RoPE (factor = 0.5):
Head Dim [0 ..................... d/2-1 | d/2 ................. d-1]
         └──── Rotated with RoPE ───────┴── Passed through unrotated ──┘
```

Some model architectures (such as GPT-J, GLM, and certain Qwen variants) hypothesize that dedicating 100% of the head dimension to position-dependent rotation forces all feature subspaces to encode positional shifts. By applying RoPE to only a **fraction** of the channels:
1. The **rotated subspace** encodes token order and relative distance.
2. The **unrotated subspace** preserves pure semantic, content-based, and frequency-invariant feature interactions without positional phase modulation.

`partial_rotary_factor` specifies the exact fraction (from $0.0$ to $1.0$) of each head's channels that receives rotary position embeddings.

---

## 2. Mechanics & Tensor Splitting

When `partial_rotary_factor` = $r \in (0, 1]$:

```text
Query / Key Head Tensor: [Batch, Seq_Len, Num_Heads, Head_Dim]
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
    Rotated Slice (dim = r * Head_Dim)     Pass-Through Slice (dim = (1-r) * Head_Dim)
                 │                               │
                 ▼                               │
      Apply 2D RoPE Rotation                     │
                 │                               │
                 └───────────────┬───────────────┘
                                 ▼
         Concatenate Along Head Dimension: [Batch, Seq_Len, Num_Heads, Head_Dim]
```

- Dimension calculation: `rotary_dim = int(head_dim * partial_rotary_factor)`.
- `rotary_dim` must be an even integer because RoPE rotates 2D coordinate pairs.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `partial_rotary_factor` | `float` | `1.0` | Range `(0.0, 1.0]`, typically `1.0`, `0.5`, or `0.25` |

---

## 4. Interactions with Related Parameters

| Related Parameter | Interaction |
| :--- | :--- |
| `head_dim` / `gdn_key_head_dim` | `head_dim * partial_rotary_factor` must evaluate to an even integer. |
| `rope_type` | Determines the RoPE algorithm (e.g. `default`, `yarn`, `linear`) applied to the rotated slice. |
| `rope_min_timescale` / `rope_max_timescale` | Base frequency calculations are computed over `rotary_dim = head_dim * partial_rotary_factor`. |

---

## 5. Practical Scenarios & Failure Modes

| Architecture / Scenario | Recommended Value | Note |
| :--- | :--- | :--- |
| **Standard LLaMA / Gemma / Mistral** | `1.0` | Standard full head rotation. |
| **GPT-J (6B)** | `0.5` | Rotates 64 out of 128 dimensions per head. |
| **GLM / ChatGLM** | `0.5` | Standard half-rotary design. |
| **Odd Rotary Dimension** | Error | If `head_dim * partial_rotary_factor` is odd, RoPE pair rotation fails on tensor reshape. |

---

### One-line intuition

> `partial_rotary_factor` specifies what fraction of each head dimension undergoes Rotary Position Embedding, leaving the remaining channels unrotated to preserve pure content representations.
