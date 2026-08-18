## 1. Why does `rope_type` exist?

Rotary Position Embedding (RoPE) encodes relative token positions by multiplying Query and Key vectors with orthogonal 2D rotation matrices in the complex plane:

$$R_{\Theta, m}^d = \text{diag}\left(R_{\theta_1, m}, R_{\theta_2, m}, \dots, R_{\theta_{d/2}, m}\right)$$

Over time, different scaling and interpolation algorithms evolved to extend the context window of pretrained models beyond their initial training context:

```text
                      RoPE Evolution & Variants
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
     "default"               "llama3.1"                 "yarn"
Standard Su et al.        Wavelength-aware          Non-uniform NTK +
Base RoPE / Linear PI     Frequency Scaling       Attention Temp Scaling
```

`rope_type` exists as the master architectural selector that switches between these rotary frequency calculation algorithms in MaxText.

---

## 2. What it actually controls

```yaml
rope_type: "default"
```

| Value | Algorithm & Implementation | Primary Configuration Parameters |
|---|---|---|
| `"default"` | Standard RoPE with optional linear scaling (Position Interpolation) | `rope_min_timescale`, `rope_max_timescale`, `rope_linear_scaling_factor` |
| `"llama3.1"` | Meta Llama 3.1 frequency-dependent wavelength scaling | `rope_min_timescale`, `rope_max_timescale`, `rope_use_scale` |
| `"yarn"` | YaRN (Yet another RoPE extensioN) with NTK-by-parts & attention temperature $m$-scale | `max_position_embeddings`, `original_max_position_embeddings`, `rope_factor`, `beta_fast`, `beta_slow`, `mscale` |

```text
Execution Flow in Attention:
Q, K Vectors ──> [ rope_type Selector ]
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
    ["default"]   ["llama3.1"]    ["yarn"]
         │             │             │
  Standard Base  Freq-Wavelength  YaRN Ramp +
  Frequencies    Interpolation    mscale Temp
         └─────────────┬─────────────┘
                       ▼
            Rotated Q, K Vectors
```

---

## 3. Options and Defaults

```yaml
rope_type: "default"    # Standard base RoPE (Llama 1/2, Gemma, Mistral)
rope_type: "llama3.1"   # Llama 3.1 8B/70B/405B context extension (up to 128k)
rope_type: "yarn"       # DeepSeek / YaRN long-context fine-tuning
```

---

## 4. Key Parameter Interactions

- When `rope_type: "default"`: MaxText computes frequencies as $\theta_i = \theta_{base}^{-2i/d}$ scaled by `rope_linear_scaling_factor`.
- When `rope_type: "llama3.1"`: MaxText activates `LLaMARotaryEmbedding`, dynamically scaling high-wavelength (low-frequency) dimensions while leaving low-wavelength (high-frequency) components intact.
- When `rope_type: "yarn"`: MaxText branches into YaRN logic, utilizing the 10 YaRN-specific parameters (`beta_fast`, `beta_slow`, `mscale`, `rope_factor`, etc.).

---

## 5. Practical Scenarios

- **Pretraining standard models (Llama 2, Mistral, Gemma 1/2)**: Use `rope_type: "default"`.
- **Running / Fine-tuning Llama 3.1 / 3.2 / 3.3 models**: Set `rope_type: "llama3.1"` with `rope_max_timescale: 500000.0`.
- **Long-context adaptation / DeepSeek V2/V3**: Set `rope_type: "yarn"` alongside context scaling parameters.

---

### One-line intuition

> **`rope_type` selects the mathematical variant of Rotary Positional Embeddings—switching between standard RoPE, Llama 3.1 wavelength scaling, and YaRN context extension.**
