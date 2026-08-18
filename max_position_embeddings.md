## 1. Why does `max_position_embeddings` exist?

In YaRN (Yet another RoPE extensioN) and long-context scaling frameworks, extending a pretrained language model requires defining both the **original context length** the model was pretrained on and the **extended target context length** it is being scaled to:

```text
Pretrained Baseline Context: [ 0 ──────── 4,096 ] (original_max_position_embeddings)
                                              │
                                              ▼ (YaRN Context Extension)
Extended Target Context:     [ 0 ──────────────────────────────── 163,840 ] (max_position_embeddings)
```

Without knowing the target context boundary, the YaRN frequency interpolation formula cannot compute the correct wavelength compression ratios and attention entropy scaling factors.

`max_position_embeddings` defines this target upper context boundary for YaRN scaling.

---

## 2. What it actually controls

```yaml
max_position_embeddings: 163840
```

Inside YaRN computation (`rope_type: "yarn"`), `max_position_embeddings` represents $L_{target}$.

It is used to compute the scale factor $s$:

$$s = \frac{\text{max\_position\_embeddings}}{\text{original\_max\_position\_embeddings}} = \frac{163{,}840}{4{,}096} = 40$$

```text
YaRN Parameter Derivation:
max_position_embeddings: 163840
original_max_position_embeddings: 4096
                 │
                 ▼
Scale Factor s = 163840 / 4096 = 40
                 │
                 ├─> Determines frequency interpolation threshold (beta_fast, beta_slow)
                 └─> Determines attention logit magnitude scaling (mscale)
```

---

## 3. Options and Defaults

| Value | Target Context | Pretrained Baseline | Effective Extension Factor ($s$) |
|---|---|---|---|
| `163840` (default) | 160k tokens | 4k tokens | $40\times$ |
| `32768` | 32k tokens | 4k tokens | $8\times$ |
| `65536` | 64k tokens | 4k tokens | $16\times$ |
| `131072` | 128k tokens | 4k tokens | $32\times$ |

---

## 4. Interactions and Dependencies

- **`original_max_position_embeddings`**: The ratio `max_position_embeddings / original_max_position_embeddings` forms the core scaling factor.
- **`rope_factor`**: Must match or be consistent with `max_position_embeddings / original_max_position_embeddings`.
- **`max_target_length`**: `max_target_length` during training/eval cannot exceed `max_position_embeddings`.

---

## 5. Practical Scenarios

- **YaRN 128k Fine-Tuning**: Set `max_position_embeddings: 131072` with `original_max_position_embeddings: 4096` ($s=32$).
- **DeepSeek 160k Context**: Uses default `max_position_embeddings: 163840` with `original_max_position_embeddings: 4096`.

---

### One-line intuition

> **`max_position_embeddings` sets the target extended context length for YaRN RoPE frequency interpolation.**
