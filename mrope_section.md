## 1. Why does `mrope_section` exist?

When M-RoPE decomposes the attention head into Temporal ($T$), Vertical ($H$), and Horizontal ($W$) rotary components, the model designer must decide what proportion of the rotary representation budget is dedicated to each coordinate axis.

Because RoPE rotates pairs of feature dimensions, each section entry represents the number of rotary coordinate pairs allocated to that axis.

```text
Head Dimension Pairs = 64 pairs (Head Dim = 128)
mrope_section: [24, 20, 20]
  ├── Temporal (T) : 24 pairs = 48 feature channels
  ├── Height (H)   : 20 pairs = 40 feature channels
  └── Width (W)    : 20 pairs = 40 feature channels
Total = 24 + 20 + 20 = 64 pairs (128 channels)
```

`mrope_section` defines the list of three integer pair counts $[d_T, d_H, d_W]$ allocated to temporal, vertical, and horizontal position rotations.

---

## 2. Constraints and Arithmetic

The sum of the three values multiplied by 2 must equal the active rotary dimension:
$$2 	imes (d_T + d_H + d_W) = D_{\text{rope}} \le \text{head\_dim}$$

For the standard default `[24, 20, 20]`:
$$2 	imes (24 + 20 + 20) = 2 	imes 64 = 128 = \text{head\_dim}$$

---

## 3. Options and Defaults

| Value | Component Breakdown | Architecture |
|---|---|---|
| `[24, 20, 20]` (Default) | 48 dims Time, 40 dims Height, 40 dims Width | Qwen2-VL / Qwen3 standard (head_dim = 128) |
| `[16, 24, 24]` | 32 dims Time, 48 dims Height, 48 dims Width | Higher spatial resolution priority |
| `[32, 16, 16]` | 64 dims Time, 32 dims Height, 32 dims Width | Long video / audio priority |

---

## 4. Failure Modes

- If $2 	imes \sum(\text{mrope\_section}) 
e D_{\text{rope}}$, rotary kernel tensor slicing will throw dimension mismatch exceptions during attention computation.

---

### One-line intuition
> **`mrope_section` specifies the channel dimension budget $[d_T, d_H, d_W]$ allocated to Temporal, Height, and Width rotary axes in M-RoPE.**
