## 1. Why does `use_mrope` exist?

Standard Rotary Position Embeddings (RoPE) assume a 1-dimensional sequential order ($0, 1, 2, \dots, S-1$). However, multimodal inputs are fundamentally multi-dimensional:
- **Text**: 1D sequential order ($t$).
- **Images**: 2D spatial coordinates (height $y$, width $x$).
- **Videos**: 3D spatiotemporal coordinates (time $t$, height $y$, width $x$).

Flattening 2D/3D visual patches into a 1D sequence under standard RoPE destroys spatial neighborhood geometry and temporal continuity. 

**M-RoPE (Multimodal Rotary Position Embedding)**, introduced by Qwen2-VL, decomposes the rotary channel dimension into three distinct sub-sections, jointly applying independent rotational frequencies for Temporal ($T$), Vertical ($H$), and Horizontal ($W$) positions.

```text
Standard 1D RoPE:
Token at pos 100 ──► Rotate all D dimensions by angle theta * 100

Multimodal M-RoPE:
Token at 3D coord (T=5, H=12, W=8):
┌───────────────────────┬───────────────────────┬───────────────────────┐
│ Section T (24 dims)   │ Section H (20 dims)   │ Section W (20 dims)   │
│ Rotate by theta_T * 5 │ Rotate by theta_H * 12│ Rotate by theta_W * 8 │
└───────────────────────┴───────────────────────┴───────────────────────┘

Text Token at pos 42:
T = 42, H = 42, W = 42 -> Uniform 1D rotation across all sections.
```

`use_mrope` activates 3D Multimodal Rotary Position Embeddings across the attention mechanism.

---

## 2. Mathematical Mechanics

The attention head dimension is divided into three sections governed by `mrope_section` $[d_T, d_H, d_W]$:
1. Channels $[0 : 2d_T]$ are rotated by position ID $p_T$.
2. Channels $[2d_T : 2(d_T + d_H)]$ are rotated by position ID $p_H$.
3. Channels $[2(d_T + d_H) : 2(d_T + d_H + d_W)]$ are rotated by position ID $p_W$.

This unifies text, 2D images, and 3D videos into a single coordinate-aware attention space.

---

## 3. Options and Defaults

| Value | Behavior | Model Family |
|---|---|---|
| `false` (Default) | Standard 1D Rotary Position Embeddings | Llama 2/3, Gemma 2, Mistral |
| `true` | 3D Multimodal RoPE (Temporal, Height, Width) | Qwen2-VL, Qwen3-OmniMoE, Omni-modal models |

---

## 4. Key Parameter Interactions

- **`mrope_section`**: Dictates the exact dimension split across $T, H, W$.
- **`position_id_per_seconds`**: Converts real-world video playback timestamps into temporal position IDs.
- **`head_dim`**: Total rotary dimension ($2(d_T + d_H + d_W)$) must not exceed `head_dim`.

---

### One-line intuition
> **`use_mrope` activates 3D Multimodal Rotary Position Embeddings, decomposing head dimensions to jointly encode temporal, vertical, and horizontal coordinates.**
