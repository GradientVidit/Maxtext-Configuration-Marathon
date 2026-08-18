## 1. Why does `engram_head_dim` exist?

Each head in the Engram memory module processes a slice of the retrieved N-gram representation.

```text
Engram Feature Width:
Total Channels = engram_num_heads (e.g. 8) × engram_head_dim (e.g. 1280) = 10,240
```

`engram_head_dim` defines the channel dimension ($d_e$) of each individual Engram head.

A wider head dimension provides more expressive capacity for modeling complex phrase interactions and non-linear gating inside each head subspace.

---

## 2. Mechanics & Memory Allocation

Inside each Engram head:
1. Retrieved N-gram embeddings are projected into a $d_e$-dimensional subspace.
2. 1D temporal convolution (`engram_kernel_size`) filters the features across neighboring sequence positions.
3. SwiGLU / Gated Linear activations modulate the feature intensity before heads are concatenated.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `engram_head_dim` | `int` | `1280` | Positive integer (e.g., `512`, `1024`, `1280`) |

---

## 4. Interactions with Related Parameters

| Related Parameter | Interaction |
| :--- | :--- |
| `engram_num_heads` | Total intermediate projection dimension is `engram_num_heads * engram_head_dim`. |
| `base_emb_dim` | Final linear out-projection maps from `engram_num_heads * engram_head_dim` to `base_emb_dim`. |
| `engram_layers` | Engram module parameters are allocated only for the layers specified in `engram_layers`. |

---

## 5. Practical Guidance

| Setting | Context | Notes |
| :--- | :--- | :--- |
| `engram_head_dim: 1280` (Default) | Standard DeepSeek Engram configuration | Designed for wide transformer backbones (e.g. hidden dim $\ge 4096$). |
| Reduced Head Dim (e.g. `512`) | Smaller model variants (1B–7B) | Prevents the Engram module from dominating the layer's total parameter count. |

---

### One-line intuition

> `engram_head_dim` sets the channel dimension of each Engram head, determining the width of the feature subspace before temporal convolution and output fusion.
