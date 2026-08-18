## 1. Why does `engram_kernel_size` exist?

When n-gram embeddings are retrieved from hash tables at each sequence position $t$, they reflect discrete token lookups.

To smooth feature transitions across adjacent sequence positions and allow the model to blend overlapping n-gram lookups before layer integration, Engram applies a **1D causal temporal convolution**:

```text
Position t-3      Position t-2      Position t-1      Position t
  Engram E_{t-3}    Engram E_{t-2}    Engram E_{t-1}    Engram E_t
        │                 │                 │                 │
        └─────────────────┴────────┬────────┴─────────────────┘
                                   ▼
          1D Causal Convolution (Kernel Size = engram_kernel_size = 4)
                                   │
                                   ▼
                   Smoothed Engram Feature Vector at t
```

`engram_kernel_size` specifies the temporal window width of this 1D causal convolution.

---

## 2. Mechanics & Causality

- **Causal Padding**: To strictly prevent future token information leakage during autoregressive training and generation, the convolution is left-padded by `engram_kernel_size - 1` steps.
- **Local Context**: A kernel width of 4 allows the Engram module to correlate n-gram representations across a 4-token window.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `engram_kernel_size` | `int` | `4` | Positive integer, typically `2`, `4`, or `8` |

---

## 4. Interactions with Related Parameters

| Related Parameter | Interaction |
| :--- | :--- |
| `engram_layers` | Determines on which layers this convolution is instantiated. |
| `engram_head_dim` | Convolutions are applied depthwise (per channel/head). |

---

## 5. Practical Guidance

| Setting | Consequence |
| :--- | :--- |
| `engram_kernel_size: 4` (Default) | Standard receptive field for local phrase smoothing. |
| `engram_kernel_size: 1` | Point-wise lookup only (no temporal smoothing across adjacent positions). |

---

### One-line intuition

> `engram_kernel_size` specifies the window width of the 1D causal convolution in Engram, smoothing and blending retrieved n-gram embeddings across neighboring sequence positions.
