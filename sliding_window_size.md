## 1. Why does `sliding_window_size` exist?

Global causal attention scales quadratically in both computation ($O(N^2)$ FLOPs) and KV cache memory ($O(N)$ per request). For long sequences (e.g. 32K–128K tokens), storing the full uncompressed KV cache for thousands of concurrent requests exhausts accelerator HBM.

**Sliding Window Attention (SWA)** bounds the attention span of each token to a local neighborhood of the nearest $W$ tokens:

$$\text{Attention Mask:}\quad M_{i, j} = egin{cases} 0 & \text{if } 0 \le i - j < W \ -\infty & \text{otherwise} \end{cases}$$

```text
Sequence of 8 Tokens with sliding_window_size W = 3:

Token  Attends to:
  t0:  [t0]
  t1:  [t0, t1]
  t2:  [t0, t1, t2]
  t3:  [t1, t2, t3]    <-- t0 falls outside the window
  t4:  [t2, t3, t4]
  t5:  [t3, t4, t5]
  t6:  [t4, t5, t6]
  t7:  [t5, t6, t7]
```

This reduces computational complexity from $O(N^2)$ to $O(N \cdot W)$ and caps the active KV cache buffer size at $W$ tokens.

---

## 2. Options & Defaults

| Value | Behavior | Notes |
|---|---|---|
| `0` | Sliding window disabled. Global attention is used. | **Default**. Standard for purely causal models. |
| Any integer $> 0$ (e.g. `4096`) | Restricts attention visibility to the current token and the preceding $W-1$ tokens. | Active when `attention_type: 'local_sliding'` or in hybrid layer configurations. |

Default in `base.yml`: `0`

---

## 3. Receptive Field Accumulation Across Layers

Even though an individual layer only attends to $W$ tokens locally, stacking $L$ sliding window layers expands the effective theoretical receptive field linearly with depth:

$$\text{Effective Receptive Field at Layer } l = l 	imes (W - 1) + 1$$

```text
Layer 3: ═══════════════════════════════════════════════> [ Receptive Field ≈ 3 * W ]
             ▲                  ▲                  ▲
Layer 2: ═══════════════> ═══════════════> ═══════════════> [ Receptive Field ≈ 2 * W ]
             ▲                  ▲                  ▲
Layer 1: ═════> ═════> ═════> ═════> ═════> ═════> ═════>   [ Receptive Field = W ]
Tokens:  t0     t1     t2     t3     t4     t5     t6
```

Modern architectures like **Gemma 2** and **Mistral 7B** interleave sliding window layers ($W=4096$) with periodic global attention layers (e.g. every 2nd or 6th layer) to combine local efficiency with instant global information routing.

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[attention_type]] | Must be set to `'local_sliding'` for global model configuration, or configured per-layer in inhomogeneous architectures. |
| [[attention_sink]] | Highly recommended for generation: prevents softmax collapse when tokens slide out of window $W$. |
| [[chunk_attn_window_size]] | Alternative localized attention pattern based on non-overlapping discrete chunks rather than a continuous sliding window. |
| [[attention]] | The selected attention kernel must support banded / sliding window masking (e.g. Splash Attention). |

---

## 5. Practical Scenarios

- **Pretraining Standard Full-Context Models:** Leave `sliding_window_size: 0`.
- **Reproducing Mistral / Gemma 2 Architectures:** Set `sliding_window_size: 4096` with `attention_type: 'local_sliding'`.
- **Serving High-Throughput Inference:** Using $W=2048$ or $W=4096$ locks KV cache consumption per sequence to a fixed upper limit, preventing OOM during continuous batching.

---

### One-line intuition

> **`sliding_window_size` limits attention span to the nearest $W$ preceding tokens, slashing attention FLOPs from $O(N^2)$ to $O(N \cdot W)$ and bounding KV cache growth during generation.**
