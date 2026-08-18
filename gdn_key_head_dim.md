## 1. Why does `gdn_key_head_dim` exist?

In standard Transformer attention, head dimension $d_k$ determines the size of query and key vectors for dot-product attention scores ($q^T k / \sqrt{d_k}$).

In Gated DeltaNet (GDN), the key head dimension $d_k$ has a fundamentally different role: it sets the **rank and memory capacity of the recurrent associative memory state**:

```text
Standard Attention:
Q, K ∈ [B, H, S, d_k] ──> Softmax(Q K^T / √d_k) ──> O(S^2) Memory

Gated DeltaNet:
Recurrent State S_t ∈ [B, H, d_k, d_v]
Memory Update: S_t = S_{t-1}(I - β_t k_t k_t^T) + β_t v_t k_t^T
```

Here, each key vector $k_t \in \mathbb{R}^{d_k}$ defines the retrieval address and erasure direction in the recurrent memory matrix $S \in \mathbb{R}^{d_k \times d_v}$.

`gdn_key_head_dim` explicitly configures $d_k$ for GDN layers independently of the standard transformer `head_dim`.

---

## 2. Mechanics & Memory Footprint

The recurrent state matrix $S$ maintained per GDN head has shape $[d_k, d_v]$:

```text
Key Head Dimension (d_k = gdn_key_head_dim)
          │
          ▼
   ┌─────────────── d_v ──────────────┐
   │ S[0, 0]    S[0, 1]   ... S[0, d_v]│
d_k│ S[1, 0]    S[1, 1]   ... S[1, d_v]│  <── Recurrent state per head
   │ ...                               │
   │ S[d_k, 0]  S[d_k, 1] ... S[d_k,d_v│
   └───────────────────────────────────┘
```

- **Associative Capacity**: A larger $d_k$ allows the memory matrix to store more orthogonal key patterns simultaneously without catastrophic interference during the delta rule updates.
- **Hardware Tiling**: TPU Matrix Multiply Units (MXUs) perform best when inner dimensions are multiples of 128 (e.g., $128 \times 128$ systolic arrays).

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `gdn_key_head_dim` | `int` | `128` | Positive integers, typically `64`, `128`, `256` |

---

## 4. Interactions with Related Parameters

| Related Parameter | Interaction |
| :--- | :--- |
| `gdn_value_head_dim` | Determines the second dimension of the state matrix $[d_k, d_v]$. If both are 128, state per head is $128 \times 128 = 16,384$ scalar elements. |
| `gdn_num_key_heads` | Total key projection dimension is `gdn_num_key_heads * gdn_key_head_dim`. |
| `use_qk_norm_in_gdn` | L2 normalization operates across this $d_k$ dimension, ensuring $\|k\|_2 = 1.0$. |
| `partial_rotary_factor` | Specifies what fraction of `gdn_key_head_dim` receives RoPE embeddings. |

---

## 5. Practical Guidance & Scenarios

| Scenario | Recommendation | Rationale |
| :--- | :--- | :--- |
| **Default Qwen3-Next Architecture** | `gdn_key_head_dim: 128` | Matches native TPU systolic array dimensions and standard Qwen3-Next hyperparameter presets. |
| **VRAM / State Constrained Inference** | `gdn_key_head_dim: 64` | Halves the recurrent state size per token sequence, reducing inference memory at the cost of associative capacity. |
| **Non-aligned Dim (e.g. 96)** | Avoid | Causes XLA padding on TPU tensor cores, degrading compute utilization. |

---

### One-line intuition

> `gdn_key_head_dim` sets the key/query vector dimension in Gated DeltaNet, directly determining the row rank of each head's recurrent associative state matrix.
