## 1. Why does `gdn_num_key_heads` exist?

In multi-head attention architectures, the number of key heads controls how many distinct subspace projections are used for addressing.

In Gated DeltaNet (GDN), having multiple key heads allows the model to partition memory into independent associative channels:

```text
Multi-Head Recurrent Architecture:
Key Projections   ──> H_k heads of dim d_k
Value Projections ──> H_v heads of dim d_v

When H_k < H_v (Grouped-Key GDN):
Each key head is shared across (H_v / H_k) value heads.
Key Head 0 ───┬───> Memory Head 0 (Value Head 0)
              └───> Memory Head 1 (Value Head 1)
```

`gdn_num_key_heads` defines the number of key/query projection heads ($H_k$). In Qwen3-Next, setting $H_k = 16$ while $H_v = 32$ establishes a 2:1 Grouped-Key ratio that cuts key projection parameters and addressing compute while preserving higher value capacity.

---

## 2. Mechanics & Grouped Addressing

GDN can operate with symmetric heads ($H_k = H_v$) or grouped heads ($H_k < H_v$):

```text
Input Tokens [Batch, Seq_Len, Hidden_Dim]
      │
      ├───> Q Projections: [Batch, Seq_Len, H_k, d_k]
      ├───> K Projections: [Batch, Seq_Len, H_k, d_k]
      └───> V Projections: [Batch, Seq_Len, H_v, d_v]
                │
                ▼
Broadcast / Repeat K, Q across groups if H_k < H_v
                │
                ▼
H_v independent state matrices: S^{(i)} ∈ ℝ^{d_k × d_v}  (i = 1 ... H_v)
```

- When $H_k < H_v$, each key head $k_t^{(j)}$ is broadcast to update and query multiple value memory blocks $S^{(i)}$.
- This mirrors Grouped Query Attention (GQA) in Transformers, but applies to linear recurrent state dynamics.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `gdn_num_key_heads` | `int` | `16` | Positive integers dividing `gdn_num_value_heads` evenly |

---

## 4. Interactions with Related Parameters

| Related Parameter | Interaction |
| :--- | :--- |
| `gdn_num_value_heads` | Must be an exact integer multiple of `gdn_num_key_heads` (e.g., $32 / 16 = 2$). |
| `gdn_key_head_dim` | Total Q and K weight dimension is `gdn_num_key_heads * gdn_key_head_dim`. |
| `partial_rotary_factor` | Applied head-wise across each of the `gdn_num_key_heads` key/query heads. |
| `logical_axis_rules` | Sharded along tensor/data parallel axes following `gdn_head` rules. |

---

## 5. Practical Guidance & Failure Modes

| Configuration | Behavior / Consequence |
| :--- | :--- |
| `gdn_num_key_heads: 16`, `gdn_num_value_heads: 32` | Standard Qwen3-Next preset; 2:1 group ratio balances parameter budget and representation capacity. |
| `gdn_num_key_heads: 32`, `gdn_num_value_heads: 32` | Full multi-head linear attention; maximum addressing expressiveness at higher compute/parameter cost. |
| `gdn_num_key_heads` does not divide `gdn_num_value_heads` | Group broadcasting fails during tensor reshape; raises assertion error on model initialization. |

---

### One-line intuition

> `gdn_num_key_heads` specifies the number of query and key addressing heads in Gated DeltaNet, enabling grouped-key linear recurrence when set smaller than the number of value heads.
