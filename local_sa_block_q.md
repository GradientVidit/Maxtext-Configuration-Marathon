## 1. Why does it exist?

Modern hybrid architectures (such as Gemma 2 and Mistral) alternate between **Global Attention layers** (full sequence context) and **Local Sliding-Window Attention layers** (where tokens attend only to a local window $W$, e.g. 4096 tokens).

In local sliding-window attention, the number of active Key/Value blocks per Query is bounded by the window size $W$. Using large 512-token tiles designed for global attention can cause significant quantization/padding overhead at the sliding window boundaries.

MaxText provides a full family of `local_sa_*` configuration parameters to tune kernel tile sizes specifically for local sliding-window attention layers independently of global layers.

```text
Global Attention Layers:             Local Sliding-Window Layers:
  Tuned via `sa_block_*`               Tuned via `local_sa_block_*`
  (e.g. sa_block_q: 512)               (e.g. local_sa_block_q: 256 for tight window tiling)
```

`local_sa_block_q` sets the Query tile size specifically for local sliding-window attention layers.

---

## 2. Inheritance Rules (`None`)

By default, all `local_sa_*` parameters are set to `None`, which means they automatically **inherit** the value of their corresponding global `sa_*` parameter:

| Local Parameter | Default | Inherited From |
|---|---|---|
| `local_sa_block_q` | `None` | `sa_block_q` (512) |
| `local_sa_block_kv` | `None` | `sa_block_kv` (512) |
| `local_sa_block_kv_compute` | `None` | `sa_block_kv_compute` (512) |
| `local_sa_block_q_dkv` | `None` | `sa_block_q_dkv` (512) |
| `local_sa_block_kv_dkv` | `None` | `sa_block_kv_dkv` (512) |
| `local_sa_block_kv_dkv_compute` | `None` | `sa_block_kv_dkv_compute` (512) |
| `local_sa_block_q_dq` | `None` | `sa_block_q_dq` (512) |
| `local_sa_block_kv_dq` | `None` | `sa_block_kv_dq` (512) |

---

## 3. Practical Usage

- **Default**: Leave as `None` to match global attention block sizes.
- **When to Override**: When profiling local sliding-window attention (e.g. in Gemma 2) and adjusting block sizes to align evenly with `sliding_window_size`.

---

### One-line intuition

> **`local_sa_block_q` (and the `local_sa_*` family) configures dedicated block tile sizes for local sliding-window attention layers, defaulting to `None` to inherit global `sa_*` values.**
