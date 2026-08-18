## 1. Why does it exist?

Standard transformer training packs variable-length documents into fixed-length batches (e.g. 8192 tokens) using padding tokens or document boundary segmentation. With standard dense attention, padding tokens and cross-document boundary masking still consume FLOPs and memory bandwidth.

**Ragged Attention** (variable-length aware attention) computes attention exclusively across valid document tokens, skipping cross-document padding and redundant masked interactions at the kernel level.

```text
Standard Packed Attention:
  [ Doc A (500 tok) | Doc B (1200 tok) | Padding (6492 tok) ]
  ──→ Computes attention matrix across all 8192x8192 positions (wasting compute on padding).

Ragged Attention (use_ragged_attention: true):
  Kernel operates directly on ragged document segment offsets
  ──→ Computes attention only within Doc A (500x500) and Doc B (1200x1200).
```

`use_ragged_attention` enables a ragged, document-aware attention kernel implementation that bypasses padding overhead.

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `false` (default) | Standard dense or flash attention kernels with standard mask packing. |
| `true` | Uses ragged attention kernel for variable-length sequence evaluation. |

Default in `base.yml`:
```yaml
use_ragged_attention: false
```

---

## 3. Companion Parameter

- **`ragged_block_size`**: Sets the tile block size (default `256`) for the ragged attention kernel.

---

### One-line intuition

> **`use_ragged_attention` enables a ragged-sequence attention kernel that processes variable-length packed documents without wasting compute on padding tokens.**
