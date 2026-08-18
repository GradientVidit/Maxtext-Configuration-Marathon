## 1. Why does it exist?

Splash Attention is Google's TPU-native FlashAttention implementation written in Pallas. FlashAttention avoids materializing the full $[S \times S]$ attention matrix in High Bandwidth Memory (HBM) by loading inputs in small tiles into ultra-fast on-chip Vector Memory (VMEM) and computing the softmax incrementally using online rescaling.

In the forward pass, the Query sequence $Q$ of length $S$ is divided into discrete chunks of size `sa_block_q`.

```text
Query Sequence (Length S):
  [ Block 0 (512) ][ Block 1 (512) ][ Block 2 (512) ] ... [ Block N (512) ]
        │
     Loaded into TPU VMEM for online softmax computation against KV blocks
```

`sa_block_q` sets the tile block size along the Query sequence dimension in the Splash Attention forward pass kernel.

---

## 2. Fundamentals & VMEM Sizing

- **Tile Sizing Trade-Off**:
  - Larger tiles (e.g. `512` or `1024`): Increase arithmetic intensity and matrix multiplication efficiency on TPU Matrix Multiply Units (MXUs), but consume more on-chip VMEM.
  - Smaller tiles (e.g. `128` or `256`): Fit more easily inside constrained VMEM budgets, but incur higher loop overhead.

---

## 3. Options & Configuration

| Value | Meaning |
|---|---|
| `512` (default) | Standard 512-token query tile size for TPU v4/v5e/v5p/v6e architectures. |
| Positive integer (e.g. `128`, `256`, `1024`) | Custom tile size (must divide or be compatible with sequence length). |

Default in `base.yml`:
```yaml
sa_block_q: 512
```

---

## 4. Interactions with Related Parameters

- **`sa_block_kv`**: Companion tile size for Key/Value tokens in the forward pass.
- **`local_sa_block_q`**: Local sliding-window attention layers inherit from `sa_block_q` unless explicitly overridden.

---

### One-line intuition

> **`sa_block_q` controls the query sequence tile size (default 512) loaded into on-chip TPU vector memory during the Splash Attention forward pass.**
