## 1. Why does it exist?

During the Splash Attention forward pass, each Query tile is multiplied against Key tokens to produce intermediate attention scores, which are then used to weight Value tokens.

To keep memory consumption within TPU Vector Memory (VMEM) limits, Key and Value sequences are streamed into VMEM in chunks of size `sa_block_kv`.

```text
For each Query Block (sa_block_q):
  Iterate over KV Blocks (sa_block_kv = 512):
    Load K_block, V_block ──→ Compute QK^T ──→ Update Softmax ──→ Accumulate Output
```

`sa_block_kv` specifies the tile block size along the Key and Value sequence dimensions loaded into memory in the Splash Attention forward pass.

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `512` (default) | Standard 512-token KV tile size for Splash Attention on TPUs. |
| Positive integer (e.g. `256`, `512`, `1024`) | Custom KV block size. |

Default in `base.yml`:
```yaml
sa_block_kv: 512
```

---

## 3. Interactions with Related Parameters

- **`sa_block_kv_compute`**: Sub-block compute tile size within each KV block.
- **`sa_block_q`**: Paired query tile size.

---

### One-line intuition

> **`sa_block_kv` defines the sequence tile size (default 512) for streaming Key and Value tensors into TPU on-chip memory during the forward attention loop.**
