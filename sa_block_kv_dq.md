## 1. Why does it exist?

While computing Query gradients ($dQ$), each stationary Query tile must attend to the full Key/Value sequence history.

`sa_block_kv_dq` specifies the tile block size of Key/Value tokens streamed into on-chip memory during the $dQ$ backward pass.

```text
Held in VMEM: Query Block (sa_block_q_dq)
  Stream KV Blocks in chunks of `sa_block_kv_dq` (512) ──→ Accumulate dQ
```

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `512` (default) | Standard 512-token KV streaming chunk size for $dQ$ backward evaluations. |
| Positive integer (e.g. `256`, `512`) | Custom KV block size. |

Default in `base.yml`:
```yaml
sa_block_kv_dq: 512
```

---

### One-line intuition

> **`sa_block_kv_dq` sets the Key/Value sequence tile size (default 512) streamed during the backward pass to evaluate Query gradients ($dQ$).**
