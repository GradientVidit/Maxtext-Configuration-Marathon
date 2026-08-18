## 1. Why does it exist?

During autoregressive generation and causal sequence processing, sequence tokens exhibit strict step-by-step causal dependencies. The `context_autoregressive` mesh axis is reserved for causal-specific context parallelism implementations.

`dcn_context_autoregressive_parallelism` sets the size of this causal context-parallel dimension across multi-slice clusters over the Data Center Network.

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `1` (default) | Causal context parallelism is not distributed across slices over DCN. |
| Integer $> 1$ | Extends causal context parallelism across `N` slices. |

Default in `base.yml`:
```yaml
dcn_context_autoregressive_parallelism: 1
```

---

## 3. Interactions with Related Parameters

- **`ici_context_autoregressive_parallelism`**: Configures the corresponding intra-slice causal axis.
- **`mesh_axes`**: Registered as `'context_autoregressive'`.

---

### One-line intuition

> **`dcn_context_autoregressive_parallelism` configures the cross-slice datacenter network degree for causal autoregressive sequence parallelism.**
