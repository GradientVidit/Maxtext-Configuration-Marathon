## 1. Why does it exist?

Context Parallelism (CP) splits input sequence lengths across devices to enable training on long context windows (e.g. 32k to 1M+ tokens) that would otherwise exceed single-device HBM memory.

In context-parallel attention (such as Ring Attention or All-Gather attention), devices must continuously circulate or gather Key/Value activation chunks for every layer. Extending context parallelism across the Data Center Network (`dcn_context_parallelism`) allows sequence lengths to scale across multi-slice clusters.

```text
Slice 0 (Tokens 0..128k) ◄──[ DCN KV Ring / All-Gather ]──► Slice 1 (Tokens 128k..256k)
```

`dcn_context_parallelism` sets the degree of context parallelism across multi-slice clusters over the Data Center Network.

---

## 2. Options & Configuration

| Value | Behavior |
|---|---|
| `1` (default) | Context parallelism is confined within each slice over fast ICI links. |
| Integer $> 1$ | Extends context parallelism across `N` slices to support extreme context lengths. |

Default in `base.yml`:
```yaml
dcn_context_parallelism: 1
```

---

## 3. Interactions with Related Parameters

- **`context_parallel_strategy`**: Determines the collective algorithm used across DCN (`"all_gather"`, `"ring"`, `"ulysses"`, or `"usp"`).
- **`context_parallel_load_balance`**: Rebalances causal mask compute across DCN context chunks.
- **`ici_context_parallelism`**: Configures the intra-slice context degree.

---

### One-line intuition

> **`dcn_context_parallelism` splits input sequence lengths across multiple TPU slices over the datacenter network to scale context windows beyond single-slice memory capacity.**
