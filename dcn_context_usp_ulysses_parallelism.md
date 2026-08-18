## 1. Why does it exist?

Unified Sequence Parallelism (USP) splits the sequence parallel dimension into a 2D mesh: a Ring Attention dimension (`context`) and an All-to-All head-exchange Ulysses dimension (`context_usp_ulysses`).

When scaling USP across multi-slice clusters, `dcn_context_usp_ulysses_parallelism` specifies the size of the Ulysses All-to-All head-exchange axis operating over the Data Center Network.

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `1` (default) | Ulysses head-exchange All-to-All is confined within slices over ICI. |
| Integer $> 1$ | Extends the Ulysses all-to-all dimension across `N` slices. |

Default in `base.yml`:
```yaml
dcn_context_usp_ulysses_parallelism: 1
```

---

## 3. Interactions with Related Parameters

- **`context_parallel_strategy: "usp"`**: Required to activate the Ulysses dimension.
- **`ulysses_context_sharding`**: Defines the physical mesh axis mapping.
- **`ici_context_usp_ulysses_parallelism`**: Configures the intra-slice Ulysses axis.

---

### One-line intuition

> **`dcn_context_usp_ulysses_parallelism` sets the cross-slice datacenter network degree for the all-to-all head-exchange dimension in Unified Sequence Parallelism.**
