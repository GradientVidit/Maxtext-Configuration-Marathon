## 1. Why does it exist?

Causal autoregressive sequence models process tokens where each position $t$ can only attend to past tokens $0 \dots t$. During autoregressive generation, distributing causal sequence attention across physical chips requires specialized causal communication topologies.

`ici_context_autoregressive_parallelism` configures the degree of causal autoregressive context parallelism within a single TPU slice over the fast Inter-Chip Interconnect.

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `1` (default) | Causal context parallelism disabled (standard context parallelism used instead). |
| Integer $> 1$ | Allocates `N` chips to autoregressive causal sequence partitioning within the slice. |

Default in `base.yml`:
```yaml
ici_context_autoregressive_parallelism: 1
```

---

### One-line intuition

> **`ici_context_autoregressive_parallelism` configures intra-slice causal sequence parallelism for autoregressive generation over high-speed chip interconnects.**
