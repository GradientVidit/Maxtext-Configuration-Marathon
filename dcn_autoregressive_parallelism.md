## 1. Why does it exist?

The `autoregressive` mesh axis is designed for specialized generation and decoding pipelines where tokens are produced sequentially one token at a time.

Because autoregressive token generation involves tiny, latency-critical operations repeated thousands of times in a loop, distributing these operations over high-latency Data Center Network (DCN) connections drastically inflates time-per-token (TPOT).

`dcn_autoregressive_parallelism` is marked as **never recommended** in `base.yml` for values $> 1$.

```text
Autoregressive Decode Step (Latency Critical):
  Token Generation ──→ Millisecond DCN roundtrips destroy generation throughput.
```

---

## 2. Options & Configuration

| Value | Status | Meaning |
|---|---|---|
| `1` (default) | **Recommended** | Autoregressive decoding parallelism remains local to each slice over ICI. |
| $> 1$ | **Never Recommended** | Distributes decoding operations across slices over DCN. |

Default in `base.yml`:
```yaml
dcn_autoregressive_parallelism: 1 # never recommended
```

---

### One-line intuition

> **`dcn_autoregressive_parallelism` configures autoregressive generation parallelism over the datacenter network — defaults to `1` and should never be scaled across DCN due to severe token-generation latency.**
