
## 1. The expert parallelism communication pattern problem

Expert parallelism (EP) shards experts across devices. In standard EP, dispatching tokens to their assigned experts requires an all-to-all collective — every device sends tokens to every other device that holds the relevant experts:

```text
Standard EP all-to-all:

device_0 → sends tokens to → device_0, device_1, device_2, device_3, ...
device_1 → sends tokens to → device_0, device_1, device_2, device_3, ...
...

Coordination required: O(N²) pairs, all at once
```

At scale, all-to-all is a bandwidth-intensive collective with high coordination overhead.

The "ring of experts" alternative:

```text
Ring-of-experts:

device_0 → device_1 → device_2 → ... → device_N → device_0

Each device processes its local experts, then passes remaining tokens to the next device
```

Tokens travel around the ring, hitting each device in sequence. Each device processes the tokens assigned to its experts, passes the rest forward. One full ring traversal = all tokens have visited all necessary experts.

---

## 2. What `use_ring_of_experts` controls

```yaml
use_ring_of_experts: false  # (default) use all-to-all dispatch
use_ring_of_experts: true   # use ring-based expert traversal
```

When `true`, MaxText replaces the all-to-all dispatch with a ring-passing pattern where tokens rotate through devices, each device handling its local experts.

---

## 3. Trade-offs

| Property | All-to-all | Ring of experts |
|---|---|---|
| Communication pattern | Single large collective | N smaller sequential passes |
| Latency | One collective (can be fast if bandwidth-saturating) | N passes × hop latency |
| Pipelinability | Hard to pipeline with compute | Natural fit for `num_moe_token_chunks` overlap |
| Implementation complexity | Simpler | More complex, but MaxText handles it |
| Scale efficiency | Can bottleneck at large EP degree | More predictable at scale |

---

## 4. The interaction with `num_moe_token_chunks`

Ring-of-experts is the natural host for token-chunk pipelining. The token batch splits into chunks; while one chunk is being processed by the current ring position, the next chunk is being dispatched to the next position:

```text
use_ring_of_experts=True + num_moe_token_chunks=4
→ chunk_0 compute ↔ chunk_1 ring-hop communication
→ effective hiding of ring communication latency
```

This combination is the full expert parallelism pipeline optimization.

---

## 5. Default

```yaml
use_ring_of_experts: false
```

The all-to-all path is the default and works well for most configurations. Switch to ring-of-experts when:
- EP degree is large and all-to-all is the bottleneck
- You want to exploit `num_moe_token_chunks` for overlap
- Network topology favors ring communication (e.g. TPU ICI)

---

## 6. Interaction with related parameters

| Related param | Interaction |
|---|---|
| `num_moe_token_chunks` | Only meaningful with `use_ring_of_experts=True` |
| `moe_chunk_barrier` | Debugging: forces sequential ring iterations |
| `use_ragged_sort` | Ragged-sort Pallas kernels support both ring and non-ring paths |
| `moe_dispatch_no_expert_sharding` | Only affects the dense (non-ring) path |

---

### One-line intuition

> **`use_ring_of_experts` replaces the all-to-all expert dispatch with a ring-passing pattern where tokens travel device-to-device — enabling natural pipelining via `num_moe_token_chunks` and potentially better scaling at large EP degrees.**
