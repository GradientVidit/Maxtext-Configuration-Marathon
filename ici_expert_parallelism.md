## 1. Why does it exist?

In Mixture of Experts (MoE) models (e.g. Mixtral 8x7B, DeepSeek-V2/V3), each layer has multiple distinct feed-forward expert networks.

Expert Parallelism (EP) assigns different experts to different physical chips within a slice. During the forward pass, router gating scores assign each token to its top-$k$ experts. An **All-to-All collective** dispatches tokens across the ICI interconnect to the destination chips holding the assigned experts, and a second All-to-All routes the expert outputs back.

```text
Tokens on Device 0 ──┐
Tokens on Device 1 ──┼──[ High-Speed ICI All-to-All ]──→ Device 0: Expert 0 & 1
Tokens on Device 2 ──┤                                    Device 1: Expert 2 & 3
Tokens on Device 3 ──┘                                    Device 2: Expert 4 & 5
                                                          Device 3: Expert 6 & 7
```

`ici_expert_parallelism` specifies the number of expert-parallel partitions across chips within a single TPU slice over the fast Inter-Chip Interconnect.

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `1` (default) | MoE experts are not partitioned along a dedicated EP axis (they are sharded via FSDP or replicated). |
| Integer $> 1$ (e.g. `8`, `16`, `32`, `64`) | Distributes experts across `N` chips in the slice. |

Default in `base.yml`:
```yaml
ici_expert_parallelism: 1
```

---

## 3. Divisibility & Hardware Alignment

- **Divisibility**: `num_experts` should ideally be an integer multiple of `ici_expert_parallelism` so each device hosts an equal number of experts:
  $$\text{Experts per device} = \frac{\text{num\_experts}}{\text{ici\_expert\_parallelism}}$$
- **Interconnect Efficiency**: Because All-to-All exchanges token embeddings across all EP ranks, keeping `ici_expert_parallelism` within a fast ICI torus provides maximum communication throughput.

---

## 4. Interactions with Related Parameters

- **`num_experts`**: Total number of experts in the model.
- **`use_ring_of_experts`**: Replaces the All-to-All collective with a circular ring shift pattern.
- **`check_vma`**: Supported for EP/FSDP ICI parallelisms.

---

### One-line intuition

> **`ici_expert_parallelism` places different subsets of MoE experts on different chips within a TPU slice, using ultra-fast all-to-all collectives to route tokens to their target experts.**
