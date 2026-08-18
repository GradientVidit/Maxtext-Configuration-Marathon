
## 1. Why does this exist?

Model parameters are the largest single tensor group in the network — for a 70B model, ~140GB in bf16. When HBM is exhausted by parameters + activations + optimizer state, the only remaining option is to not keep all of them on device at once.

`parameter_memory_host_offload` moves parameters to CPU host RAM. Unlike optimizer state offloading (which only transfers at optimizer steps), **parameters must be transferred every forward and backward pass** — this is an aggressive strategy with significant bandwidth cost.

---

## 2. When parameters need to be offloaded

The raw (unsharded) memory breakdown for a 70B model in bf16:

```text
Total un-sharded footprint = parameters + optimizer_state + activations + gradients

Large model (~70B), bf16:
  Parameters:      ~140 GB (bf16)
  Optimizer state: ~280 GB (fp32 Adam moments)
  Gradients:       ~140 GB (bf16)
```

With FSDP sharding across N devices, these quantities are divided by N. However, when total model size is huge relative to device count or sequence length is very long, parameter offloading provides the ultimate safety margin.

---

## 3. The bandwidth cost

With parameter offloading:

```text
Each forward pass:
  Host → Device: transfer parameters for each layer (or all at once)
  Compute layer
  Device → Host: (or keep on device temporarily if memory allows one layer at a time)

Each backward pass:
  Host → Device: transfer parameters again for gradient computation
```

This creates a memory-bandwidth bottleneck that didn't exist before. The TPU/GPU's compute sits partially idle waiting for parameter transfers. Whether the net effect is acceptable depends on the host→device bandwidth vs. layer compute time ratio.

---

## 4. Options

| Value | Behavior |
|---|---|
| `false` | Default — parameters live in device HBM |
| `true` | Parameters offloaded to host (CPU) RAM |

Default: `false`.

---

## 5. Comparison with optimizer offloading

| | `optimizer_memory_host_offload` | `parameter_memory_host_offload` |
|---|---|---|
| What's moved | Adam moments | Model weights |
| Transfer frequency | Once per optimizer step | Every forward + backward pass |
| Bandwidth cost | Low (optimizer step is infrequent) | High (every step) |
| Memory saving | Large (moments are 2× params in fp32) | Large (params are bf16) |
| Typical use | Common, recommended for large models | More aggressive, higher cost |

---

## 6. Interaction with `scan_layers`

When `scan_layers=true`, parameter offloading uses JAX sharding annotations (`memory_kind='pinned_host'`) to guide XLA parameter placement. While the stacked parameters live in CPU memory, XLA attempts to stage and prefetch parameter slices during scan execution. Always profile host↔device PCIe bandwidth when combining `scan_layers` with parameter offloading to ensure transfers hide behind computation.

---

## 7. When to enable

Enable only after:
1. `optimizer_memory_host_offload=true` is already set and still not enough
2. FSDP sharding is maximized and still not enough
3. Remat policy is at `'full'` and still not enough
4. You've confirmed HBM is the binding constraint (not compute throughput)

This is the last resort in the memory hierarchy, not the first.

---

### One-line intuition

> **`parameter_memory_host_offload` moves model weights themselves to CPU RAM — the most aggressive memory-saving option available, but it incurs host↔device parameter transfers on every forward and backward pass, making it the last resort after optimizer offloading and FSDP are exhausted.**
