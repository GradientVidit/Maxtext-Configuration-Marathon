
## 1. Why does this exist?

Optimizer state (Adam's first and second moments, potentially EMA buffers) is typically 2–3× the size of the model parameters in bytes. For a 70B model in fp32, that's ~800GB+ of optimizer state alone — impossible to hold in HBM at large scale.

Two standard solutions exist: shard optimizer state across devices (ZeRO-1), or **move it off the accelerator entirely**. `optimizer_memory_host_offload` takes the second approach.

```text
Without offload (unsharded baseline):
  TPU/GPU HBM holds: parameters (bf16) + optimizer state (fp32 moments) + activations + gradients
  Typical split: ~33% params, ~66% optimizer state (with remat_policy='full' keeping activations minimal)

With offload:
  Host RAM (CPU) holds: optimizer state (Adam 1st and 2nd moments)
  TPU/GPU HBM holds: parameters + activations + gradients (no optimizer state)
```

---

## 2. How it works mechanically

JAX/XLA supports host memory arrays via `jax.device_put(..., jax.devices('cpu')[0])` or through sharded pytree placement. When optimizer state is offloaded:

1. **Forward/backward**: optimizer state stays on host, not visible to accelerator computation
2. **Optimizer step**: optimizer state is brought to device, update computed, sent back to host
3. **Next step**: device proceeds with updated parameters; optimizer state lives on host until next update

The transfer happens at the optimizer step boundary — it's not free, but the step frequency is low relative to forward/backward compute.

---

## 3. The cost

- **Host↔Device PCIe bandwidth**: transferring optimizer states every step consumes host-to-device interconnect bandwidth.
- **Step latency**: the optimizer step itself is slower waiting for host↔device transfers.
- **Async overlap**: transfers can be partially overlapped with gradient accumulation or compute, depending on XLA scheduling and host system setup.

For large models where optimizer state simply doesn't fit in HBM, this cost is irrelevant — the alternative is OOM.

---

## 4. Options

| Value | Behavior |
|---|---|
| `false` | Default — optimizer state lives in device HBM |
| `true` | Optimizer state offloaded to host (CPU) RAM |

Default: `false`.

---

## 5. Interaction with `parameter_memory_host_offload`

These are two separate offloading knobs:

```text
optimizer_memory_host_offload: offloads Adam moments and optimizer state
parameter_memory_host_offload: offloads model parameters themselves
```

Offloading parameters is more aggressive and unusual (parameters must be transferred every forward pass, not just at optimizer steps). Offloading optimizer state is far more common and the standard approach for fitting large models.

Pairing both:
```yaml
optimizer_memory_host_offload: true
parameter_memory_host_offload: true
```
maximally reduces HBM requirements but incurs substantial host↔device traffic on every step.

---

## 6. Interaction with pipeline parallelism

MaxText's comments on `pipeline_fsdp_ag_once` mention optimizer offloading as an alternative to FSDP within pipeline stages:

> "An alternative to setting this to true may be to replace any FSDP with DP and use optimizer offloading if necessary."

Meaning: if per-stage FSDP weight communication is expensive, consider replacing FSDP with replicated DP and using optimizer offloading to recover the HBM that FSDP sharding used to free.

---

## 7. When to enable

- Running out of HBM and optimizer state is the culprit (verify via profiler — optimizer state is usually the largest HBM consumer after activations)
- Model is large enough that FSDP + offloading combined is the only way to fit
- Host RAM is abundant (TPU VMs typically have 192GB+ host RAM vs. HBM limits)

---

### One-line intuition

> **`optimizer_memory_host_offload` moves Adam's first and second moments to CPU RAM, freeing the single largest category of HBM consumers — the right lever when optimizer state (not parameters or activations) is what's causing OOM.**
