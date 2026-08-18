
## 1. Why this exists

When `scan_layers=true`, all decoder layer parameters are **stacked** into a single array. For a transformer with weights `W` per layer and `L` layers, the stacked array has shape:

```text
(dim_0, ..., L, ..., dim_k)
       ↑
  param_scan_axis = the axis containing L
```

JAX's `lax.scan` needs to know which axis to scan over (slice along). `param_scan_axis` tells it exactly that.

---

## 2. Default: `1`

```yaml
param_scan_axis: 1
```

The stacked array has the layer dimension at index 1. This is MaxText's default layout for stacked parameters.

```text
shape: (fsdp_shards, num_layers, hidden_dim, ...)
              axis 0        axis 1
                         ↑
                    param_scan_axis = 1
```

---

## 3. What changing it does

If your stacked parameter layout differs — perhaps from a checkpoint produced by different code, or a custom sharding that places the layer axis elsewhere — `param_scan_axis` must match that layout.

Mismatching this value causes `lax.scan` to slice along the wrong axis, producing incorrect parameter shapes per layer, leading to shape errors or silently wrong computation.

---

## 4. When would you change it?

Rarely. The only scenarios:
- Loading a stacked checkpoint from external code that used a different axis convention
- Custom model implementations that stack layers along axis 0 instead of 1
- Debugging checkpoint compatibility between different MaxText versions

In practice, if you set `scan_layers=true` (default) and let MaxText produce the checkpoint, you never need to touch this.

---

## 5. Options

Integer — the index of the layer (scan) dimension in the stacked parameter array.

| Value | Meaning |
|---|---|
| `0` | Layer axis is the first array dimension |
| `1` | Layer axis is the second array dimension (default) |
| `N` | Layer axis is the Nth dimension |

Default: `1`.

---

## 6. Interaction with `scan_layers`

```yaml
scan_layers: false  → param_scan_axis is irrelevant (no stacking)
scan_layers: true   → param_scan_axis must match the stacked layout
```

No effect when `scan_layers=false`.

---

## 7. Checkpoint implications

The scan axis is recorded in checkpoint metadata so that when MaxText auto-detects `scan_layers` from a checkpoint on resume, it also recovers the correct axis. You generally don't need to set this manually even when switching between `scan_layers` values — unless you're bridging between checkpoints from different systems.

---

### One-line intuition

> **`param_scan_axis` tells `jax.lax.scan` which array axis corresponds to the layer dimension in the stacked parameter tensor — a low-level bookkeeping detail that defaults to `1` and only needs changing when loading checkpoints with a non-standard layer stacking convention.**
