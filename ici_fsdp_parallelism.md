## 1. Why does it exist?

Modern foundation models (e.g. Llama 3 70B/405B, Gemma 2 27B) have billions of parameters that require hundreds of gigabytes or terabytes of High Bandwidth Memory (HBM) for weights, master weights, and optimizer momentum/variance states. No single TPU chip has enough HBM to hold a 70B+ model in memory.

Fully Sharded Data Parallelism (FSDP / ZeRO-3) solves this by partitioning parameters, gradients, and optimizer states across all chips within the high-speed Inter-Chip Interconnect (ICI) domain:

```text
70B Model on 64 TPU Chips with ici_fsdp_parallelism: 64
  - Each chip stores only 1/64th of the parameters (~1.1B params).
  - Before forward pass of Layer N: High-speed ICI All-Gather fetches Layer N weights.
  - Compute forward pass: Matrix Multiply executes at full speed.
  - After forward pass: Discard gathered Layer N weights to free HBM!
```

`ici_fsdp_parallelism` sets the degree of Fully Sharded Data Parallelism within a single TPU slice, and is the **recommended auto-sharded ICI axis** in MaxText.

---

## 2. Fundamentals & Auto-Derivation (`-1`)

The product of all ICI axes must equal the total number of physical devices in a single slice:

$$\prod \text{all ICI axes} = \text{devices\_per\_slice}$$

When `ici_fsdp_parallelism: -1`, MaxText automatically allocates all remaining accelerator chips in the slice to FSDP:

$$\text{ici\_fsdp\_parallelism} = \frac{\text{devices\_per\_slice}}{\prod (\text{other ICI axes})}$$

```text
Example: Single v5e-256 slice (256 chips)
  ici_tensor_parallelism: 8
  ici_expert_parallelism: 4
  ici_fsdp_parallelism: -1  ──→ Auto-resolves to: 256 / (8 * 4) = 8
```

---

## 3. Options & Configuration

| Value | Meaning |
|---|---|
| `-1` (default) | Auto-derived from available chips per slice (Recommended standard). |
| Positive integer (e.g. `8`, `64`, `256`) | Explicit FSDP degree within the slice. |

Default in `base.yml`:
```yaml
ici_fsdp_parallelism: -1 # recommended ICI axis to be auto-sharded
```

---

## 4. Interactions with Related Parameters

- **`logical_axis_rules`**: Rules mapping embedding/MLP dimensions to `fsdp` and `fsdp_transpose` resolve against this axis.
- **`dcn_data_parallelism`**: Pairs with `ici_fsdp_parallelism` for multi-slice scaling (ICI FSDP within slice + DCN Data Parallelism across slices).
- **`dense_fsdp_use_two_stage_all_gather`**: Optimizes communication when combining FSDP with 2D transpose axes.

---

## 5. Practical Guidelines

- **Default Recommendation**: Always leave `ici_fsdp_parallelism: -1`. If you specify tensor parallelism (`ici_tensor_parallelism: 4`) or expert parallelism (`ici_expert_parallelism: 8`), FSDP will automatically absorb the remaining mesh capacity.

---

### One-line intuition

> **`ici_fsdp_parallelism` shards model parameters, gradients, and optimizer states across the high-speed chip interconnect within a TPU slice, serving as MaxText's default auto-derived ICI axis.**
