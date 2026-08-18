## 1. Why does it exist?

In 2D-FSDP, weight matrices are sharded across both the `fsdp` and `fsdp_transpose` physical mesh axes (forming a 2D grid of accelerator devices).

Before executing the matrix multiplication in each dense layer, the weight matrix must be gathered from all participating devices. Gathering across a 2D mesh in a single flat all-gather collective causes devices to contend across both dimensions of the interconnect simultaneously.

Splitting the weight gathering into two sequential stages (first gathering along `fsdp_transpose`, then gathering along `fsdp`) allows XLA to align communications with the physical 2D/3D torus ring topologies of TPU pods. MaxText inserts optimization barriers to prevent XLA from inadvertently re-combining the two separate operations into a single flat collective.

```text
Flat 2D All-Gather:
  All devices in 2D grid contend on global interconnect simultaneously.

Two-Stage All-Gather (dense_fsdp_use_two_stage_all_gather: true):
  Stage 1: Explicit All-Gather along local row (fsdp_transpose ring)
  Stage 2: Explicit All-Gather along local column (fsdp ring)
  ──→ Clean, contention-free communication along physical hardware rings.
```

`dense_fsdp_use_two_stage_all_gather` configures MaxText to issue two separate sequential all-gather calls for dense MLP weights sharded across both FSDP and FSDP-transpose axes.

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `false` (default) | Single-stage all-gather collective. |
| `true` | Issues two-stage sequential all-gather calls with barriers to prevent XLA collective fusion. |

Default in `base.yml`:
```yaml
dense_fsdp_use_two_stage_all_gather: false
```

---

## 3. MoE Counterpart

This parameter is the dense-layer counterpart to `moe_fsdp_use_two_stage_all_gather` (which handles 2D-sharded MoE expert weights in `src/maxtext/layers/moe.py`).

---

### One-line intuition

> **`dense_fsdp_use_two_stage_all_gather` decomposes 2D-FSDP dense MLP weight gathering into two sequential all-gathers, preventing XLA from merging them into a single contention-heavy collective.**
