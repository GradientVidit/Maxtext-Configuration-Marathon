## 1. Why does `max_segments_per_seq` exist?

When sequence packing is active (`packing: true`), dozens of small documents can be packed into a single sequence of length `max_target_length`. 

In GPU training with NVIDIA's **TransformerEngine** (`DotProductAttention`), the underlying fused FlashAttention / cuDNN C++ kernels allocate fixed-size auxiliary memory buffers to store sequence offsets and segment metadata.

```text
Packed Sequence: [ Doc 1 | Doc 2 | Doc 3 | ... | Doc 64 ] (64 segments packed!)
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
    max_segments_per_seq: -1 (Unbounded)     max_segments_per_seq: 32 (Cap)
                 │                                       │
     TransformerEngine Kernel Overflow       TransformerEngine Buffer Allocated
                 ▼                                       ▼
    💥 Silent Memory Corruption / Crash       ✅ Safe, Validated Execution
```

If the number of packed segments exceeds TransformerEngine's internal buffer, it leads to **silent numerical corruption or kernel crashes**. `max_segments_per_seq` caps the maximum number of packed segments permitted per sequence.

---

## 2. Mechanics & Grain Enforcement

When `max_segments_per_seq` is set (e.g. `32`), Grain's packing algorithms (`_grain_data_processing.py`) will stop adding new documents to the current sequence once the segment count reaches this limit, even if unused token slots remain.

```text
Grain Packing Loop:
  if len(current_segments) >= max_segments_per_seq:
      finalize_sequence_and_pad()
```

---

## 3. Options & Default

| Parameter | Type | Default | Recommended GPU Value |
| :--- | :--- | :--- | :--- |
| `max_segments_per_seq` | `int` | `-1` | `32` (when training on GPUs with TransformerEngine) |

- `-1`: Unbounded (standard on TPUs using Pallas / native JAX attention).
- Positive integer (e.g. `32`): Explicit segment ceiling for GPU safety.

---

## 4. Interactions with Related Parameters

- **`packing`**: Only active when `packing: true`.
- **`attention: cudnn_flash_te`**: Fused cuDNN attention from TransformerEngine requires this constraint.
- **`grain_packing_type`**: Enforced during Grain segment assembly.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **GPU training with TransformerEngine crashes randomly on short text** | CUDA illegal memory access or NaN loss caused by segment metadata overflow | Set `max_segments_per_seq: 32`. |
| **TPU pretraining with Pallas / SplashAttention** | TPU kernels handle arbitrary segment counts dynamically | Keep default `-1` for maximum packing density. |

---

### One-line intuition

> `max_segments_per_seq` sets an upper bound on how many documents can be packed into a single sequence, preventing memory corruption and kernel crashes in TransformerEngine on GPUs.
