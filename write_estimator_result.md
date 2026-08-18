## 1. Why does `write_estimator_result` exist?

Before launching costly training runs on thousands of TPU chips, ML engineers need to estimate theoretical Model FLOPs Utilization (MFU), memory bandwidth requirements, execution latency, and financial training cost.

MaxText includes an analytical **Estimator tool** that computes theoretical forward/backward FLOPs, memory footprint, and estimated step times based on model dimensions and hardware specs:

```text
Model Configuration (Dims, Layers, Sharding) + Hardware Specs (TFLOPS, HBM Bandwidth)
                                     │
                                     ▼ (MaxText Estimator)
                    Theoretical MFU, Latency & Cost Breakdown
                                     │
                                     ▼ (write_estimator_result: true)
                         Saved JSON / Text Metrics Artifact
```

`write_estimator_result` controls whether the results of this estimator analysis are written out to persistent storage.

---

## 2. What it actually controls

```yaml
write_estimator_result: false
```

- When `false` (default): Estimator metrics are printed to stdout or suppressed during standard runs.
- When `true`: MaxText exports the structured analytical estimation results (FLOP counts, memory bounds, expected throughput) into a file in `base_output_directory`.

---

## 3. Options and Defaults

| Value | Behavior | Output Location |
|---|---|---|
| `false` (default) | Does not write estimator file to disk | None |
| `true` | Writes detailed estimation JSON/summary | `<base_output_directory>/estimator_result.json` |

---

## 4. Interactions

- **`cost_estimate_flops_fwd`, `cost_estimate_flops_bwd`**: The estimator uses analytical or user-specified FLOPs formulas.
- **`base_output_directory`**: Specifies the destination path for estimator output.

---

## 5. Practical Scenarios

- **Pre-flight Cluster Sizing**: Enable `write_estimator_result: true` during capacity planning to document theoretical vs actual MFU metrics across cluster scale iterations.

---

### One-line intuition

> **`write_estimator_result` writes the MaxText analytical cost, throughput, and FLOP estimation report to disk for capacity planning.**
