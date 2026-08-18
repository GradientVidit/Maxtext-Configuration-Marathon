## 1. Why it exists: injecting external experiment metadata into benchmark records

When running automated benchmarks across hardware generations, different compiler versions (XLA/JAX), or deployment topologies, the benchmark harness itself only knows internal runtime parameters (e.g. step time, batch size):

```text
Benchmark Execution Data:
{ "ttft_ms": 14.2, "tps": 1240.5, "prefill_length": 512 }
                             +
External Context Data (inference_metadata_file):
{ "accelerator": "v6e-8", "topology": "2x4", "git_commit": "a1b2c3d", "model_version": "v1.2-fp8" }
                             │
                             ▼
Complete Benchmark Artifact (Exported to JSON / Data Warehouse):
{
  "accelerator": "v6e-8",
  "topology": "2x4",
  "git_commit": "a1b2c3d",
  "model_version": "v1.2-fp8",
  "ttft_ms": 14.2,
  "tps": 1240.5,
  "prefill_length": 512
}
```

Without an external metadata injection mechanism, tracking external variables—such as cluster ID, host VM OS version, quantization calibration recipe, or upstream pipeline ID—requires custom ad-hoc scripting or manual post-processing of log files.

`inference_metadata_file` specifies a path to a JSON file containing user-defined metadata that is parsed and merged into the benchmark record output.

---

## 2. Mechanics: file loading and schema merging

At startup of the benchmark runner:

```text
 1. Read JSON file at `inference_metadata_file`
    ┌────────────────────────────────────────────────┐
    │ { "test_environment": "nightly_ci", ... }     │
    └───────────────────────┬────────────────────────┘
                            │
 2. Run Microbenchmark Sweep Loop
    ┌────────────────────────────────────────────────┐
    │ Execute prefill & decode passes, measure stats │
    └───────────────────────┬────────────────────────┘
                            │
 3. Merge & Serialize
    ┌────────────────────────────────────────────────┐
    │ output_dict = {**metadata_dict, **perf_stats}  │
    │ Write to `inference_microbenchmark_log_file`   │
    └────────────────────────────────────────────────┘
```

1. MaxText validates and parses the JSON file at `inference_metadata_file`.
2. The key-value pairs are stored in the runtime configuration context.
3. Upon completion of the benchmark runs, the external dictionary is merged directly into the final performance summary payload.

---

## 3. Options & Configuration

Default in `base.yml`:

```yaml
inference_metadata_file: ""
```

| Value | Behavior | Example |
|---|---|---|
| Empty string `""` (default) | No external metadata is loaded or merged into the output log. | `""` |
| Local JSON filepath | Loads and parses JSON from local disk on host 0. | `"/tmp/ci_metadata.json"` |
| GCS JSON path | Loads external metadata directly from a Google Cloud Storage bucket. | `"gs://my-bucket/benchmarks/metadata/run_42.json"` |

### Example Metadata JSON Payload:
```json
{
  "pipeline_id": "nightly-maxtext-2026-08-18",
  "git_sha": "f78a2e1",
  "tpu_cluster": "us-central2-b-v6e-slice-128",
  "quantization_scheme": "int8_kvcache_fp8_weights",
  "benchmark_owner": "ml-infra-team"
}
```

---

## 4. Parameter Interactions

```text
┌───────────────────────────────────────────────────────────┐
│                  inference_metadata_file                  │
└─────────────┬───────────────────────────────┬─────────────┘
              │                               │
              ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│inference_microbenchmark_  │   │log_config                 │
│log_file_path              │   │Logs config details        │
│Receives the merged        │   │including the loaded       │
│metadata + perf dict.      │   │metadata path.             │
└───────────────────────────┘   └───────────────────────────┘
```

- **`inference_microbenchmark_log_file_path`**: The destination where the merged metadata + performance metrics dictionary is written. If no log file path is specified, the merged dictionary is printed to `stdout`.

---

## 5. Practical Scenarios & Failure Modes

### Automated Nightly Regression Tracking
In automated performance tracking pipelines, CI systems generate a small JSON file containing build metadata (compiler flags, PR number, hardware node ID) before launching the benchmark. Passing `inference_metadata_file` ensures all downstream BigQuery or database ingestion pipelines have full provenance attached to each latency measurement.

### What breaks if misconfigured:
- **Malformed JSON**: If the target file contains invalid JSON syntax, the benchmark process fails during initialization with a `json.JSONDecodeError`.
- **Missing File**: Specifying a non-existent filepath causes a `FileNotFoundError` during setup before any benchmark runs execute.

---

### One-line intuition

> **`inference_metadata_file` supplies a path to a JSON file of external contextual metadata (e.g., git commit, cluster topology, dataset ID) that MaxText merges directly into the final benchmark output for end-to-end provenance.**
