## 1. Why it exists: programmatic capture of inference performance metrics

During inference optimization and automated regression testing, printing benchmark results solely to `stdout` is insufficient for automated analysis, regression tracking, and dashboard ingestion:

```text
Without Log File (Ephemeral):
[Microbenchmark Run] ──> stdout / Console logs ──> Lost when terminal closes or pod dies

With Log File (Structured Persistence):
[Microbenchmark Run] ──> Structured Output (JSON / CSV) ──> GCS / Local Storage
                                                                   │
                                                                   ▼
                                                     Regression Dashboards /
                                                     Automated Perf Assertions
```

When profiling multiple model architectures, hardware generations (e.g. TPU v5e vs v5p vs v6e), or sharding configurations, performance engineers need a dedicated target file where detailed timing breakdown (TTFT, ITL, throughput, per-iteration stats) is serialized in a structured, parseable format.

`inference_microbenchmark_log_file_path` defines the exact destination file path where microbenchmark results are saved.

---

## 2. Mechanics: logging pipeline and payload serialization

When `inference_microbenchmark_log_file_path` is specified, MaxText collects performance telemetry after all stage iterations complete and writes structured data to the designated path:

```text
 Microbenchmark Loop Runs
 ├── Collect Prefill Latencies [ms]
 ├── Collect Decode Step Latencies [ms]
 ├── Calculate TFLOP/s & Tokens/sec
 └── Extract Memory Footprint
                │
                ▼
 Format: Structured JSON / Key-Value Metrics
                │
                ▼
 Write to `inference_microbenchmark_log_file_path`
 (Local disk or GCS bucket `gs://...`)
```

### Metrics captured in the log output:
- **Prefill metrics**: Prompt length, batch size, average TTFT (ms), prefill TFLOP/s, prefill throughput (tokens/sec).
- **Decode metrics**: Context length, batch size, average inter-token latency (ms/token), generation throughput (tokens/sec).
- **Environment metadata**: Device type, device mesh shape, precision (`bfloat16`/`fp8`), timestamp, and run configuration.

---

## 3. Options & Configuration

Default in `base.yml`:

```yaml
inference_microbenchmark_log_file_path: ""
```

| Path Format | Example | Behavior |
|---|---|---|
| Empty string `""` (default) | `""` | Logs results exclusively to `stdout`/console. No persistent file is written. |
| Local filesystem path | `"/tmp/benchmarks/llama3_70b_v5p.json"` | Writes structured benchmark metrics to a local file on host 0. |
| Google Cloud Storage (GCS) path | `"gs://my-bucket/maxtext/perf_logs/v6e_sweep.json"` | Streams the formatted benchmark output directly to cloud storage for persistent multi-node tracking. |

---

## 4. Parameter Interactions

```text
┌───────────────────────────────────────────────────────────┐
│          inference_microbenchmark_log_file_path           │
└─────────────┬───────────────────────────────┬─────────────┘
              │                               │
              ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│inference_metadata_file    │   │inference_microbenchmark_  │
│Supplies static workload   │   │stages / prefill_lengths   │
│metadata merged into the   │   │Determines the schema and  │
│output file.               │   │content rows written.      │
└───────────────────────────┘   └───────────────────────────┘
```

- **`inference_metadata_file`**: If provided, custom metadata from this JSON file (e.g. Git commit hash, dataset name, cluster topology) is merged directly into the benchmark log output alongside performance measurements.
- **`log_config`**: Independent flag; `log_config` prints the parsed configuration parameters at startup, whereas `inference_microbenchmark_log_file_path` writes the measured execution results.

---

## 5. Practical Scenarios & Failure Modes

### CI/CD Performance Regression Testing
In continuous integration pipelines, automated benchmarks run on new MaxText commits:
```bash
python3 src/maxtext/inference_microbenchmark.py src/maxtext/configs/base.yml \
  model_name=llama3-8b \
  inference_microbenchmark_log_file_path=/artifacts/perf_results.json
```
A downstream CI step parses `/artifacts/perf_results.json` and fails the build if throughput drops by >3% compared to baseline.

### What breaks if misconfigured:
- **Invalid path permissions**: If the specified directory does not exist or lacks write permissions, the benchmark will crash at the end of the run during file I/O.
- **GCS authentication failures**: If specifying a `gs://` URI without proper Google Cloud IAM storage write permissions on the host, upload will fail.

---

### One-line intuition

> **`inference_microbenchmark_log_file_path` specifies the local or GCS file destination for serializing structured benchmark metrics (TTFT, TPS, TFLOPs), enabling automated regression tracking and performance dashboards.**
