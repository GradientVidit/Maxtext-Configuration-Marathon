## 1. Why does `dump_hlo_upload_all` exist?

In SPMD (Single Program, Multiple Data) distributed training, all hosts are supposed to compile identical HLO computation graphs.

However, subtle compiler bugs, environment discrepancies, or host-specific sharding logic can cause different hosts to compile divergent HLO programs, leading to desynchronization or performance degradation:

```text
SPMD Divergence Check:
Host 0 HLO Graph  ──┐
Host 1 HLO Graph  ──┼──>[ dump_hlo_upload_all: true ]──> Diff HLOs in GCS to detect compiler bugs
Host 2 HLO Graph  ──┘
```

`dump_hlo_upload_all` enables dumping and uploading HLO modules from all cluster hosts instead of host 0 alone.

---

## 2. Fundamentals & Mechanics

- **`false` (Default):** Only `jax.process_index() == 0` dumps and uploads HLO.
- **`true`:** Every host dumps and uploads its HLO graph, prefixing filenames with the process rank.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `false` | Dumps HLO only from host 0. |
| Multi-Host | `true` | Dumps and uploads HLO from all cluster processes. |

---

## 4. Interactions & Dependencies

- Active when `dump_hlo: true`.

---

## 5. Practical Scenarios & Failure Modes

- Use `dump_hlo_upload_all: true` when diagnosing mysterious collective deadlocks where rank 0 proceeds but worker ranks stall during execution.

---

### One-line intuition

> **`dump_hlo_upload_all` dumps and uploads HLO graphs from every host in the cluster to detect multi-host SPMD compilation divergence.**
