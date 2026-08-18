## 1. Why does `dump_hlo` exist?

JAX code lowers through multiple intermediate representations before executing on TPU/GPU hardware:

$$\text{Python} \longrightarrow \text{jaxpr} \longrightarrow \text{StableHLO} \longrightarrow \text{XLA HLO IR} \longrightarrow \text{Hardware Binary}$$

When compiler fusions fail, collective communications are scheduled inefficiently, or memory layouts cause excessive transpose copies, developers must inspect the actual compiled XLA High-Level Optimizer (HLO) instructions:

```text
Compilation Pipeline:
MaxText Python Code
        │
        ▼
XLA Compiler Optimization Passes
        │
   dump_hlo: true ?
        │
   ┌────┴────┐
   ▼         ▼
 [No]      [Yes]
         Serialize HLO Text (.txt) & Proto (.pb)
         Save to Local & Upload to GCS
```

`dump_hlo` serves as the master switch to trigger dumping and uploading of XLA HLO module representations.

---

## 2. Fundamentals & Mechanics

When `dump_hlo: true`:
1. MaxText configures XLA dumping environment flags (`XLA_FLAGS`).
2. The XLA compiler writes serialized HLO modules (before and after compiler optimizations) to `dump_hlo_local_dir`.
3. Filtered module files matching `dump_hlo_module_name` are uploaded to `dump_hlo_gcs_dir`.
4. If `dump_hlo_delete_local_after: true`, local files are purged to save host disk space.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `false` | HLO dumping disabled (standard training). |
| Debugging | `true` | Dumps XLA HLO IR graphs for compiler inspection. |

---

## 4. Interactions & Dependencies

```text
dump_hlo: true
    │
    ├──> dump_step (Step selector)
    ├──> dump_hlo_local_dir & dump_hlo_delete_local_after
    ├──> dump_hlo_gcs_dir
    ├──> dump_hlo_local_module_name & dump_hlo_module_name
    ├──> dump_hlo_xla_flags
    └──> dump_hlo_upload_all
```

---

## 5. Practical Scenarios & Failure Modes

- **Inspecting Sharding & Fusions:** Enable `dump_hlo: true` to verify whether FSDP all-gathers and attention projections were fused as expected by the XLA compiler.

---

### One-line intuition

> **`dump_hlo` acts as the master toggle for serializing and uploading XLA HLO compiler intermediate representations for low-level performance analysis.**
