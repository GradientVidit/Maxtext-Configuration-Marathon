## 1. Why does `dump_jaxpr` exist?

Before XLA optimizes computation graphs, JAX traces Python operations into **jaxpr** (JAX expressions)—the high-level typed intermediate representation.

When debugging whether a model architecture bug originates in the Python frontend (Flax/JAX transformations, scan loops, rematerialization decorators) or inside the XLA backend compiler, inspecting jaxpr isolates the frontend representation:

```text
Transformation Hierarchy:
Python Model (Flax) ──>[ Traced into jaxpr ] ──>[ Lowered to StableHLO ] ──>[ Compiled by XLA ]
                              │
                        dump_jaxpr: true
                              │
                              ▼
                 Serialized Text Dump of JAXPR
```

`dump_jaxpr` is the master switch to serialize and upload JAX expression graphs.

---

## 2. Fundamentals & Mechanics

When `dump_jaxpr: true`:
1. MaxText serializes the traced `jaxpr` representation of the training step.
2. Writes text files to `dump_jaxpr_local_dir`.
3. Uploads to `dump_jaxpr_gcs_dir`.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `false` | Jaxpr dumping disabled. |
| Enabled | `true` | Dumps jaxpr representation for frontend debugging. |

---

## 4. Interactions & Dependencies

```text
dump_jaxpr: true
     ├──> dump_jaxpr_local_dir
     ├──> dump_jaxpr_delete_local_after
     └──> dump_jaxpr_gcs_dir
```

---

## 5. Practical Scenarios & Failure Modes

- Inspecting whether `jax.lax.scan` or `remat` decorators properly structured the layer loop before compiler lowering.

---

### One-line intuition

> **`dump_jaxpr` enables serializing and uploading JAX expression (jaxpr) IR graphs to debug Python frontend tracing and model transformations.**
