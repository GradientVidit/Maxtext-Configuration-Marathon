
## 1. The traditional checkpoint file layout problem

Before OCDBT, each JAX array (or array shard) was stored as a separate Zarr file:

```text
checkpoint/
  params/
    embedding/
      0.0    (file: chunk 0,0 of embedding)
      0.1    (file: chunk 0,1 of embedding)
    attention/
      query/
        0.0.0  (file)
        0.0.1  (file)
        ...
```

For a large model across many hosts, this produces **millions of small files** on GCS. GCS metadata operations (LIST, GET, PUT) have fixed per-operation latency regardless of file size — millions of files means millions of slow metadata calls.

---

## 2. What OCDBT is

**OCDBT = Optionally-Cooperative Distributed B-Tree**

It's a storage format built on top of TensorStore that fundamentally changes how checkpoint data is organized on disk/cloud.

### How it works:

**Phase 1 — Uncoordinated parallel writes**

Each host writes its array shards independently into its own local manifest directory:

```text
checkpoint/ocdbt.process_0/  ← host 0 writes here
checkpoint/ocdbt.process_1/  ← host 1 writes here
checkpoint/ocdbt.process_2/  ← host 2 writes here
```

No coordination needed during writing — hosts never wait for each other.

**Phase 2 — Atomic manifest merge**

Once all hosts finish writing, the per-process B-tree structures are merged into a single global manifest. This merge is **atomic** — readers either see a fully-committed checkpoint or nothing.

```text
ocdbt.process_0/ + ocdbt.process_1/ + ... → global manifest
```

---

## 3. Key benefits

| Property | Without OCDBT | With OCDBT |
|---|---|---|
| Write coordination | Hosts contend for shared files | Each host writes independently |
| File count | Millions of small files | Far fewer, larger data files |
| Read performance | Many GCS GET calls | Few large reads |
| Atomicity | Partial checkpoint possible | Atomic — all-or-nothing |
| Load speed at scale | Degrades with file count | Scales better |

---

## 4. "Optionally cooperative" — what that means

The "optionally cooperative" part refers to the fact that:
- The **write phase** is fully uncoordinated (optional cooperation — hosts don't need to cooperate)
- The **merge phase** requires coordination (but it's brief and happens once)

This is why it scales: N hosts each write independently, then a single merge step consolidates.

---

## 5. Default and recommendation

```yaml
checkpoint_storage_use_ocdbt: true
```

MaxText defaults this to `true`. It's the recommended production setting. The only reason to disable it:
- Compatibility with a tool/reader that doesn't support the OCDBT format
- Debugging/inspecting checkpoints with older Zarr-only tooling

---

## 6. OCDBT + Zarr3

These are complementary:

```text
OCDBT  → controls how the B-tree index and data files are structured
Zarr3  → controls the array chunk format and codec within those files
```

Both enabled together = the recommended, highest-performance checkpoint configuration.

---

### One-line intuition

> **`checkpoint_storage_use_ocdbt` switches from a "one-file-per-array-shard" layout to a B-tree-indexed format where hosts write independently then merge atomically — dramatically reducing GCS file count and improving checkpoint I/O throughput at scale.**
