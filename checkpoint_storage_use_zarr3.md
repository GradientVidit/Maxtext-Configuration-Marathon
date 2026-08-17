
## 1. What Zarr is

Zarr is an open format for storing chunked, compressed N-dimensional arrays. TensorStore (which Orbax uses) natively supports both Zarr v2 and Zarr v3 as storage backends for model parameters.

In checkpoint terms:
- Each model parameter array gets stored as a Zarr array
- The array is split into chunks
- Each chunk is stored (possibly compressed) as an object in GCS/local storage

---

## 2. Zarr v2 limitations

Zarr v2's spec has a fundamental issue for cloud-native storage:

```text
chunk size = physical file size
```

If you use small chunks (for parallelism), you get many small files. If you use large chunks (to reduce file count), you lose fine-grained parallel loading. You're forced to choose.

This is the "small file problem" — GCS object metadata operations dominate over actual data transfer time when files are small.

---

## 3. What Zarr v3 adds

Zarr v3's key innovation for checkpointing is **chunk sharding**:

```text
Zarr v2: 1 chunk = 1 file
Zarr v3: many chunks → packed into 1 "shard" (physical file)
```

The logical chunk size (which determines parallelism granularity for reading) is **decoupled** from the physical file size.

```text
logical chunk: 4MB  (fine-grained parallel access)
shard size:    2GB  (few large physical files)
```

You get both:
- High parallelism (many logical chunks can be read independently)
- Low file count (chunks are packed into shards)

---

## 4. Additional v3 improvements

| Feature | Zarr v2 | Zarr v3 |
|---|---|---|
| Chunk sharding | ✗ | ✓ |
| Cloud-native metadata | Limited | Designed for high-latency stores |
| Codec extensibility | numcodecs-only | Plugin-based, extensible |
| Async I/O design | Retrofitted | Built-in |
| TensorStore driver | `zarr` | `zarr3` |

---

## 5. Default and recommendation

```yaml
checkpoint_storage_use_zarr3: true
```

MaxText defaults this to `true` — Zarr v3 is the recommended format. The only reason to switch back to Zarr v2:
- You need to share checkpoints with a tool that only reads Zarr v2
- You're loading a checkpoint written by an older MaxText version using Zarr v2

---

## 6. Zarr3 + OCDBT together

These operate at different layers of the stack:

```text
Your parameter (JAX array)
        ↓
Zarr3 chunking + sharding
        ↓
TensorStore storage layer
        ↓
OCDBT B-tree index
        ↓
GCS objects
```

Zarr3 handles **how the array data is chunked and packed**. OCDBT handles **how those packed data files are indexed and addressed**. Together they solve both the file-count problem and the metadata-overhead problem.

---

### One-line intuition

> **`checkpoint_storage_use_zarr3` switches to Zarr v3's sharding format — decoupling logical chunk size from physical file size, solving the cloud storage "small file problem" while preserving fine-grained parallel access.**
