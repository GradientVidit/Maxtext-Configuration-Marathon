
## 1. The physical storage problem

Orbax stores model parameters as arrays (via TensorStore). Large arrays — say, a 70B model's embedding table or weight matrices — are split into **chunks** and written as individual files.

Without a target file size, each chunk may produce one tiny file:

```text
embedding/chunk_0  (4MB)
embedding/chunk_1  (4MB)
embedding/chunk_2  (4MB)
...
embedding/chunk_N  (4MB)
```

On GCS (and any object store), file-count overhead dominates at scale:
- Metadata operations per file (LIST, GET, PUT) have latency
- Millions of small files → checkpoint save/load slows dramatically

---

## 2. What this parameter does

```yaml
checkpoint_storage_target_data_file_size_bytes: 2147483648  # 2GB
```

Tells Orbax's storage layer to **aggregate chunks into fewer, larger physical files** until each file approaches 2GB.

```text
Before: 512 × 4MB chunks = 512 files
After:  8 × ~256MB files (or similar, depending on chunking)
```

Fewer files → faster parallel I/O, reduced metadata overhead, faster listing.

---

## 3. Why 2GB specifically

2GB = `2 × 1024³ = 2,147,483,648` bytes is a common practical ceiling:
- Below the 4GB limit of some older filesystems/tools
- Large enough to amortize GCS per-object overhead
- Small enough that a single file isn't too large for network-level retry granularity

This is the default. It's rarely changed unless you have very specific storage constraints.

---

## 4. The interaction with distributed loading

Chunking into larger files also directly affects **distributed checkpoint load speed**. When loading across many hosts:

```text
fewer large files → each host can independently fetch a large file
                    and decompose chunks locally
→ better parallelism, fewer API calls to GCS
```

From the MaxText config comment itself:
> "chunking a single large array into small physical files (<2GB) can speed up distributed and over-the-network loading enormously"

The framing is: you want files that are large enough to amortize overhead but not so large that they can't be parallelized.

---

## 5. Interaction with OCDBT and Zarr3

This parameter works within the storage stack:

```text
checkpoint_storage_target_data_file_size_bytes
    ↓
    used by Orbax/TensorStore layer
    ↓
OCDBT (if enabled): coalesces chunks into B-tree data files
Zarr3 (if enabled): uses sharding codecs to pack chunks into shard files
```

All three work together to reduce file count and improve I/O performance. OCDBT and Zarr3 have their own internal coalescing logic that also respects this target size hint.

---

## 6. When to change it

Almost never. Reasons you might:
- Very limited per-file size constraints (some network filesystems cap file sizes below 2GB)
- Debugging: smaller size produces more granular files for inspection
- Very small model where even 2GB is overkill and you want faster per-file operations

---

## 7. Options

Any positive integer (bytes):

```yaml
checkpoint_storage_target_data_file_size_bytes: 2147483648   # 2GB (default)
checkpoint_storage_target_data_file_size_bytes: 1073741824   # 1GB
checkpoint_storage_target_data_file_size_bytes: 536870912    # 512MB
```

---

### One-line intuition

> **`checkpoint_storage_target_data_file_size_bytes` tells Orbax to coalesce checkpoint chunks into physical files up to ~2GB each — reducing the file count in GCS and dramatically improving distributed load throughput.**
