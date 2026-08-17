
## 1. The problem: redundant I/O at scale

In a distributed training job with many hosts (e.g., a TPU v4-512 with 64 hosts), the default checkpoint restore behavior is:

```text
host_0  → reads full checkpoint from GCS
host_1  → reads full checkpoint from GCS
host_2  → reads full checkpoint from GCS
...
host_63 → reads full checkpoint from GCS
```

All 64 hosts hammer storage simultaneously. On GCS, this creates:

- High concurrent read load → throughput degrades
- Per-host latency spikes → all-or-nothing barrier means the slowest host determines total restore time
- Storage API costs scale linearly with host count

For a 70B model checkpoint (~140GB), doing this across 512 hosts wastes enormous bandwidth.

---

## 2. What single-replica restore does

```yaml
enable_single_replica_ckpt_restoring: true
```

Changes the strategy to:

```text
host_0  → reads full checkpoint from GCS
                ↓
        broadcasts to all other hosts
        via high-speed interconnect (ICI/DCN)
host_1  ← receives from host_0
host_2  ← receives from host_0
...
host_63 ← receives from host_0
```

Only **one replica** (the designated leader) reads from storage. All others receive the data via the TPU/GPU high-bandwidth interconnect.

---

## 3. Why the interconnect is faster than GCS for this

TPU ICI (Inter-Chip Interconnect) bandwidth is measured in hundreds of GB/s. GCS throughput per host is typically capped at tens of GB/s, and degrades under concurrent load. So:

```text
GCS concurrent read (64 hosts × checkpoint):
  → high API contention, degraded bandwidth per host

ICI broadcast from one reader:
  → one fast GCS read, then distributed via low-latency high-bandwidth fabric
```

The math favors single-replica when the interconnect is significantly faster than the storage layer — which is true for TPU pods.

---

## 4. Trade-offs

| Aspect | Without (default) | With single-replica |
|---|---|---|
| Storage bandwidth pressure | High — all hosts read | Low — one host reads |
| Interconnect usage | None | High — broadcast across fabric |
| Restore time at scale | Degrades with host count | More predictable |
| Complexity | Simple | Requires coordination logic |

The broadcast means the interconnect is busy during restore — not a problem since training isn't running yet, but worth noting.

---

## 5. When to enable

- Large multi-host training (>8 hosts / large TPU pods)
- Storage bandwidth is a measured bottleneck on restore
- You're using reserved interconnects (TPU ICI / high-bandwidth DCN)

Default is `false` because for small runs the overhead of coordination outweighs the benefit.

---

## 6. Default

```yaml
enable_single_replica_ckpt_restoring: false
```

---

### One-line intuition

> **`enable_single_replica_ckpt_restoring` has only one host read the checkpoint from storage then broadcast it over the fast interconnect to all others — replacing N concurrent GCS reads with 1 read + a fast intra-pod distribution.**
