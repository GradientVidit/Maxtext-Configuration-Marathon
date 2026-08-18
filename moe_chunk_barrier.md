
## 1. Why this exists

`num_moe_token_chunks > 1` splits the token batch into pipeline chunks to overlap communication and compute (see `num_moe_token_chunks.md`). But XLA's compiler is free to schedule these chunk iterations in any order — potentially interleaving chunks in ways that don't respect the intended pipeline order.

This interleaving is the point for performance. But when debugging, it makes behavior harder to understand: if chunk_2 runs before chunk_1's result is ready, is that a bug or an optimization?

`moe_chunk_barrier` is a correctness-preserving scheduling constraint. It inserts an `optimization_barrier` (identity operation that XLA cannot reorder through) between chunks:

```text
chunk_0 output → optimization_barrier → chunk_1 input
chunk_1 output → optimization_barrier → chunk_2 input
```

This forces strictly sequential, non-interleaved execution. Math is unchanged — barriers are identity operations. Only scheduling changes.

---

## 2. What it controls

```yaml
moe_chunk_barrier: false  # (default) allow XLA to schedule chunks freely
moe_chunk_barrier: true   # force strictly sequential chunk execution
```

---

## 3. Preconditions

`moe_chunk_barrier` has no effect unless **both**:
- `num_moe_token_chunks > 1` — there are chunks to sequence
- `use_ring_of_experts=True` — the ring path provides the chunking structure

With `num_moe_token_chunks=1`, chunks don't exist, and the barrier is irrelevant.

---

## 4. When to use it

**Production:** leave `false`. The whole point of chunking is overlapped execution — the barrier defeats that.

**Debugging chunk interleaving issues:** if you suspect XLA's scheduling of chunks is causing correctness problems (e.g. a chunk reading stale data), enable `moe_chunk_barrier=True` to force sequential execution and see if the bug disappears.

**Isolating performance baseline:** with `moe_chunk_barrier=True` and multiple chunks, you get chunked-but-sequential execution — useful to measure the overhead of the chunking machinery without overlap, as a baseline for the overlapped case.

---

## 5. Options

| Value | Behavior |
|---|---|
| `false` (default) | XLA schedules chunks freely — overlap possible |
| `true` | Strictly sequential chunk execution — no overlap |

---

### One-line intuition

> **`moe_chunk_barrier` forces sequential (non-interleaved) execution of the ring-of-experts chunk loop by fencing each chunk on the previous — a debugging tool that removes overlap without changing correctness; leave `false` in production.**
