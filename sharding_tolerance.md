## 1. Why does it exist?

When configuring complex sharding strategies across thousands of TPU cores, small auxiliary parameter arrays (such as layer normalization scale biases, rotary position frequency buffers, or scalar multipliers) might not divide evenly across the mesh axes or may be left unsharded (replicated) by design.

However, if a significant portion of large weight tensors (like attention projections or MLP matrices) fails to shard properly due to a rule typo or dimension mismatch, massive memory duplication occurs, leading to Out-Of-Memory (OOM) crashes.

MaxText audits the compiled model weights to calculate what fraction of total model parameters ended up unsharded.

```text
Total Model Parameters: 70 Billion
Unsharded (Replicated) Parameters: 700 Million (1.0%)

Audit against `sharding_tolerance`:
  Threshold = 0.02 (2.0%)
  1.0% <= 2.0% ──→ PASS (Training proceeds)
  
If Unsharded = 5.0 Billion (7.1%):
  7.1% > 2.0%  ──→ FAIL (Raises Sharding Verification Error)
```

`sharding_tolerance` sets the maximum allowable fraction of parameters that can remain unsharded before MaxText raises a fatal configuration error.

---

## 2. Fundamentals & Verification Logic

During initialization and AOT compilation, MaxText computes:

$$\text{Unsharded Fraction} = \frac{\sum \text{Params in fully replicated arrays}}{\sum \text{Total model parameters}}$$

- If $\text{Unsharded Fraction} \le \text{sharding\_tolerance}$, training starts cleanly.
- If $\text{Unsharded Fraction} > \text{sharding\_tolerance}$, MaxText logs a detailed report of which layers failed to shard and halts execution to protect against catastrophic OOMs.

---

## 3. Options & Configuration

| Value | Meaning |
|---|---|
| Float in `[0.0, 1.0]` | The maximum tolerated fraction of unsharded weights. |
| `0.02` (default) | Allows up to 2% of parameters (typically small norms, biases, and RoPE tables) to remain unsharded. |

Default in `base.yml`:
```yaml
sharding_tolerance: 0.02
```

---

## 4. Practical Scenarios & Pitfalls

- **Why not set to `0.0`?** Exact `0.0` tolerance would fail on almost all modern transformer models because 1D LayerNorm scales (`[dim]`) or scalar embeddings are intentionally replicated across tensor-parallel heads.
- **When to increase?** In very small models (e.g. `< 100M` parameters) or specialized architectures with large shared unsharded lookup tables, the unsharded fraction may naturally exceed 2%, requiring `sharding_tolerance: 0.05` or `0.10`.
- **Debugging Catch**: If you accidentally misspell an axis rule (e.g. `'head'` instead of `'heads'`), the entire 70B attention block remains replicated. `sharding_tolerance` immediately catches this bug before GPU/TPU hours are wasted.

---

### One-line intuition

> **`sharding_tolerance` is a safety guardrail (default 2%) that halts training if too high a fraction of model parameters fails to shard across devices, preventing silent memory duplication.**
