## 1. Why does `decode_sampling_strategy` exist?

During autoregressive generation, the model produces next-token logit distributions over the vocabulary $V$. Converting these logits into the next predicted token requires a sampling policy:

```text
Logit Vector z ∈ R^V
          │
          ▼ [ decode_sampling_strategy ]
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
▼                 ▼                 ▼                 ▼                 ▼
"greedy"          "topk"            "nucleus"         "weighted"        "composite"
argmax(z)         Filter Top-K      Filter Cumulative Categorical       Chained:
Deterministic     Tokens            Probability p     Sample by Temp    Top-K → Nucleus → Temp
```

Different tasks require different trade-offs: code and math require deterministic precision (`greedy`), while creative writing and open dialogue require controlled diversity (`nucleus` or `composite`).

`decode_sampling_strategy` selects which token selection algorithm `decode.py` executes at each generation step.

---

## 2. What it actually controls

```yaml
decode_sampling_strategy: "greedy"
```

| Strategy | Selection Algorithm | Controlling Hyperparameters |
|---|---|---|
| `"greedy"` (default) | $\aargmax_{i} z_i$ (Deterministic highest probability token) | None |
| `"topk"` | Truncates to $k$ highest-probability logits, then samples | `decode_sampling_top_k`, `decode_sampling_temperature` |
| `"nucleus"` | Truncates to the smallest set of tokens whose cumulative probability $\ge p$, then samples | `decode_sampling_nucleus_p`, `decode_sampling_temperature` |
| `"weighted"` | Categorical random sampling from full softmax distribution scaled by temperature | `decode_sampling_temperature` |
| `"composite"` | Chains transformations sequentially: Top-$K$ filter $ \to $ Top-$P$ filter $ \to $ Temperature softmax sampling | `decode_sampling_top_k`, `decode_sampling_nucleus_p`, `decode_sampling_temperature` |

---

## 3. Options and Defaults

```yaml
decode_sampling_strategy: "greedy"    # Default: deterministic argmax
decode_sampling_strategy: "nucleus"   # Standard top-p generation
decode_sampling_strategy: "composite" # Production-grade chained filtering
```

---

## 4. Interactions and Dependencies

```text
decode_sampling_strategy Selection Tree:
  ├── "greedy"    ──> Ignores temperature, top_k, nucleus_p
  ├── "nucleus"   ──> Requires decode_sampling_nucleus_p > 0
  ├── "topk"      ──> Requires decode_sampling_top_k > 0
  └── "composite" ──> Uses decode_sampling_top_k, decode_sampling_nucleus_p, and temperature
```

---

## 5. Practical Scenarios

- **Exact Verification / Regression Tests**: Always use `"greedy"` with `autoregressive_decode_assert`.
- **Creative / Chat Generation**: Use `"composite"` with `decode_sampling_top_k: 40`, `decode_sampling_nucleus_p: 0.95`, `decode_sampling_temperature: 0.7`.

---

### One-line intuition

> **`decode_sampling_strategy` chooses the token generation policy—selecting between greedy argmax, top-k, nucleus (top-p), weighted, or composite chained sampling.**
