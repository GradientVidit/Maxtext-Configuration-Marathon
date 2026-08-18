
## 1. The backward pass efficiency problem in sparse MoE

In sparse MoE, the token sort/permute step assigns each token to its routed expert:

```text
Forward:
  tokens [t_0, t_1, t_2, ...]  →  sort by expert assignment  →  [t_2, t_0, t_1, ...]

Backward:
  gradient w.r.t. sorted tokens  →  inverse sort  →  gradient w.r.t. original tokens
```

The sort is not differentiable in the standard sense — it involves discrete index operations. JAX's autodiff can still differentiate through it using a custom VJP (vector-Jacobian product) rule, but the default approach may be inefficient.

`use_custom_sort_vjp` enables a hand-written backward-pass rule for the sort operation, optimized for the sparse MoE permute case.

---

## 2. What it does

```yaml
use_custom_sort_vjp: true   # use custom VJP for token sort in sparse matmul
use_custom_sort_vjp: false  # use JAX's autodiff default
```

The custom VJP is specifically designed to:
- Avoid materializing unnecessary intermediate tensors during backward
- Exploit the structure of the permutation (it's a gather, not a general sort)
- Be efficient at the scale of MoE token permutations

---

## 3. This is a backward-pass optimization, not a correctness change

The custom VJP produces the same gradient as JAX's default — it's mathematically equivalent. The difference is compute and memory efficiency in the backward pass.

```text
forward:   identical with or without custom VJP
backward:  custom VJP → fewer intermediate tensors, potentially faster
```

---

## 4. Default

```yaml
use_custom_sort_vjp: true
```

Enabled by default. It's the production-quality implementation. The only reason to disable:

- Debugging: verify custom VJP matches default autodiff (gradient check)
- Compatibility: if a new JAX version breaks the custom VJP

---

## 5. Preconditions

Only relevant when `sparse_matmul=True` — the sort operation being optimized is part of the sparse path. With dense padded MoE (`sparse_matmul=False`), there's no token sort to optimize.

---

## 6. Options

| Value | Behavior |
|---|---|
| `true` (default) | Custom VJP — efficient backward for token sort |
| `false` | JAX default autodiff through sort — correct but potentially less efficient |

---

### One-line intuition

> **`use_custom_sort_vjp` replaces JAX's default autodiff through the MoE token sort with a hand-optimized VJP that avoids unnecessary intermediates in the backward pass — keep it `true` unless debugging gradients.**
