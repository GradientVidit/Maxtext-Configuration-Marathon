
## 1. The memory problem that remat solves

Training a transformer requires holding all intermediate activations in HBM for the backward pass. For a model with L layers, the activation memory scales roughly as:

```text
O(L × batch × seq_len × hidden_dim)
```

At large scale this dominates HBM — a 70B model at a reasonable batch size will OOM without intervention. Rematerialization solves this by a trade: **discard activations during the forward pass, recompute them during the backward pass**.

```text
Standard:
  Forward:  [layer 0 acts] [layer 1 acts] ... [layer L acts] all in HBM
  Backward: read and free acts in reverse order

Rematerialization:
  Forward:  [keep checkpoint] [discard intermediates] [keep checkpoint] ...
  Backward: [recompute from checkpoint] [update weights] [move to next checkpoint]
```

`remat_policy` selects how aggressively to apply this trade.

---

## 2. The policy spectrum

```text
Fastest ←──────────────────────────────────────────────────────→ Most memory-efficient

'minimal_with_context'
      'minimal'
            'save_dot_with_context_except_mlp'
                      'save_dot_except_mlpwi'
                                'save_dot_except_mlp'
                                          'save_qkv_proj'
                                                    'qkv_proj_offloaded'
                                                              'save_out_proj'
                                                                        'full'
```

And the two special modes: `'custom'` (manual per-tensor), `'minimal_offloaded'` (minimal + host offload).

**Critical naming trap**: `'full'` means "fully rematerialize everything" — it is the **most memory-efficient** (keeps least in HBM), despite the name suggesting "keep everything." MaxText comments explicitly warn about this.

> **Note on the middle entries**: the relative ordering of `save_dot_*` variants is approximate and architecture-dependent (e.g. whether MLP or attention intermediates are larger for your specific model/sequence length). The endpoints — `'minimal_with_context'` (fastest) and `'full'` (most memory-efficient) — are consistent.

---

## 3. Policy definitions

| Policy | What it keeps in HBM | Memory | Speed |
|---|---|---|---|
| `'minimal_with_context'` | Most activations + context | Highest | Fastest |
| `'minimal'` | Most activations | High | Fast |
| `'save_dot_with_context_except_mlp'` | Dot-product results + context, not MLP | Medium-high | Medium |
| `'save_dot_except_mlpwi'` | Dot results, not MLP input projection | Medium | Medium |
| `'save_dot_except_mlp'` | Dot results, not MLP at all | Medium | Medium |
| `'save_qkv_proj'` | QKV projections only | Lower | Slower |
| `'qkv_proj_offloaded'` | QKV projections + host offload | Lower | Slower |
| `'save_out_proj'` | Output projections only | Low | Slower |
| `'full'` | Nothing (fully recompute) | **Lowest** | **Slowest** |
| `'custom'` | Per-tensor specified | Depends | Depends |
| `'minimal_offloaded'` | Minimal + host offload | Very low | Variable |

---

## 4. Default: `'full'`

```yaml
remat_policy: 'full'
```

Default is the most memory-efficient option — recompute everything during backward. This is a conservative choice: it lets you fit large models without thinking about which activations to keep. The speed cost is real: recomputation during backward runs sequentially alongside gradient computation, adding FLOPs but drastically reducing peak HBM.

For high-throughput production runs where memory is not the binding constraint, moving to `'minimal'` or `'save_dot_except_mlp'` significantly improves step time.

---

## 5. The `'custom'` escape hatch

`'custom'` activates per-tensor placement: each intermediate tensor gets assigned `'remat'`, `'device'`, or `'offload'` via its own config key. See the custom remat table in `5. Pipeline Parallelism + Rematerialization.md` for the full tensor list.

```yaml
remat_policy: 'custom'
query_proj: 'device'    # keep in HBM
mlpwi: 'offload'        # send to CPU RAM
attention_out: 'remat'  # recompute on backward
```

---

## 6. Practical guidance

| Situation | Recommended policy |
|---|---|
| Debugging / just getting it to run | `'full'` (default) — don't worry about speed yet |
| Good batch size, want speed | `'minimal'` or `'save_dot_except_mlp'` |
| Memory-constrained, need max capacity | `'full'` or `'minimal_offloaded'` |
| Fine-tuning which ops are bottlenecks | `'custom'` + profiler |

---

## 7. Interaction with pipeline remat settings

```yaml
remat_policy: 'save_dot_except_mlp'
set_remat_policy_on_pipeline_iterations: true
```

The `remat_policy` specifies *which tensors* to discard; `set_remat_policy_on_pipeline_iterations` specifies *where in the loop structure* to apply it. Both must be set correctly — the policy does nothing if no scan has the policy attached.

---

### One-line intuition

> **`remat_policy` selects how aggressively to trade forward-pass recomputation for HBM memory: `'full'` (the default) recomputes everything and is the most memory-efficient despite the confusing name — work up from there toward `'minimal'` as you verify your run fits in memory.**
