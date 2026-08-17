
## 1. Why does it exist?

Applying N:M sparsity from the very first training step is aggressive and harmful. Early in training, weight values are random initializations — magnitude-based pruning at this stage removes weights arbitrarily (there's no signal about which weights matter yet). This damages the model's ability to learn.

`weight_sparsity_start_step` implements a **warm-up for sparsity**: train densely long enough for weights to develop meaningful magnitudes, then begin pruning.

```text
step 0 → step 49:       dense training (weights develop signal)
step 50 (start_step):   first mask computed, sparsity applied
step 60:                mask refreshed (update_step=10)
step 70:                mask refreshed
...
```

---

## 2. The rationale for warm-up

In the early phase of training:
- Weights are near their initialization values (small random numbers)
- All weights have roughly equal importance (magnitude provides no useful signal for pruning)
- The loss is dropping rapidly — gradients are large, learning is fastest

Applying sparsity here removes random weights and permanently restricts which connections can carry information, severely hurting the model's capacity to learn.

By step 50 (the default), the model has adapted weights to the data distribution. The largest-magnitude weights are meaningfully important. Pruning now targets the least-important connections based on real signal.

---

## 3. Options

| Value | Behavior |
|---|---|
| Integer >= 0 | Number of dense training steps before sparsity |
| `50` | Default |
| `0` | Apply sparsity from the very first step (not recommended) |

Default in base.yml:
```yaml
weight_sparsity_start_step: 50
```

---

## 4. How long to warm up

The right `weight_sparsity_start_step` depends on training dynamics:

```text
Very short warm-up (0–50 steps):
  Weights still near init → bad sparsity pattern → accuracy loss
  Only acceptable if resuming from a dense pre-trained checkpoint

Moderate warm-up (50–1000 steps):
  Default range. Weights have started to differentiate.
  Suitable for training from scratch on most tasks.

Long warm-up (1000+ steps):
  Near-converged weights → stable magnitude rankings
  High-quality initial mask
  Better final accuracy, especially for aggressive sparsity levels
```

For sparsity applied to a pretrained model (fine-tuning with sparsity), a very short warm-up (or 0) is appropriate since weights are already meaningful.

---

## 5. Dependency

Only relevant when:
```yaml
weight_sparsity_n: 2  # (or any non-null N)
weight_sparsity_m: 4  # (or any non-null M)
```

---

### One-line intuition

> **`weight_sparsity_start_step=50` delays activating the sparsity mask until training has run for 50 steps — letting weights develop meaningful magnitudes before pruning so that early mask decisions are based on signal, not noise.**
