## 1. Why does `eval_start_step` exist?

During the initial warmup phase of training (steps $0 \dots N$), model weights undergo rapid gradient updates, optimizer state warms up, and loss drops precipitously. Running full evaluation passes on validation datasets during step 0 or early warmup provides little actionable insight while interrupting training throughput and delaying JIT optimization:

```text
Training Timeline:
Step 0 ────────────── Step 100 ────────────── Step 1,000 ────────────── Step 100,000
 [ Rapid Warmup ]            [ First Eval Pass ]            [ Regular Periodic Evals ]
                             ▲
                             │
                      eval_start_step = 100
```

`eval_start_step` sets the initial training step barrier before which no evaluation passes are executed.

---

## 2. What it actually controls

```yaml
eval_start_step: 0
```

- MaxText evaluates the condition:
$$    ext{train\_step} \ge     ext{eval\_start\_step} \quad     ext{and} \quad (    ext{train\_step} -     ext{eval\_start\_step}) \pmod{    ext{eval\_interval}} == 0$$
- If `train_step < eval_start_step`, the evaluation loop is skipped entirely.

---

## 3. Options and Defaults

| Value | Behavior | Use Case |
|---|---|---|
| `0` (default) | Evaluations can trigger from step 0 onward | Standard runs when initial baseline validation loss is desired |
| `1000` | Skips validation during first 1,000 warmup steps | Large-scale pretraining runs to avoid early eval overhead |

---

## 4. Interactions

- **`eval_interval`**: Evaluation frequency once `eval_start_step` is reached.
- **`save_checkpoint_on_start`**: If saving a baseline checkpoint at step 0, setting `eval_start_step: 0` allows evaluating the initial un-finetuned checkpoint.

---

## 5. Practical Scenarios

- **Pretraining from Scratch**: Set `eval_start_step: 500` to let learning rate warmup stabilize before measuring validation metrics.

---

### One-line intuition

> **`eval_start_step` delays the start of periodic validation passes until training reaches a specified step threshold.**
