## 1. Why does `target_eval_loss` exist?

In fine-tuning, knowledge distillation, and early-stopping workflows, training beyond a specific target validation loss can lead to overfitting or waste expensive accelerator hours.

```text
Training Progress:
Step 0    ──> Eval Loss: 4.82
Step 500  ──> Eval Loss: 2.15
Step 1000 ──> Eval Loss: 1.48  ──(Crosses target_eval_loss: 1.50)──> EARLY STOPPING TRIGGERED!
                                                                     Save Checkpoint & Exit 0
```

`target_eval_loss` provides an automated early-stopping threshold that halts training as soon as the validation loss drops below this target value.

---

## 2. What it actually controls

```yaml
target_eval_loss: 0.
```

- When `0.` (default): Early stopping is **disabled** (a loss of $0.0$ is practically unreachable with cross-entropy).
- When `> 0.` (e.g. `1.50`): At the end of every evaluation pass, MaxText checks:
$$    ext{eval\_loss} \le     ext{target\_eval\_loss}$$
If satisfied, MaxText saves the final checkpoint and cleanly terminates training.

---

## 3. Options and Defaults

| Value | Behavior | Use Case |
|---|---|---|
| `0.` (default) | Early stopping disabled; runs until `steps` | Standard pretraining and fixed-budget training |
| `> 0.` (e.g., `1.85`) | Halts training when validation loss $\le 1.85$ | Convergence-targeted fine-tuning, regression testing |

---

## 4. Interactions

- **Requires Active Eval Loop**: Only evaluated when `eval_interval > 0` and `eval_steps > 0`.
- **`save_checkpoint_on_completion`**: If `true`, saving the final checkpoint occurs immediately upon reaching `target_eval_loss`.

---

## 5. Practical Scenarios

- **Automated CI Convergence Tests**: Set `target_eval_loss: 2.0` in CI to ensure code changes still converge to expected loss levels within a step budget.

---

### One-line intuition

> **`target_eval_loss` sets an early-stopping validation loss threshold that cleanly terminates training once reached.**
