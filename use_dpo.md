## 1. Why does `use_dpo` exist?

Direct Preference Optimization (DPO) aligns language models to human preferences without requiring a separate reinforcement learning reward model (PPO/RLHF). DPO optimizes the policy directly on pairs of prompt-completion examples labeled as "chosen" (preferred) and "rejected" (dispreferred).

```text
Standard Pretrain/SFT: Single text sequence ──> Cross-Entropy Loss
                                                       │
                                                       ▼
DPO Mode: [ Chosen Sequence, Rejected Sequence ] ──> DPO Margin Loss
                                                       │
                                                       ▼
                           Increases log-prob(chosen) - log-prob(rejected)
```

`use_dpo: true` switches MaxText from standard next-token cross-entropy into the paired DPO loss computation graph.

---

## 2. Mechanics & Loss Computation

When `use_dpo: true`:
1. The input pipeline extracts two streams using `train_data_columns: ['chosen', 'rejected']`.
2. MaxText evaluates both sequences through the active policy model and a frozen reference model.
3. The DPO objective maximizes the implicit reward:
$$\mathcal{L}_{ ext{DPO}}(\pi_ heta; \pi_{ ext{ref}}) = -\mathbb{E}_{(x, y_w, y_l)}\left[ \log \sigma \left( eta \log rac{\pi_ heta(y_w|x)}{\pi_{ ext{ref}}(y_w|x)} - eta \log rac{\pi_ heta(y_l|x)}{\pi_{ ext{ref}}(y_l|x)} 
ight) 
ight]$$

```text
Input Pair: (Chosen, Rejected)
       │
       ├── Forward Policy Model ──> Log-probs (Policy)
       │
       └── Forward Ref Model   ──> Log-probs (Reference)
               │
               ▼
         DPO Loss Kernel
```

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `use_dpo` | `bool` | `false` | `true` (enable DPO preference alignment), `false` (standard SFT/pretrain) |

---

## 4. Interactions with Related Parameters

- **`train_data_columns` / `eval_data_columns`**: Must provide both chosen and rejected keys (`['chosen', 'rejected']`).
- **`dpo_beta`**: The temperature / implicit reward scaling parameter $eta$.
- **`use_sft`**: Incompatible; SFT and DPO represent distinct training objectives.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **`use_dpo: true` with default `train_data_columns: ['text']`** | Pipeline crash: missing chosen/rejected columns | Set `train_data_columns: ['chosen', 'rejected']`. |
| **DPO loss collapsing / NaN** | Reference model log-probabilities unstable | Verify reference checkpoint weights match policy base weights. |

---

### One-line intuition

> `use_dpo` switches training from next-token prediction to pairwise Direct Preference Optimization, aligning model outputs to human preference pairs without reinforcement learning.
