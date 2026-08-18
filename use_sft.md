## 1. Why does `use_sft` exist?

Supervised Fine-Tuning (SFT) adapts a general pretrained base model into an instruction-following assistant by training on high-quality demonstration datasets (user prompts paired with desired assistant completions).

```text
Pretraining: Unstructured text corpus ──> Predict every next token
                                               │
                                               ▼
SFT Mode:    Structured (Prompt, Completion) ──> Formatted dialogue & selective masking
```

`use_sft: true` toggles MaxText's instruction fine-tuning mode, configuring chat templates, loss masks, and evaluation metrics tailored for instruction datasets.

---

## 2. Mechanics

In SFT mode:
1. Datasets are parsed through instruction templates or chat formatting (`use_chat_template`).
2. Loss computation can optionally mask prompt tokens (`sft_train_on_completion_only: true`), ensuring gradients are computed only over assistant tokens.
3. Checkpoint loading defaults to parameter-only weights (`load_parameters_path`).

```text
Prompt Tokens (Loss Mask = 0.0) ──> Completion Tokens (Loss Mask = 1.0) ──> Cross-Entropy
```

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `use_sft` | `bool` | `false` | `true` (enable SFT instruction tuning), `false` (standard pretraining) |

---

## 4. Interactions with Related Parameters

- **`sft_train_on_completion_only`**: Determines whether to backpropagate through prompt tokens.
- **`use_chat_template`**: Formats multi-turn conversations into tokenized prompt-response pairs.
- **`load_parameters_path`**: Path to the base pretrained model checkpoint to fine-tune.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Fine-tuning on instruction pairs without SFT flags** | Model trains on entire prompt with equal loss weight, degrading prompt comprehension | Set `use_sft: true` and `sft_train_on_completion_only: true`. |
| **Simultaneously enabling SFT and DPO** | Config conflict: DPO and SFT objectives collide | Enable only one objective per run. |

---

### One-line intuition

> `use_sft` configures MaxText for Supervised Fine-Tuning, enabling conversation formatting and prompt-aware loss masking.
