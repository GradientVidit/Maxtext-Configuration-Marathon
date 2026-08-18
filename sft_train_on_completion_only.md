## 1. Why does `sft_train_on_completion_only` exist?

During instruction fine-tuning, the user's prompt is a fixed conditioning prefix; the model is only evaluated on its ability to produce the correct assistant completion. 

```text
Training Input: "Prompt: What is the capital of France? Response: Paris."

When sft_train_on_completion_only: false
Tokens:    [ What, is, the, capital, of, France, ?, Paris, . ]
Loss Mask: [   1,   1,   1,    1,    1,    1,    1,   1,   1 ]  <-- Penalizes model for prompt wording

When sft_train_on_completion_only: true
Tokens:    [ What, is, the, capital, of, France, ?, Paris, . ]
Loss Mask: [   0,   0,   0,    0,    0,    0,    0,   1,   1 ]  <-- Gradients computed ONLY on response!
```

If the loss mask includes prompt tokens, the model wastes gradient capacity memorizing user prompt phrasings instead of learning reasoning and response generation.

`sft_train_on_completion_only: true` zeros out the loss mask across all prompt tokens, computing cross-entropy gradients solely on the completion.

---

## 2. Mechanics

The tokenizer or chat template marks token spans with a binary mask:
- Prompt tokens: $ ext{mask}_i = 0$
- Completion tokens: $ ext{mask}_i = 1$

```text
Cross-Entropy Loss = - rac{\sum_i  ext{mask}_i \log P(x_i | x_{<i})}{\sum_i  ext{mask}_i}
```

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `sft_train_on_completion_only` | `bool` | `false` | `true` (mask prompt, train on completion), `false` (train on prompt + completion) |

---

## 4. Interactions with Related Parameters

- **`use_sft`**: Must be `true` for SFT loss masking to operate.
- **`use_chat_template`**: Template identifies the boundary where prompt ends and assistant generation starts.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Instruction-tuning a chat model** | Model regurgitates prompt tokens or fails to follow instructions | Set `sft_train_on_completion_only: true`. |
| **Empty completion or misidentified boundary** | All tokens masked to 0, resulting in `0/0 = NaN` loss | Ensure chat template properly wraps assistant responses. |

---

### One-line intuition

> `sft_train_on_completion_only` masks out prompt tokens so that backpropagation gradients update weights strictly based on assistant completions.
