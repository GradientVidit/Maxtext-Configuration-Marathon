## 1. Why does `use_chat_template` exist?

Instruction-tuned and conversational language models are trained on structured multi-turn dialogues containing distinct roles (`system`, `user`, `assistant`).

```text
Raw Structured Input:
[{"role": "user", "content": "Explain JAX."}]
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
use_chat_template: false      use_chat_template: true
         │                             │
(Raw string serialization)     <|im_start|>user
"[{'role': 'user' ...}]"       Explain JAX.<|im_end|>
                               <|im_start|>assistant
```

Without chat templating, multi-turn messages would be tokenized as plain string representations of Python dictionaries or JSON. `use_chat_template: true` activates Jinja2-based prompt formatting to render structured conversations into standardized token strings containing correct role tags and delimiters.

---

## 2. Mechanics

When `use_chat_template: true`, MaxText's input pipeline passes raw list-of-dict messages through HuggingFace's `apply_chat_template()` utility or a custom Jinja2 template renderer before tokenization:

```text
Structured Messages (JSON / Dicts)
                │
                ▼
   Jinja2 Template Engine ◄── [chat_template / chat_template_path]
                │
        Formatted String (e.g. ChatML format)
                │
                ▼
            Tokenizer ──> [Token IDs, Loss Mask]
```

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `use_chat_template` | `bool` | `false` | `true`, `false` |

---

## 4. Interactions with Related Parameters

- **`chat_template`**: Raw Jinja2 template string provided inline.
- **`chat_template_path`**: Path to a `.json` or `.jinja` file containing the chat template.
- **`tokenizer_type`**: Works seamlessly with `tokenizer_type: "huggingface"` which has built-in `apply_chat_template` support.
- **`use_sft` / `sft_train_on_completion_only`**: In conversational SFT, the chat template defines assistant turn boundaries, allowing the loss mask to mask out user prompts and compute gradients solely on assistant responses.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **SFT dataset has `[{"role": "...", "content": "..."}]` but `use_chat_template: false`** | Model memorizes python dictionary syntax (`'role'`, `'content'`) instead of chat conversation tags | Set `use_chat_template: true` and specify the model's template. |
| **No template supplied when enabled** | Tokenizer raises `ValueError: Cannot use chat template without template defined` | Provide `chat_template` or `chat_template_path`. |

---

### One-line intuition

> `use_chat_template` toggles Jinja2 role-based formatting for conversational datasets, converting multi-turn structured messages into standard tokenized dialogues with system/user/assistant delimiters.
