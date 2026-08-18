## 1. Why does `chat_template` exist?

When fine-tuning on conversational data without maintaining an external template file, users need a direct way to specify Jinja2 prompt formatting rules directly inside the MaxText config or command line.

```text
chat_template: "{% for message in messages %}{{'<|im_start|>' + message['role'] + '
' + message['content'] + '<|im_end|>
'}}{% endfor %}"
                               │
                               ▼
            Applied inline to all input conversation batches
```

`chat_template` allows injecting a raw Jinja2 template string directly into the tokenizer environment.

---

## 2. Mechanics

During dataset preprocessing, the HuggingFace `PreTrainedTokenizerFast` or Grain template processor uses `jinja2.Template(chat_template).render(messages=...)` to generate formatted dialogue strings before subword tokenization.

```text
Input: [{"role": "user", "content": "Hi"}]
             │
             ▼ [chat_template]
Output: "<|im_start|>user
Hi<|im_end|>
<|im_start|>assistant
"
```

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `chat_template` | `str` | `""` | Valid Jinja2 template string |

---

## 4. Interactions with Related Parameters

- **`use_chat_template`**: Must be `true` for `chat_template` to take effect.
- **`chat_template_path`**: Alternative mechanism; `chat_template` is used when specifying templates inline.
- **`use_sft` & `sft_train_on_completion_only`**: The template defines token boundaries that separate user prompts from assistant completions for loss masking.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Jinja2 syntax error (missing bracket/tag)** | `jinja2.exceptions.TemplateSyntaxError` at data loading time | Test the Jinja2 template string in Python before running training. |
| **Escaped quotes in CLI overrides** | Shell parser strips quotation marks, corrupting Jinja syntax | Place complex templates inside YAML or use `chat_template_path`. |

---

### One-line intuition

> `chat_template` holds the raw Jinja2 template string used by the tokenizer to serialize multi-turn message dictionaries into formatted dialogue text.
