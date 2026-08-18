## 1. Why does `hf_access_token` exist?

Many frontier foundation models and curated datasets on HuggingFace Hub (e.g. Meta-Llama, Gemma, StarCoder) are gated behind terms-of-service agreements or stored in private organizational repositories.

```text
Gated HF Repo ──[ Request ]──> Authentication Check
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
     hf_access_token: ""                             hf_access_token: "hf_..."
              │                                               │
   401 Unauthorized / Gated Error                     Access Granted & Streamed
```

`hf_access_token` passes the Hugging Face User Access Token (`hf_...`) to authorize downloads and API calls.

---

## 2. Mechanics

Passed into `datasets.load_dataset(..., token=hf_access_token)` and `AutoTokenizer.from_pretrained(..., token=hf_access_token)`.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `hf_access_token` | `str` | `''` | Hugging Face API user token string |

---

## 4. Interactions with Related Parameters

- **`hf_path`**: Authenticates access to this repository.
- **`tokenizer_path`**: Used if loading Hugging Face tokenizers directly from gated Hub repos.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Loading gated dataset without token** | `huggingface_hub.utils.GatedRepoError` | Provide token in `hf_access_token` or set `HF_TOKEN` environment variable. |

---

### One-line intuition

> `hf_access_token` provides the Hugging Face API authentication token required to download gated or private datasets.
