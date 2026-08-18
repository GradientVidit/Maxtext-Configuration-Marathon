## 1. Why does `adam_eps_root` exist?

Some framework implementations of Adam (such as TensorFlow/PAX legacy implementations) apply epsilon inside the square root rather than outside:

$$\text{Denominator} = \sqrt{v_t + \epsilon_{\text{root}}} + \epsilon$$

```text
Placement of Epsilon:
  Standard AdamW:  1 / (sqrt(v_t) + adam_eps)
  Root Epsilon:    1 / (sqrt(v_t + adam_eps_root) + adam_eps)
```

`adam_eps_root` provides a numerical stability constant added inside the square root for exact mathematical alignment with legacy PAX configurations.

---

## 2. Fundamentals & Mechanics

- When `adam_eps_root: 0.0` (default), standard AdamW formulation is used.
- When set positive (e.g. `1e-16` or `1e-30`), it acts inside the radical before root extraction.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `0.0` | Disabled; standard AdamW denominator. |
| PAX Compatibility | `1.e-30` or `1.e-16` | Matches PAX-specific Adam formulations. |

---

## 4. Interactions & Dependencies

- Interacts directly with `opt_type: "adam_pax"`.

---

## 5. Practical Scenarios & Failure Modes

- Keep at `0.0` unless porting exact bitwise training checkpoints from legacy PAX models.

---

### One-line intuition

> **`adam_eps_root` adds a numerical stabilizer inside the square root in the Adam denominator for compatibility with PAX optimizer formulations.**
