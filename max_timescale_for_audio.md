## 1. Why does `max_timescale_for_audio` exist?

When the audio encoder uses sinusoidal positional encodings, the timescale determines the maximum wavelength period of the sinusoidal basis functions:
$$\omega_k = \frac{1}{\text{max\_timescale}^{2k / D}}$$

`max_timescale_for_audio` sets the maximum timescale constant for the audio encoder's positional encodings.

```text
Positional Encoding Frequencies:
omega_k = 1.0 / (max_timescale_for_audio ^ (2k / D))
```

---

## 2. Options and Defaults

| Value | Behavior |
|---|---|
| `10000.0` (Default) | Standard Transformer sinusoidal timescale base constant |
| `100000.0` | Extended timescale for ultra-long audio sequences |

---

### One-line intuition
> **`max_timescale_for_audio` specifies the base timescale constant (default 10000.0) for sinusoidal positional encoding in the audio encoder.**
