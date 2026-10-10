## Multimodal model: fixed 35 m, three-seed check

Trained a fresh six-mode attention GNN for each seed, using the existing 35 m train/validation/test datasets. Protocol: 30 epochs, batch size 64, Adam lr=0.001, best-of-six ADE loss, and best-validation checkpoint selection. These runs used CPU. No radius sweep or radius change.

Errors in metres; lower is better. FDE is evaluated on the mode selected by minimum ADE, rather than independently minimized FDE.

| Seed | Val minADE6 | Val FDE at best ADE | Test minADE6 | Test FDE at best ADE |
|---|---:|---:|---:|---:|
| 42 | 1.782 | 4.169 | 1.704 | 3.921 |
| 123 | 1.809 | 4.285 | 1.718 | 3.997 |
| 2026 | 1.803 | 4.219 | 1.721 | 3.977 |
| Mean ± sample std | 1.798 ± 0.014 | 4.224 ± 0.058 | 1.714 ± 0.009 | 3.965 ± 0.039 |

**Conclusion:** Test minADE6 is consistent across these seeds (1.704–1.721 m, mean 1.714 ± 0.009 m). This supports that the strong result is not driven by one lucky initialization in this small check. Retain 35 m. These best-of-six metrics use ground truth to select the best mode and are not directly equivalent to single-trajectory ADE.

Validation: all notebook code cells compile; all three completed runs have finite results and 30-epoch histories; all three seed checkpoints and the existing notebook checkpoint load into the shared model; a forward/backward smoke test passes.
