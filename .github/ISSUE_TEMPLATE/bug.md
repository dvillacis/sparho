---
name: Bug report
about: Something doesn't work the way the docs say it should
labels: bug
---

## What happened

<!-- One sentence. -->

## What you expected

<!-- One sentence. -->

## Minimal reproducer

```python
import numpy as np
import sparho

# The smallest snippet that demonstrates the bug.
# Synthetic data is fine — we don't need (and don't want) your private dataset.
```

## Environment

- sparho version: <!-- python -c "import sparho; print(sparho.__version__)" -->
- core version:   <!-- python -c "import sparho; print(sparho._core.version())" -->
- Python:         <!-- python --version -->
- OS / arch:      <!-- e.g. macOS 14 / arm64, Ubuntu 22.04 / x86_64, Windows 11 / AMD64 -->
- numpy / scipy / scikit-learn versions: <!-- pip show numpy scipy scikit-learn | grep Version -->

## Full traceback (if applicable)

<details><summary>traceback</summary>

```
(paste here)
```

</details>

## Notes

<!-- Anything else useful: was this working in a previous version? Does it
     reproduce with a different solver adapter? Have you tried `--repeat 5`
     to rule out reproducibility jitter? -->
