# Seed variance

Three seeds for the matched FP32 control and the ternary QAT baseline.

| run | seed 42 | seed 43 | seed 44 | mean | std |
|---|---:|---:|---:|---:|---:|
| FP32, 600 steps | 19.3580 | 19.2622 | 20.0638 | **19.5613** | 0.4378 |
| Ternary QAT, 600 steps | 21.4886 | 21.2718 | 21.5735 | **21.4446** | 0.1556 |

QAT gap vs FP32: **1.883 PPL**, about 9.6 percent.

## Consequence

- The 0.06 PPL difference between 21.49 and 21.42 is below the QAT seed std of
  0.16. It is not a ranking.
- The 2-bit outlier gain does not survive seed noise.
- The real ternary tax is about 1.9 PPL, not 5.3.
- Any future format must beat 21.44 by more than about 0.3 PPL to be credible
  with three seeds.
