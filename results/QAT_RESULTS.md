# Tiny MoE QAT results

Model: `ggml-org/stories15M_MOE`

Mixtral architecture: 6 layers, hidden 288, 4 experts, top-2, FF 768, 36M params.

Corpus: first 2 MB of `TinyStories-valid.txt` for training, next 0.5 MB held out.
PPL is on the held-out 40 chunks of 128 tokens.

## Results

| variant | PPL | vs FP |
|---|---:|---:|
| FP32 baseline | 16.20 | 1.00x |
| ternary PTQ, no training | 3404.56 | 210x |
| QAT experts only, fixed scales, 600 steps | 30.36 | 1.87x |
| QAT experts only, fixed scales, 1000 steps | 29.82 | 1.84x |
| QAT all params, fixed scales, 600 steps | 24.67 | 1.52x |
| QAT all params, learnable scales, 600 steps | 24.19 | 1.49x |
| QAT all params, learnable scales + block-32 Hadamard, 600 steps | 25.18 | 1.55x |
| QAT all params + outlier 0.1% + p99.9 calibration | 21.95 | 1.35x |
| same + layers 0 and 5 kept FP32 | **21.78** | **1.34x** |

## What this says

1. **Post-training ternary is unusable.** PPL 3404 vs 16.2.
2. **QAT is the main lever.** 3404 -> 24.2 in 600 steps, 1.5x FP.
3. **Unfreezing the whole model matters.** Experts only: 29.8. All params: 24.2.
4. **Learnable per-group scales are a small gain.** 24.67 -> 24.19, about 2%.
5. **Block-32 Hadamard does not help when QAT is used.** 24.19 vs 25.18, slightly worse.
6. **Outlier protection plus calibration is the largest single gain.** 24.19 -> 21.95.
7. **Keeping layers 0 and 5 in FP32 gives a small extra gain.** 21.95 -> 21.78.
8. **Longer training on the same 2 MB plateaued.** 600 and 1000 steps are both around 30 for experts-only.

## Caveats

- 36M is a toy. A 7B MoE may behave differently.
- The Hadamard here is block-diagonal 32, not a full-width learned transform.
- Only 2 MB of training text and 600 steps. Longer training with more data may
  change the ranking.
- Full QAT of a 7B model does not fit a 24 GB desktop GPU: weights + gradients +
  Adam states exceed 24 GB.

## Reproduce

```bash
python train_qat.py --preset best
```

The script writes `runs/best/metrics.json` and `runs/best/history.csv`.

## Reproduced 2026-11-23

The best recipe was rerun on CPU with the scale-gradient fix.

| stage | PPL |
|---|---:|
| FP32 | 16.1958 |
| ternary PTQ + outlier + mixed | 349.5014 |
| after 600 steps QAT | **21.4886** |

Earlier 21.78 and 21.88 runs trained weights but not the scales. The fixed
estimator gives 21.49. Full report in `BEST_RUN_2026-11-23.md`.
