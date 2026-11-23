# FP32 matched-compute control

The original FP32 baseline was evaluated before any continued training. That is
not the right control for a 600-step QAT run. This control trains the FP32 model
for the same 600 steps, same data order settings and same batch size.

## Results

| seed | FP32 PPL after 600 steps |
|---:|---:|
| 42 | 19.3580 |
| 43 | 19.2622 |
| 44 | 20.0638 |

Mean: **19.56 PPL**. Seed standard deviation: about **0.44 PPL**.

## Consequence

The old comparison was QAT 21.49 vs FP32 16.20, a 5.3 PPL gap. The matched
control is 19.56, so the actual QAT gap is about **1.9 PPL**, not 5.3.

The FP32 model overfits the 2 MB training slice. Validation loss rises during
continued training. That means the earlier "QAT damage" number was inflated by
comparing against an untrained model.

## What this changes

- Format rankings based on 21.49 vs 16.20 are invalid.
- The correct comparison is QAT vs FP32 at the same step count.
- A 0.06 PPL difference between QAT variants is below the FP32 seed noise of
  0.44 PPL. Do not treat 21.42 vs 21.49 as a real ranking without more seeds.
