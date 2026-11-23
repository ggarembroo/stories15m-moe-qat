# Scale LR sweep

Both runs use the 36M MoE, ternary experts, group-32 scales, outlier 0.1%,
p99.9 calibration, layers 0 and 5 in FP32, 600 steps, batch 4, seq 128.

## PPL

| scale_lr | PPL | wall time |
|---:|---:|---:|
| 3e-4 (1x) | **21.4886** | 439.8 s |
| 3e-3 (10x) | 21.5284 | 565.0 s |

10x scale LR is slightly worse. The difference is small, but there is no reason
to pay for it.

## Scale movement

1x run

| metric | value |
|---|---:|
| mean relative change | 0.53 to 0.56% |
| median | 0.45 to 0.48% |
| p90 | about 1.1% |
| max | 3.7% |
| groups below 1% | 85 to 87% |
| groups above 10% | 0% |
| collapsed groups | 0 |

10x run

| metric | value |
|---|---:|
| mean relative change | 5.2 to 5.5% |
| median | 4.4 to 4.8% |
| p90 | about 17% |
| max | 28 to 42% |
| groups below 1% | 11 to 15% |
| groups above 10% | 12 to 15% |
| collapsed groups | 0 |

## What this means

The 10x run moves the scales about ten times more, but PPL does not improve. The
1x run is already near the optimum for this recipe. More scale movement is not
the bottleneck.

No group collapsed in either run. Outlier groups did not move more than the
rest.

## Next

Do not keep tuning scale LR. Change the quantization scheme instead:

- per-channel ternary instead of per-group
- 2-bit for outlier channels
- separate treatment for router weights
