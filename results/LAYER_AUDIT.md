# Per-layer ternary audit

Baseline FP32 PPL: 16.1958.

For each layer:
- `only` means only this layer's experts are ternary.
- `all-but` means every layer except this one is ternary.

## Results

| layer | only PPL | only delta | all-but PPL | all-but delta | quant error |
|---:|---:|---:|---:|---:|---:|
| 0 | 75.9573 | +59.7615 | 925.2269 | +909.0312 | 0.4278 |
| 1 | 36.1246 | +19.9288 | 1516.2579 | +1500.0621 | 0.4236 |
| 2 | 30.8638 | +14.6680 | 1721.9775 | +1705.7817 | 0.4261 |
| 3 | 28.4103 | +12.2145 | 2157.3521 | +2141.1564 | 0.4286 |
| 4 | 28.0374 | +11.8416 | 2238.5966 | +2222.4009 | 0.4292 |
| 5 | 46.3727 | +30.1769 | 1590.3202 | +1574.1244 | 0.4312 |

## What this says

- Layer 0 and layer 5 are the most sensitive. This justifies keeping them in
  FP32 in the best recipe.
- The middle layers 2, 3 and 4 are the least sensitive individually, but
  removing any one layer from the ternary set is still catastrophic. The damage
  accumulates across depth.
- Quantization error is almost identical across layers, about 0.43. The
  difference in sensitivity is not explained by the error magnitude. It is a
  depth/position effect.

## Next

A selective precision scheme should protect layers 0 and 5 first. The middle
layers are the place to test a cheaper format, but only after the matched
FP32 control and seed variance are understood.
