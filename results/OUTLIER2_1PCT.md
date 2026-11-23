# 2-bit outlier groups, 1 percent

Top 1 percent of groups by max absolute weight are quantized with a fixed
4-level 2-bit quantizer. The rest stays ternary. `bpw` grows by about 0.01.

## PPL

| variant | PTQ | after 600 steps |
|---|---:|---:|
| ternary baseline | 349.50 | 21.4886 |
| 1% groups at 2-bit | **329.52** | **21.4242** |

The PTQ gain is real, 5.7 percent. After QAT the gain shrinks to 0.064 PPL.
That is below the 21.3 threshold set for adopting a new baseline.

## Scale movement

Overall mean relative change is 0.53 to 0.54 percent, the same as the baseline.

The split is different:

| group type | mean relative change |
|---|---:|
| outlier groups | 0.12 to 0.26 percent |
| rest | 0.53 to 0.55 percent |

The outlier groups barely move because their 2-bit quantizer is fixed, not
learnable. The QAT does not interact with them.

## Conclusion

The 2-bit outlier gain is mostly a PTQ effect. It does not stack with QAT.
Keep 21.49 as the baseline. The next test is 5 percent outlier groups, then a
per-channel 2-bit variant if 5 percent does not move below 21.3.
