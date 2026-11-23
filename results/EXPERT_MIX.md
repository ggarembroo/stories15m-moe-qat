# Per-expert mixed precision

In each layer, the most-used FP32 expert is ternary. The other three are binary.
Router, norms, embeddings and attention stay FP32. Aux loss 0.02.

Top expert per layer: L0 E0, L1 E0, L2 E2, L3 E2, L4 E3, L5 E1.

## PPL

| format | bpw | PPL | delta vs FP32 |
|---|---:|---:|---:|
| FP32 control, 3 seeds | 16 | 19.56 | 0 |
| Ternary, 3 seeds | 2.08 | 21.44 | +1.88 |
| Mixed 2T4B | 1.397 | 23.25 | +3.69 |
| **Expert mix 1T3B** | **1.311** | **23.81** | **+4.25** |
| Binary | 1.055 | 25.17 | +5.61 |

## Routing

Layer 5 expert 3:

| format | expert 3 | entropy |
|---|---:|---:|
| Ternary | 8.90% | 1.210 |
| Mixed 2T4B | 1.71% | 1.054 |
| Mixed + aux 0.02 | 4.25% | 1.110 |
| Expert mix + aux 0.02 | **5.61%** | **1.116** |
| Binary | 0.15% | 1.006 |

Per-expert mixing gives the best routing among the 1.3 to 1.4 bpw formats, but
the PPL is worse than mixed 2T4B by 0.56.

## Verdict

At 1.311 bpw the expert-only weight file for 35B is about 5.74 GB. That leaves
almost nothing for attention, embeddings, router, lm head, vision encoder and KV
cache. So even this format is not a guaranteed 6 GB model.

Quality-wise, 23.81 is 4.25 PPL above the FP32 control. That is outside the
strict pass threshold of 4 PPL, and routing is not fully restored. Per-layer and
per-expert mixing are saturated at this budget.

The next step is not another mixing scheme. It is a full 6 GB accounting with
non-expert weights, then a decision: smaller architecture or a different
quantizer class.
