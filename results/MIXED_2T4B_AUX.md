# Mixed 2T4B with router auxiliary loss

Same format as `MIXED_2T4B.md`, plus router load-balancing loss with coefficient
0.02.

## PPL

The raw training log prints 24.53, but that loss includes the auxiliary router
term. The fair cross-entropy PPL, evaluated with `output_router_logits=False`,
is **23.55**.

| variant | PPL |
|---|---:|
| Mixed 2T4B, no aux | **23.25** |
| Mixed 2T4B, aux 0.02 | 23.55 |
| Ternary reference | 21.44 |
| FP32 control | 19.56 |

Aux loss costs about 0.29 PPL.

## Routing

Layer 5 expert usage:

| expert | ternary | binary | mixed no aux | mixed aux 0.02 |
|---|---:|---:|---:|---:|
| 0 | 22.7% | 14.0% | 13.5% | 12.8% |
| 1 | 50.0% | 38.7% | 34.9% | 32.9% |
| 2 | 18.4% | 47.2% | 49.9% | 50.0% |
| 3 | 8.9% | 0.15% | 1.71% | **4.25%** |

Entropy L5: ternary 1.21, binary 1.006, mixed no aux 1.054, mixed aux 1.110.

Aux loss revives expert 3 from 1.71 to 4.25 percent and improves entropy, but it
does not fully restore the ternary routing. The 50 percent concentration on
expert 2 remains.

## Verdict

Aux loss helps routing at a small PPL cost. It is a conditional pass, not a
clean win. Next options:

- smaller aux coefficient, 0.005 to 0.01
- freeze the router during the format search
- per-expert mixed precision instead of per-layer
- longer training with the aux term

The 1.4 bpw mixed path is plausible but not settled.
