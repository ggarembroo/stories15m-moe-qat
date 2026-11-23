# Mixed binary/ternary, 2 ternary + 4 binary layers

Layers 0 and 5 stay ternary. Layers 1 to 4 are binary. Effective expert bpw is
about 1.397.

## PPL

| format | bpw | PPL |
|---|---:|---:|
| FP32 control, 3 seeds | 16 | 19.56 В± 0.44 |
| Ternary, 3 seeds | 2.08 | 21.44 В± 0.16 |
| Binary | 1.055 | 25.17 |
| Mixed 2T4B | 1.397 | **23.25** |

Mixed is 3.69 PPL above the FP32 control, about 18.9 percent. It is within the
4 PPL pass threshold, but not below the 15 percent informal target.

## Routing

Layer 5 expert usage:

| expert | ternary | binary | mixed 2T4B |
|---|---:|---:|---:|
| 0 | 22.7% | 14.0% | 13.5% |
| 1 | 50.0% | 38.7% | 34.9% |
| 2 | 18.4% | 47.2% | 49.9% |
| 3 | 8.9% | 0.15% | 1.71% |

Layer 5 entropy: ternary 1.21, binary 1.006, mixed 1.054.

The mixed format revives expert 3 from 0.15 to 1.71 percent, but it is still far
below the ternary 8.9 percent. Other layers have entropy 1.33 to 1.38, close to
ternary. The layer 5 collapse is not fixed by keeping layer 5 ternary, because
the upstream binary layers shift the representations the router sees.

## Verdict

PPL is a conditional pass. Routing is still a fail at layer 5. The next test is
the same mixed format with a router load-balancing auxiliary loss, to see
whether routing can be kept healthy without changing the bit budget.
