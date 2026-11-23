# Mixed 2T4B aux sweep

All runs are 600 steps, seed 42. PPL is the fair cross-entropy, evaluated with
`output_router_logits=False`. Raw training logs include the aux term and are not
comparable.

| variant | fair PPL | L5 expert 3 | L5 entropy |
|---|---:|---:|---:|
| mixed, no aux | 23.25 | 1.71% | 1.054 |
| aux 0.005 | **23.14** | 2.50% | 1.076 |
| aux 0.01 | 23.77 | 2.11% | 1.097 |
| aux 0.02 | 23.55 | **4.25%** | **1.110** |
| aux 0.01, frozen router | 23.23 | 1.95% | 1.037 |
| ternary reference | 21.44 | 8.90% | 1.210 |

## What this says

- Aux 0.005 gives the best PPL, 23.14, slightly better than no aux. Routing is
  still poor, expert 3 at 2.5 percent.
- Aux 0.02 gives the best routing, expert 3 at 4.25 percent, but costs 0.4 PPL.
- Freezing the router does not help. The collapse is driven by upstream binary
  representations, not by router adaptation alone.
- No setting restores expert 3 to the ternary 8.9 percent. Layer 2 still takes
  about 50 percent of tokens.

## Verdict

Per-layer mixed is saturated. The next experiment is per-expert mixing: keep
the most-used expert in each layer ternary, make the rest binary. That is about
1.31 bpw per expert, inside the 1.37 budget, and it protects the expert that
actually carries the tokens.
