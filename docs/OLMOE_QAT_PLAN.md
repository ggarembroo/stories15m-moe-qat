# OLMoE-1B-7B QAT plan

OLMoE is 6.9B total parameters, 1B active. Full QAT is a different class of job
from the 36M testbed.

## Why a 2-layer slice is not enough

Removing 14 of 16 layers destroys the residual stream. A 2-layer slice is not a
valid language model and its PPL is meaningless. Use the slice only to check
that the QAT module works with `OlmoeExperts`. Do not report its PPL as a result.

## Memory math

Approximate, BF16 weights and FP32 Adam.

| Item | Full model |
|---|---:|
| Weights | 13.8 GB |
| Gradients | 13.8 GB |
| Adam m + v | 55.3 GB |
| **Total** | **~83 GB** |
| Activations | extra |

With 8-bit Adam the optimizer state is ~13.8 GB, so the total is still ~41 GB
before activations. A 40 GB GPU is tight. An 80 GB GPU is comfortable.

Freezing attention, embeddings and the router and training experts only does not
save enough. The expert weights are most of the model.

## Options

### A. Partial layer QAT on CPU or 24 GB GPU

Keep all 16 layers for a valid forward pass. Quantize and train only selected
layers, for example 2 layers. Freeze the rest. This fits 24 GB but is slow on
CPU. It answers a narrow question: does QAT on a subset improve PPL without
breaking the model?

This is the cheapest valid experiment. It is not full QAT. The script is
`olmoe_partial_qat.py`.

### B. Full QAT on one 80 GB GPU

Load BF16, train all experts, use 8-bit Adam and gradient checkpointing. This is
the minimum for a real full-model run. Expect tight memory and a slow step.

### C. Full QAT on 4x H100 80 GB

Use FSDP or ZeRO-3. This is the safe path for 10B tokens. Budget from the
earlier estimate: 300 to 800 USD for a 7B-class run, depending on provider and
spot pricing.

## Recommended order

1. Run partial layer QAT on 2 layers. Check that PPL improves and that the
   quantized model still generates coherent text.
2. If the direction is good, rent one 80 GB GPU and run expert-only full QAT.
3. If that works, run 4x H100 for the full recipe and more tokens.
4. Only then decide whether 35B-A3B is realistic.

## What is already done

- 36M MoE QAT is reproduced at 21.88 PPL.
- OLMoE PTQ baselines exist: FP 6.43, ternary 69.07, mixed 41.13, calibration
  70.60.
- The QAT module works with `OlmoeExperts`; the slice smoke test runs.

## What is not done

- No valid full OLMoE QAT run.
- No multi-GPU FSDP script yet.
- No 35B QAT run.

The next honest step is partial layer QAT, not another slice PPL number.
