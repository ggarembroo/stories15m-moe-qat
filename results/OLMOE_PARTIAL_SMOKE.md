# OLMoE partial QAT smoke

Goal: check that the valid 16-layer partial QAT path works and measure memory.

Setup

- Model: `OLMoE-1B-7B`, BF16, CPU
- Quantized and trained layers: 0 and 15, the worst-case first and last layers
- Eval: 16 chunks x 128 tokens = 2048 tokens
- One training step, batch 1, seq 128

## Memory

| stage | peak RAM |
|---|---:|
| after model load | 13.02 GB |
| after quantization | 15.21 GB |

The base weights are shared, not cloned. `copy_weights=False` keeps the peak
close to the model size instead of doubling it.

## PPL

| stage | PPL |
|---|---:|
| FP32 | 12.10 |
| ternary PTQ + outlier + mixed | 14.44 |

This is a real 2048-token eval, not a 32-token noise sample.

## Step time

- 32-token step: 349 s
- 128-token step: did not finish within 30 minutes, killed

The 128-token full 16-layer backward is too slow on CPU. The process was killed
before grad norm and routing histogram could be written.

## Conclusion

Memory fits a 24 GB GPU. Compute does not fit CPU. The next run must be on a GPU,
for example L4 24 GB with gradient checkpointing or A100 40 GB.

## Next

- Run 100 steps on 2 layers on a GPU, eval 2048 tokens.
- Log grad norm and routing histogram for layers 0 and 15.
- If PPL moves from 14.44 toward 10 to 11, scale to more layers.
- Compare with QLoRA on a TQ1_0 base, LoRA r=16 on all 16 layers.
