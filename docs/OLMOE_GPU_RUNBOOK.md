# OLMoE GPU runbook

CPU is closed. The 128-token step did not finish in 30 minutes. Use a GPU.

## Memory math

Base BF16 weights: 13.8 GB.

| trainable | grads BF16 | Adam FP32 | total with base | Adam 8-bit total |
|---|---:|---:|---:|---:|
| 2 layers, 0.83B | 1.7 GB | 6.6 GB | ~22 GB | ~18 GB |
| 8 layers, 3.3B | 6.6 GB | 26.4 GB | ~47 GB | ~27 GB |
| 16 layers, 6.4B | 12.8 GB | 51.2 GB | ~78 GB | ~39 GB |

So "16 layers full partial on L4 24 GB" is not possible with expert weights
trainable. L4 24 GB fits:

- 2 layers full expert QAT, with checkpointing if needed
- 16 layers with only scales trainable
- 16 layers with LoRA adapters, if the adapter path is implemented

## L4 24 GB

Recommended first run: layers 0 and 15, 100 steps, eval 2048 tokens.

```bash
python olmoe_partial_qat.py \
  --model-id allenai/OLMoE-1B-7B-0924 \
  --quant-layers 0,15 \
  --steps 100 \
  --batch-size 1 \
  --grad-accum 4 \
  --seq-len 128 \
  --max-val-chunks 16 \
  --device cuda \
  --dtype float16 \
  --outlier-frac 0.001 \
  --clip-percentile 0.999 \
  --output-dir runs/olmoe-l4-2layer
```

Expected: about 90 s/step after the L4 vs CPU correction, 100 steps about 2.5
hours, 0.5 to 1 USD at spot pricing.

## A100 40 GB

Same command with `--dtype bfloat16`. Two layers fit without checkpointing.
Four layers may fit with 8-bit Adam, but the script does not implement 8-bit
Adam yet.

## A100 80 GB

16 layers with 8-bit Adam is about 39 GB. That is the first configuration where
all experts can train. It still needs a real optimizer change, not just a flag.

## 4x H100 80 GB

Full QAT with FSDP or ZeRO-3. This is the only comfortable path for the full
recipe and more tokens.

## QLoRA on T4

Base in 4-bit, LoRA r=16 on all 16 layers. Adapters are 200 to 400 MB, Adam for
adapters about 1 GB, activations small. It fits 16 GB.

Caveat: OLMoE stores routed experts as fused 3D tensors, not `nn.Linear`. PEFT
LoRA does not attach to them out of the box. Attention and router LoRA is easy;
expert LoRA needs a custom wrapper.

## What to log

`olmoe_partial_qat.py` already logs:

- peak RAM after load and after quantization
- grad norm
- routing histogram for quantized layers
- PPL on `--max-val-chunks x seq_len` tokens

Add peak RAM after Adam and after the first step before trusting the memory
number.

## Next

1. L4, 2 layers, 100 steps, 2048-token eval.
2. If PPL moves from 14.44 toward 10 to 11, scale to 8 layers or add LoRA.
3. If PPL sticks at 14, the problem is the quantization scheme, not the number
   of layers.
