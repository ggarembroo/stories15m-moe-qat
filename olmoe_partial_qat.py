"""Valid partial-layer QAT for OLMoE-1B-7B.

All 16 layers stay in the forward pass, so PPL is meaningful. Only the routed
experts of selected layers are quantized and trained. The rest is frozen.

This is the cheapest valid OLMoE experiment. It fits a 24 GB CPU/GPU when two
layers are selected. Full QAT needs an 80 GB GPU or multi-GPU FSDP; see
docs/OLMOE_QAT_PLAN.md.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, OlmoeForCausalLM

from qat import QATExperts, count_parameters


MODEL_ID = "allenai/OLMoE-1B-7B-0924"


def parse_layers(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(x) for x in value.split(",") if x.strip())


def choose_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.bfloat16


def ensure_corpus(path: Path) -> Path:
    if path.exists():
        return path
    from datasets import load_dataset

    path.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(row["text"] for row in ds)
    path.write_text(text, encoding="utf-8")
    return path


def make_chunks(tokenizer, text: str, seq_len: int, limit: int | None = None):
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    chunks = []
    for start in range(0, len(ids) - seq_len, seq_len):
        chunks.append(ids[start : start + seq_len])
        if limit is not None and len(chunks) >= limit:
            break
    return chunks


@torch.no_grad()
def evaluate(model, chunks, device, batch_size: int = 1):
    model.eval()
    losses = []
    for start in range(0, len(chunks), batch_size):
        batch = torch.stack(chunks[start : start + batch_size]).to(device)
        losses.append(model(batch, labels=batch).loss.item())
    loss = sum(losses) / len(losses)
    return loss, math.exp(loss)


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = choose_dtype(args.dtype)

    corpus = ensure_corpus(Path(args.corpus))
    raw = corpus.read_text(encoding="utf-8", errors="ignore")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.model_max_length = 10_000_000

    print(f"loading {args.model_id} dtype={dtype}", flush=True)
    model = OlmoeForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)

    train_chunks = make_chunks(tokenizer, raw[: args.train_chars], args.seq_len, args.max_train_chunks)
    val_chunks = make_chunks(
        tokenizer,
        raw[args.train_chars : args.train_chars + args.val_chars],
        args.seq_len,
        args.max_val_chunks,
    )
    print(f"train chunks {len(train_chunks)} val chunks {len(val_chunks)}", flush=True)
    if not val_chunks:
        raise SystemExit("no validation chunks, increase --val-chars or lower --seq-len")

    fp32_loss, fp32_ppl = evaluate(model, val_chunks, device, args.eval_batch_size)
    print(f"fp32 ppl {fp32_ppl:.4f}", flush=True)

    quant_layers = set(parse_layers(args.quant_layers))
    replaced = []
    for i, layer in enumerate(model.model.layers):
        if i not in quant_layers:
            continue
        layer.mlp.experts = QATExperts(
            layer.mlp.experts,
            bits=args.bits,
            group=args.group,
            scale_mode=args.scale_mode,
            learnable_scales=not args.fixed_scales,
            outlier_frac=args.outlier_frac,
            clip_percentile=args.clip_percentile,
            rotate=args.rotate,
            block=args.block,
            copy_weights=False,
        )
        replaced.append(layer.mlp.experts)
    gc.collect()
    print(f"quantized layers {sorted(quant_layers)}", flush=True)

    ptq_loss, ptq_ppl = evaluate(model, val_chunks, device, args.eval_batch_size)
    print(f"ptq ppl {ptq_ppl:.4f}", flush=True)

    if args.train_scope == "all":
        for p in model.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = False
        for expert in replaced:
            for p in expert.parameters():
                p.requires_grad = True

    params = [p for p in model.parameters() if p.requires_grad]
    total, trainable = count_parameters(model)
    print(f"params {total:,} trainable {trainable:,}", flush=True)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, foreach=True)

    rng = random.Random(args.seed)
    model.train()
    started = time.time()
    history = []

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for _ in range(args.grad_accum):
            ids = torch.stack([train_chunks[rng.randrange(len(train_chunks))] for _ in range(args.batch_size)]).to(device)
            out = model(ids, labels=ids)
            (out.loss / args.grad_accum).backward()
            loss_value += out.loss.item() / args.grad_accum
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        elapsed = time.time() - started
        history.append({"step": step + 1, "loss": loss_value, "elapsed_s": elapsed})
        if (step + 1) % args.log_every == 0:
            print(f"step {step + 1}/{args.steps} loss {loss_value:.5f} {elapsed:.1f}s", flush=True)

    after_loss, after_ppl = evaluate(model, val_chunks, device, args.eval_batch_size)
    print(f"after ppl {after_ppl:.4f}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "olmoe_partial_qat.pt")
    import csv

    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "loss", "elapsed_s"])
        writer.writeheader()
        writer.writerows(history)

    metrics = {
        "model_id": args.model_id,
        "quant_layers": sorted(quant_layers),
        "device": str(device),
        "dtype": str(dtype),
        "bits": args.bits,
        "group": args.group,
        "scale_mode": args.scale_mode,
        "learnable_scales": not args.fixed_scales,
        "outlier_frac": args.outlier_frac,
        "clip_percentile": args.clip_percentile,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "seq_len": args.seq_len,
        "lr": args.lr,
        "parameters": total,
        "trainable": trainable,
        "fp32_ppl": fp32_ppl,
        "ptq_ppl": ptq_ppl,
        "after_ppl": after_ppl,
        "wall_time_s": time.time() - started,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--quant-layers", default="0,15")
    parser.add_argument("--train-scope", choices=["experts", "all"], default="experts")
    parser.add_argument("--corpus", default="data/wikitext2.txt")
    parser.add_argument("--output-dir", default="runs/olmoe-partial")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--train-chars", type=int, default=1_000_000)
    parser.add_argument("--val-chars", type=int, default=200_000)
    parser.add_argument("--max-train-chunks", type=int, default=2000)
    parser.add_argument("--max-val-chunks", type=int, default=40)
    parser.add_argument("--bits", type=int, default=1)
    parser.add_argument("--group", type=int, default=32)
    parser.add_argument("--scale-mode", choices=["group", "row", "col"], default="group")
    parser.add_argument("--fixed-scales", action="store_true")
    parser.add_argument("--outlier-frac", type=float, default=0.0)
    parser.add_argument("--clip-percentile", type=float, default=0.999)
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--block", type=int, default=32)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
