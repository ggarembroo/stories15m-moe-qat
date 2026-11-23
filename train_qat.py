"""Quantization-aware training for ggml-org/stories15M_MOE.

Presets:
    best      ternary, learnable group scales, outlier 0.1%, p99.9 calibration,
              layers 0 and 5 kept FP32 (mixed precision), 600 steps.
    ternary   ternary, learnable group scales, no outlier, no mixed precision.
    fp32      no quantization, evaluation only (use --steps 0).

Example:
    python train_qat.py --preset best
    python train_qat.py --preset best --steps 100 --batch-size 2
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, MixtralForCausalLM

from qat import QATExperts, count_parameters, replace_experts


MODEL_ID = "ggml-org/stories15M_MOE"
CORPUS_REPO = "roneneldan/TinyStories"
CORPUS_FILE = "TinyStories-valid.txt"


def parse_skip_layers(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(x) for x in value.split(",") if x.strip())


def ensure_corpus(path: Path) -> Path:
    if path.exists():
        return path
    from huggingface_hub import hf_hub_download

    path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=CORPUS_REPO,
        filename=CORPUS_FILE,
        repo_type="dataset",
    )
    return Path(downloaded)


def make_chunks(tokenizer, text: str, seq_len: int, limit: int | None = None):
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    chunks = []
    for start in range(0, len(ids) - seq_len, seq_len):
        chunks.append(ids[start : start + seq_len])
        if limit is not None and len(chunks) >= limit:
            break
    return chunks


@torch.no_grad()
def evaluate(model, chunks, device, batch_size: int = 8):
    model.eval()
    losses = []
    for start in range(0, len(chunks), batch_size):
        batch = torch.stack(chunks[start : start + batch_size]).to(device)
        loss = model(batch, labels=batch).loss
        losses.append(loss.item())
    mean_loss = sum(losses) / len(losses)
    return mean_loss, math.exp(mean_loss)


def load_model(args, device):
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.model_max_length = 10_000_000
    model = MixtralForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float32)
    return tokenizer, model.to(device)


def quantize_model(args, model):
    if args.preset == "fp32":
        return model
    replace_experts(
        model,
        bits=args.bits,
        group=args.group,
        scale_mode=args.scale_mode,
        learnable_scales=not args.fixed_scales,
        outlier_frac=args.outlier_frac,
        outlier_group_frac=args.outlier_group_frac,
        outlier_bits=args.outlier_bits,
        clip_percentile=args.clip_percentile,
        rotate=args.rotate,
        block=args.block,
        skip_layers=parse_skip_layers(args.skip_layers),
    )
    return model


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device}")

    corpus_path = ensure_corpus(Path(args.corpus))
    raw = corpus_path.read_text(encoding="utf-8", errors="ignore")
    train_text = raw[: args.train_chars]
    val_text = raw[args.train_chars : args.train_chars + args.val_chars]

    tokenizer, model = load_model(args, device)
    train_chunks = make_chunks(tokenizer, train_text, args.seq_len, args.max_train_chunks)
    val_chunks = make_chunks(tokenizer, val_text, args.seq_len, args.max_val_chunks)
    print(f"train chunks: {len(train_chunks)}  val chunks: {len(val_chunks)}")

    fp32_loss, fp32_ppl = evaluate(model, val_chunks, device)
    print(f"fp32: loss {fp32_loss:.5f}  ppl {fp32_ppl:.4f}")

    model = quantize_model(args, model)
    ptq_loss, ptq_ppl = evaluate(model, val_chunks, device)
    print(f"ptq:  loss {ptq_loss:.5f}  ppl {ptq_ppl:.4f}")

    total, trainable = count_parameters(model)
    print(f"parameters: {total:,} total  {trainable:,} trainable")

    if args.steps == 0:
        metrics = {
            "preset": args.preset,
            "fp32_loss": fp32_loss,
            "fp32_ppl": fp32_ppl,
            "ptq_loss": ptq_loss,
            "ptq_ppl": ptq_ppl,
            "after_loss": ptq_loss,
            "after_ppl": ptq_ppl,
            "parameters": total,
            "trainable": trainable,
            "steps": 0,
        }
        print(json.dumps(metrics, indent=2))
        return metrics

    if args.train_scope == "experts":
        for name, p in model.named_parameters():
            p.requires_grad = "experts" in name
    else:
        for p in model.parameters():
            p.requires_grad = True

    params = [p for p in model.parameters() if p.requires_grad]
    scale_params = [p for n, p in model.named_parameters() if p.requires_grad and "log_scale" in n]
    other_params = [p for n, p in model.named_parameters() if p.requires_grad and "log_scale" not in n]
    scale_lr = args.scale_lr if args.scale_lr is not None else args.lr
    groups = [{"params": other_params, "lr": args.lr}]
    if scale_params:
        groups.append({"params": scale_params, "lr": scale_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    rng = random.Random(args.seed)
    model.train()
    started = time.time()
    history = []

    for step in range(args.steps):
        ids = torch.stack([train_chunks[rng.randrange(len(train_chunks))] for _ in range(args.batch_size)]).to(device)
        out = model(ids, labels=ids)
        optimizer.zero_grad(set_to_none=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        elapsed = time.time() - started
        history.append({"step": step + 1, "loss": out.loss.item(), "elapsed_s": elapsed})
        if (step + 1) % args.log_every == 0:
            print(f"step {step + 1}/{args.steps}  loss {out.loss.item():.5f}  {elapsed:.1f}s")

    after_loss, after_ppl = evaluate(model, val_chunks, device)
    print(f"after: loss {after_loss:.5f}  ppl {after_ppl:.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"qat_{args.preset}.pt"
    torch.save(model.state_dict(), checkpoint)
    import csv

    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "loss", "elapsed_s"])
        writer.writeheader()
        writer.writerows(history)
    metrics = {
        "preset": args.preset,
        "model_id": args.model_id,
        "device": str(device),
        "bits": args.bits,
        "group": args.group,
        "scale_mode": args.scale_mode,
        "learnable_scales": not args.fixed_scales,
        "outlier_frac": args.outlier_frac,
        "outlier_group_frac": args.outlier_group_frac,
        "outlier_bits": args.outlier_bits,
        "clip_percentile": args.clip_percentile,
        "rotate": args.rotate,
        "skip_layers": list(parse_skip_layers(args.skip_layers)),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "lr": args.lr,
        "scale_lr": scale_lr,
        "parameters": total,
        "trainable": trainable,
        "fp32_loss": fp32_loss,
        "fp32_ppl": fp32_ppl,
        "ptq_loss": ptq_loss,
        "ptq_ppl": ptq_ppl,
        "after_loss": after_loss,
        "after_ppl": after_ppl,
        "wall_time_s": time.time() - started,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved: {checkpoint}")
    return metrics


def apply_preset(args):
    if args.preset == "fp32":
        return args
    if args.preset == "best":
        args.bits = 1
        args.group = 32
        args.scale_mode = "group"
        args.outlier_frac = 0.001
        args.clip_percentile = 0.999
        args.skip_layers = "0,5"
    elif args.preset == "ternary":
        args.bits = 1
        args.group = 32
        args.scale_mode = "group"
        args.outlier_frac = 0.0
        args.clip_percentile = 1.0
        args.skip_layers = ""
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["best", "ternary", "fp32"], default="best")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--corpus", default="data/TinyStories-valid.txt")
    parser.add_argument("--output-dir", default="runs/best")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--scale-lr", type=float, default=None, help="separate LR for log_scale parameters")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--train-chars", type=int, default=2_000_000)
    parser.add_argument("--val-chars", type=int, default=500_000)
    parser.add_argument("--max-train-chunks", type=int, default=6000)
    parser.add_argument("--max-val-chunks", type=int, default=40)
    parser.add_argument("--train-scope", choices=["all", "experts"], default="all")
    parser.add_argument("--bits", type=int, default=1)
    parser.add_argument("--group", type=int, default=32)
    parser.add_argument("--scale-mode", choices=["group", "row", "col"], default="group")
    parser.add_argument("--fixed-scales", action="store_true")
    parser.add_argument("--outlier-frac", type=float, default=0.001)
    parser.add_argument("--outlier-group-frac", type=float, default=0.0)
    parser.add_argument("--outlier-bits", type=int, default=2)
    parser.add_argument("--clip-percentile", type=float, default=0.999)
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--block", type=int, default=32)
    parser.add_argument("--skip-layers", default="0,5")
    args = parser.parse_args()
    args = apply_preset(args)
    metrics = train(args)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
