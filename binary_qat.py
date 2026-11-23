"""Binary-expert QAT for the 36M MoE.

All routed experts are binary. Each output row has one learnable absmean scale.
Router, norms, embeddings, attention and LM head stay FP32 in this arm.

This is the first format test for a 6 GB 35B target: binary experts are about
1.0 bpw before non-expert weights.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, MixtralForCausalLM

from qat import QATExperts, count_parameters
from train_qat import evaluate, make_chunks


class BinaryExperts(nn.Module):
    def __init__(self, orig, copy_weights: bool = True):
        super().__init__()
        self.num_experts = orig.num_experts
        self.hidden_dim = orig.hidden_dim
        self.intermediate_dim = orig.intermediate_dim
        self.act_fn = orig.act_fn
        if copy_weights:
            self.gate_up_proj = nn.Parameter(orig.gate_up_proj.detach().clone())
            self.down_proj = nn.Parameter(orig.down_proj.detach().clone())
        else:
            self.gate_up_proj = orig.gate_up_proj
            self.down_proj = orig.down_proj
        self.gate_log_scale = nn.Parameter(self._init_absmean(self.gate_up_proj))
        self.down_log_scale = nn.Parameter(self._init_absmean(self.down_proj))

    @staticmethod
    def _init_absmean(w: torch.Tensor) -> torch.Tensor:
        scale = w.detach().float().abs().mean(dim=2, keepdim=True).clamp_min(1e-4)
        return torch.log(scale)

    def _quant(self, w: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
        scale = torch.exp(log_scale.clamp(-12, 4))
        q = torch.where(w >= 0, scale, -scale).to(w.dtype)
        return (w + (q - w).detach() + (q - q.detach())).to(w.dtype)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        wg = self._quant(self.gate_up_proj, self.gate_log_scale)
        wd = self._quant(self.down_proj, self.down_log_scale)
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            x = hidden_states[token_idx]
            gate, up = F.linear(x, wg[expert_idx]).chunk(2, dim=-1)
            h = self.act_fn(gate) * up
            y = F.linear(h, wd[expert_idx])
            y = y * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, y.to(final.dtype))
        return final


def parse_layers(value: str) -> set[int]:
    if not value:
        return set()
    return {int(x) for x in value.split(",") if x.strip()}


def replace_experts(model, ternary_layers: set[int], outlier_frac: float, clip_percentile: float):
    for i, layer in enumerate(model.model.layers):
        if i in ternary_layers:
            layer.mlp.experts = QATExperts(
                layer.mlp.experts,
                bits=1,
                group=32,
                scale_mode="group",
                learnable_scales=True,
                outlier_frac=outlier_frac,
                clip_percentile=clip_percentile,
                copy_weights=True,
            )
        else:
            layer.mlp.experts = BinaryExperts(layer.mlp.experts)
    return model


def run(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    raw = Path(args.corpus).read_text(encoding="utf-8", errors="ignore")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.model_max_length = 10_000_000
    train_chunks = make_chunks(tokenizer, raw[: args.train_chars], args.seq_len, args.max_train_chunks)
    val_chunks = make_chunks(
        tokenizer,
        raw[args.train_chars : args.train_chars + args.val_chars],
        args.seq_len,
        args.max_val_chunks,
    )

    model = MixtralForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float32).to(device)
    model.config.output_router_logits = args.router_aux_coef > 0
    fp32_loss, fp32_ppl = evaluate(model, val_chunks, device, batch_size=8)
    print(f"fp32 ppl {fp32_ppl:.4f}", flush=True)

    ternary_layers = parse_layers(args.ternary_layers)
    replace_experts(model, ternary_layers, args.outlier_frac, args.clip_percentile)
    ptq_loss, ptq_ppl = evaluate(model, val_chunks, device, batch_size=8)
    print(f"binary ptq ppl {ptq_ppl:.4f}", flush=True)

    for p in model.parameters():
        p.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad]
    scale_params = [p for n, p in model.named_parameters() if p.requires_grad and "log_scale" in n]
    other_params = [p for n, p in model.named_parameters() if p.requires_grad and "log_scale" not in n]
    groups = [{"params": other_params, "lr": args.lr}]
    if scale_params:
        groups.append({"params": scale_params, "lr": args.scale_lr if args.scale_lr is not None else args.lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    total, trainable = count_parameters(model)
    print(f"params {total:,} trainable {trainable:,}", flush=True)

    rng = random.Random(args.seed)
    model.train()
    started = time.time()
    history = []
    for step in range(args.steps):
        ids = torch.stack([train_chunks[rng.randrange(len(train_chunks))] for _ in range(args.batch_size)]).to(device)
        out = model(ids, labels=ids)
        loss = out.loss
        aux = getattr(out, "aux_loss", None)
        if args.router_aux_coef > 0 and aux is not None:
            loss = loss + args.router_aux_coef * aux
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        elapsed = time.time() - started
        history.append({"step": step + 1, "loss": loss.item(), "elapsed_s": elapsed})
        if (step + 1) % args.log_every == 0:
            print(f"step {step + 1}/{args.steps} loss {loss.item():.5f} {elapsed:.1f}s", flush=True)

    after_loss, after_ppl = evaluate(model, val_chunks, device, batch_size=8)
    print(f"after ppl {after_ppl:.4f}", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "binary_qat.pt")
    import csv

    with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "loss", "elapsed_s"])
        writer.writeheader()
        writer.writerows(history)

    metrics = {
        "format": "mixed_binary_ternary" if ternary_layers else "binary_experts",
        "ternary_layers": sorted(ternary_layers),
        "model_id": args.model_id,
        "device": str(device),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "lr": args.lr,
        "scale_lr": args.scale_lr if args.scale_lr is not None else args.lr,
        "router_aux_coef": args.router_aux_coef,
        "seed": args.seed,
        "parameters": total,
        "trainable": trainable,
        "fp32_ppl": fp32_ppl,
        "ptq_ppl": ptq_ppl,
        "after_ppl": after_ppl,
        "wall_time_s": time.time() - started,
        "approx_bpw": (
            len(ternary_layers) * 2.08
            + (len(model.model.layers) - len(ternary_layers)) * 1.055
        )
        / len(model.model.layers),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=r"C:\kruko\models\stories15m-moe")
    parser.add_argument("--corpus", default=r"C:\kruko\models\stories15m-moe\TinyStories-valid.txt")
    parser.add_argument("--output-dir", default="runs/binary-s42")
    parser.add_argument("--ternary-layers", default="", help="comma-separated layers kept ternary")
    parser.add_argument("--outlier-frac", type=float, default=0.001)
    parser.add_argument("--clip-percentile", type=float, default=0.999)
    parser.add_argument("--router-aux-coef", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--scale-lr", type=float, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--train-chars", type=int, default=2_000_000)
    parser.add_argument("--val-chars", type=int, default=500_000)
    parser.add_argument("--max-train-chunks", type=int, default=6000)
    parser.add_argument("--max-val-chunks", type=int, default=40)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
