"""Per-expert mixed precision QAT for the 36M MoE.

In each layer, the most-used expert stays ternary. The other experts are binary.
The router, norms, embeddings and attention stay FP32.

For 4 experts this is about 1.31 bpw, inside the 1.37 bpw budget for 35B in
6 GB before non-expert weights are counted.
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

from qat import count_parameters
from train_qat import evaluate, make_chunks


class ExpertMixExperts(nn.Module):
    def __init__(self, orig, ternary_idx: tuple[int, ...], group: int = 32):
        super().__init__()
        self.num_experts = orig.num_experts
        self.hidden_dim = orig.hidden_dim
        self.intermediate_dim = orig.intermediate_dim
        self.act_fn = orig.act_fn
        self.group = group
        self.gate_up_proj = nn.Parameter(orig.gate_up_proj.detach().clone())
        self.down_proj = nn.Parameter(orig.down_proj.detach().clone())
        mask = torch.zeros(self.num_experts, dtype=torch.bool)
        mask[list(ternary_idx)] = True
        self.register_buffer("ternary_mask", mask)

        self.gate_tern_log_scale = nn.Parameter(self._init_tern(self.gate_up_proj))
        self.down_tern_log_scale = nn.Parameter(self._init_tern(self.down_proj))
        self.gate_bin_log_scale = nn.Parameter(self._init_bin(self.gate_up_proj))
        self.down_bin_log_scale = nn.Parameter(self._init_bin(self.down_proj))

    def _init_tern(self, w: torch.Tensor) -> torch.Tensor:
        e, rows, cols = w.shape
        blocks = w.detach().float().reshape(e, rows, cols // self.group, self.group)
        scale = torch.quantile(blocks.abs(), 0.999, dim=-1).clamp_min(1e-4)
        return torch.log(scale)

    def _init_bin(self, w: torch.Tensor) -> torch.Tensor:
        scale = w.detach().float().abs().mean(dim=2, keepdim=True).clamp_min(1e-4)
        return torch.log(scale)

    def _ternary(self, w: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
        e, rows, cols = w.shape
        blocks = w.reshape(e, rows, cols // self.group, self.group)
        scale = torch.exp(log_scale.clamp(-12, 4)).unsqueeze(-1)
        q = torch.clamp(torch.round(blocks / scale), -1, 1) * scale
        return q.reshape(e, rows, cols)

    def _binary(self, w: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
        scale = torch.exp(log_scale.clamp(-12, 4))
        return torch.where(w >= 0, scale, -scale).to(w.dtype)

    def _quant(self, w: torch.Tensor, tern_log: torch.Tensor, bin_log: torch.Tensor) -> torch.Tensor:
        q_t = self._ternary(w, tern_log)
        q_b = self._binary(w, bin_log)
        q = torch.where(self.ternary_mask[:, None, None], q_t, q_b)
        return (w + (q - w).detach() + (q - q.detach())).to(w.dtype)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        wg = self._quant(self.gate_up_proj, self.gate_tern_log_scale, self.gate_bin_log_scale)
        wd = self._quant(self.down_proj, self.down_tern_log_scale, self.down_bin_log_scale)
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


def fp32_top_expert(model, chunks, device):
    counts = {i: torch.zeros(model.config.num_local_experts, dtype=torch.long) for i in range(len(model.model.layers))}
    hooks = []

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            idx = output[2]
            counts[layer_idx] += torch.bincount(idx.flatten().cpu(), minlength=model.config.num_local_experts)
        return hook

    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.mlp.gate.register_forward_hook(make_hook(i)))
    model.eval()
    with torch.no_grad():
        for chunk in chunks:
            model(chunk.unsqueeze(0).to(device))
    for hook in hooks:
        hook.remove()
    return {i: int(c.argmax()) for i, c in counts.items()}, {i: c.tolist() for i, c in counts.items()}


def fair_eval(model, chunks, device, batch_size=8):
    old = model.config.output_router_logits
    model.config.output_router_logits = False
    loss, ppl = evaluate(model, chunks, device, batch_size=batch_size)
    model.config.output_router_logits = old
    return loss, ppl


def routing_stats(model, chunks, device, num_experts):
    counts = {i: torch.zeros(num_experts, dtype=torch.long) for i in range(len(model.model.layers))}
    hooks = []

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            idx = output[2]
            counts[layer_idx] += torch.bincount(idx.flatten().cpu(), minlength=num_experts)
        return hook

    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.mlp.gate.register_forward_hook(make_hook(i)))
    model.eval()
    with torch.no_grad():
        for chunk in chunks:
            model(chunk.unsqueeze(0).to(device))
    for hook in hooks:
        hook.remove()
    out = {}
    for i, c in counts.items():
        total = c.sum().item()
        p = c.float() / max(total, 1)
        out[i] = {
            "counts": c.tolist(),
            "fractions": [round(float(x), 4) for x in p],
            "entropy": round(float(-(p * torch.log(p.clamp_min(1e-12))).sum()), 4),
            "max_fraction": round(float(p.max()), 4),
        }
    return out


def run(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.model_max_length = 10_000_000
    raw = Path(args.corpus).read_text(encoding="utf-8", errors="ignore")
    chunks = make_chunks(tokenizer, raw[: args.train_chars], args.seq_len, args.max_train_chunks)
    val_chunks = make_chunks(tokenizer, raw[args.train_chars : args.train_chars + args.val_chars], args.seq_len, args.max_val_chunks)

    model = MixtralForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float32).to(device)
    model.config.output_router_logits = args.router_aux_coef > 0
    fp32_loss, fp32_ppl = fair_eval(model, val_chunks, device, batch_size=8)
    print(f"fp32 ppl {fp32_ppl:.4f}", flush=True)

    top_expert, usage = fp32_top_expert(model, chunks[:8], device)
    print("top expert per layer", top_expert, flush=True)

    for i, layer in enumerate(model.model.layers):
        layer.mlp.experts = ExpertMixExperts(layer.mlp.experts, (top_expert[i],), group=args.group)
    ptq_loss, ptq_ppl = fair_eval(model, val_chunks, device, batch_size=8)
    print(f"expert-mix ptq ppl {ptq_ppl:.4f}", flush=True)

    for name, p in model.named_parameters():
        p.requires_grad = not (args.freeze_router and ".gate." in name)
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
        ids = torch.stack([chunks[rng.randrange(len(chunks))] for _ in range(args.batch_size)]).to(device)
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

    after_loss, after_ppl = fair_eval(model, val_chunks, device, batch_size=8)
    routing = routing_stats(model, val_chunks, device, model.config.num_local_experts)
    print(f"after ppl {after_ppl:.4f}", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "expert_mix_qat.pt")
    (out_dir / "routing.json").write_text(json.dumps(routing, indent=2), encoding="utf-8")
    metrics = {
        "format": "expert_mix",
        "top_expert": top_expert,
        "fp32_usage": usage,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "seed": args.seed,
        "router_aux_coef": args.router_aux_coef,
        "freeze_router": bool(args.freeze_router),
        "fp32_ppl": fp32_ppl,
        "ptq_ppl": ptq_ppl,
        "after_ppl": after_ppl,
        "parameters": total,
        "trainable": trainable,
        "approx_bpw": (2.08 + 3 * 1.055) / 4,
        "routing": routing,
        "wall_time_s": time.time() - started,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "routing"}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=r"C:\kruko\models\stories15m-moe")
    parser.add_argument("--corpus", default=r"C:\kruko\models\stories15m-moe\TinyStories-valid.txt")
    parser.add_argument("--output-dir", default="runs/expert-mix-s42")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--scale-lr", type=float, default=None)
    parser.add_argument("--group", type=int, default=32)
    parser.add_argument("--router-aux-coef", type=float, default=0.02)
    parser.add_argument("--freeze-router", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--train-chars", type=int, default=2_000_000)
    parser.add_argument("--val-chars", type=int, default=500_000)
    parser.add_argument("--max-train-chunks", type=int, default=6000)
    parser.add_argument("--max-val-chunks", type=int, default=40)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
