"""Per-layer ternary sensitivity audit for the 36M MoE.

For each layer:
- only this layer quantized
- all layers except this one quantized

Also reports per-layer weight statistics and relative quantization error.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoTokenizer, MixtralForCausalLM

from qat import QATExperts
from train_qat import evaluate, make_chunks


def quant_error(w: torch.Tensor, group: int, percentile: float, outlier_frac: float) -> float:
    e, rows, cols = w.shape
    blocks = w.detach().float().reshape(e, rows, cols // group, group)
    scale = torch.quantile(blocks.abs(), percentile, dim=-1, keepdim=True).clamp_min(1e-8)
    q = torch.clamp(torch.round(blocks / scale), -1, 1) * scale
    q = q.reshape(e, rows, cols)
    if outlier_frac > 0:
        threshold = torch.quantile(w.detach().abs().float().flatten(), 1.0 - outlier_frac)
        q = torch.where(w.detach().abs() >= threshold, w.detach(), q)
    return float(((w.detach() - q).pow(2).sum() / w.detach().pow(2).sum()).item())


def weight_stats(w: torch.Tensor) -> dict:
    x = w.detach().float().flatten()
    mu = x.mean()
    sigma = x.std().clamp_min(1e-12)
    kurt = ((x - mu) ** 4).mean() / sigma.pow(4)
    absx = x.abs()
    return {
        "kurtosis": float(kurt),
        "max_over_median": float(absx.max() / absx.median().clamp_min(1e-12)),
        "max": float(absx.max()),
        "median": float(absx.median()),
    }


def replace_layer(model, layer_idx: int, args) -> None:
    layer = model.model.layers[layer_idx]
    layer.mlp.experts = QATExperts(
        layer.mlp.experts,
        bits=args.bits,
        group=args.group,
        scale_mode=args.scale_mode,
        learnable_scales=False,
        outlier_frac=args.outlier_frac,
        outlier_group_frac=0.0,
        clip_percentile=args.clip_percentile,
    )


def eval_variant(args, chunks, layers_to_quantize: tuple[int, ...]) -> float:
    model = MixtralForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float32)
    for i in layers_to_quantize:
        replace_layer(model, i, args)
    _, ppl = evaluate(model, chunks, torch.device("cpu"), batch_size=8)
    return ppl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=r"C:\kruko\models\stories15m-moe")
    parser.add_argument("--corpus", default=r"C:\kruko\models\stories15m-moe\TinyStories-valid.txt")
    parser.add_argument("--output-dir", default="results/layer-audit")
    parser.add_argument("--chunks", type=int, default=40)
    parser.add_argument("--bits", type=int, default=1)
    parser.add_argument("--group", type=int, default=32)
    parser.add_argument("--scale-mode", default="group")
    parser.add_argument("--outlier-frac", type=float, default=0.001)
    parser.add_argument("--clip-percentile", type=float, default=0.999)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = Path(args.corpus).read_text(encoding="utf-8", errors="ignore")
    val_text = raw[2_000_000:2_500_000]
    tok = AutoTokenizer.from_pretrained(args.model_id)
    tok.model_max_length = 10_000_000
    chunks = make_chunks(tok, val_text, 128, args.chunks)

    base_model = MixtralForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float32)
    _, baseline = evaluate(base_model, chunks, torch.device("cpu"), batch_size=8)
    print(f"baseline PPL {baseline:.4f}", flush=True)

    layers = len(base_model.model.layers)
    rows = []
    for i in range(layers):
        last = tuple(range(layers))
        only = (i,)
        all_but = tuple(j for j in last if j != i)
        only_ppl = eval_variant(args, chunks, only)
        all_but_ppl = eval_variant(args, chunks, all_but)
        experts = base_model.model.layers[i].mlp.experts
        err_gate = quant_error(experts.gate_up_proj, args.group, args.clip_percentile, args.outlier_frac)
        err_down = quant_error(experts.down_proj, args.group, args.clip_percentile, args.outlier_frac)
        stats_gate = weight_stats(experts.gate_up_proj)
        stats_down = weight_stats(experts.down_proj)
        row = {
            "layer": i,
            "only_ppl": only_ppl,
            "only_delta": only_ppl - baseline,
            "all_but_ppl": all_but_ppl,
            "all_but_delta": all_but_ppl - baseline,
            "quant_error_gate": err_gate,
            "quant_error_down": err_down,
            "quant_error_avg": (err_gate + err_down) / 2,
            "kurtosis_gate": stats_gate["kurtosis"],
            "kurtosis_down": stats_down["kurtosis"],
            "max_over_median_gate": stats_gate["max_over_median"],
            "max_over_median_down": stats_down["max_over_median"],
        }
        rows.append(row)
        print(
            f"L{i}: only {only_ppl:.4f} (+{row['only_delta']:.4f}) "
            f"all-but {all_but_ppl:.4f} (+{row['all_but_delta']:.4f}) "
            f"err {row['quant_error_avg']:.4f}",
            flush=True,
        )

    (out_dir / "layer_audit.json").write_text(
        json.dumps({"baseline_ppl": baseline, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Layer audit",
        "",
        f"Baseline PPL: {baseline:.4f}",
        "",
        "| layer | only PPL | only delta | all-but PPL | all-but delta | quant err | kurtosis | max/median |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['layer']} | {r['only_ppl']:.4f} | {r['only_delta']:.4f} | "
            f"{r['all_but_ppl']:.4f} | {r['all_but_delta']:.4f} | "
            f"{r['quant_error_avg']:.4f} | {r['kurtosis_gate']:.2f} | "
            f"{r['max_over_median_gate']:.2f} |"
        )
    (out_dir / "layer_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
