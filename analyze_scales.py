"""Analyze learnable scales after QAT.

Compares the scale initialization from the FP32 model with the scales stored in
a QAT checkpoint. Reports collapse, relative change, per-layer movement and the
outlier-group split.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import MixtralForCausalLM


def init_log_scale(w: torch.Tensor, group: int, percentile: float) -> torch.Tensor:
    e, rows, cols = w.shape
    blocks = w.detach().float().reshape(e, rows, cols // group, group)
    scale = torch.quantile(blocks.abs(), percentile, dim=-1)
    return torch.log(scale.clamp_min(1e-4))


def rel_change(old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
    return (new - old).abs() / old.clamp_min(1e-12)


def summarize(old: torch.Tensor, new: torch.Tensor) -> dict:
    rel = rel_change(old, new)
    return {
        "n": int(old.numel()),
        "old_mean": float(old.mean()),
        "old_std": float(old.std()),
        "old_min": float(old.min()),
        "old_max": float(old.max()),
        "new_mean": float(new.mean()),
        "new_std": float(new.std()),
        "new_min": float(new.min()),
        "new_max": float(new.max()),
        "rel_mean": float(rel.mean()),
        "rel_median": float(rel.median()),
        "rel_p90": float(torch.quantile(rel, 0.90)),
        "rel_p99": float(torch.quantile(rel, 0.99)),
        "rel_max": float(rel.max()),
        "frac_lt_1pct": float((rel < 0.01).float().mean()),
        "frac_lt_5pct": float((rel < 0.05).float().mean()),
        "frac_gt_10pct": float((rel > 0.10).float().mean()),
        "frac_gt_50pct": float((rel > 0.50).float().mean()),
        "collapsed": int((new < 1e-6).sum()),
        "clamped_low": int((new <= math.exp(-11.999)).sum()),
        "clamped_high": int((new >= math.exp(3.999)).sum()),
    }


def layer_rows(sd: dict, model, group: int, percentile: float) -> tuple[list[dict], dict]:
    rows = []
    rels = {}
    layers = sorted({int(k.split(".")[2]) for k in sd if "mlp.experts" in k and "log_scale" in k})
    for layer_idx in layers:
        orig = model.model.layers[layer_idx].mlp.experts
        for kind, weight, scale_key in [
            ("gate", orig.gate_up_proj, f"model.layers.{layer_idx}.mlp.experts.gate_log_scale"),
            ("down", orig.down_proj, f"model.layers.{layer_idx}.mlp.experts.down_log_scale"),
        ]:
            old_log = init_log_scale(weight, group, percentile)
            new_log = sd[scale_key].float()
            old = torch.exp(old_log)
            new = torch.exp(new_log)
            stats = summarize(old, new)
            stats["layer"] = layer_idx
            stats["kind"] = kind
            stats["log_delta_mean"] = float((new_log - old_log).mean())
            stats["log_delta_std"] = float((new_log - old_log).std())

            # Outlier-group split: top 1% groups by max absolute weight.
            blocks = weight.detach().float().reshape(weight.shape[0], weight.shape[1], -1, group)
            max_abs = blocks.abs().amax(dim=-1)
            thr = torch.quantile(max_abs.flatten(), 0.99)
            top = max_abs >= thr
            rel = rel_change(old, new)
            rels[(layer_idx, kind)] = rel.flatten()
            stats["outlier_groups"] = int(top.sum())
            stats["rel_outlier_mean"] = float(rel[top].mean()) if top.any() else 0.0
            stats["rel_rest_mean"] = float(rel[~top].mean()) if (~top).any() else 0.0
            rows.append(stats)
    return rows, rels


def write_markdown(rows: list[dict], path: Path, checkpoint: Path) -> None:
    lines = [
        "# Scale analysis",
        "",
        f"Checkpoint `{checkpoint.as_posix()}`",
        "",
        "## Per layer",
        "",
        "| layer | kind | n | mean rel | median rel | p90 | <1% | >10% | collapsed | outlier rel | rest rel |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['layer']} | {r['kind']} | {r['n']} | {r['rel_mean']:.4f} | "
            f"{r['rel_median']:.4f} | {r['rel_p90']:.4f} | {r['frac_lt_1pct']:.2%} | "
            f"{r['frac_gt_10pct']:.2%} | {r['collapsed']} | {r['rel_outlier_mean']:.4f} | "
            f"{r['rel_rest_mean']:.4f} |"
        )
    lines += [
        "",
        "## Scale values",
        "",
        "| layer | kind | old mean | old min | old max | new mean | new min | new max |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['layer']} | {r['kind']} | {r['old_mean']:.4e} | {r['old_min']:.4e} | "
            f"{r['old_max']:.4e} | {r['new_mean']:.4e} | {r['new_min']:.4e} | {r['new_max']:.4e} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(rows: list[dict], rels: dict, out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"matplotlib unavailable: {exc}")
        return

    layers = sorted({r["layer"] for r in rows})
    for kind in ["gate", "down"]:
        subset = [r for r in rows if r["kind"] == kind]
        if not subset:
            continue
        plt.figure(figsize=(9, 4))
        plt.bar([f"L{r['layer']}" for r in subset], [r["rel_mean"] for r in subset], color="#5eead4")
        plt.axhline(0.01, color="#ef4444", linestyle="--", linewidth=1, label="1%")
        plt.ylabel("mean relative scale change")
        plt.title(f"{kind} scale change by layer")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"scale_change_{kind}.png", dpi=160)
        plt.close()

    # Histogram of log10 relative change for every group, one subplot per layer.
    for kind in ["gate", "down"]:
        layers = sorted({layer for (layer, k) in rels if k == kind})
        if not layers:
            continue
        fig, axes = plt.subplots(len(layers), 1, figsize=(9, 2.2 * len(layers)), sharex=True)
        if len(layers) == 1:
            axes = [axes]
        for ax, layer in zip(axes, layers):
            vals = torch.log10(rels[(layer, kind)] + 1e-8).numpy()
            ax.hist(vals, bins=80, color="#5eead4" if kind == "gate" else "#818cf8")
            ax.set_ylabel("groups")
            ax.set_title(f"L{layer} {kind} relative scale change")
        axes[-1].set_xlabel("log10 relative change")
        plt.tight_layout()
        plt.savefig(out_dir / f"scale_hist_{kind}.png", dpi=160)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/best-full-scale-grad/qat_best.pt")
    parser.add_argument("--model-id", default=r"C:\kruko\models\stories15m-moe")
    parser.add_argument("--output-dir", default="results/scale-analysis")
    parser.add_argument("--group", type=int, default=32)
    parser.add_argument("--clip-percentile", type=float, default=0.999)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading checkpoint", flush=True)
    sd = torch.load(checkpoint, map_location="cpu")
    print("loading fp32 model", flush=True)
    model = MixtralForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float32)
    model.eval()

    rows, rels = layer_rows(sd, model, args.group, args.clip_percentile)
    (out_dir / "scale_analysis.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_markdown(rows, out_dir / "scale_analysis.md", checkpoint)
    make_plots(rows, rels, out_dir)

    for r in rows:
        print(
            f"L{r['layer']} {r['kind']}: mean {r['rel_mean']:.4f} "
            f"median {r['rel_median']:.4f} <1% {r['frac_lt_1pct']:.2%} "
            f"collapsed {r['collapsed']} outlier {r['rel_outlier_mean']:.4f} "
            f"rest {r['rel_rest_mean']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
