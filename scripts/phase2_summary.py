#!/usr/bin/env python3
"""
Phase 2 Part E: Aggregate results under results/phase2 into PHASE2_SUMMARY.md and stdout.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.logger import setup_logger


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2 summary markdown + stdout")
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "results" / "phase2"),
        help="Directory to scan and where PHASE2_SUMMARY.md is written",
    )
    return p.parse_args()


def read_csv_rows(path: Path) -> tuple[Optional[List[str]], List[Dict[str, str]]]:
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        fields = r.fieldnames
        rows = list(r)
    return fields, rows


def md_table(headers: List[str], rows: List[Dict[str, Any]], float_fmt: str = "{:.4f}") -> str:
    if not headers:
        return "_No columns._\n"

    def fmt_cell(h: str, v: Any) -> str:
        if v is None or v == "":
            return ""
        try:
            x = float(v)
            if x != x:  # nan
                return "nan"
            return float_fmt.format(x)
        except (TypeError, ValueError):
            return str(v)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt_cell(h, row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines) + "\n"


def caption_for_png(name: str) -> str:
    if name == "calibration_summary.png":
        return "Grouped ECE before vs after temperature scaling across models."
    if name == "calibration_triage_interaction.png":
        return "ResNet18: system accuracy, error reduction, and ECE on AI-accepted cases vs deferral rate (strategies A–C)."
    if name.startswith("reliability_") and name.endswith(".png"):
        return f"Reliability diagram with bin counts for {name.replace('reliability_', '').replace('.png', '')}."
    if name.startswith("gradcam_summary"):
        return "EfficientNet-B3 Grad-CAM overview (one panel per class)."
    if name.startswith("gradcam_"):
        return "Per-sample Grad-CAM grid (RGB, overlay, labels, confidence)."
    if name.startswith("uncertainty_heatmap_"):
        return "MC variance of Grad-CAM heatmaps for the predicted class."
    if name.startswith("attention_"):
        return "ViT last-layer attention: original, mean over heads, per-head maps."
    return "Phase 2 figure."


def build_summary(root: Path) -> str:
    parts: List[str] = []
    parts.append("# Phase 2 summary\n")
    parts.append(f"_Generated from artifacts under `{root}`._\n")

    cal_csv = root / "calibration_results.csv"
    if cal_csv.is_file():
        headers, rows = read_csv_rows(cal_csv)
        parts.append("## Temperature scaling and calibration (PathMNIST)\n")
        parts.append(md_table(list(headers or []), rows))
        ece_after = []
        models = []
        tri_b, tri_a = [], []
        for row in rows:
            try:
                ece_after.append(float(row["ece_after"]))
                models.append(row.get("model", "?"))
            except (KeyError, ValueError, TypeError):
                pass
            try:
                tri_b.append(float(row["triage_acc_before"]))
                tri_a.append(float(row["triage_acc_after"]))
            except (KeyError, ValueError, TypeError):
                pass
        if ece_after and models:
            i = min(range(len(ece_after)), key=lambda j: ece_after[j])
            parts.append(
                f"- **Best post-calibration ECE:** `{models[i]}` with ECE ≈ {ece_after[i]:.4f}.\n"
            )
        if tri_b and tri_a:
            deltas = [float(ta) - float(tb) for ta, tb in zip(tri_a, tri_b)]
            mean_d = sum(deltas) / len(deltas)
            parts.append(
                "- **Calibration vs triage:** After scaling, triage accuracy uses `1 − max p` "
                f"(sweep); mean Δ triage vs MC-entropy baseline ≈ {mean_d:.4f} "
                "across models in `calibration_results.csv` (sign varies by model).\n"
            )
    else:
        parts.append("## Calibration\n\n_No `calibration_results.csv` found._\n")

    tri_csv = root / "calibration_triage_comparison.csv"
    if tri_csv.is_file():
        headers, rows = read_csv_rows(tri_csv)
        parts.append("## Calibration–triage interaction (ResNet18)\n")
        parts.append(md_table(list(headers or []), rows))
    else:
        parts.append("## Calibration–triage interaction\n\n_No `calibration_triage_comparison.csv` found._\n")

    attn_json = root / "attention_stats.json"
    if attn_json.is_file():
        with open(attn_json) as f:
            stats = json.load(f)
        parts.append("## ViT attention statistics\n")
        parts.append("```json\n")
        parts.append(json.dumps(stats, indent=2)[:8000])
        if len(json.dumps(stats)) > 8000:
            parts.append("\n… (truncated)")
        parts.append("\n```\n")

    pngs = sorted({p for p in root.rglob("*.png")})
    parts.append("## Figures\n")
    if pngs:
        for p in pngs:
            rel = p.relative_to(root) if p.is_relative_to(root) else p
            parts.append(f"- `{rel}` — {caption_for_png(p.name)}\n")
    else:
        parts.append("_No PNG files found._\n")

    return "".join(parts)


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    setup_logger(ROOT / "logs" / "phase2" / "summary.log")

    text = build_summary(out)
    out_path = out / "PHASE2_SUMMARY.md"
    out_path.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
