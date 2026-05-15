#!/usr/bin/env python3
"""
Phase 3 Part E: Aggregate results/phase3 into PHASE3_SUMMARY.md and stdout.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.logger import setup_logger


def parse_args():
    p = argparse.ArgumentParser(description="Phase 3 summary")
    p.add_argument("--output-dir", type=str, default=str(ROOT / "results" / "phase3"))
    return p.parse_args()


def md_table(headers: List[str], rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._\n"
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines) + "\n"


def read_csv(path: Path) -> tuple[List[str], List[Dict[str, str]]]:
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def main():
    args = parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs" / "phase3").mkdir(parents=True, exist_ok=True)
    setup_logger(ROOT / "logs" / "phase3" / "summary.log")

    parts: List[str] = []
    parts.append("# Phase 3 summary\n")
    parts.append(f"_Artifacts under `{root}`._\n")

    # A — Robustness ranking
    rob = root / "robustness_results.csv"
    if rob.is_file():
        from collections import defaultdict

        agg: Dict[str, List[float]] = defaultdict(list)
        with open(rob, newline="") as f:
            for row in csv.DictReader(line for line in f if not line.lstrip().startswith("#")):
                if not row.get("model"):
                    continue
                try:
                    agg[row["model"]].append(float(row["accuracy_drop"]))
                except (KeyError, ValueError):
                    pass
        ranked = sorted(((m, sum(v) / len(v)) for m, v in agg.items()), key=lambda x: x[1])
        parts.append("## Part A — Robustness\n")
        parts.append("Mean `accuracy_drop` across all corruption×severity (lower is better):\n\n")
        for m, v in ranked:
            parts.append(f"- `{m}`: {v:.4f}\n")
        parts.append("\n")
    else:
        parts.append("## Part A — Robustness\n\n_No robustness_results.csv._\n\n")

    # B — DermaMNIST
    dcsv = root / "dermamnist_comparison.csv"
    if dcsv.is_file():
        hdr, drows = read_csv(dcsv)
        parts.append("## Part B — DermaMNIST\n\n")
        parts.append(md_table(hdr, drows))
        parts.append("\n")
    else:
        parts.append("## Part B — DermaMNIST\n\n_No dermamnist_comparison.csv._\n\n")

    # C — Statistical tests
    st = root / "statistical_tests.json"
    if st.is_file():
        data = json.loads(st.read_text(encoding="utf-8"))
        parts.append("## Part C — Statistical validation\n\n")
        parts.append("| test | statistic | p_value | significant |\n| --- | --- | --- | --- |\n")
        for name, block in data.items():
            if not isinstance(block, dict):
                continue
            sig = block.get("significant", "")
            parts.append(
                f"| {name} | {block.get('statistic', '')} | {block.get('p_value', '')} | {sig} |\n"
            )
        parts.append("\n")
    else:
        parts.append("## Part C — Statistical validation\n\n_No statistical_tests.json._\n\n")

    # D — OOD
    ood = root / "ood_detection_results.csv"
    if ood.is_file():
        hdr, orows = read_csv(ood)
        parts.append("## Part D — OOD detection (AUROC)\n\n")
        parts.append(md_table(hdr, orows))
        parts.append("\n")
    else:
        parts.append("## Part D — OOD detection\n\n_No ood_detection_results.csv._\n\n")

    # Key bullets
    parts.append("## Key findings\n\n")
    parts.append(
        "- **Part A:** Corruption stress-test reports accuracy drop and whether MC uncertainty tracks severity (see CSV header comment).\n"
    )
    parts.append(
        "- **Part B:** Compare zero-shot head training vs full fine-tune vs ResNet18-from-scratch on DermaMNIST.\n"
    )
    parts.append(
        "- **Part C:** Bootstrap CIs and McNemar / Spearman tests quantify significance of baseline vs triage and vs best Phase 1 model.\n"
    )
    parts.append(
        "- **Part D:** AUROC of MC entropy separates clean PathMNIST from severely corrupted inputs (EfficientNet-B3 ROC figure).\n"
    )
    parts.append("- **Part E:** This file aggregates all Phase 3 CSV/JSON/PNG outputs.\n")

    # Figures list
    pngs = sorted(root.rglob("*.png"))
    if pngs:
        parts.append("\n## Figures\n\n")
        for p in pngs:
            rel = p.relative_to(root)
            parts.append(f"- `{rel}`\n")

    text = "".join(parts)
    out_md = root / "PHASE3_SUMMARY.md"
    out_md.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
