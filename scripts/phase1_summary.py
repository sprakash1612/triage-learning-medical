#!/usr/bin/env python3
"""
Phase 1 Part D: Aggregate results from results/phase1/*.csv and *.json to stdout + PHASE1_SUMMARY.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_csv(path: Path) -> list:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(ROOT / "results" / "phase1"),
    )
    args = parser.parse_args()
    d = Path(args.results_dir)
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("# Phase 1 summary")
    emit()

    mc_path = d / "model_comparison.csv"
    if mc_path.is_file():
        emit("## PathMNIST model comparison")
        rows = load_csv(mc_path)
        cols = list(rows[0].keys()) if rows else []
        if cols:
            widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
            header = " | ".join(c.ljust(widths[c]) for c in cols)
            emit(header)
            emit("-+-".join("-" * widths[c] for c in cols))
            for r in rows:
                emit(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
        emit()

    ens_path = d / "ensemble_vs_mcdropout.csv"
    if ens_path.is_file():
        emit("## Ensemble vs MC Dropout")
        rows = load_csv(ens_path)
        cols = list(rows[0].keys()) if rows else []
        if cols:
            widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
            emit(" | ".join(c.ljust(widths[c]) for c in cols))
            emit("-+-".join("-" * widths[c] for c in cols))
            for r in rows:
                emit(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
        emit()

    chest_path = d / "chestmnist_results.json"
    if chest_path.is_file():
        emit("## ChestMNIST")
        data = json.loads(chest_path.read_text())
        emit(f"- Task: {data.get('task')}")
        emit(f"- Test exact-match accuracy: {data.get('test_subset_accuracy_exact_match')}")
        emit(f"- Mean uncertainty (MC): {data.get('mean_uncertainty')}")
        tri = data.get("triage", {})
        emit(f"- Triage accuracy (optimal sweep): {tri.get('triage_accuracy')}")
        emit(f"- Automation rate: {tri.get('automation_rate')}")
        emit(f"- Error reduction: {tri.get('error_reduction')}")
        emit(f"- PathMNIST reference: {data.get('pathmnist_reference')}")
        emit()

    out_md = d / "PHASE1_SUMMARY.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out_md}", file=sys.stderr)


if __name__ == "__main__":
    main()
