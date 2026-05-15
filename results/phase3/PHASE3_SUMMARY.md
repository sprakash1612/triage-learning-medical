# Phase 3 summary
_Artifacts under `/home/aryan/Desktop/triage-learning-medical-diagnosis/results/phase3`._
## Part A — Robustness

_No robustness_results.csv._

## Part B — DermaMNIST

_No dermamnist_comparison.csv._

## Part C — Statistical validation

_No statistical_tests.json._

## Part D — OOD detection

_No ood_detection_results.csv._

## Key findings

- **Part A:** Corruption stress-test reports accuracy drop and whether MC uncertainty tracks severity (see CSV header comment).
- **Part B:** Compare zero-shot head training vs full fine-tune vs ResNet18-from-scratch on DermaMNIST.
- **Part C:** Bootstrap CIs and McNemar / Spearman tests quantify significance of baseline vs triage and vs best Phase 1 model.
- **Part D:** AUROC of MC entropy separates clean PathMNIST from severely corrupted inputs (EfficientNet-B3 ROC figure).
- **Part E:** This file aggregates all Phase 3 CSV/JSON/PNG outputs.
