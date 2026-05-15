#!/usr/bin/env python3
"""
Phase 2 Part C: ViT-tiny CLS attention maps on PathMNIST test samples.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.model_factory import _unpack_logits
from src.utils.logger import setup_logger
from src.utils.phase2_common import (
    default_pathmnist_yaml,
    get_pathmnist_loaders,
    load_model_from_checkpoint,
    softmax_np,
)

logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2 ViT attention maps")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--output-dir", type=str, default=str(ROOT / "results" / "phase2"))
    p.add_argument("--model-path", type=str, default=str(ROOT / "experiments/phase1/vit_tiny/best_model.pt"))
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def device_arg(s: str) -> torch.device:
    if s == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def denormalize_chw(t: torch.Tensor) -> np.ndarray:
    x = t.detach().cpu().numpy() * IMAGENET_STD + IMAGENET_MEAN
    x = np.clip(x, 0, 1)
    return np.transpose(x, (1, 2, 0))


def find_attn_dropout_module(vit_backbone: nn.Module) -> nn.Module:
    """timm ViT: attention probs pass through attn_drop (Dropout)."""
    blocks = getattr(vit_backbone, "blocks", None)
    if blocks is None:
        raise ValueError("No blocks on backbone")
    last_attn = blocks[-1].attn
    if hasattr(last_attn, "attn_drop"):
        return last_attn.attn_drop
    raise ValueError("Could not find attn_drop on last attention block")


def capture_cls_patch_attention(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, int, int]:
    """
    Returns attn_np shape (num_heads, n_patches) for CLS row (excluding CLS column self),
    i.e. CLS -> each patch token (excluding CLS token index 0).
    """
    bb = model.backbone
    captured: List[torch.Tensor] = []

    def pre_hook(_mod, inputs):
        t = inputs[0]
        if t is None:
            return
        if t.dim() == 4:
            return
        # (B, num_heads, N, N) attention weights before dropout
        captured.append(t.detach().cpu())

    attn_drop = find_attn_dropout_module(bb)
    h = attn_drop.register_forward_pre_hook(pre_hook)
    try:
        model.eval()
        with torch.no_grad():
            _ = model(x.to(device))
    finally:
        h.remove()

    if not captured:
        raise RuntimeError("Attention weights not captured; check timm ViT structure")
    attn = captured[-1]
    if attn.dim() != 4:
        raise ValueError(f"Unexpected attn shape {attn.shape}")
    # (B, H, N, N)
    attn_b = attn[0]
    cls_to_all = attn_b[:, 0, :].numpy()
    cls_to_patches = cls_to_all[:, 1:]
    n_patches = cls_to_patches.shape[1]
    g = int(np.sqrt(n_patches))
    if g * g != n_patches:
        g = int(np.round(np.sqrt(n_patches)))
        if g * g != n_patches:
            raise ValueError(f"Non-square patches: {n_patches}")
    return cls_to_patches, g, g


def attention_to_28(attn_heads_patch: np.ndarray, g: int) -> np.ndarray:
    """(num_heads, n_patches) -> (num_heads, 28, 28)"""
    H = attn_heads_patch.shape[0]
    out = []
    for h in range(H):
        m = attn_heads_patch[h].reshape(g, g).astype(np.float32)
        t = torch.from_numpy(m)[None, None, ...]
        up = F.interpolate(t, size=(28, 28), mode="bilinear", align_corners=False)
        out.append(up.squeeze().numpy())
    return np.stack(out, axis=0)


def main():
    args = parse_args()
    device = device_arg(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(ROOT / "logs" / "phase2" / "attention.log")

    yaml = Path(args.config) if args.config else default_pathmnist_yaml(ROOT)
    (_, _, test_loader), cfg = get_pathmnist_loaders(ROOT, yaml, args.seed)
    class_names = cfg["dataset"].get("class_names", [str(i) for i in range(9)])

    ck = Path(args.model_path)
    model = load_model_from_checkpoint(ck, device, "vit", "vit_tiny")

    # 2 indices per class (any test sample with that label)
    dataset = test_loader.dataset
    by_class: List[List[int]] = [[] for _ in range(9)]
    for idx in range(len(dataset)):
        _, y = dataset[idx]
        c = int(y.item()) if y.dim() == 0 else int(np.asarray(y).ravel()[0])
        if c < 0 or c > 8:
            continue
        if len(by_class[c]) < 2:
            by_class[c].append(idx)

    stats_per_class = {str(c): {"entropies": [], "max_attn": [], "conf": []} for c in range(9)}
    all_max_attn = []
    all_conf = []

    for c in range(9):
        indices = by_class[c][:2]
        for j, idx in enumerate(indices):
            img, y = dataset[idx]
            x = img.unsqueeze(0)
            cls_patch, gh, gw = capture_cls_patch_attention(model, x, device)
            heads_28 = attention_to_28(cls_patch, gh)
            mean_map = heads_28.mean(axis=0)
            p_patch = cls_patch / (cls_patch.sum(axis=1, keepdims=True) + 1e-10)
            ent = float(-(p_patch * np.log(p_patch + 1e-10)).sum(axis=1).mean())
            max_attn = float(cls_patch.max())
            with torch.no_grad():
                conf = float(torch.softmax(_unpack_logits(model(x.to(device))), dim=1).max().item())
            stats_per_class[str(c)]["entropies"].append(ent)
            stats_per_class[str(c)]["max_attn"].append(max_attn)
            stats_per_class[str(c)]["conf"].append(conf)
            all_max_attn.append(max_attn)
            all_conf.append(conf)

            cname = class_names[c] if c < len(class_names) else str(c)
            safe = "".join(ch if ch.isalnum() else "_" for ch in str(cname))[:40]
            nh = heads_28.shape[0]
            fig = plt.figure(figsize=(3 * (2 + nh), 5))
            gs = fig.add_gridspec(1, 2 + nh, wspace=0.25)
            ax0 = fig.add_subplot(gs[0, 0])
            ax0.imshow(denormalize_chw(img))
            ax0.set_title("Original")
            ax0.axis("off")
            ax1 = fig.add_subplot(gs[0, 1])
            ax1.imshow(mean_map, cmap="viridis")
            ax1.set_title("Mean attention")
            ax1.axis("off")
            for hi in range(nh):
                axh = fig.add_subplot(gs[0, 2 + hi])
                axh.imshow(heads_28[hi], cmap="magma")
                axh.set_title(f"head {hi}", fontsize=8)
                axh.axis("off")
            fig.suptitle(f"{safe} sample {j} (idx={idx})", fontsize=11)
            plt.tight_layout()
            fig.savefig(out_dir / f"attention_{safe}_{j}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    corr = float("nan")
    if len(all_max_attn) > 2:
        corr, _ = spearmanr(all_max_attn, all_conf)

    summary = {
        "per_class_mean_entropy": {
            str(c): float(np.mean(stats_per_class[str(c)]["entropies"]))
            if stats_per_class[str(c)]["entropies"]
            else None
            for c in range(9)
        },
        "spearman_max_attn_vs_confidence": corr,
        "per_class_detail": stats_per_class,
    }
    with open(out_dir / "attention_stats.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote attention maps and stats to %s", out_dir)


if __name__ == "__main__":
    main()
