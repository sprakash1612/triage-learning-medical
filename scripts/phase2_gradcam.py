#!/usr/bin/env python3
"""
Phase 2 Part B: Grad-CAM for ResNet18, DenseNet121, EfficientNet-B3 on PathMNIST.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import cm
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.model_factory import _unpack_logits
from src.utils.logger import setup_logger
from src.utils.phase2_common import (
    DEFAULT_PHASE2_MODELS,
    default_pathmnist_yaml,
    get_pathmnist_loaders,
    load_model_from_checkpoint,
    softmax_np,
)

logger = logging.getLogger(__name__)

CNN_MODELS = [
    ("resnet18", "experiments/resnet18_pathmnist/best_model.pt", "resnet", "18", "layer4"),
    ("densenet121", "experiments/phase1/densenet121/best_model.pt", "densenet", "121", "features"),
    ("efficientnet_b3", "experiments/phase1/efficientnet_b3/best_model.pt", "efficientnet", "b3", None),
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class GradCAM:
    def __init__(self, model: nn.Module, target_layer_name: Optional[str] = None, target_module: Optional[nn.Module] = None):
        self.model = model
        self.target_layer_name = target_layer_name
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.hooks: List = []
        self.target_module: Optional[nn.Module] = None

        if target_module is not None:
            self.target_module = target_module
        else:
            if not target_layer_name:
                raise ValueError("Need target_layer_name or target_module")
            candidates = []
            for name, module in model.named_modules():
                if not name:
                    continue
                if name == target_layer_name or name.endswith("." + target_layer_name):
                    candidates.append((len(name), name, module))
                elif name.split(".")[-1] == target_layer_name:
                    candidates.append((len(name), name, module))
            if not candidates:
                raise ValueError(f"No layer matching {target_layer_name!r} in model")
            candidates.sort(reverse=True)
            _, resolved_name, self.target_module = candidates[0]
            logger.info("GradCAM target layer: %s", resolved_name)

        self.hooks.append(
            self.target_module.register_forward_hook(self._forward_hook)
        )
        self.hooks.append(
            self.target_module.register_full_backward_hook(self._backward_hook)
        )

    def _forward_hook(self, _m, _inp, out):
        self.activations = out.detach()

    def _backward_hook(self, _m, _gi, go):
        if go[0] is not None:
            self.gradients = go[0].detach()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None
        self.model.eval()
        x = input_tensor.clone().detach()
        x.requires_grad_(True)
        with torch.enable_grad():
            out = self.model(x)
            logits = _unpack_logits(out)
            if target_class is None:
                target_class = int(logits.argmax(dim=1).item())
            sc = logits[0, target_class]
            sc.backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("GradCAM hooks did not capture activations/gradients")
        act = self.activations
        grad = self.gradients
        if act.dim() != 4 or grad.dim() != 4:
            raise ValueError(f"Expected 4D activations, got act={act.shape}, grad={grad.shape}")
        w = grad.mean(dim=(2, 3), keepdim=True)
        cam = (w * act).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(28, 28), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().float().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def try_cv2_jet_heatmap(cam: np.ndarray) -> np.ndarray:
    """cam (H,W) in [0,1] -> RGB (H,W,3) uint8 JET"""
    try:
        import cv2

        cm_u8 = (cam * 255).astype(np.uint8)
        color = cv2.applyColorMap(cm_u8, cv2.COLORMAP_JET)
        return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    except Exception:
        rgba = cm.jet(np.clip(cam, 0, 1))
        return (rgba[:, :, :3] * 255).astype(np.uint8)


def denormalize_chw(t: torch.Tensor) -> np.ndarray:
    x = t.detach().cpu().numpy() * IMAGENET_STD + IMAGENET_MEAN
    x = np.clip(x, 0, 1)
    return np.transpose(x, (1, 2, 0))


def blend_overlay(img_rgb: np.ndarray, heat_rgb: np.ndarray, alpha_img: float = 0.4) -> np.ndarray:
    img_f = img_rgb.astype(np.float32)
    h_f = heat_rgb.astype(np.float32) / 255.0
    out = alpha_img * img_f + (1 - alpha_img) * h_f
    return np.clip(out, 0, 1)


def collect_test_predictions(model, loader, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds, ys, confs = [], [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Pred test"):
            x = x.to(device)
            logits = _unpack_logits(model(x))
            pr = logits.argmax(dim=1).cpu().numpy()
            p = torch.softmax(logits, dim=1).max(dim=1).values.cpu().numpy()
            preds.append(pr)
            ys.append(y.numpy().ravel())
            confs.append(p)
    return np.concatenate(preds), np.concatenate(ys), np.concatenate(confs)


def sample_indices_per_class(
    preds: np.ndarray, y: np.ndarray, class_id: int, k: int = 5
) -> Tuple[List[int], List[int]]:
    """Up to k correct (pred==y==c) and k wrong (y==c, pred!=c) global indices."""
    mask_c = y == class_id
    idx_all = np.where(mask_c)[0]
    correct = [i for i in idx_all if preds[i] == class_id][:k]
    wrong = [i for i in idx_all if preds[i] != class_id][:k]
    return correct, wrong


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2 Grad-CAM")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--output-dir", type=str, default=str(ROOT / "results" / "phase2"))
    p.add_argument("--model-path", type=str, default=None, help="Single checkpoint (debug)")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def device_arg(s: str) -> torch.device:
    if s == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_one_model(
    disp: str,
    ck_rel: str,
    arch: str,
    var: str,
    layer: Optional[str],
    model: nn.Module,
    test_loader,
    dataset,
    device: torch.device,
    out_dir: Path,
    class_names: List[str],
):
    preds, ys, confs = collect_test_predictions(model, test_loader, device)
    if layer is None and "efficientnet" in disp:
        last_block = model.backbone.blocks[-1]
        gradcam = GradCAM(model, target_module=last_block)
    else:
        gradcam = GradCAM(model, target_layer_name=layer or "")

    for c in range(9):
        correct_idx, wrong_idx = sample_indices_per_class(preds, ys, c, 5)
        rows = []
        for tag, indices in [("ok", correct_idx), ("err", wrong_idx)]:
            for gi in indices:
                img_t, y_true = dataset[gi]
                img_b = img_t.unsqueeze(0).to(device)
                pred_c = int(preds[gi])
                conf = float(confs[gi])
                with torch.enable_grad():
                    hm = gradcam.generate(img_b, target_class=pred_c)
                img_rgb = denormalize_chw(img_t)
                heat_rgb = try_cv2_jet_heatmap(hm)
                img_u8 = (np.clip(img_rgb, 0, 1) * 255).astype(np.uint8)
                overlay = blend_overlay(img_u8.astype(np.float32) / 255.0, heat_rgb, 0.4)
                rows.append((img_rgb, overlay, pred_c, int(y_true), conf, hm))

        # MC variance maps for same class grid
        var_rows = []
        for tag, indices in [("ok", correct_idx), ("err", wrong_idx)]:
            for gi in indices:
                img_t, _ = dataset[gi]
                img_b = img_t.unsqueeze(0).to(device)
                pred_c = int(preds[gi])
                maps = []
                model.train()
                for _ in range(10):
                    with torch.enable_grad():
                        m = gradcam.generate(img_b, target_class=pred_c)
                    maps.append(m)
                model.eval()
                var_map = np.var(np.stack(maps, axis=0), axis=0)
                var_map = (var_map - var_map.min()) / (var_map.max() - var_map.min() + 1e-8)
                var_rows.append(var_map)

        _save_class_grid(rows, out_dir / f"gradcam_{disp}_class{c}.png", class_names[c])
        if var_rows:
            _save_variance_grid(var_rows, out_dir / f"uncertainty_heatmap_{disp}_class{c}.png", class_names[c])

    gradcam.remove_hooks()


def _save_class_grid(rows, path: Path, cname: str):
    if not rows:
        return
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(9, 2.2 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)
    for i, (img_rgb, overlay, pred, yt, conf, _hm) in enumerate(rows):
        axes[i, 0].imshow(np.clip(img_rgb, 0, 1))
        axes[i, 0].set_title("Input")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(np.clip(overlay, 0, 1))
        axes[i, 1].set_title("Grad-CAM")
        axes[i, 1].axis("off")
        axes[i, 2].axis("off")
        axes[i, 2].text(
            0.1,
            0.5,
            f"pred={pred}\ntrue={yt}\nconf={conf:.3f}",
            fontsize=11,
            va="center",
            transform=axes[i, 2].transAxes,
        )
    fig.suptitle(f"class {cname}", fontsize=12)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_variance_grid(var_rows: List[np.ndarray], path: Path, cname: str):
    if not var_rows:
        return
    n = len(var_rows)
    fig, axes = plt.subplots(n, 1, figsize=(3, 2 * n))
    if n == 1:
        axes = [axes]
    for i, vm in enumerate(var_rows):
        axes[i].imshow(vm, cmap="magma")
        axes[i].set_title(f"var MC Grad-CAM #{i}")
        axes[i].axis("off")
    fig.suptitle(f"uncertainty heatmaps {cname}")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def gradcam_summary_efficientnet(model, test_loader, dataset, device, out_path: Path, class_names):
    preds, ys, _ = collect_test_predictions(model, test_loader, device)
    last_block = model.backbone.blocks[-1]
    gc = GradCAM(model, target_module=last_block)
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for c in range(9):
        r, cc = divmod(c, 3)
        ax = axes[r, cc]
        ok, _ = sample_indices_per_class(preds, ys, c, 1)
        if not ok:
            ax.axis("off")
            continue
        gi = ok[0]
        img_t, _ = dataset[gi]
        img_b = img_t.unsqueeze(0).to(device)
        pred_c = int(preds[gi])
        with torch.enable_grad():
            hm = gc.generate(img_b, target_class=pred_c)
        img_rgb = denormalize_chw(img_t)
        heat = try_cv2_jet_heatmap(hm)
        ov = blend_overlay((np.clip(img_rgb, 0, 1) * 255).astype(np.uint8).astype(np.float32) / 255.0, heat, 0.4)
        side = np.hstack([np.clip(img_rgb, 0, 1), np.clip(ov, 0, 1)])
        ax.imshow(side)
        ax.set_title(class_names[c][:18] if c < len(class_names) else str(c), fontsize=9)
        ax.axis("off")
    gc.remove_hooks()
    plt.suptitle("EfficientNet-B3: original | Grad-CAM overlay (one sample per class)", fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    device = device_arg(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(ROOT / "logs" / "phase2" / "gradcam.log")

    yaml = Path(args.config) if args.config else default_pathmnist_yaml(ROOT)
    (_, _, test_loader), cfg = get_pathmnist_loaders(ROOT, yaml, args.seed)
    dataset = test_loader.dataset
    class_names = cfg["dataset"].get("class_names", [str(i) for i in range(9)])

    for disp, ck_rel, arch, var, layer in CNN_MODELS:
        ck = Path(args.model_path) if args.model_path else ROOT / ck_rel
        if not ck.is_file():
            logger.warning("Skip %s — missing %s", disp, ck)
            continue
        model = load_model_from_checkpoint(ck, device, arch, var)
        run_one_model(disp, ck_rel, arch, var, layer, model, test_loader, dataset, device, out_dir, class_names)

    # Summary: EfficientNet-B3 only
    ck_e = ROOT / "experiments/phase1/efficientnet_b3/best_model.pt"
    if ck_e.is_file():
        m = load_model_from_checkpoint(ck_e, device, "efficientnet", "b3")
        gradcam_summary_efficientnet(
            m, test_loader, dataset, device, out_dir / "gradcam_summary.png", class_names
        )


if __name__ == "__main__":
    main()
