"""
Build Trainer-compatible config dicts from MedMNIST YAML files (pathmnist / chestmnist).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _deep_update(base: Dict[str, Any], updates: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not updates:
        return base
    for k, v in updates.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _loss_type_from_yaml(loss_block: Dict[str, Any]) -> str:
    t = loss_block.get("type") or loss_block.get("name") or "cross_entropy"
    return str(t).lower().replace("-", "_").replace(" ", "_")


def _optimizer_block(raw_training: Dict[str, Any]) -> Dict[str, Any]:
    opt = raw_training.get("optimizer", {})
    name = str(opt.get("name", "adam")).lower()
    lr = float(raw_training.get("learning_rate", opt.get("lr", 1e-3)))
    wd = float(raw_training.get("weight_decay", opt.get("weight_decay", 1e-4)))
    return {
        "type": name,
        "lr": lr,
        "weight_decay": wd,
        "momentum": float(opt.get("momentum", 0.9)),
        "betas": [
            float(opt.get("beta1", 0.9)),
            float(opt.get("beta2", 0.999)),
        ],
    }


def _scheduler_block(raw_training: Dict[str, Any]) -> Dict[str, Any]:
    sch = raw_training.get("scheduler", {})
    name = str(sch.get("name", sch.get("type", "cosine"))).lower()
    num_epochs = int(raw_training.get("max_epochs", raw_training.get("num_epochs", 100)))
    return {
        "type": name,
        "step_size": int(sch.get("step_size", 30)),
        "gamma": float(sch.get("gamma", 0.1)),
        "T_max": int(sch.get("T_max", num_epochs)),
        "patience": int(sch.get("patience", 10)),
    }


def _training_block(raw: Dict[str, Any]) -> Dict[str, Any]:
    rt = raw["training"]
    loss_src = rt.get("loss", {})
    loss_block: Dict[str, Any] = {
        "type": _loss_type_from_yaml(loss_src),
        "label_smoothing": float(loss_src.get("label_smoothing", 0.0)),
        "class_weights": loss_src.get("class_weights"),
        "pos_weight": loss_src.get("pos_weight"),
    }
    num_epochs = int(rt.get("max_epochs", rt.get("num_epochs", 100)))
    return {
        "num_epochs": num_epochs,
        "batch_size": int(rt.get("batch_size", 128)),
        "num_workers": int(rt.get("num_workers", 4)),
        "pin_memory": bool(rt.get("pin_memory", True)),
        "optimizer": _optimizer_block(rt),
        "scheduler": _scheduler_block(rt),
        "loss": loss_block,
        "early_stopping": rt.get("early_stopping", {}),
    }


def build_trainer_config_from_pathmnist_yaml(
    yaml_path: str | Path,
    seed: int = 42,
    checkpoint_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    save_every_n_epochs: int = 10,
) -> Dict[str, Any]:
    yaml_path = Path(yaml_path)
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    config: Dict[str, Any] = {
        "experiment": dict(raw.get("experiment", {"name": "pathmnist"})),
        "paths": dict(raw.get("paths", {})),
        "dataset": dict(raw["dataset"]),
        "data_split": dict(raw.get("data_split", {})),
        "augmentation": copy.deepcopy(raw.get("augmentation", {})),
        "model": dict(raw["model"]),
        "training": _training_block(raw),
        "reproducibility": {
            "seed": seed,
            "deterministic": bool(raw.get("deterministic", True)),
            "benchmark": bool(raw.get("reproducibility", {}).get("benchmark", False)),
        },
        "triage": copy.deepcopy(raw.get("triage", {})),
        "checkpoint": {"save_every_n_epochs": int(save_every_n_epochs)},
    }
    if checkpoint_dir is not None:
        config["paths"]["checkpoint_dir"] = str(checkpoint_dir)
    if log_dir is not None:
        config["paths"]["log_dir"] = str(log_dir)
    _deep_update(config, overrides)
    config["reproducibility"]["seed"] = int(seed)
    return config


def build_trainer_config_from_chestmnist_yaml(
    yaml_path: str | Path,
    seed: int = 42,
    checkpoint_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    save_every_n_epochs: int = 10,
    force_rgb: bool = True,
) -> Dict[str, Any]:
    yaml_path = Path(yaml_path)
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    config: Dict[str, Any] = {
        "experiment": dict(raw.get("experiment", {"name": "chestmnist"})),
        "paths": dict(raw.get("paths", {})),
        "dataset": dict(raw["dataset"]),
        "data_split": dict(raw.get("data_split", {})),
        "augmentation": copy.deepcopy(raw.get("augmentation", {})),
        "model": dict(raw["model"]),
        "training": _training_block(raw),
        "reproducibility": {
            "seed": int(seed),
            "deterministic": bool(raw.get("deterministic", True)),
            "benchmark": bool(raw.get("reproducibility", {}).get("benchmark", False)),
        },
        "triage": copy.deepcopy(raw.get("triage", {})),
        "checkpoint": {"save_every_n_epochs": int(save_every_n_epochs)},
    }
    if checkpoint_dir is not None:
        config["paths"]["checkpoint_dir"] = str(checkpoint_dir)
    if log_dir is not None:
        config["paths"]["log_dir"] = str(log_dir)

    if force_rgb:
        config["dataset"]["as_rgb"] = True
        config["dataset"]["input_channels"] = 3
        config["model"]["input_channels"] = 3
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        for split in ("train", "val", "test"):
            if split in config["augmentation"]:
                if "normalize" in config["augmentation"][split]:
                    config["augmentation"][split]["normalize"] = {
                        "mean": img_mean,
                        "std": img_std,
                    }

    _deep_update(config, overrides)
    config["reproducibility"]["seed"] = int(seed)
    return config


def build_trainer_config_from_dermamnist_yaml(
    yaml_path: str | Path,
    seed: int = 42,
    checkpoint_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    save_every_n_epochs: int = 999,
) -> Dict[str, Any]:
    """Trainer config from DermaMNIST YAML; loss block forced to cross_entropy for Trainer compatibility."""
    yaml_path = Path(yaml_path)
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    raw_training = dict(raw["training"])
    raw_training["loss"] = {
        "name": "cross_entropy",
        "type": "cross_entropy",
        "label_smoothing": 0.0,
        "class_weights": None,
    }
    raw_copy = dict(raw)
    raw_copy["training"] = raw_training

    config: Dict[str, Any] = {
        "experiment": dict(raw.get("experiment", {"name": "dermamnist"})),
        "paths": dict(raw.get("paths", {})),
        "dataset": dict(raw["dataset"]),
        "data_split": dict(raw.get("data_split", {})),
        "augmentation": copy.deepcopy(raw.get("augmentation", {})),
        "model": dict(raw["model"]),
        "training": _training_block(raw_copy),
        "reproducibility": {
            "seed": seed,
            "deterministic": bool(raw.get("deterministic", True)),
            "benchmark": bool(raw.get("reproducibility", {}).get("benchmark", False)),
        },
        "triage": copy.deepcopy(raw.get("triage", {})),
        "checkpoint": {"save_every_n_epochs": int(save_every_n_epochs)},
    }
    if checkpoint_dir is not None:
        config["paths"]["checkpoint_dir"] = str(checkpoint_dir)
    if log_dir is not None:
        config["paths"]["log_dir"] = str(log_dir)
    _deep_update(config, overrides)
    config["reproducibility"]["seed"] = int(seed)
    return config
