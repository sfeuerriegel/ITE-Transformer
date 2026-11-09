"""Shared utilities for IPW training scripts and analysis notebooks."""
from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import wandb
except Exception:
    wandb = None

from src.DR_Learner.prepare_treatment_effect_dataset import (
    create_treatment_effect_dataset_from_pickles,
    load_pickle_dataframe,
)
from src.data_prep.build_sequences import Vocab, _build_compat_stoi
from src.model import Transformer

@dataclass
class SplitIndices:
    train: list[int]
    val: list[int]
    test: list[int]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_vocab(path: Path) -> Vocab:
    payload = json.loads(path.read_text(encoding="utf-8"))
    itos = payload.get("itos")
    if not isinstance(itos, list) or not itos:
        raise ValueError(f"{path} does not contain a valid vocab.json payload.")
    stoi = _build_compat_stoi(itos)
    return Vocab(stoi=stoi, itos=itos)


def infer_transformer_hparams(state_dict: dict[str, torch.Tensor], nhead: int, dropout: float) -> dict[str, int | float]:
    # Rebuild the encoder architecture directly from checkpoint tensor shapes so
    # the linear head can be trained without separately storing model hparams.
    d_model = int(state_dict["tok_emb.weight"].shape[1])
    vocab_size = int(state_dict["tok_emb.weight"].shape[0])
    layer_ids = {
        int(key.split(".")[2])
        for key in state_dict
        if key.startswith("transformer.layers.") and key.split(".")[2].isdigit()
    }
    if not layer_ids:
        raise ValueError("Could not infer the number of transformer layers from the checkpoint.")

    ff_dim = int(state_dict["transformer.layers.0.linear1.weight"].shape[0])
    if ff_dim % d_model != 0:
        raise ValueError("Could not infer ff_mult cleanly from the checkpoint.")

    return {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": max(layer_ids) + 1,
        "ff_mult": ff_dim // d_model,
        "dropout": dropout,
    }


def load_transformer(checkpoint_path: Path, vocab: Vocab, nhead: int, dropout: float, device: torch.device) -> Transformer:
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    hparams = infer_transformer_hparams(state_dict=state_dict, nhead=nhead, dropout=dropout)

    if hparams["vocab_size"] != len(vocab):
        raise ValueError(
            f"Checkpoint vocab size ({hparams['vocab_size']}) does not match vocab.json size ({len(vocab)}).",
        )

    encoder = Transformer(
        vocab_size=len(vocab),
        d_model=int(hparams["d_model"]),
        nhead=int(hparams["nhead"]),
        num_layers=int(hparams["num_layers"]),
        ff_mult=int(hparams["ff_mult"]),
        dropout=float(hparams["dropout"]),
        pad_idx=vocab.pad_idx,
    )
    incompatible = encoder.load_state_dict(state_dict, strict=False)
    allowed_missing = {
        "pool_ffn.0.weight",
        "pool_ffn.0.bias",
        "pool_ffn.2.weight",
        "pool_ffn.2.bias",
        "time_head.weight",
        "time_head.bias",
    }
    missing_keys = set(incompatible.missing_keys)
    if incompatible.unexpected_keys or (missing_keys - allowed_missing):
        error_messages = []
        if missing_keys - allowed_missing:
            error_messages.append(f"Missing key(s) in state_dict: {sorted(missing_keys)!r}.")
        if incompatible.unexpected_keys:
            error_messages.append(f"Unexpected key(s) in state_dict: {sorted(incompatible.unexpected_keys)!r}.")
        raise RuntimeError("Error(s) in loading state_dict for Transformer:\n\t" + "\n\t".join(error_messages))
    encoder.to(device)
    return encoder


def load_samples_dataframe(args: argparse.Namespace, vocab: Vocab) -> pd.DataFrame:
    if args.samples_path is not None:
        samples_path = args.samples_path
        if samples_path.suffix.lower() == ".parquet":
            return pd.read_parquet(samples_path)
        return load_pickle_dataframe(samples_path)

    required_paths = [
        args.treatment_pkl,
        args.comparator_pkl,
        args.conditions_pkl,
        args.persons_pkl,
    ]
    if any(path is None for path in required_paths):
        raise ValueError(
            "Provide --samples-path or all of --treatment-pkl, --comparator-pkl, --conditions-pkl, and --persons-pkl.",
        )

    return create_treatment_effect_dataset_from_pickles(
        treatment_pkl=args.treatment_pkl,
        comparator_pkl=args.comparator_pkl,
        conditions_pkl=args.conditions_pkl,
        persons_pkl=args.persons_pkl,
        vocab=vocab,
        treatment_name=args.treatment_name,
        comparator_name=args.comparator_name,
        outcome_horizon_days=args.outcome_horizon_days,
        min_history_events=args.min_history_events,
        max_history_events=args.max_history_events,
    )


def stratified_split_indices(
    labels: Sequence[int],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> SplitIndices:
    if not 0.0 < train_frac < 1.0:
        raise ValueError("--train-frac must be between 0 and 1.")
    if not 0.0 <= val_frac < 1.0:
        raise ValueError("--val-frac must be between 0 and 1.")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0.")

    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        by_class.setdefault(int(label), []).append(idx)

    splits = {"train": [], "val": [], "test": []}
    for class_indices in by_class.values():
        # Split each treatment class separately so train/val/test keep roughly
        # the same treated vs comparator ratio.
        rng.shuffle(class_indices)
        n_total = len(class_indices)
        n_train = max(1, int(round(n_total * train_frac)))
        n_val = int(round(n_total * val_frac))
        if n_train + n_val >= n_total:
            n_val = max(0, n_total - n_train - 1)

        splits["train"].extend(class_indices[:n_train])
        splits["val"].extend(class_indices[n_train:n_train + n_val])
        splits["test"].extend(class_indices[n_train + n_val:])

    for split_name in splits:
        splits[split_name].sort()

    return SplitIndices(train=splits["train"], val=splits["val"], test=splits["test"])


def binary_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    labels_np = np.asarray(labels, dtype=np.int64)
    scores_np = np.asarray(scores, dtype=np.float64)
    pos = int(labels_np.sum())
    neg = int((1 - labels_np).sum())
    if pos == 0 or neg == 0:
        return float("nan")

    # AUC is computed from score ranks (equivalent to the Mann-Whitney U
    # statistic), which avoids adding a hard dependency on sklearn.
    ranks = pd.Series(scores_np).rank(method="average").to_numpy(dtype=np.float64)
    rank_sum_pos = float(ranks[labels_np == 1].sum())
    return (rank_sum_pos - (pos * (pos + 1) / 2.0)) / (pos * neg)


def build_propensity_criterion(
    train_loader: DataLoader,
    device: torch.device,
    disable_pos_weight: bool = False,
) -> nn.Module:
    # Estimate class imbalance from the training split only and upweight the
    # treated class in the loss when treatment assignment is rare.
    if disable_pos_weight:
        pos_weight = None
    else:
        train_targets = [
            int(train_loader.dataset.dataset.base_dataset.samples_df.iloc[idx]["treatment"])
            for idx in train_loader.dataset.indices
        ]
        pos = sum(train_targets)
        neg = len(train_targets) - pos
        pos_weight = torch.tensor([neg / max(1, pos)], dtype=torch.float32, device=device)

    return nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")


def init_wandb(args: argparse.Namespace) -> object | None:
    if wandb is None:
        return None

    config = serialize_args(args)
    project_name = getattr(args, "wandb_project", "IPW")
    run = wandb.init(project=project_name, config=config)
    return run


def _slugify_run_part(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "na"


def build_wandb_run_name(args: argparse.Namespace, task_name: str) -> str:
    parts = [_slugify_run_part(task_name)]

    treatment_name = getattr(args, "treatment_name", None)
    comparator_name = getattr(args, "comparator_name", None)
    if treatment_name and comparator_name:
        parts.append(f"{_slugify_run_part(treatment_name)}-vs-{_slugify_run_part(comparator_name)}")

    pseudo_outcome_column = getattr(args, "pseudo_outcome_column", None)
    if pseudo_outcome_column:
        parts.append(_slugify_run_part(pseudo_outcome_column))

    num_folds = getattr(args, "num_folds", None)
    if num_folds is not None:
        parts.append(f"{int(num_folds)}fold")

    seed = getattr(args, "seed", None)
    if seed is not None:
        parts.append(f"seed{int(seed)}")

    parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
    return "__".join(parts)


def configure_wandb_run(
    run: object | None,
    args: argparse.Namespace,
    task_name: str,
    *,
    define_default_epoch_metrics: bool = True,
    define_global_step_metrics: bool = False,
) -> object | None:
    if run is None or wandb is None:
        return run

    try:
        run.name = build_wandb_run_name(args=args, task_name=task_name)
    except Exception:
        pass

    try:
        run.config.update({"task_name": task_name}, allow_val_change=True)
    except Exception:
        pass

    try:
        if define_default_epoch_metrics:
            wandb.define_metric("epoch")
            wandb.define_metric("train/*", step_metric="epoch")
            wandb.define_metric("val/*", step_metric="epoch")
            wandb.define_metric("test/*", step_metric="epoch")
            wandb.define_metric("summary/*", step_metric="epoch")
            wandb.define_metric("optimizer/*", step_metric="epoch")
        if define_global_step_metrics:
            wandb.define_metric("trainer/global_step")
            wandb.define_metric("trainer/*", step_metric="trainer/global_step")
            wandb.define_metric("crossfit/*", step_metric="trainer/global_step")
    except Exception:
        pass

    return run


def serialize_args(args: argparse.Namespace) -> dict[str, object]:
    serializable: dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            serializable[key] = str(value)
        else:
            serializable[key] = value
    return serializable


