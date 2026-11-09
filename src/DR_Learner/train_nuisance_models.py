from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

try:
    import wandb
except Exception:
    wandb = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.DR_Learner.prepare_treatment_effect_dataset import TreatmentEffectDataset
from src.DR_Learner.helpers import (
    binary_auc,
    configure_wandb_run,
    build_wandb_run_name,
    build_propensity_criterion,
    init_wandb,
    load_samples_dataframe,
    load_transformer,
    load_vocab,
    serialize_args,
    set_seed,
)
from src.model import Transformer


class IndexedOutcomeDataset(Dataset):
    """Wrap the treatment-effect dataset and keep stable row ids for exports."""

    def __init__(self, base_dataset: TreatmentEffectDataset):
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict[str, object]:
        item = self.base_dataset[idx]
        item["row_index"] = int(idx)
        return item


def collate_outcome_batch(batch: list[dict[str, object]]) -> dict[str, torch.Tensor]:
    tokens = [item["history_tokens"] for item in batch]
    ages = [item["history_ages"] for item in batch]
    tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=0)
    ages_padded = pad_sequence(ages, batch_first=True, padding_value=0)

    return {
               'tokens': tokens_padded,
               'ages': ages_padded,
               'attention_mask': (tokens_padded != 0).long(),
               'treatment': torch.stack([item['treatment'] for item in batch]).to(torch.float32),
               'outcome': torch.stack([item['outcome'] for item in batch]).to(torch.float32),
               'row_index': torch.tensor([int(item['row_index']) for item in batch], dtype=torch.long),
               'person_id': torch.tensor([int(item['person_id']) if item['person_id'] is not None else -1 for item in batch], dtype=torch.long),
           }


def make_loader(
    dataset: IndexedOutcomeDataset,
    indices: list[int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    subset = torch.utils.data.Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_outcome_batch,
    )


class SingleHeadTreatmentEffectModel(nn.Module):
    """Shared transformer representation with one 3-logit nuisance head."""

    def __init__(
        self,
        encoder: Transformer,
        freeze_encoder: bool = False,
        head_hidden_dim: int | None = None,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        self.head_input_dim = int(encoder.d_model)
        self.head_hidden_dim = int(head_hidden_dim or encoder.d_model)
        self.head_output_dim = 3
        self.head = nn.Sequential(
            nn.Linear(self.head_input_dim, self.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(self.head_hidden_dim, self.head_output_dim),
        )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

    def train(self, mode: bool = True) -> "SingleHeadTreatmentEffectModel":
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    @staticmethod
    def _last_valid_hidden(hidden: torch.Tensor, attention_mask: torch.LongTensor) -> torch.Tensor:
        valid_lengths = attention_mask.sum(dim=1)
        last_idx = (valid_lengths - 1).clamp(min=0)
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_idx, last_idx, :]

    def encode_history(
        self,
        tokens: torch.LongTensor,
        ages: torch.FloatTensor,
        attention_mask: torch.LongTensor,
    ) -> torch.Tensor:
        key_padding_mask = attention_mask == 0

        if self.freeze_encoder:
            with torch.no_grad():
                encoder_out = self.encoder(
                    seq_tokens=tokens,
                    seq_age=ages,
                    key_padding_mask=key_padding_mask,
                )
        else:
            encoder_out = self.encoder(
                seq_tokens=tokens,
                seq_age=ages,
                key_padding_mask=key_padding_mask,
            )

        return self._last_valid_hidden(encoder_out["hidden"], attention_mask)

    def forward(
        self,
        tokens: torch.LongTensor,
        ages: torch.FloatTensor,
        attention_mask: torch.LongTensor,
    ) -> dict[str, torch.Tensor]:
        history_rep = self.encode_history(tokens=tokens, ages=ages, attention_mask=attention_mask)
        logits = self.head(history_rep)
        return {
            "history_rep": history_rep,
            "propensity_logits": logits[:, 0],
            "comparator_outcome_logits": logits[:, 1],
            "treated_outcome_logits": logits[:, 2],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-fit nuisance models and export IPW / AIPW pseudo-outcomes.",
    )

    data_group = parser.add_argument_group("Dataset inputs")
    data_group.add_argument("--samples-path", type=Path, default=None, help="Optional prebuilt treatment-effect dataset (.pkl or .parquet).")
    data_group.add_argument("--treatment-pkl", type=Path, default=None, help="Pickle containing the treatment cohort.")
    data_group.add_argument("--comparator-pkl", type=Path, default=None, help="Pickle containing the comparator cohort.")
    data_group.add_argument("--conditions-pkl", type=Path, default=None, help="Pickle containing condition history.")
    data_group.add_argument("--persons-pkl", type=Path, default=None, help="Pickle containing person demographics.")
    data_group.add_argument("--treatment-name", type=str, default="treatment")
    data_group.add_argument("--comparator-name", type=str, default="comparator")
    data_group.add_argument("--min-history-events", type=int, default=0)
    data_group.add_argument("--max-history-events", type=int, default=None)


    data_group.add_argument(
        "--split-indices-path",
        type=Path,
        default=None,
        help="Optional JSON path containing a top-level {'train': [...], 'test': [...]} split.",
    )

    model_group = parser.add_argument_group("Transformer encoder")
    model_group.add_argument("--transformer-checkpoint", type=Path, default=None, help="Path to the pretrained transformer state dict.")
    model_group.add_argument("--vocab-path", type=Path, required=True, help="Path to the vocab.json used by the transformer.")
    model_group.add_argument("--random-init-transformer", action="store_true", help="Initialize the transformer encoder from scratch instead of loading checkpoint weights.")
    model_group.add_argument("--d-model", type=int, default=128, help="Transformer hidden width used with --random-init-transformer.")
    model_group.add_argument("--nhead", type=int, default=4, help="Number of attention heads used by the pretrained transformer.")
    model_group.add_argument("--num-layers", type=int, default=6, help="Transformer layer count used with --random-init-transformer.")
    model_group.add_argument("--ff-mult", type=int, default=6, help="Transformer feed-forward multiplier used with --random-init-transformer.")
    model_group.add_argument("--dropout", type=float, default=0.1, help="Dropout to use when rebuilding the transformer module.")
    model_group.add_argument("--freeze-transformer", action="store_true", help="Freeze the transformer and only train the nuisance head.")
    model_group.add_argument("--head-hidden-dim", type=int, default=None, help="Hidden width of the two-layer prediction head. Defaults to the transformer d_model.")
    model_group.add_argument("--head-dropout", type=float, default=0.1, help="Dropout used inside the nuisance head.")

    train_group = parser.add_argument_group("Optimization")
    train_group.add_argument("--output-dir", type=Path, required=True)
    train_group.add_argument("--batch-size", type=int, default=64)
    train_group.add_argument("--epochs", type=int, default=20)
    train_group.add_argument("--lr", type=float, default=1e-4)
    train_group.add_argument("--transformer-lr", type=float, default=None, help="Optional learning rate for transformer parameters. Defaults to --lr.")
    train_group.add_argument("--head-lr", type=float, default=None, help="Optional learning rate for head parameters. Defaults to --lr.")
    train_group.add_argument("--min-transformer-lr", type=float, default=None, help="Final transformer learning rate for linear decay. Defaults to --transformer-lr.")
    train_group.add_argument("--max-transformer-lr", type=float, default=None, help="Initial transformer learning rate for linear decay. Defaults to --transformer-lr.")
    train_group.add_argument("--min-head-lr", type=float, default=None, help="Final nuisance head learning rate for linear decay. Defaults to --head-lr.")
    train_group.add_argument("--max-head-lr", type=float, default=None, help="Initial nuisance head learning rate for linear decay. Defaults to --head-lr.")
    train_group.add_argument("--weight-decay", type=float, default=1e-4)
    train_group.add_argument("--grad-clip-norm", type=float, default=None, help="Optional max gradient norm applied before each optimizer step.")
    train_group.add_argument("--disable-propensity-pos-weight", action="store_true", help="Disable the default positive-class reweighting in the propensity loss.")
    train_group.add_argument("--outcome-loss-weight-start", type=float, default=0.1, help="Initial multiplier applied to the factual outcome loss at epoch 1.")
    train_group.add_argument("--outcome-loss-weight-end", type=float, default=1.0, help="Final multiplier applied to the factual outcome loss at the last epoch.")

    train_group.add_argument("--num-folds", type=int, default=5, help="Number of outer folds used for nuisance cross-fitting.")
    train_group.add_argument("--fold-val-frac", type=float, default=0.15, help="Validation fraction carved out from each fold's training pool for early stopping.")
    train_group.add_argument("--propensity-clip", type=float, default=0.01, help="Lower and upper clipping threshold used when building pseudo-outcomes.")
    train_group.add_argument("--skip-pseudo-outcomes", action="store_true", help="Skip IPW/AIPW pseudo-outcome columns when only nuisance predictions are needed.")
    train_group.add_argument("--save-fold-checkpoints", action=argparse.BooleanOptionalAction, default=True, help="Whether to save the best checkpoint for each outer fold.")
    train_group.add_argument("--wandb-project", type=str, default="nuisance models")
    train_group.add_argument("--seed", type=int, default=42)
    train_group.add_argument("--num-workers", type=int, default=0)
    train_group.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def accuracy_from_probs(labels: list[int], probs: list[float]) -> float:
    total = max(1, len(labels))
    preds = [int(prob >= 0.5) for prob in probs]
    return sum(int(pred == label) for pred, label in zip(preds, labels)) / total


def select_factual_outcome_logits(
    treated_logits: torch.Tensor,
    comparator_logits: torch.Tensor,
    treatment: torch.Tensor,
) -> torch.Tensor:
    treated_mask = treatment.to(torch.bool)
    return torch.where(treated_mask, treated_logits, comparator_logits)


def outcome_loss_weight_for_epoch(epoch: int, epochs: int, start: float, end: float) -> float:
    if epochs <= 1:
        return float(end)
    progress = float(epoch - 1) / float(epochs - 1)
    return float(start + (end - start) * progress)


def linear_schedule_value(epoch: int, epochs: int, start: float, end: float) -> float:
    if epochs <= 1:
        return float(start)
    progress = float(epoch - 1) / float(epochs - 1)
    return float(start + (end - start) * progress)


def validate_lr_range(name: str, min_lr: float, max_lr: float) -> None:
    if min_lr < 0 or max_lr < 0:
        raise ValueError(f"{name} learning rates must be non-negative.")
    if min_lr > max_lr:
        raise ValueError(f"{name} min learning rate must be less than or equal to max learning rate.")


def set_optimizer_group_lr(optimizer: torch.optim.Optimizer, group_name: str, lr: float) -> None:
    for group in optimizer.param_groups:
        if group.get("name") == group_name:
            group["lr"] = float(lr)


def get_optimizer_group_lr(optimizer: torch.optim.Optimizer, group_name: str) -> float:
    for group in optimizer.param_groups:
        if group.get("name") == group_name:
            return float(group["lr"])
    return 0.0


def compute_losses(
    outputs: dict[str, torch.Tensor],
    treatment: torch.Tensor,
    outcome: torch.Tensor,
    propensity_criterion: nn.Module,
    outcome_criterion: nn.Module,
) -> dict[str, torch.Tensor]:
    propensity_losses = propensity_criterion(outputs["propensity_logits"], treatment)
    factual_outcome_logits = select_factual_outcome_logits(
        treated_logits=outputs["treated_outcome_logits"],
        comparator_logits=outputs["comparator_outcome_logits"],
        treatment=treatment,
    )
    outcome_losses = outcome_criterion(factual_outcome_logits, outcome)


    return {
        "propensity_loss": propensity_losses.mean(),
        "outcome_loss": outcome_losses.mean(),
        "factual_outcome_logits": factual_outcome_logits,
    }


def evaluate(
    model: SingleHeadTreatmentEffectModel,
    loader: DataLoader,
    propensity_criterion: nn.Module,
    outcome_criterion: nn.Module,
    device: torch.device,
    outcome_loss_weight: float,
) -> dict[str, float]:
    model.eval()
    propensity_loss_sum = 0.0
    outcome_loss_sum = 0.0
    sample_count = 0.0
    all_treatment_labels: list[int] = []
    all_propensity_probs: list[float] = []
    all_outcome_labels: list[int] = []
    all_factual_outcome_probs: list[float] = []

    with torch.no_grad():
        for batch in loader:
            tokens = batch["tokens"].to(device)
            ages = batch["ages"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            treatment = batch["treatment"].to(device)
            outcome = batch["outcome"].to(device)


            outputs = model(tokens=tokens, ages=ages, attention_mask=attention_mask)
            losses = compute_losses(
                outputs=outputs,
                treatment=treatment,
                outcome=outcome,
                propensity_criterion=propensity_criterion,
                outcome_criterion=outcome_criterion,
            )

            batch_size = float(batch["treatment"].numel())
            propensity_loss_sum += float(losses["propensity_loss"].item() * batch_size)
            outcome_loss_sum += float(losses["outcome_loss"].item() * batch_size)
            sample_count += batch_size
            all_treatment_labels.extend(treatment.cpu().to(torch.int64).tolist())
            all_propensity_probs.extend(torch.sigmoid(outputs["propensity_logits"]).cpu().tolist())
            all_outcome_labels.extend(outcome.cpu().to(torch.int64).tolist())
            all_factual_outcome_probs.extend(torch.sigmoid(losses["factual_outcome_logits"]).cpu().tolist())

    propensity_loss = propensity_loss_sum / max(sample_count, 1e-12)
    outcome_loss = outcome_loss_sum / max(sample_count, 1e-12)
    total_loss = propensity_loss + float(outcome_loss_weight) * outcome_loss

    return {
        "loss": total_loss,
        "propensity_loss": propensity_loss,
        "outcome_loss": outcome_loss,
        "propensity_accuracy": accuracy_from_probs(all_treatment_labels, all_propensity_probs),
        "propensity_auc": binary_auc(all_treatment_labels, all_propensity_probs),
        "outcome_accuracy": accuracy_from_probs(all_outcome_labels, all_factual_outcome_probs),
        "outcome_auc": binary_auc(all_outcome_labels, all_factual_outcome_probs),
    }


def fit(
    model: SingleHeadTreatmentEffectModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    transformer_lr: float,
    head_lr: float,
    weight_decay: float,
    propensity_criterion: nn.Module,
    outcome_criterion: nn.Module,
    device: torch.device,
    outcome_loss_weight_start: float,
    outcome_loss_weight_end: float,
    min_transformer_lr: float | None = None,
    max_transformer_lr: float | None = None,
    min_head_lr: float | None = None,
    max_head_lr: float | None = None,
    grad_clip_norm: float | None = None,
    wandb_run: object | None = None,
    log_namespace: str | None = None,
    log_step_offset: int = 0,
    fold_index: int | None = None,
) -> tuple[SingleHeadTreatmentEffectModel, dict[str, float]]:
    if grad_clip_norm is not None and grad_clip_norm <= 0:
        raise ValueError("grad_clip_norm must be positive when provided.")

    max_transformer_lr = float(transformer_lr if max_transformer_lr is None else max_transformer_lr)
    min_transformer_lr = float(max_transformer_lr if min_transformer_lr is None else min_transformer_lr)
    max_head_lr = float(head_lr if max_head_lr is None else max_head_lr)
    min_head_lr = float(max_head_lr if min_head_lr is None else min_head_lr)
    validate_lr_range("Transformer", min_transformer_lr, max_transformer_lr)
    validate_lr_range("Head", min_head_lr, max_head_lr)

    optimizer_param_groups: list[dict[str, object]] = []
    encoder_params = [param for param in model.encoder.parameters() if param.requires_grad]
    head_params = [param for param in model.head.parameters() if param.requires_grad]
    if encoder_params:
        optimizer_param_groups.append(
            {
                "params": encoder_params,
                "lr": max_transformer_lr,
                "weight_decay": weight_decay,
                "name": "transformer",
            }
        )
    if head_params:
        optimizer_param_groups.append(
            {
                "params": head_params,
                "lr": max_head_lr,
                "weight_decay": weight_decay,
                "name": "head",
            }
        )
    if not optimizer_param_groups:
        raise ValueError("No trainable parameters were found for optimization.")

    optimizer = torch.optim.AdamW(optimizer_param_groups)
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    best_val_selection_loss = math.inf
    num_train_batches = len(train_loader)
    metric_prefix = f"{log_namespace}/" if log_namespace else ""

    for epoch in range(1, epochs + 1):
        model.train()
        running_propensity_loss_sum = 0.0
        running_outcome_loss_sum = 0.0
        running_sample_count = 0.0
        current_transformer_lr = linear_schedule_value(
            epoch=epoch,
            epochs=epochs,
            start=max_transformer_lr,
            end=min_transformer_lr,
        )
        current_head_lr = linear_schedule_value(
            epoch=epoch,
            epochs=epochs,
            start=max_head_lr,
            end=min_head_lr,
        )
        set_optimizer_group_lr(optimizer, "transformer", current_transformer_lr)
        set_optimizer_group_lr(optimizer, "head", current_head_lr)
        outcome_loss_weight = outcome_loss_weight_for_epoch(
            epoch=epoch,
            epochs=epochs,
            start=outcome_loss_weight_start,
            end=outcome_loss_weight_end,
        )

        print(
            f"{log_namespace or 'fit'} | epoch {epoch:03d} | training on {num_train_batches} batches "
            f"({len(train_loader.dataset)} samples) on {device.type} | "
            f"transformer_lr={get_optimizer_group_lr(optimizer, 'transformer'):.6g} | "
            f"head_lr={get_optimizer_group_lr(optimizer, 'head'):.6g} | "
            f"outcome_loss_weight={outcome_loss_weight:.4f}",
            flush=True,
        )

        for batch in train_loader:
            tokens = batch["tokens"].to(device)
            ages = batch["ages"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            treatment = batch["treatment"].to(device)
            outcome = batch["outcome"].to(device)


            optimizer.zero_grad(set_to_none=True)
            outputs = model(tokens=tokens, ages=ages, attention_mask=attention_mask)
            losses = compute_losses(
                outputs=outputs,
                treatment=treatment,
                outcome=outcome,
                propensity_criterion=propensity_criterion,
                outcome_criterion=outcome_criterion,
            )
            loss = losses["propensity_loss"] + outcome_loss_weight * losses["outcome_loss"]
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            batch_size = float(batch["treatment"].numel())
            running_propensity_loss_sum += float(losses["propensity_loss"].item() * batch_size)
            running_outcome_loss_sum += float(losses["outcome_loss"].item() * batch_size)
            running_sample_count += batch_size

        train_propensity_loss = running_propensity_loss_sum / max(running_sample_count, 1e-12)
        train_outcome_loss = running_outcome_loss_sum / max(running_sample_count, 1e-12)
        train_total_loss = train_propensity_loss + outcome_loss_weight * train_outcome_loss

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            propensity_criterion=propensity_criterion,
            outcome_criterion=outcome_criterion,
            device=device,
            outcome_loss_weight=outcome_loss_weight,
        )
        print(
            f"{log_namespace or 'fit'} | epoch {epoch:03d} | train_loss={train_total_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | val_outcome_loss={val_metrics['outcome_loss']:.4f} | "
            f"val_propensity_auc={val_metrics['propensity_auc']:.4f} | val_outcome_auc={val_metrics['outcome_auc']:.4f}",
            flush=True,
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "trainer/global_step": log_step_offset + epoch,
                    "trainer/fold_epoch": epoch,
                    "trainer/fold_index": -1 if fold_index is None else int(fold_index),
                    "train/loss": train_total_loss,
                    "train/propensity_loss": train_propensity_loss,
                    "train/outcome_loss": train_outcome_loss,
                    "train/outcome_loss_weight": outcome_loss_weight,
                    "val/loss": val_metrics["loss"],
                    "val/propensity_loss": val_metrics["propensity_loss"],
                    "val/outcome_loss": val_metrics["outcome_loss"],
                    "val/propensity_accuracy": val_metrics["propensity_accuracy"],
                    "val/propensity_auc": val_metrics["propensity_auc"],
                    "val/outcome_accuracy": val_metrics["outcome_accuracy"],
                    "val/outcome_auc": val_metrics["outcome_auc"],
                    "optimizer/transformer_lr": get_optimizer_group_lr(optimizer, "transformer"),
                    "optimizer/head_lr": get_optimizer_group_lr(optimizer, "head"),
                    f"{metric_prefix}epoch": epoch,
                    f"{metric_prefix}train/loss": train_total_loss,
                    f"{metric_prefix}train/propensity_loss": train_propensity_loss,
                    f"{metric_prefix}train/outcome_loss": train_outcome_loss,
                    f"{metric_prefix}train/outcome_loss_weight": outcome_loss_weight,
                    f"{metric_prefix}val/loss": val_metrics["loss"],
                    f"{metric_prefix}val/propensity_loss": val_metrics["propensity_loss"],
                    f"{metric_prefix}val/outcome_loss": val_metrics["outcome_loss"],
                    f"{metric_prefix}val/propensity_accuracy": val_metrics["propensity_accuracy"],
                    f"{metric_prefix}val/propensity_auc": val_metrics["propensity_auc"],
                    f"{metric_prefix}val/outcome_accuracy": val_metrics["outcome_accuracy"],
                    f"{metric_prefix}val/outcome_auc": val_metrics["outcome_auc"],
                    f"{metric_prefix}optimizer/transformer_lr": get_optimizer_group_lr(optimizer, "transformer"),
                    f"{metric_prefix}optimizer/head_lr": get_optimizer_group_lr(optimizer, "head"),
                },
                step=log_step_offset + epoch,
            )

        val_selection_loss = val_metrics["outcome_loss"] + val_metrics["propensity_loss"]
        if val_selection_loss < best_val_selection_loss:
            best_val_selection_loss = val_selection_loss
            best_metrics = {
                "train_loss": train_total_loss,
                "train_propensity_loss": train_propensity_loss,
                "train_outcome_loss": train_outcome_loss,
                "outcome_loss_weight": outcome_loss_weight,
                "selection_metric": "outcome_loss_plus_propensity_loss",
                "selection_loss": val_selection_loss,
                **val_metrics,
                "best_epoch": epoch,
            }
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None or best_metrics is None:
        raise RuntimeError("Training did not produce a valid model state.")

    model.load_state_dict(best_state)
    return model, best_metrics


def predict_outputs(
    model: SingleHeadTreatmentEffectModel,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, float | int]] = []

    with torch.no_grad():
        for batch in loader:
            tokens = batch["tokens"].to(device)
            ages = batch["ages"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            treatment = batch["treatment"].to(device)

            outputs = model(tokens=tokens, ages=ages, attention_mask=attention_mask)
            propensity_probs = torch.sigmoid(outputs["propensity_logits"]).cpu().tolist()
            y1_probs = torch.sigmoid(outputs["treated_outcome_logits"]).cpu().tolist()
            y0_probs = torch.sigmoid(outputs["comparator_outcome_logits"]).cpu().tolist()
            factual_probs = torch.sigmoid(
                select_factual_outcome_logits(
                    treated_logits=outputs["treated_outcome_logits"],
                    comparator_logits=outputs["comparator_outcome_logits"],
                    treatment=treatment,
                )
            ).cpu().tolist()

            for row_index, person_id, treatment_value, propensity_prob, factual_prob, y0_prob, y1_prob in zip(
                batch["row_index"].tolist(),
                batch["person_id"].tolist(),
                batch["treatment"].tolist(),
                propensity_probs,
                factual_probs,
                y0_probs,
                y1_probs,
            ):
                rows.append(
                    {
                        "row_index": int(row_index),
                        "person_id": int(person_id),
                        "treatment": int(treatment_value),
                        "propensity": float(propensity_prob),
                        "outcome_probability_observed_treatment": float(factual_prob),
                        "outcome_probability_treatment_0": float(y0_prob),
                        "outcome_probability_treatment_1": float(y1_prob),
                        "individual_treatment_effect": float(y1_prob - y0_prob),
                    }
                )

    return pd.DataFrame(rows).sort_values("row_index").reset_index(drop=True)


def stratified_kfold_indices(labels: Sequence[int], num_folds: int, seed: int) -> list[list[int]]:
    if num_folds < 2:
        raise ValueError("--num-folds must be at least 2.")
    by_class: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        by_class.setdefault(int(label), []).append(idx)

    folds = [[] for _ in range(num_folds)]
    rng = random.Random(seed)
    for class_indices in by_class.values():
        shuffled = class_indices[:]
        rng.shuffle(shuffled)
        for offset, idx in enumerate(shuffled):
            folds[offset % num_folds].append(idx)

    for fold in folds:
        fold.sort()
    if any(len(fold) == 0 for fold in folds):
        raise ValueError("At least one cross-fitting fold is empty. Reduce --num-folds or increase dataset size.")
    return folds


def stratified_train_val_split(indices: Sequence[int], labels: Sequence[int], val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    if not 0.0 < val_frac < 1.0:
        raise ValueError("--fold-val-frac must be between 0 and 1.")
    subset_by_class: dict[int, list[int]] = {}
    for idx in indices:
        subset_by_class.setdefault(int(labels[idx]), []).append(int(idx))

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for class_indices in subset_by_class.values():
        shuffled = class_indices[:]
        rng.shuffle(shuffled)
        n_total = len(shuffled)
        if n_total == 1:
            train_indices.extend(shuffled)
            continue
        n_val = max(1, int(round(n_total * val_frac)))
        if n_val >= n_total:
            n_val = n_total - 1
        val_indices.extend(shuffled[:n_val])
        train_indices.extend(shuffled[n_val:])

    train_indices.sort()
    val_indices.sort()
    if not train_indices or not val_indices:
        raise ValueError("Fold train/validation split is empty. Reduce --fold-val-frac or dataset stratification granularity.")
    return train_indices, val_indices


def build_pseudo_outcomes(predictions_df: pd.DataFrame, clip_value: float) -> pd.DataFrame:
    if not 0.0 < clip_value < 0.5:
        raise ValueError("--propensity-clip must be between 0 and 0.5.")

    output_df = predictions_df.copy()
    output_df["propensity_clipped"] = output_df["propensity"].clip(clip_value, 1.0 - clip_value)

    treatment = output_df["treatment"].to_numpy(dtype=np.float64)
    outcome = output_df["outcome"].to_numpy(dtype=np.float64)
    propensity = output_df["propensity_clipped"].to_numpy(dtype=np.float64)
    mu0 = output_df["outcome_probability_treatment_0"].to_numpy(dtype=np.float64)
    mu1 = output_df["outcome_probability_treatment_1"].to_numpy(dtype=np.float64)

    output_df["ipw_pseudo_outcome"] = (
        (treatment * outcome / propensity)
        - (((1.0 - treatment) * outcome) / (1.0 - propensity))
    )
    output_df["aipw_pseudo_outcome"] = (
        (mu1 - mu0)
        + (treatment * (outcome - mu1) / propensity)
        - (((1.0 - treatment) * (outcome - mu0)) / (1.0 - propensity))
    )
    output_df["dr_pseudo_outcome"] = output_df["aipw_pseudo_outcome"]
    return output_df


def dataset_rows_for_merge(samples_df: pd.DataFrame) -> pd.DataFrame:
    dataset_rows = samples_df.reset_index(drop=True).copy()
    if "_dataset_row_index" in dataset_rows.columns:
        dataset_rows = dataset_rows.drop(columns=["_dataset_row_index"])
    dataset_rows.insert(0, "_dataset_row_index", np.arange(len(dataset_rows), dtype=np.int64))
    return dataset_rows


def ensure_row_index_column(samples_df: pd.DataFrame) -> pd.DataFrame:
    output_df = samples_df.copy()
    if "row_index" not in output_df.columns:
        output_df.insert(0, "row_index", np.arange(len(output_df), dtype=np.int64))
    return output_df


def subset_to_training_partition(samples_df: pd.DataFrame, split_indices_path: Path | None) -> tuple[pd.DataFrame, dict[str, object] | None]:
    if split_indices_path is None:
        return samples_df.reset_index(drop=True).copy(), None

    split_payload = json.loads(split_indices_path.read_text(encoding="utf-8"))
    if "train" not in split_payload:
        raise KeyError(f"Split file must include a 'train' key: {split_indices_path}")

    train_indices = [int(idx) for idx in split_payload["train"]]
    subset_df = samples_df.iloc[train_indices].reset_index(drop=True).copy()
    split_summary = {
        "split_indices_path": str(split_indices_path.resolve()),
        "train_size": int(len(train_indices)),
        "test_size": int(len(split_payload.get("test", []))),
    }
    return subset_df, split_summary


def summarize_crossfit_predictions(predictions_df: pd.DataFrame) -> dict[str, float]:
    summary = {
        "num_rows": int(len(predictions_df)),
        "treated_fraction": float(predictions_df["treatment"].mean()),
        "outcome_fraction": float(predictions_df["outcome"].mean()),
        "mean_propensity": float(predictions_df["propensity"].mean()),
        "mean_outcome_probability_treatment_0": float(predictions_df["outcome_probability_treatment_0"].mean()),
        "mean_outcome_probability_treatment_1": float(predictions_df["outcome_probability_treatment_1"].mean()),
        "mean_individual_treatment_effect": float(predictions_df["individual_treatment_effect"].mean()),
    }
    if {"ipw_pseudo_outcome", "aipw_pseudo_outcome"}.issubset(predictions_df.columns):
        summary.update({
            "ipw_ate": float(predictions_df["ipw_pseudo_outcome"].mean()),
            "aipw_ate": float(predictions_df["aipw_pseudo_outcome"].mean()),
            "mean_abs_ipw_pseudo_outcome": float(predictions_df["ipw_pseudo_outcome"].abs().mean()),
            "mean_abs_aipw_pseudo_outcome": float(predictions_df["aipw_pseudo_outcome"].abs().mean()),
        })
    return summary


def build_fold_metric_record(
    fold_idx: int,
    fold_holdout_size: int,
    fold_train_size: int,
    fold_val_size: int,
    best_metrics: dict[str, float],
    holdout_metrics: dict[str, float],
) -> dict[str, float | int]:
    row: dict[str, float | int] = {
        "fold": int(fold_idx),
        "holdout_size": int(fold_holdout_size),
        "train_size": int(fold_train_size),
        "val_size": int(fold_val_size),
    }
    for key, value in best_metrics.items():
        row[f"best_val/{key}"] = float(value) if isinstance(value, (int, float)) else value
    for key, value in holdout_metrics.items():
        row[f"holdout/{key}"] = float(value)
    return row


def save_fold_checkpoint(
    output_dir: Path,
    fold_idx: int,
    model: SingleHeadTreatmentEffectModel,
    args: argparse.Namespace,
    best_metrics: dict[str, float],
    holdout_metrics: dict[str, float],
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    holdout_indices: Sequence[int],
) -> None:
    fold_dir = output_dir / "folds" / f"fold_{fold_idx:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_val_metrics": best_metrics,
            "holdout_metrics": holdout_metrics,
            "args": serialize_args(args),
            "train_indices": list(map(int, train_indices)),
            "val_indices": list(map(int, val_indices)),
            "holdout_indices": list(map(int, holdout_indices)),
        },
        fold_dir / "nuisance_model.pt",
    )


def log_crossfit_summary_to_wandb(
    wandb_run: object | None,
    fold_metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    summary_metrics: dict[str, float],
) -> None:
    if wandb_run is None:
        return
    payload: dict[str, object] = {f"crossfit/{key}": value for key, value in summary_metrics.items()}
    payload["crossfit/folds"] = int(fold_metrics_df.shape[0])
    if "holdout/loss" in fold_metrics_df.columns:
        payload["crossfit/mean_holdout_loss"] = float(fold_metrics_df["holdout/loss"].mean())
    if "holdout/propensity_auc" in fold_metrics_df.columns:
        payload["crossfit/mean_holdout_propensity_auc"] = float(fold_metrics_df["holdout/propensity_auc"].mean())
    if "holdout/outcome_auc" in fold_metrics_df.columns:
        payload["crossfit/mean_holdout_outcome_auc"] = float(fold_metrics_df["holdout/outcome_auc"].mean())

    if wandb is not None:
        try:
            payload["crossfit/propensity_hist"] = wandb.Histogram(predictions_df["propensity"].to_numpy(dtype=np.float64))
            if {"ipw_pseudo_outcome", "aipw_pseudo_outcome"}.issubset(predictions_df.columns):
                payload["crossfit/ipw_pseudo_outcome_hist"] = wandb.Histogram(
                    predictions_df["ipw_pseudo_outcome"].to_numpy(dtype=np.float64)
                )
                payload["crossfit/aipw_pseudo_outcome_hist"] = wandb.Histogram(
                    predictions_df["aipw_pseudo_outcome"].to_numpy(dtype=np.float64)
                )
        except Exception:
            pass

    wandb.log(payload)


def finish_active_wandb_run() -> None:
    if wandb is None or wandb.run is None:
        return
    try:
        wandb.finish()
    except Exception:
        pass


def build_transformer_encoder(args: argparse.Namespace, vocab, device: torch.device) -> Transformer:
    if args.random_init_transformer:
        return Transformer(
            vocab_size=len(vocab),
            d_model=int(args.d_model),
            nhead=int(args.nhead),
            num_layers=int(args.num_layers),
            ff_mult=int(args.ff_mult),
            dropout=float(args.dropout),
            pad_idx=vocab.pad_idx,
        ).to(device)

    if args.transformer_checkpoint is None:
        raise ValueError("--transformer-checkpoint is required unless --random-init-transformer is set.")
    return load_transformer(
        checkpoint_path=args.transformer_checkpoint,
        vocab=vocab,
        nhead=args.nhead,
        dropout=args.dropout,
        device=device,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    vocab = load_vocab(args.vocab_path)
    samples_df = load_samples_dataframe(args=args, vocab=vocab)
    samples_df = ensure_row_index_column(samples_df)
    samples_df, split_summary = subset_to_training_partition(samples_df, args.split_indices_path)

    if samples_df.empty:
        raise ValueError("The constructed treatment-effect dataset is empty.")
    required_cols = {"treatment", "outcome"}
    missing_cols = sorted(required_cols - set(samples_df.columns))
    if missing_cols:
        raise KeyError(f"Expected the samples dataframe to include columns: {missing_cols}")
    if args.outcome_loss_weight_start < 0 or args.outcome_loss_weight_end < 0:
        raise ValueError("Outcome loss weights must be non-negative.")

    split_labels = (samples_df["treatment"].astype(int) * 2 + samples_df["outcome"].astype(int)).tolist()
    fold_indices = stratified_kfold_indices(labels=split_labels, num_folds=args.num_folds, seed=args.seed)

    base_dataset = TreatmentEffectDataset(samples_df)
    indexed_dataset = IndexedOutcomeDataset(base_dataset)
    transformer_lr = args.lr if args.transformer_lr is None else args.transformer_lr
    head_lr = args.lr if args.head_lr is None else args.head_lr
    max_transformer_lr = transformer_lr if args.max_transformer_lr is None else args.max_transformer_lr
    min_transformer_lr = max_transformer_lr if args.min_transformer_lr is None else args.min_transformer_lr
    max_head_lr = head_lr if args.max_head_lr is None else args.max_head_lr
    min_head_lr = max_head_lr if args.min_head_lr is None else args.min_head_lr
    outcome_criterion = nn.BCEWithLogitsLoss(reduction="none")
    wandb_run = configure_wandb_run(
        init_wandb(args),
        args,
        task_name="nuisance-crossfit",
        define_default_epoch_metrics=True,
        define_global_step_metrics=True,
    )

    if wandb_run is not None:
        try:
            wandb_run.config.update(
                {
                    "crossfit/num_folds": int(args.num_folds),
                    "crossfit/fold_val_frac": float(args.fold_val_frac),
                    "crossfit/propensity_clip": float(args.propensity_clip),
                    "display_run_name": build_wandb_run_name(args=args, task_name="nuisance-crossfit"),
                },
                allow_val_change=True,
            )
        except Exception:
            pass

    fold_metric_rows: list[dict[str, float | int]] = []
    oof_prediction_frames: list[pd.DataFrame] = []
    serialized_folds: list[dict[str, object]] = []

    for fold_idx, holdout_indices in enumerate(fold_indices):
        holdout_index_set = set(holdout_indices)
        train_pool_indices = sorted(idx for idx in range(len(indexed_dataset)) if idx not in holdout_index_set)
        train_indices, val_indices = stratified_train_val_split(
            indices=train_pool_indices,
            labels=split_labels,
            val_frac=args.fold_val_frac,
            seed=args.seed + fold_idx,
        )

        print(
            f"Cross-fit fold {fold_idx + 1}/{args.num_folds} | train={len(train_indices)} | "
            f"val={len(val_indices)} | holdout={len(holdout_indices)}",
            flush=True,
        )

        train_loader = make_loader(indexed_dataset, train_indices, args.batch_size, True, args.num_workers)
        val_loader = make_loader(indexed_dataset, val_indices, args.batch_size, False, args.num_workers)
        holdout_loader = make_loader(indexed_dataset, holdout_indices, args.batch_size, False, args.num_workers)

        encoder = build_transformer_encoder(args=args, vocab=vocab, device=device)
        model = SingleHeadTreatmentEffectModel(
            encoder=encoder,
            freeze_encoder=args.freeze_transformer,
            head_hidden_dim=args.head_hidden_dim,
            head_dropout=args.head_dropout,
        ).to(device)

        propensity_criterion = build_propensity_criterion(
            train_loader=train_loader,
            device=device,
            disable_pos_weight=args.disable_propensity_pos_weight,
        )
        model, best_metrics = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            transformer_lr=transformer_lr,
            head_lr=head_lr,
            weight_decay=args.weight_decay,
            min_transformer_lr=min_transformer_lr,
            max_transformer_lr=max_transformer_lr,
            min_head_lr=min_head_lr,
            max_head_lr=max_head_lr,
            propensity_criterion=propensity_criterion,
            outcome_criterion=outcome_criterion,
            device=device,
            outcome_loss_weight_start=args.outcome_loss_weight_start,
            outcome_loss_weight_end=args.outcome_loss_weight_end,
            grad_clip_norm=args.grad_clip_norm,
            wandb_run=wandb_run,
            log_namespace=f"fold_{fold_idx}",
            log_step_offset=fold_idx * args.epochs,
            fold_index=fold_idx,
        )

        holdout_metrics = evaluate(
            model=model,
            loader=holdout_loader,
            propensity_criterion=propensity_criterion,
            outcome_criterion=outcome_criterion,
            device=device,
            outcome_loss_weight=float(best_metrics["outcome_loss_weight"]),
        )
        holdout_predictions = predict_outputs(model=model, loader=holdout_loader, device=device)
        holdout_predictions["crossfit_fold"] = int(fold_idx)
        oof_prediction_frames.append(holdout_predictions)

        fold_metric_rows.append(
            build_fold_metric_record(
                fold_idx=fold_idx,
                fold_holdout_size=len(holdout_indices),
                fold_train_size=len(train_indices),
                fold_val_size=len(val_indices),
                best_metrics=best_metrics,
                holdout_metrics=holdout_metrics,
            )
        )
        serialized_folds.append(
            {
                "fold": int(fold_idx),
                "train": list(map(int, train_indices)),
                "val": list(map(int, val_indices)),
                "holdout": list(map(int, holdout_indices)),
            }
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "trainer/global_step": (fold_idx + 1) * args.epochs,
                    "trainer/fold_index": int(fold_idx),
                    "summary/best_epoch": int(best_metrics["best_epoch"]),
                    "summary/holdout_loss": float(holdout_metrics["loss"]),
                    "summary/holdout_propensity_auc": float(holdout_metrics["propensity_auc"]),
                    "summary/holdout_outcome_auc": float(holdout_metrics["outcome_auc"]),
                    f"fold_{fold_idx}/summary/best_epoch": int(best_metrics["best_epoch"]),
                    f"fold_{fold_idx}/summary/holdout_loss": float(holdout_metrics["loss"]),
                    f"fold_{fold_idx}/summary/holdout_propensity_auc": float(holdout_metrics["propensity_auc"]),
                    f"fold_{fold_idx}/summary/holdout_outcome_auc": float(holdout_metrics["outcome_auc"]),
                },
                step=(fold_idx + 1) * args.epochs,
            )

        if args.save_fold_checkpoints:
            save_fold_checkpoint(
                output_dir=output_dir,
                fold_idx=fold_idx,
                model=model,
                args=args,
                best_metrics=best_metrics,
                holdout_metrics=holdout_metrics,
                train_indices=train_indices,
                val_indices=val_indices,
                holdout_indices=holdout_indices,
            )

    oof_predictions_df = pd.concat(oof_prediction_frames, ignore_index=True).sort_values("row_index").reset_index(drop=True)
    if oof_predictions_df["row_index"].nunique() != len(samples_df):
        raise RuntimeError("Cross-fitting did not produce exactly one out-of-fold prediction per sample.")

    merged_predictions_df = oof_predictions_df.rename(columns={"row_index": "_dataset_row_index"}).merge(
        dataset_rows_for_merge(samples_df),
        on=["_dataset_row_index", "person_id", "treatment"],
        how="left",
        validate="one_to_one",
    )
    if "row_index" not in merged_predictions_df.columns:
        merged_predictions_df = merged_predictions_df.rename(columns={"_dataset_row_index": "row_index"})
    if args.skip_pseudo_outcomes:
        merged_predictions_df["propensity_clipped"] = merged_predictions_df["propensity"].clip(
            args.propensity_clip,
            1.0 - args.propensity_clip,
        )
    else:
        merged_predictions_df = build_pseudo_outcomes(merged_predictions_df, clip_value=args.propensity_clip)

    fold_metrics_df = pd.DataFrame(fold_metric_rows).sort_values("fold").reset_index(drop=True)
    summary_metrics = summarize_crossfit_predictions(merged_predictions_df)

    merged_predictions_path = output_dir / "patient_nuisance_predictions.parquet"
    fold_metrics_path = output_dir / "fold_metrics.parquet"
    metrics_path = output_dir / "metrics.json"
    fold_indices_path = output_dir / "crossfit_fold_indices.json"

    merged_predictions_df.to_parquet(merged_predictions_path, index=False)
    fold_metrics_df.to_parquet(fold_metrics_path, index=False)
    fold_indices_path.write_text(json.dumps(serialized_folds, indent=2), encoding="utf-8")
    metrics_path.write_text(
        json.dumps(
            {
                "summary": summary_metrics,
                "fold_metrics": fold_metric_rows,
                "train_split": split_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    log_crossfit_summary_to_wandb(
        wandb_run=wandb_run,
        fold_metrics_df=fold_metrics_df,
        predictions_df=merged_predictions_df,
        summary_metrics=summary_metrics,
    )

    if wandb is not None and wandb.run is not None:
        try:
            wandb.save(str(fold_metrics_path))
            wandb.save(str(metrics_path))
        except Exception:
            pass
        finally:
            finish_active_wandb_run()

    print(f"Saved nuisance predictions to: {merged_predictions_path}")
    print(f"Saved fold metrics to: {fold_metrics_path}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        finish_active_wandb_run()
