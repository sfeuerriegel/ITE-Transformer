from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
import sys

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
    SplitIndices,
    build_wandb_run_name,
    configure_wandb_run,
    init_wandb,
    load_transformer,
    load_vocab,
    serialize_args,
    set_seed,
    stratified_split_indices,
)
from src.model import Transformer


class IndexedPseudoOutcomeDataset(Dataset):
    def __init__(self, base_dataset: TreatmentEffectDataset, pseudo_targets: pd.Series):
        self.base_dataset = base_dataset
        targets = pd.to_numeric(pseudo_targets, errors="coerce")
        if targets.isna().any():
            raise ValueError("Pseudo-outcome target column contains missing or non-numeric values.")
        self.targets = targets.astype("float32").tolist()

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict[str, object]:
        item = self.base_dataset[idx]
        item["row_index"] = int(idx)
        item["pseudo_outcome"] = torch.tensor(self.targets[int(idx)], dtype=torch.float32)
        return item


def collate_pseudo_outcome_batch(batch: list[dict[str, object]]) -> dict[str, torch.Tensor]:
    tokens = [item["history_tokens"] for item in batch]
    ages = [item["history_ages"] for item in batch]
    tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=0)
    ages_padded = pad_sequence(ages, batch_first=True, padding_value=0)

    return {
               'tokens': tokens_padded,
               'ages': ages_padded,
               'attention_mask': (tokens_padded != 0).long(),
               'pseudo_outcome': torch.stack([item['pseudo_outcome'] for item in batch]).to(torch.float32),
               'row_index': torch.tensor([int(item['row_index']) for item in batch], dtype=torch.long),
               'person_id': torch.tensor([int(item['person_id']) if item['person_id'] is not None else -1 for item in batch], dtype=torch.long),
               'treatment': torch.stack([item['treatment'] for item in batch]).to(torch.float32),
               'outcome': torch.stack([item['outcome'] for item in batch]).to(torch.float32),
           }


def make_loader(
    dataset: IndexedPseudoOutcomeDataset,
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
        collate_fn=collate_pseudo_outcome_batch,
    )


class PseudoOutcomeCateModel(nn.Module):
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
        self.head = nn.Sequential(
            nn.Linear(self.head_input_dim, self.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(self.head_hidden_dim, 1),
        )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

    def train(self, mode: bool = True) -> "PseudoOutcomeCateModel":
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

    def forward(
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
        pooled = self._last_valid_hidden(encoder_out["hidden"], attention_mask)
        return self.head(pooled).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a CATE learner against cross-fitted pseudo-outcomes.",
    )
    parser.add_argument("--samples-path", type=Path, required=True, help="Parquet produced by train_nuisance_models.py.")

    parser.add_argument(
        "--pseudo-outcome-column",
        type=str,
        default="aipw_pseudo_outcome",
        choices=["ipw_pseudo_outcome", "aipw_pseudo_outcome", "dr_pseudo_outcome"],
    )
    parser.add_argument(
        "--split-indices-path",
        type=Path,
        default=None,
        help="Optional JSON path containing a top-level {'train': [...], 'test': [...]} split.",
    )
    parser.add_argument("--transformer-checkpoint", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--freeze-transformer", action="store_true")
    parser.add_argument("--head-hidden-dim", type=int, default=None)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--transformer-lr", type=float, default=None)
    parser.add_argument("--head-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)

    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "huber"])
    parser.add_argument("--wandb-project", type=str, default="CATE learner")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def compute_regression_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    per_example_loss = criterion(preds, targets)

    return per_example_loss.mean()


def regression_metrics(
    preds: list[float],
    targets: list[float],
) -> dict[str, float]:
    preds_np = np.asarray(preds, dtype=np.float64)
    targets_np = np.asarray(targets, dtype=np.float64)
    errors = preds_np - targets_np
    mse = float(np.mean(np.square(errors)))
    mae = float(np.mean(np.abs(errors)))
    corr = float(np.corrcoef(preds_np, targets_np)[0, 1]) if len(preds_np) > 1 else float("nan")
    return {
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mae": mae,
        "corr": corr,
        "prediction_mean": float(preds_np.mean()),
        "target_mean": float(targets_np.mean()),
    }


def evaluate(
    model: PseudoOutcomeCateModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    sample_count = 0.0
    preds: list[float] = []
    targets: list[float] = []

    with torch.no_grad():
        for batch in loader:
            output = model(
                tokens=batch["tokens"].to(device),
                ages=batch["ages"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            target = batch["pseudo_outcome"].to(device)

            loss = compute_regression_loss(
                preds=output,
                targets=target,
                criterion=criterion,
            )
            batch_size = float(target.numel())
            loss_sum += float(loss.item() * batch_size)
            sample_count += batch_size
            preds.extend(output.cpu().tolist())
            targets.extend(target.cpu().tolist())

    metrics = regression_metrics(preds=preds, targets=targets)
    metrics["loss"] = loss_sum / max(sample_count, 1e-12)
    return metrics


def fit(
    model: PseudoOutcomeCateModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    transformer_lr: float,
    head_lr: float,
    weight_decay: float,
    criterion: nn.Module,
    device: torch.device,
    grad_clip_norm: float | None,
    wandb_run: object | None,
) -> tuple[PseudoOutcomeCateModel, dict[str, float]]:
    optimizer_param_groups: list[dict[str, object]] = []
    encoder_params = [param for param in model.encoder.parameters() if param.requires_grad]
    head_params = [param for param in model.head.parameters() if param.requires_grad]
    if encoder_params:
        optimizer_param_groups.append({"params": encoder_params, "lr": transformer_lr, "weight_decay": weight_decay})
    if head_params:
        optimizer_param_groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
    optimizer = torch.optim.AdamW(optimizer_param_groups)

    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    best_val_loss = math.inf
    num_train_batches = len(train_loader)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss_sum = 0.0
        running_sample_count = 0.0
        print(
            f"Epoch {epoch:03d} | training on {num_train_batches} batches ({len(train_loader.dataset)} samples)",
            flush=True,
        )
        for batch in train_loader:
            preds = model(
                tokens=batch["tokens"].to(device),
                ages=batch["ages"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            targets = batch["pseudo_outcome"].to(device)


            optimizer.zero_grad(set_to_none=True)
            loss = compute_regression_loss(
                preds=preds,
                targets=targets,
                criterion=criterion,
            )
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            batch_size = float(targets.numel())
            running_loss_sum += float(loss.item() * batch_size)
            running_sample_count += batch_size

        train_loss = running_loss_sum / max(running_sample_count, 1e-12)
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | "
            f"val_rmse={val_metrics['rmse']:.4f} | val_mae={val_metrics['mae']:.4f} | val_corr={val_metrics['corr']:.4f}",
            flush=True,
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "trainer/global_step": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_metrics["loss"],
                    "val/rmse": val_metrics["rmse"],
                    "val/mae": val_metrics["mae"],
                    "val/corr": val_metrics["corr"],
                    "val/prediction_mean": val_metrics["prediction_mean"],
                    "val/target_mean": val_metrics["target_mean"],
                },
                step=epoch,
            )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_metrics = {"train_loss": train_loss, **val_metrics, "best_epoch": epoch}
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None or best_metrics is None:
        raise RuntimeError("Training did not produce a valid model state.")
    model.load_state_dict(best_state)
    return model, best_metrics


def predict(
    model: PseudoOutcomeCateModel,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, float | int]] = []
    with torch.no_grad():
        for batch in loader:
            preds = model(
                tokens=batch["tokens"].to(device),
                ages=batch["ages"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            for row_index, person_id, treatment, outcome, target, pred in zip(
                batch["row_index"].tolist(),
                batch["person_id"].tolist(),
                batch["treatment"].tolist(),
                batch["outcome"].tolist(),
                batch["pseudo_outcome"].tolist(),
                preds.cpu().tolist(),
            ):
                rows.append(
                    {
                        "row_index": int(row_index),
                        "person_id": int(person_id),
                        "treatment": int(treatment),
                        "outcome": int(outcome),
                        "pseudo_outcome": float(target),
                        "predicted_treatment_effect": float(pred),
                    }
                )
    return pd.DataFrame(rows).sort_values("row_index").reset_index(drop=True)


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


def stratified_train_val_split(indices: list[int], labels: list[int], val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    if not 0.0 < val_frac < 1.0:
        raise ValueError("--val-frac must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    by_class: dict[int, list[int]] = {}
    for idx in indices:
        by_class.setdefault(int(labels[idx]), []).append(int(idx))

    train_indices: list[int] = []
    val_indices: list[int] = []
    for class_indices in by_class.values():
        shuffled = list(class_indices)
        rng.shuffle(shuffled)
        n_total = len(shuffled)
        n_val = max(1, int(round(n_total * val_frac)))
        if n_val >= n_total:
            n_val = n_total - 1
        if n_val <= 0:
            raise ValueError("Validation split is empty for at least one class. Reduce --val-frac or adjust the split.")
        val_indices.extend(shuffled[:n_val])
        train_indices.extend(shuffled[n_val:])

    train_indices.sort()
    val_indices.sort()
    if not train_indices or not val_indices:
        raise ValueError("Training and validation splits must both be non-empty.")
    return train_indices, val_indices


def build_development_splits(
    samples_df: pd.DataFrame,
    split_indices_path: Path | None,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, SplitIndices, dict[str, object]]:
    if split_indices_path is None:
        labels = (samples_df["treatment"].astype(int) * 2 + samples_df["outcome"].astype(int)).tolist()
        splits = stratified_split_indices(labels=labels, train_frac=train_frac, val_frac=val_frac, seed=seed)
        development_df = samples_df.reset_index(drop=True).copy()
        return development_df, splits, {
            "split_indices_path": None,
            "development_size": int(len(development_df)),
            "train_size": len(splits.train),
            "val_size": len(splits.val),
            "test_size": len(splits.test),
        }

    split_payload = json.loads(split_indices_path.read_text(encoding="utf-8"))
    if "train" not in split_payload:
        raise KeyError(f"Split file must include a 'train' key: {split_indices_path}")

    train_row_indices = {int(idx) for idx in split_payload["train"]}
    development_df = samples_df.loc[samples_df["row_index"].astype(int).isin(train_row_indices)].reset_index(drop=True).copy()
    if development_df.empty:
        raise ValueError("The provided split file did not match any row_index values in the CATE samples dataframe.")
    labels = (development_df["treatment"].astype(int) * 2 + development_df["outcome"].astype(int)).tolist()
    train_indices, val_indices = stratified_train_val_split(
        indices=list(range(len(development_df))),
        labels=labels,
        val_frac=val_frac,
        seed=seed,
    )
    splits = SplitIndices(train=train_indices, val=val_indices, test=[])
    summary = {
        "split_indices_path": str(split_indices_path.resolve()),
        "development_size": int(len(development_df)),
        "heldout_test_size": int(len(split_payload.get("test", []))),
        "train_size": int(len(train_indices)),
        "val_size": int(len(val_indices)),
    }
    return development_df, splits, summary


def finish_active_wandb_run() -> None:
    if wandb is None or wandb.run is None:
        return
    try:
        wandb.finish()
    except Exception:
        pass


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples_df = pd.read_parquet(args.samples_path)

    samples_df = ensure_row_index_column(samples_df)
    if args.pseudo_outcome_column not in samples_df.columns:
        raise KeyError(f"Missing pseudo-outcome column: {args.pseudo_outcome_column}")

    vocab = load_vocab(args.vocab_path)
    device = torch.device(args.device)
    samples_df, splits, split_summary = build_development_splits(
        samples_df=samples_df,
        split_indices_path=args.split_indices_path,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    if not splits.val:
        raise ValueError("Validation split must be non-empty.")

    base_dataset = TreatmentEffectDataset(samples_df)
    dataset = IndexedPseudoOutcomeDataset(base_dataset=base_dataset, pseudo_targets=samples_df[args.pseudo_outcome_column])
    train_loader = make_loader(dataset, splits.train, args.batch_size, True, args.num_workers)
    val_loader = make_loader(dataset, splits.val, args.batch_size, False, args.num_workers)
    full_loader = make_loader(dataset, list(range(len(dataset))), args.batch_size, False, args.num_workers)

    encoder = load_transformer(
        checkpoint_path=args.transformer_checkpoint,
        vocab=vocab,
        nhead=args.nhead,
        dropout=args.dropout,
        device=device,
    )
    model = PseudoOutcomeCateModel(
        encoder=encoder,
        freeze_encoder=args.freeze_transformer,
        head_hidden_dim=args.head_hidden_dim,
        head_dropout=args.head_dropout,
    ).to(device)

    if args.loss == "mse":
        criterion: nn.Module = nn.MSELoss(reduction="none")
    else:
        criterion = nn.HuberLoss(reduction="none")

    wandb_run = configure_wandb_run(
        init_wandb(args),
        args,
        task_name="pseudo-outcome-cate",
        define_default_epoch_metrics=True,
        define_global_step_metrics=True,
    )
    if wandb_run is not None:
        try:
            wandb_run.config.update(
                {
                    "pseudo_outcome_column": args.pseudo_outcome_column,
                    "loss_name": args.loss,
                    "display_run_name": build_wandb_run_name(args=args, task_name="pseudo-outcome-cate"),
                },
                allow_val_change=True,
            )
        except Exception:
            pass

    transformer_lr = args.lr if args.transformer_lr is None else args.transformer_lr
    head_lr = args.lr if args.head_lr is None else args.head_lr
    model, best_metrics = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        transformer_lr=transformer_lr,
        head_lr=head_lr,
        weight_decay=args.weight_decay,
        criterion=criterion,
        device=device,
        grad_clip_norm=args.grad_clip_norm,
        wandb_run=wandb_run,
    )

    development_metrics = evaluate(
        model=model,
        loader=full_loader,
        criterion=criterion,
        device=device,
    )
    predictions_df = predict(model=model, loader=full_loader, device=device).rename(
        columns={"row_index": "_dataset_row_index"}
    ).merge(
        dataset_rows_for_merge(samples_df),
        on=["_dataset_row_index", "person_id", "treatment", "outcome"],
        how="left",
        validate="one_to_one",
    )
    if "row_index" not in predictions_df.columns:
        predictions_df = predictions_df.rename(columns={"_dataset_row_index": "row_index"})

    checkpoint_path = output_dir / "pseudo_outcome_cate_model.pt"
    predictions_path = output_dir / "patient_cate_predictions.parquet"
    metrics_path = output_dir / "metrics.json"
    split_path = output_dir / "split_indices.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_val_metrics": best_metrics,
            "development_metrics": development_metrics,
            "args": serialize_args(args),
        },
        checkpoint_path,
    )
    predictions_df.to_parquet(predictions_path, index=False)
    split_path.write_text(
        json.dumps(
            {
                "train": splits.train,
                "val": splits.val,
                "test": splits.test,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps(
            {
                "best_validation": best_metrics,
                "development_population": development_metrics,
                "data_split": split_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if wandb is not None and wandb.run is not None:
        try:
            wandb_run.log(
                {
                    "epoch": int(best_metrics["best_epoch"]),
                    "trainer/global_step": int(best_metrics["best_epoch"]),
                    "development/loss": development_metrics["loss"],
                    "development/rmse": development_metrics["rmse"],
                    "development/mae": development_metrics["mae"],
                    "development/corr": development_metrics["corr"],
                    "development/prediction_mean": development_metrics["prediction_mean"],
                    "development/target_mean": development_metrics["target_mean"],
                },
                step=int(best_metrics["best_epoch"]),
            )
            wandb.save(str(checkpoint_path))
            wandb.save(str(metrics_path))
        except Exception:
            pass
        finally:
            finish_active_wandb_run()

    print(f"Saved model checkpoint to: {checkpoint_path}")
    print(f"Saved patient predictions to: {predictions_path}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        finish_active_wandb_run()
