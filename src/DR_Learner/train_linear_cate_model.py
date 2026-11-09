from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd

try:
    import wandb
except Exception:
    wandb = None

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.DR_Learner.random_forest import TopTokenFeatureizer
from src.DR_Learner.helpers import (
    build_wandb_run_name,
    configure_wandb_run,
    init_wandb,
    serialize_args,
    set_seed,
)
from src.DR_Learner.train_pseudo_outcome_cate_model import (
    build_development_splits,
    dataset_rows_for_merge,
    ensure_row_index_column,
    finish_active_wandb_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a linear CATE baseline against cross-fitted pseudo-outcomes.",
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)

    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "huber"])
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--top-k-tokens", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--wandb-project", type=str, default="cate-linear-baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def regression_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    *,
    loss_name: str,
    huber_delta: float,
) -> dict[str, float]:
    preds_np = np.asarray(preds, dtype=np.float64)
    targets_np = np.asarray(targets, dtype=np.float64)
    errors = preds_np - targets_np
    squared_error = np.square(errors)
    absolute_error = np.abs(errors)
    mse = squared_error.mean()
    mae = absolute_error.mean()
    if loss_name == "mse":
        loss = mse
    else:
        quadratic = np.minimum(absolute_error, huber_delta)
        linear = absolute_error - quadratic
        huber = 0.5 * np.square(quadratic) + huber_delta * linear
        loss = huber.mean()
    corr = float(np.corrcoef(preds_np, targets_np)[0, 1]) if len(preds_np) > 1 else float("nan")
    return {
        "loss": float(loss),
        "mse": float(mse),
        "rmse": float(math.sqrt(mse)),
        "mae": float(mae),
        "corr": corr,
        "prediction_mean": float(preds_np.mean()),
        "target_mean": float(targets_np.mean()),
    }


def build_feature_matrices(
    samples_df: pd.DataFrame,
    *,
    top_k_tokens: int,
    train_indices: list[int],
) -> tuple[TopTokenFeatureizer, np.ndarray]:
    featureizer = TopTokenFeatureizer(top_k_tokens).fit(
        samples_df.iloc[list(train_indices)]["history_tokens"].tolist()
    )
    features = featureizer.transform(samples_df, include_treatment=False)
    return featureizer, features


def predict_frame(samples_df: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    prediction_df = pd.DataFrame(
        {
            "_dataset_row_index": np.arange(len(samples_df), dtype=np.int64),
            "predicted_treatment_effect": np.asarray(predictions, dtype=np.float64),
        }
    )
    prediction_df = prediction_df.merge(
        dataset_rows_for_merge(samples_df),
        on="_dataset_row_index",
        how="left",
        validate="one_to_one",
    )
    if "row_index" not in prediction_df.columns:
        prediction_df = prediction_df.rename(columns={"_dataset_row_index": "row_index"})
    return prediction_df


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples_df = pd.read_parquet(args.samples_path)

    samples_df = ensure_row_index_column(samples_df)
    if args.pseudo_outcome_column not in samples_df.columns:
        raise KeyError(f"Missing pseudo-outcome column: {args.pseudo_outcome_column}")

    samples_df, splits, split_summary = build_development_splits(
        samples_df=samples_df,
        split_indices_path=args.split_indices_path,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    if not splits.val:
        raise ValueError("Validation split must be non-empty.")

    featureizer, features = build_feature_matrices(
        samples_df,
        top_k_tokens=args.top_k_tokens,
        train_indices=splits.train,
    )
    targets = pd.to_numeric(samples_df[args.pseudo_outcome_column], errors="coerce").to_numpy(dtype=np.float64)
    if np.isnan(targets).any():
        raise ValueError("Pseudo-outcome target column contains missing or non-numeric values.")


    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("regressor", Ridge(alpha=float(args.alpha), random_state=args.seed)),
        ]
    )

    X_train = features[splits.train]
    y_train = targets[splits.train]

    model.fit(X_train, y_train)

    X_val = features[splits.val]
    y_val = targets[splits.val]

    val_preds = model.predict(X_val)
    best_metrics = regression_metrics(
        val_preds,
        y_val,
        loss_name=args.loss,
        huber_delta=float(args.huber_delta),
    )
    best_metrics["best_epoch"] = 0

    full_preds = model.predict(features)
    development_metrics = regression_metrics(
        full_preds,
        targets,
        loss_name=args.loss,
        huber_delta=float(args.huber_delta),
    )

    predictions_df = predict_frame(samples_df, full_preds)

    checkpoint_path = output_dir / "linear_cate_model.pkl"
    predictions_path = output_dir / "patient_cate_predictions.parquet"
    metrics_path = output_dir / "metrics.json"
    split_path = output_dir / "split_indices.json"

    with checkpoint_path.open("wb") as f:
        pickle.dump(
            {
                "model": model,
                "featureizer": featureizer,
                "args": serialize_args(args),
            },
            f,
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

    wandb_run = configure_wandb_run(
        init_wandb(args),
        args,
        task_name="linear-cate-baseline",
        define_default_epoch_metrics=False,
        define_global_step_metrics=False,
    )
    if wandb_run is not None:
        try:
            wandb_run.config.update(
                {
                    "pseudo_outcome_column": args.pseudo_outcome_column,
                    "loss_name": args.loss,
                    "display_run_name": build_wandb_run_name(args=args, task_name="linear-cate-baseline"),
                },
                allow_val_change=True,
            )
            wandb_run.log(
                {
                    "validation/loss": best_metrics["loss"],
                    "validation/rmse": best_metrics["rmse"],
                    "validation/corr": best_metrics["corr"],
                    "development/loss": development_metrics["loss"],
                    "development/rmse": development_metrics["rmse"],
                    "development/corr": development_metrics["corr"],
                }
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
