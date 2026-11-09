from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.DR_Learner.helpers import (
    binary_auc,
    load_samples_dataframe,
    load_vocab,
    serialize_args,
    set_seed,
    stratified_split_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train random forest propensity and outcome baselines on the "
            "treatment-effect dataset."
        ),
    )

    data_group = parser.add_argument_group("Dataset inputs")
    data_group.add_argument(
        "--samples-path",
        type=Path,
        default=None,
        help="Optional prebuilt treatment-effect dataset (.pkl or .parquet).",
    )
    data_group.add_argument(
        "--treatment-pkl",
        type=Path,
        default=None,
        help="Pickle containing the treatment cohort.",
    )
    data_group.add_argument(
        "--comparator-pkl",
        type=Path,
        default=None,
        help="Pickle containing the comparator cohort.",
    )
    data_group.add_argument(
        "--conditions-pkl",
        type=Path,
        default=None,
        help="Pickle containing condition history.",
    )
    data_group.add_argument(
        "--persons-pkl",
        type=Path,
        default=None,
        help="Pickle containing person demographics.",
    )
    data_group.add_argument("--treatment-name", type=str, default="treatment")
    data_group.add_argument("--comparator-name", type=str, default="comparator")
    data_group.add_argument("--min-history-events", type=int, default=0)
    data_group.add_argument("--max-history-events", type=int, default=None)


    vocab_group = parser.add_argument_group("Vocabulary")
    vocab_group.add_argument(
        "--vocab-path",
        type=Path,
        required=True,
        help="Path to vocab.json used to build the dataset.",
    )

    feature_group = parser.add_argument_group("Featureization")
    feature_group.add_argument(
        "--top-k-tokens",
        type=int,
        default=512,
        help=(
            "Number of most frequent history tokens to use as binary "
            "bag-of-codes features."
        ),
    )

    rf_group = parser.add_argument_group("Random forest")
    rf_group.add_argument("--n-estimators", type=int, default=300)
    rf_group.add_argument(
        "--max-depth-options",
        type=int,
        nargs="+",
        default=[8, 16, 24],
    )
    rf_group.add_argument(
        "--min-samples-leaf-options",
        type=int,
        nargs="+",
        default=[1, 10, 25],
    )
    rf_group.add_argument("--max-features", type=str, default="sqrt")
    rf_group.add_argument("--n-jobs", type=int, default=-1)

    train_group = parser.add_argument_group("Splits")
    train_group.add_argument("--output-dir", type=Path, required=True)
    train_group.add_argument("--train-frac", type=float, default=0.7)
    train_group.add_argument("--val-frac", type=float, default=0.15)
    train_group.add_argument("--seed", type=int, default=42)
    train_group.add_argument(
        "--disable-propensity-pos-weight",
        action="store_true",
        help=(
            "Disable the positive-class reweighting used by the transformer "
            "propensity baseline."
        ),
    )

    return parser.parse_args()


def clip_probs(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0 - eps)


def binary_log_loss(
    labels: np.ndarray,
    probs: np.ndarray,
    pos_weight: float | None = None,
) -> float:
    y = np.asarray(labels, dtype=np.float64)
    p = clip_probs(probs)


    if pos_weight is None:
        losses = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    else:
        losses = -(pos_weight * y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float(losses.mean())


def accuracy_from_probs(labels: np.ndarray, probs: np.ndarray) -> float:
    preds = (np.asarray(probs) >= 0.5).astype(np.int64)
    y = np.asarray(labels, dtype=np.int64)
    return float(np.mean(preds == y)) if len(y) else float("nan")


class TopTokenFeatureizer:
    """Fixed-width tabular features from longitudinal histories."""

    def __init__(self, top_k_tokens: int):
        self.top_k_tokens = int(top_k_tokens)
        self.selected_tokens_: list[int] = []
        self.token_to_col_: dict[int, int] = {}

    def fit(self, histories: list[list[int]]) -> "TopTokenFeatureizer":
        counts: dict[int, int] = {}
        for tokens in histories:
            for tok in set(int(tok) for tok in tokens[1:]):
                counts[tok] = counts.get(tok, 0) + 1

        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        self.selected_tokens_ = [tok for tok, _ in ranked[: self.top_k_tokens]]
        self.token_to_col_ = {
            tok: idx for idx, tok in enumerate(self.selected_tokens_)
        }
        return self

    def transform(self, df: pd.DataFrame, include_treatment: bool = False) -> np.ndarray:
        n_rows = len(df)
        base_width = 7 + len(self.selected_tokens_)
        width = base_width + int(include_treatment)
        features = np.zeros((n_rows, width), dtype=np.float32)

        for row_idx, row in enumerate(df.itertuples(index=False)):
            tokens = [int(tok) for tok in row.history_tokens]
            ages = [float(age) for age in row.history_ages]
            history_tokens = tokens[1:]
            history_ages = ages[1:]

            features[row_idx, 0] = float(tokens[0]) if tokens else 0.0
            features[row_idx, 1] = float(len(history_tokens))
            features[row_idx, 2] = float(len(set(history_tokens)))
            features[row_idx, 3] = float(history_ages[-1]) if history_ages else 0.0
            features[row_idx, 4] = float(np.mean(history_ages)) if history_ages else 0.0
            features[row_idx, 5] = float(np.std(history_ages)) if history_ages else 0.0
            features[row_idx, 6] = (
                float(np.mean(np.diff(history_ages))) if len(history_ages) >= 2 else 0.0
            )

            for tok in set(history_tokens):
                col_idx = self.token_to_col_.get(tok)
                if col_idx is not None:
                    features[row_idx, 7 + col_idx] = 1.0

            if include_treatment:
                features[row_idx, -1] = float(row.treatment)

        return features


def subset_df(df: pd.DataFrame, indices: list[int]) -> pd.DataFrame:
    return df.iloc[indices].reset_index(drop=True).copy()


def propensity_pos_weight(train_df: pd.DataFrame, disable_pos_weight: bool) -> float | None:
    if disable_pos_weight:
        return None
    labels = train_df["treatment"].astype(int).to_numpy()
    pos = int(labels.sum())
    neg = int(len(labels) - pos)
    return float(neg / max(1, pos))


def compute_ipw_from_propensity(
    df: pd.DataFrame, propensity_col: str = "propensity"
) -> pd.DataFrame:
    out = df.copy()
    out["propensity_clipped"] = clip_probs(out[propensity_col].to_numpy(), eps=1e-3)
    treatment_rate = float(out["treatment"].mean())
    treat_mask = out["treatment"].astype(int).to_numpy() == 1
    p = out["propensity_clipped"].to_numpy(dtype=np.float64)
    out["ipw"] = np.where(treat_mask, 1.0 / p, 1.0 / (1.0 - p))
    out["stabilized_ipw"] = np.where(
        treat_mask,
        treatment_rate / p,
        (1.0 - treatment_rate) / (1.0 - p),
    )
    return out


def evaluate_classifier(
    model: RandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
    *,
    pos_weight: float | None = None,
) -> dict[str, float]:
    probs = model.predict_proba(X)[:, 1]
    return {
        "loss": binary_log_loss(y, probs, pos_weight=pos_weight),
        "accuracy": accuracy_from_probs(y, probs),
        "auc": binary_auc(y.tolist(), probs.tolist()),
    }


def train_rf_with_val_selection(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    featureizer: TopTokenFeatureizer,
    *,
    label_col: str,
    include_treatment: bool,
    n_estimators: int,
    max_depth_options: list[int],
    min_samples_leaf_options: list[int],
    max_features: str,
    n_jobs: int,
    seed: int,
    pos_weight: float | None = None,
) -> tuple[RandomForestClassifier, dict[str, float], dict[str, int | None]]:
    X_train = featureizer.transform(train_df, include_treatment=include_treatment)
    X_val = featureizer.transform(val_df, include_treatment=include_treatment)
    y_train = train_df[label_col].astype(int).to_numpy()
    y_val = val_df[label_col].astype(int).to_numpy()


    best_model: RandomForestClassifier | None = None
    best_metrics: dict[str, float] | None = None
    best_params: dict[str, int | None] | None = None

    for max_depth in max_depth_options:
        for min_samples_leaf in min_samples_leaf_options:
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                n_jobs=n_jobs,
                random_state=seed,
            )
            model.fit(X_train, y_train)
            metrics = evaluate_classifier(model, X_val, y_val, pos_weight=pos_weight)

            if best_metrics is None or metrics["loss"] < best_metrics["loss"]:
                best_model = model
                best_metrics = metrics
                best_params = {
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                }

    if best_model is None or best_metrics is None or best_params is None:
        raise RuntimeError(
            "Validation model selection did not produce a fitted random forest."
        )

    return best_model, best_metrics, best_params


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_propensity_outputs(
    model: RandomForestClassifier,
    featureizer: TopTokenFeatureizer,
    samples_df: pd.DataFrame,
) -> pd.DataFrame:
    probs = model.predict_proba(
        featureizer.transform(samples_df, include_treatment=False)
    )[:, 1]
    out = samples_df.reset_index(drop=True).copy()
    out.insert(0, "row_index", np.arange(len(out), dtype=np.int64))
    out["propensity"] = probs.astype(np.float64)
    return compute_ipw_from_propensity(out, propensity_col="propensity")


def build_outcome_outputs(
    model: RandomForestClassifier,
    featureizer: TopTokenFeatureizer,
    samples_df: pd.DataFrame,
) -> pd.DataFrame:
    out = samples_df.reset_index(drop=True).copy()
    out.insert(0, "row_index", np.arange(len(out), dtype=np.int64))

    X_obs = featureizer.transform(out, include_treatment=True)
    y_obs = model.predict_proba(X_obs)[:, 1]

    y0_df = out.copy()
    y0_df["treatment"] = 0
    y1_df = out.copy()
    y1_df["treatment"] = 1

    y0 = model.predict_proba(featureizer.transform(y0_df, include_treatment=True))[:, 1]
    y1 = model.predict_proba(featureizer.transform(y1_df, include_treatment=True))[:, 1]

    out["outcome_probability_observed_treatment"] = y_obs.astype(np.float64)
    out["outcome_probability_treatment_0"] = y0.astype(np.float64)
    out["outcome_probability_treatment_1"] = y1.astype(np.float64)
    out["individual_treatment_effect"] = (
        out["outcome_probability_treatment_1"]
        - out["outcome_probability_treatment_0"]
    )
    return out


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab = load_vocab(args.vocab_path)
    base_samples_df = load_samples_dataframe(args=args, vocab=vocab)
    if base_samples_df.empty:
        raise ValueError("The constructed treatment-effect dataset is empty.")

    propensity_samples_df = base_samples_df


    propensity_labels = propensity_samples_df["treatment"].astype(int).tolist()
    propensity_splits = stratified_split_indices(
        labels=propensity_labels,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    propensity_train_df = subset_df(propensity_samples_df, propensity_splits.train)
    propensity_val_df = subset_df(propensity_samples_df, propensity_splits.val)
    propensity_test_df = subset_df(propensity_samples_df, propensity_splits.test)

    propensity_featureizer = TopTokenFeatureizer(args.top_k_tokens).fit(
        propensity_train_df["history_tokens"].tolist()
    )
    pos_weight = propensity_pos_weight(
        propensity_train_df,
        args.disable_propensity_pos_weight,
    )
    propensity_model, propensity_best_val_metrics, propensity_best_params = (
        train_rf_with_val_selection(
            train_df=propensity_train_df,
            val_df=propensity_val_df,
            featureizer=propensity_featureizer,
            label_col="treatment",
            include_treatment=False,
            n_estimators=args.n_estimators,
            max_depth_options=args.max_depth_options,
            min_samples_leaf_options=args.min_samples_leaf_options,
            max_features=args.max_features,
            n_jobs=args.n_jobs,
            seed=args.seed,
            pos_weight=pos_weight,
        )
    )
    propensity_test_metrics = evaluate_classifier(propensity_model, propensity_featureizer.transform(propensity_test_df, include_treatment=False), propensity_test_df["treatment"].astype(int).to_numpy(), pos_weight=pos_weight)
    propensity_outputs = build_propensity_outputs(
        propensity_model,
        propensity_featureizer,
        propensity_samples_df,
    )

    outcome_samples_df = base_samples_df.copy()


    outcome_split_labels = (
        outcome_samples_df["treatment"].astype(int) * 2
        + outcome_samples_df["outcome"].astype(int)
    ).tolist()
    outcome_splits = stratified_split_indices(
        labels=outcome_split_labels,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    outcome_train_df = subset_df(outcome_samples_df, outcome_splits.train)
    outcome_val_df = subset_df(outcome_samples_df, outcome_splits.val)
    outcome_test_df = subset_df(outcome_samples_df, outcome_splits.test)

    outcome_featureizer = TopTokenFeatureizer(args.top_k_tokens).fit(
        outcome_train_df["history_tokens"].tolist()
    )
    outcome_model, outcome_best_val_metrics, outcome_best_params = (
        train_rf_with_val_selection(
            train_df=outcome_train_df,
            val_df=outcome_val_df,
            featureizer=outcome_featureizer,
            label_col="outcome",
            include_treatment=True,
            n_estimators=args.n_estimators,
            max_depth_options=args.max_depth_options,
            min_samples_leaf_options=args.min_samples_leaf_options,
            max_features=args.max_features,
            n_jobs=args.n_jobs,
            seed=args.seed,
            pos_weight=None,
        )
    )
    outcome_test_metrics = evaluate_classifier(outcome_model, outcome_featureizer.transform(outcome_test_df, include_treatment=True), outcome_test_df["outcome"].astype(int).to_numpy(), pos_weight=None)
    outcome_outputs = build_outcome_outputs(
        outcome_model,
        outcome_featureizer,
        outcome_samples_df,
    )

    joblib.dump(
        {
            'model': propensity_model,
            'featureizer': propensity_featureizer,
            'best_val_metrics': propensity_best_val_metrics,
            'test_metrics': propensity_test_metrics,
            'best_params': propensity_best_params,
            'args': serialize_args(args),
        },
        output_dir / "rf_propensity_model.joblib",
    )
    joblib.dump(
        {
            'model': outcome_model,
            'featureizer': outcome_featureizer,
            'best_val_metrics': outcome_best_val_metrics,
            'test_metrics': outcome_test_metrics,
            'best_params': outcome_best_params,
            'args': serialize_args(args),
        },
        output_dir / "rf_outcome_model.joblib",
    )

    propensity_outputs.to_parquet(
        output_dir / "patient_propensities.parquet",
        index=False,
    )
    outcome_outputs.to_parquet(
        output_dir / "patient_outcome_predictions.parquet",
        index=False,
    )
    save_json(
        output_dir / "propensity_split_indices.json",
        {
            "train": propensity_splits.train,
            "val": propensity_splits.val,
            "test": propensity_splits.test,
        },
    )
    save_json(
        output_dir / "outcome_split_indices.json",
        {
            "train": outcome_splits.train,
            "val": outcome_splits.val,
            "test": outcome_splits.test,
        },
    )
    save_json(
        output_dir / "metrics.json",
        {
            "propensity": {"best_validation": propensity_best_val_metrics, "test": propensity_test_metrics, "best_params": propensity_best_params},
            "outcome": {"best_validation": outcome_best_val_metrics, "test": outcome_test_metrics, "best_params": outcome_best_params},
        },
    )

    print(f"Saved propensity model to: {output_dir / 'rf_propensity_model.joblib'}")
    print(f"Saved outcome model to: {output_dir / 'rf_outcome_model.joblib'}")
    print(f"Saved patient propensities to: {output_dir / 'patient_propensities.parquet'}")
    print(
        "Saved patient outcome predictions to: "
        f"{output_dir / 'patient_outcome_predictions.parquet'}"
    )
    print(f"Saved metrics to: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
