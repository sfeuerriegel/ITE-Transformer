from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.DR_Learner.prepare_treatment_effect_dataset import DOB_COL, PERSON_ID_COL, load_pickle_dataframe
from src.DR_Learner.random_forest import accuracy_from_probs, compute_ipw_from_propensity, binary_log_loss
from src.DR_Learner.train_linear_cate_model import regression_metrics
from src.DR_Learner.helpers import (
    binary_auc,
    load_samples_dataframe,
    load_vocab,
    serialize_args,
    set_seed,
)
from src.DR_Learner.train_nuisance_models import build_pseudo_outcomes, summarize_crossfit_predictions
from src.DR_Learner.train_pseudo_outcome_cate_model import build_development_splits, dataset_rows_for_merge, ensure_row_index_column


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train age-at-index and sex-only nuisance/CATE baselines.")

    data_group = parser.add_argument_group("Dataset inputs")
    data_group.add_argument("--samples-path", type=Path, default=None)
    data_group.add_argument("--treatment-pkl", type=Path, default=None)
    data_group.add_argument("--comparator-pkl", type=Path, default=None)
    data_group.add_argument("--conditions-pkl", type=Path, default=None)
    data_group.add_argument("--persons-pkl", type=Path, default=None)
    data_group.add_argument("--treatment-name", type=str, default="treatment")
    data_group.add_argument("--comparator-name", type=str, default="comparator")
    data_group.add_argument("--outcome-horizon-days", type=int, default=None)
    data_group.add_argument("--min-history-events", type=int, default=0)
    data_group.add_argument("--max-history-events", type=int, default=None)


    parser.add_argument("--vocab-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-indices-path", type=Path, default=None)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--propensity-clip", type=float, default=0.1)

    parser.add_argument("--cate-alpha", type=float, default=1.0)
    parser.add_argument(
        "--allow-history-age-fallback",
        action="store_true",
        help="If age_at_index is unavailable, use the last pre-index history age as an approximation.",
    )
    return parser.parse_args()


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def compute_age_at_index_from_persons(samples_df: pd.DataFrame, persons_pkl: Path | None) -> pd.Series | None:
    if persons_pkl is None:
        return None

    people = load_pickle_dataframe(persons_pkl)
    required = {PERSON_ID_COL, DOB_COL}
    missing = sorted(required - set(people.columns))
    if missing:
        raise KeyError(f"Missing required person columns for age_at_index: {missing}")

    people = people[[PERSON_ID_COL, DOB_COL]].copy()
    people[PERSON_ID_COL] = pd.to_numeric(people[PERSON_ID_COL], errors="coerce")
    people[DOB_COL] = pd.to_datetime(people[DOB_COL], utc=True, errors="coerce")
    people = people.dropna(subset=[PERSON_ID_COL]).drop_duplicates(subset=[PERSON_ID_COL])
    people[PERSON_ID_COL] = people[PERSON_ID_COL].astype("int64")

    merged = samples_df[[PERSON_ID_COL, "index_datetime"]].copy()
    merged[PERSON_ID_COL] = pd.to_numeric(merged[PERSON_ID_COL], errors="coerce").astype("Int64")
    merged["index_datetime"] = pd.to_datetime(merged["index_datetime"], utc=True, errors="coerce")
    merged = merged.merge(people, on=PERSON_ID_COL, how="left", validate="many_to_one")
    delta_days = (merged["index_datetime"] - merged[DOB_COL]).dt.total_seconds() / (3600.0 * 24.0)
    return (delta_days / 365.25).fillna(0.0).clip(lower=0.0).astype("float32")


def ensure_age_at_index(samples_df: pd.DataFrame, persons_pkl: Path | None, allow_history_fallback: bool) -> pd.DataFrame:
    out = samples_df.copy()
    if "age_at_index" in out.columns:
        out["age_at_index"] = pd.to_numeric(out["age_at_index"], errors="coerce").fillna(0.0).astype("float32")
        out["age_at_index_source"] = "samples.age_at_index"
        return out

    computed = compute_age_at_index_from_persons(out, persons_pkl)
    if computed is not None:
        out["age_at_index"] = computed
        out["age_at_index_source"] = "persons.date_of_birth"
        return out

    if allow_history_fallback:
        out["age_at_index"] = out["history_ages"].map(
            lambda ages: float(ages[-1]) if isinstance(ages, (list, tuple, np.ndarray)) and len(ages) else 0.0
        ).astype("float32")
        out["age_at_index_source"] = "last_history_age_approximation"
        return out

    raise KeyError(
        "Missing age_at_index. Rebuild samples with the updated dataset code, pass --persons-pkl so age can be "
        "computed from DOB, or use --allow-history-age-fallback for an approximation."
    )


class AgeSexFeatureizer:
    def __init__(self) -> None:
        self.gender_categories_: list[int] = []

    def fit(self, df: pd.DataFrame) -> "AgeSexFeatureizer":
        genders = pd.to_numeric(df["gender_token"], errors="coerce").fillna(-1).astype(int)
        self.gender_categories_ = sorted(genders.unique().tolist())
        return self

    def transform(self, df: pd.DataFrame, include_treatment: bool = False, include_interactions: bool = False) -> np.ndarray:
        age = pd.to_numeric(df["age_at_index"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        gender = pd.to_numeric(df["gender_token"], errors="coerce").fillna(-1).astype(int).to_numpy()
        cols = [age.reshape(-1, 1)]
        for category in self.gender_categories_:
            cols.append((gender == int(category)).astype(np.float64).reshape(-1, 1))

        if include_treatment:
            treatment = pd.to_numeric(df["treatment"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            cols.append(treatment.reshape(-1, 1))
            if include_interactions:
                cols.append((age * treatment).reshape(-1, 1))
                for category in self.gender_categories_:
                    cols.append(((gender == int(category)).astype(np.float64) * treatment).reshape(-1, 1))

        return np.hstack(cols).astype(np.float64)


def subset_df(df: pd.DataFrame, indices: list[int]) -> pd.DataFrame:
    return df.iloc[list(indices)].reset_index(drop=True).copy()


def fit_logistic_classifier(
    train_df: pd.DataFrame,
    featureizer: AgeSexFeatureizer,
    *,
    label_col: str,
    include_treatment: bool,
    include_interactions: bool,
    seed: int,
) -> Pipeline:
    y = train_df[label_col].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError(f"{label_col} has only one class in the training split.")

    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed)),
        ]
    )

    model.fit(
        featureizer.transform(train_df, include_treatment=include_treatment, include_interactions=include_interactions),
        y,
    )
    return model


def evaluate_classifier(
    model: Pipeline,
    featureizer: AgeSexFeatureizer,
    df: pd.DataFrame,
    *,
    label_col: str,
    include_treatment: bool,
    include_interactions: bool,
) -> dict[str, float]:
    probs = model.predict_proba(
        featureizer.transform(df, include_treatment=include_treatment, include_interactions=include_interactions)
    )[:, 1]
    y = df[label_col].astype(int).to_numpy()

    return {
        "loss": binary_log_loss(y, probs),
        "accuracy": accuracy_from_probs(y, probs),
        "auc": binary_auc(y.tolist(), probs.tolist()),
    }


def build_nuisance_predictions(
    samples_df: pd.DataFrame,
    featureizer: AgeSexFeatureizer,
    propensity_model: Pipeline,
    outcome_model: Pipeline,
    propensity_clip: float,
) -> pd.DataFrame:
    out = samples_df.reset_index(drop=True).copy()
    if "row_index" not in out.columns:
        out.insert(0, "row_index", np.arange(len(out), dtype=np.int64))

    out["propensity"] = propensity_model.predict_proba(featureizer.transform(out, include_treatment=False))[:, 1].astype(np.float64)

    observed = outcome_model.predict_proba(
        featureizer.transform(out, include_treatment=True, include_interactions=True)
    )[:, 1]
    y0_df = out.copy()
    y0_df["treatment"] = 0
    y1_df = out.copy()
    y1_df["treatment"] = 1
    y0 = outcome_model.predict_proba(featureizer.transform(y0_df, include_treatment=True, include_interactions=True))[:, 1]
    y1 = outcome_model.predict_proba(featureizer.transform(y1_df, include_treatment=True, include_interactions=True))[:, 1]

    out["outcome_probability_observed_treatment"] = observed.astype(np.float64)
    out["outcome_probability_treatment_0"] = y0.astype(np.float64)
    out["outcome_probability_treatment_1"] = y1.astype(np.float64)
    out["individual_treatment_effect"] = out["outcome_probability_treatment_1"] - out["outcome_probability_treatment_0"]

    out = compute_ipw_from_propensity(out, propensity_col="propensity")
    return build_pseudo_outcomes(out, clip_value=propensity_clip)


def fit_cate_model(
    samples_df: pd.DataFrame,
    splits,
    featureizer: AgeSexFeatureizer,
    *,
    pseudo_outcome_column: str,
    alpha: float,
    seed: int,
) -> tuple[Pipeline, pd.DataFrame, dict[str, float]]:
    targets = pd.to_numeric(samples_df[pseudo_outcome_column], errors="coerce").to_numpy(dtype=np.float64)
    if np.isnan(targets).any():
        raise ValueError(f"{pseudo_outcome_column} contains missing or non-numeric values.")

    features = featureizer.transform(samples_df, include_treatment=False)
    model = Pipeline(steps=[("scale", StandardScaler()), ("regressor", Ridge(alpha=float(alpha), random_state=seed))])

    model.fit(features[splits.train], targets[splits.train])


    val_metrics = regression_metrics(
        model.predict(features[splits.val]),
        targets[splits.val],
        loss_name="mse",
        huber_delta=1.0,
    )
    val_metrics["best_epoch"] = 0

    predictions = model.predict(features)
    prediction_df = pd.DataFrame(
        {
            "_dataset_row_index": np.arange(len(samples_df), dtype=np.int64),
            "predicted_treatment_effect": np.asarray(predictions, dtype=np.float64),
        }
    ).merge(dataset_rows_for_merge(samples_df), on="_dataset_row_index", how="left", validate="one_to_one")
    if "row_index" not in prediction_df.columns:
        prediction_df = prediction_df.rename(columns={"_dataset_row_index": "row_index"})
    return model, prediction_df, val_metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab = load_vocab(args.vocab_path)
    samples_df = load_samples_dataframe(args=args, vocab=vocab)
    if samples_df.empty:
        raise ValueError("The treatment-effect dataset is empty.")

    samples_df = ensure_row_index_column(samples_df)
    samples_df = ensure_age_at_index(samples_df, args.persons_pkl, args.allow_history_age_fallback)


    development_df, splits, split_summary = build_development_splits(
        samples_df=samples_df,
        split_indices_path=args.split_indices_path,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    if not splits.val:
        raise ValueError("Validation split must be non-empty.")

    train_df = subset_df(development_df, splits.train)
    val_df = subset_df(development_df, splits.val)

    featureizer = AgeSexFeatureizer().fit(train_df)
    propensity_model = fit_logistic_classifier(
        train_df,
        featureizer,
        label_col="treatment",
        include_treatment=False,
        include_interactions=False,
        seed=args.seed,
    )
    outcome_model = fit_logistic_classifier(
        train_df,
        featureizer,
        label_col="outcome",
        include_treatment=True,
        include_interactions=True,
        seed=args.seed,
    )

    nuisance_predictions = build_nuisance_predictions(
        development_df,
        featureizer,
        propensity_model,
        outcome_model,
        propensity_clip=float(args.propensity_clip),
    )
    cate_model, cate_predictions, cate_val_metrics = fit_cate_model(
        nuisance_predictions,
        splits,
        featureizer,
        pseudo_outcome_column="aipw_pseudo_outcome",
        alpha=float(args.cate_alpha),
        seed=args.seed,
    )

    metrics = {
        "propensity": {
            "validation": evaluate_classifier(
                propensity_model,
                featureizer,
                val_df,
                label_col="treatment",
                include_treatment=False,
                include_interactions=False,
            )
        },
        "outcome": {
            "validation": evaluate_classifier(
                outcome_model,
                featureizer,
                val_df,
                label_col="outcome",
                include_treatment=True,
                include_interactions=True,
            )
        },
        "cate": {"best_validation": cate_val_metrics, "pseudo_outcome_column": "aipw_pseudo_outcome"},
        "summary": summarize_crossfit_predictions(nuisance_predictions),
        "data_split": split_summary,
        "features": {
            "covariates": ["age_at_index", "gender_token"],
            "age_at_index_source": str(development_df["age_at_index_source"].iloc[0]),
            "gender_categories": featureizer.gender_categories_,
            "outcome_model_includes_treatment_interactions": True,
        },
        "args": serialize_args(args),
    }

    nuisance_predictions.to_parquet(output_dir / "patient_nuisance_predictions.parquet", index=False)
    cate_predictions.to_parquet(output_dir / "patient_cate_predictions.parquet", index=False)
    save_json(output_dir / "split_indices.json", {"train": splits.train, "val": splits.val, "test": splits.test})
    save_json(output_dir / "metrics.json", metrics)
    joblib.dump(
        {
            "featureizer": featureizer,
            "propensity_model": propensity_model,
            "outcome_model": outcome_model,
            "cate_model": cate_model,
            "args": serialize_args(args),
        },
        output_dir / "age_sex_baseline_models.joblib",
    )

    print(f"Saved nuisance predictions to: {output_dir / 'patient_nuisance_predictions.parquet'}")
    print(f"Saved CATE predictions to: {output_dir / 'patient_cate_predictions.parquet'}")
    print(f"Saved metrics to: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
