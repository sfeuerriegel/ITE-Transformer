"""Run downstream treatment-effect experiments without a notebook.

Existing samples, splits, and predictions are reused. Use a new output directory
when changing data or model settings. Analysis and plots remain in the notebook.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runnables.treatment_effect_config import EXPERIMENT_CHOICES, build_config
import numpy as np
import pandas as pd
import torch
from src.DR_Learner.helpers import load_vocab, stratified_split_indices
from src.DR_Learner.prepare_treatment_effect_dataset import create_treatment_effect_dataset_from_pickles
from src.DR_Learner.random_forest import TopTokenFeatureizer as RFTopTokenFeatureizer, propensity_pos_weight as rf_propensity_pos_weight, train_rf_with_val_selection, evaluate_classifier as evaluate_rf_classifier, save_json as save_rf_json, set_seed as set_rf_seed
from src.DR_Learner.helpers import set_seed as set_age_sex_seed
from src.DR_Learner.train_nuisance_models import build_pseudo_outcomes
from src.DR_Learner.train_age_sex_baseline import AgeSexFeatureizer, ensure_age_at_index, fit_logistic_classifier as fit_age_sex_logistic_classifier, evaluate_classifier as evaluate_age_sex_classifier


class ExperimentRunner:
    """Prepare data and run model stages using shared experiment settings."""

    def __init__(self, config):
        self.config = config


    def require_files(self, paths) -> None:
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError('Required input or saved result files are missing:\n' + '\n'.join(missing))


    def run_command(self, cmd, workdir: Path) -> None:
        env = os.environ.copy()
        env['WANDB_MODE'] = self.config.WANDB_MODE
        print('Running:', ' '.join(map(str, cmd)))
        subprocess.run(cmd, cwd=workdir, env=env, check=True)


    def saved_predictions_exist(self, path: Path, stage: str) -> bool:
        if path.is_file():
            print(f'Using saved {stage}: {path}')
            return True
        return False


    def load_crossfit_fold_specs(self, seed: int) -> list[dict[str, list[int]]]:
        fold_path = self.config.NUISANCE_ROOT / f'seed_{seed}' / 'crossfit_fold_indices.json'
        if not fold_path.exists():
            raise FileNotFoundError(
                f'Missing transformer cross-fit fold indices for seed {seed}: {fold_path}. '
                'Run the pretrained nuisance stage first.'
            )
        return json.loads(fold_path.read_text())


    def prepare_baseline_samples(self, ) -> pd.DataFrame:
        baseline_source_df = ensure_age_at_index(
            self.development_samples_df.copy(),
            persons_pkl=self.config.PERSONS_PKL,
            allow_history_fallback=False,
        )
        baseline_df = baseline_source_df.reset_index(drop=True)


        return baseline_df


    def subset_rows(self, df: pd.DataFrame, indices: list[int]) -> pd.DataFrame:
        return df.iloc[list(indices)].reset_index(drop=True).copy()


    def build_holdout_prediction_frame(self,
        *,
        holdout_indices: list[int],
        holdout_df: pd.DataFrame,
        propensity_probs: np.ndarray,
        outcome_probs_obs: np.ndarray,
        outcome_probs_t0: np.ndarray,
        outcome_probs_t1: np.ndarray,
        fold_idx: int,
    ) -> pd.DataFrame:
        prediction_df = pd.DataFrame({
            '_dataset_row_index': np.asarray(holdout_indices, dtype=int),
            'row_index': holdout_df['row_index'].astype(int).to_numpy(),
            'person_id': holdout_df['person_id'].astype(int).to_numpy(),
            'treatment': holdout_df['treatment'].astype(int).to_numpy(),
            'outcome': holdout_df['outcome'].astype(int).to_numpy(),
            'propensity': np.asarray(propensity_probs, dtype=float),
            'outcome_probability_observed_treatment': np.asarray(outcome_probs_obs, dtype=float),
            'outcome_probability_treatment_0': np.asarray(outcome_probs_t0, dtype=float),
            'outcome_probability_treatment_1': np.asarray(outcome_probs_t1, dtype=float),
            'crossfit_fold': int(fold_idx),
        })
        prediction_df['individual_treatment_effect'] = (
            prediction_df['outcome_probability_treatment_1']
            - prediction_df['outcome_probability_treatment_0']
        )
        return prediction_df


    def train_rf_crossfit_baseline(self, seed: int, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        set_rf_seed(seed)
        baseline_df = self.prepare_baseline_samples()
        fold_specs = self.load_crossfit_fold_specs(seed)

        fold_prediction_frames = []
        fold_metric_rows = []
        fold_artifacts = []

        for fold_spec in fold_specs:
            fold_idx = int(fold_spec['fold'])
            train_indices = list(map(int, fold_spec['train']))
            val_indices = list(map(int, fold_spec['val']))
            holdout_indices = list(map(int, fold_spec['holdout']))

            train_df = self.subset_rows(baseline_df, train_indices)
            val_df = self.subset_rows(baseline_df, val_indices)
            holdout_df = self.subset_rows(baseline_df, holdout_indices)

            propensity_featureizer = RFTopTokenFeatureizer(self.config.RF_TOP_K_TOKENS).fit(train_df['history_tokens'].tolist())
            propensity_pos_weight = rf_propensity_pos_weight(
                train_df,
                disable_pos_weight=self.config.NUISANCE_DISABLE_PROPENSITY_POS_WEIGHT,
            )
            propensity_model, propensity_val_metrics, propensity_best_params = train_rf_with_val_selection(
                train_df=train_df,
                val_df=val_df,
                featureizer=propensity_featureizer,
                label_col='treatment',
                include_treatment=False,
                n_estimators=self.config.RF_N_ESTIMATORS,
                max_depth_options=self.config.RF_MAX_DEPTH_OPTIONS,
                min_samples_leaf_options=self.config.RF_MIN_SAMPLES_LEAF_OPTIONS,
                max_features=self.config.RF_MAX_FEATURES,
                n_jobs=self.config.RF_N_JOBS,
                seed=seed,
                pos_weight=propensity_pos_weight,
            )
            propensity_holdout_metrics = evaluate_rf_classifier(propensity_model, propensity_featureizer.transform(holdout_df, include_treatment=False), holdout_df['treatment'].astype(int).to_numpy(), pos_weight=propensity_pos_weight)
            propensity_holdout_probs = propensity_model.predict_proba(
                propensity_featureizer.transform(holdout_df, include_treatment=False)
            )[:, 1]

            outcome_featureizer = RFTopTokenFeatureizer(self.config.RF_TOP_K_TOKENS).fit(train_df['history_tokens'].tolist())
            outcome_model, outcome_val_metrics, outcome_best_params = train_rf_with_val_selection(
                train_df=train_df,
                val_df=val_df,
                featureizer=outcome_featureizer,
                label_col='outcome',
                include_treatment=True,
                n_estimators=self.config.RF_N_ESTIMATORS,
                max_depth_options=self.config.RF_MAX_DEPTH_OPTIONS,
                min_samples_leaf_options=self.config.RF_MIN_SAMPLES_LEAF_OPTIONS,
                max_features=self.config.RF_MAX_FEATURES,
                n_jobs=self.config.RF_N_JOBS,
                seed=seed,
                pos_weight=None,
            )
            outcome_holdout_metrics = evaluate_rf_classifier(outcome_model, outcome_featureizer.transform(holdout_df, include_treatment=True), holdout_df['outcome'].astype(int).to_numpy(), pos_weight=None)

            outcome_holdout_obs_df = holdout_df.copy()
            outcome_holdout_t0_df = holdout_df.copy()
            outcome_holdout_t0_df['treatment'] = 0
            outcome_holdout_t1_df = holdout_df.copy()
            outcome_holdout_t1_df['treatment'] = 1
            outcome_holdout_probs_obs = outcome_model.predict_proba(
                outcome_featureizer.transform(outcome_holdout_obs_df, include_treatment=True)
            )[:, 1]
            outcome_holdout_probs_t0 = outcome_model.predict_proba(
                outcome_featureizer.transform(outcome_holdout_t0_df, include_treatment=True)
            )[:, 1]
            outcome_holdout_probs_t1 = outcome_model.predict_proba(
                outcome_featureizer.transform(outcome_holdout_t1_df, include_treatment=True)
            )[:, 1]

            fold_prediction_frames.append(
                self.build_holdout_prediction_frame(
                    holdout_indices=holdout_indices,
                    holdout_df=holdout_df,
                    propensity_probs=propensity_holdout_probs,
                    outcome_probs_obs=outcome_holdout_probs_obs,
                    outcome_probs_t0=outcome_holdout_probs_t0,
                    outcome_probs_t1=outcome_holdout_probs_t1,
                    fold_idx=fold_idx,
                )
            )
            fold_metric_rows.extend([
                {
                    'fold': fold_idx,
                    'task': 'propensity',
                    'train_size': len(train_indices),
                    'val_size': len(val_indices),
                    'holdout_size': len(holdout_indices),
                    **{f'best_val/{key}': value for key, value in propensity_val_metrics.items()},
                    **{f'holdout/{key}': value for key, value in propensity_holdout_metrics.items()},
                    'best_params': str(propensity_best_params),
                },
                {
                    'fold': fold_idx,
                    'task': 'outcome',
                    'train_size': len(train_indices),
                    'val_size': len(val_indices),
                    'holdout_size': len(holdout_indices),
                    **{f'best_val/{key}': value for key, value in outcome_val_metrics.items()},
                    **{f'holdout/{key}': value for key, value in outcome_holdout_metrics.items()},
                    'best_params': str(outcome_best_params),
                },
            ])
            fold_artifacts.append({
                'fold': fold_idx,
                'propensity_best_params': propensity_best_params,
                'outcome_best_params': outcome_best_params,
            })

        prediction_df = pd.concat(fold_prediction_frames, ignore_index=True).sort_values('row_index').reset_index(drop=True)
        if prediction_df['row_index'].nunique() != len(baseline_df):
            raise RuntimeError(f'RF cross-fitting for seed {seed} did not produce one holdout prediction per row.')
        prediction_df = build_pseudo_outcomes(prediction_df, clip_value=self.config.PROPENSITY_CLIP)
        fold_metrics_df = pd.DataFrame(fold_metric_rows).sort_values(['fold', 'task']).reset_index(drop=True)

        save_rf_json(
            output_dir / 'metrics.json',
            {
                'seed': seed,
                'folds': fold_artifacts,
                'summary': {
                    'mean_propensity': float(prediction_df['propensity'].mean()),
                    'mean_individual_treatment_effect': float(prediction_df['individual_treatment_effect'].mean()),
                    'ipw_ate': float(prediction_df['ipw_pseudo_outcome'].mean()),
                    'aipw_ate': float(prediction_df['aipw_pseudo_outcome'].mean()),
                    'mean_abs_aipw_pseudo_outcome': float(prediction_df['aipw_pseudo_outcome'].abs().mean()),
                },
            },
        )
        prediction_df.to_parquet(output_dir / 'patient_nuisance_predictions.parquet', index=False)
        fold_metrics_df.to_parquet(output_dir / 'fold_metrics.parquet', index=False)
        return prediction_df, fold_metrics_df


    def train_age_sex_crossfit_baseline(self, seed: int, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        set_age_sex_seed(seed)
        baseline_df = self.prepare_baseline_samples()
        fold_specs = self.load_crossfit_fold_specs(seed)

        fold_prediction_frames = []
        fold_metric_rows = []
        fold_artifacts = []

        for fold_spec in fold_specs:
            fold_idx = int(fold_spec['fold'])
            train_indices = list(map(int, fold_spec['train']))
            val_indices = list(map(int, fold_spec['val']))
            holdout_indices = list(map(int, fold_spec['holdout']))

            train_df = self.subset_rows(baseline_df, train_indices)
            holdout_df = self.subset_rows(baseline_df, holdout_indices)

            featureizer = AgeSexFeatureizer().fit(train_df)
            propensity_model = fit_age_sex_logistic_classifier(
                train_df,
                featureizer,
                label_col='treatment',
                include_treatment=False,
                include_interactions=False,
                seed=seed,
            )
            outcome_model = fit_age_sex_logistic_classifier(
                train_df,
                featureizer,
                label_col='outcome',
                include_treatment=True,
                include_interactions=True,
                seed=seed,
            )

            propensity_probs = propensity_model.predict_proba(
                featureizer.transform(holdout_df, include_treatment=False)
            )[:, 1]
            outcome_probs_obs = outcome_model.predict_proba(
                featureizer.transform(holdout_df, include_treatment=True, include_interactions=True)
            )[:, 1]
            holdout_t0_df = holdout_df.copy()
            holdout_t0_df['treatment'] = 0
            holdout_t1_df = holdout_df.copy()
            holdout_t1_df['treatment'] = 1
            outcome_probs_t0 = outcome_model.predict_proba(
                featureizer.transform(holdout_t0_df, include_treatment=True, include_interactions=True)
            )[:, 1]
            outcome_probs_t1 = outcome_model.predict_proba(
                featureizer.transform(holdout_t1_df, include_treatment=True, include_interactions=True)
            )[:, 1]

            fold_prediction_df = self.build_holdout_prediction_frame(
                holdout_indices=holdout_indices,
                holdout_df=holdout_df,
                propensity_probs=propensity_probs,
                outcome_probs_obs=outcome_probs_obs,
                outcome_probs_t0=outcome_probs_t0,
                outcome_probs_t1=outcome_probs_t1,
                fold_idx=fold_idx,
            )
            fold_prediction_frames.append(fold_prediction_df)

            propensity_holdout_metrics = evaluate_age_sex_classifier(
                propensity_model,
                featureizer,
                holdout_df,
                label_col='treatment',
                include_treatment=False,
                include_interactions=False,
            )
            outcome_holdout_metrics = evaluate_age_sex_classifier(
                outcome_model,
                featureizer,
                holdout_df,
                label_col='outcome',
                include_treatment=True,
                include_interactions=True,
            )
            fold_metric_rows.extend([
                {
                    'fold': fold_idx,
                    'task': 'propensity',
                    'holdout_size': len(holdout_indices),
                    'train_size': len(train_indices),
                    'val_size': len(val_indices),
                    **{f'holdout/{key}': value for key, value in propensity_holdout_metrics.items()},
                },
                {
                    'fold': fold_idx,
                    'task': 'outcome',
                    'holdout_size': len(holdout_indices),
                    'train_size': len(train_indices),
                    'val_size': len(val_indices),
                    **{f'holdout/{key}': value for key, value in outcome_holdout_metrics.items()},
                },
            ])
            fold_artifacts.append({
                'fold': fold_idx,
                'train_size': len(train_indices),
                'val_size': len(val_indices),
                'holdout_size': len(holdout_indices),
                'gender_categories': featureizer.gender_categories_,
            })

        prediction_df = pd.concat(fold_prediction_frames, ignore_index=True).sort_values('row_index').reset_index(drop=True)
        prediction_df = build_pseudo_outcomes(prediction_df, clip_value=self.config.PROPENSITY_CLIP)
        fold_metrics_df = pd.DataFrame(fold_metric_rows).sort_values(['fold', 'task']).reset_index(drop=True)
        save_rf_json(
            output_dir / 'metrics.json',
            {
                'seed': seed,
                'folds': fold_artifacts,
                'summary': {
                    'mean_propensity': float(prediction_df['propensity'].mean()),
                    'mean_individual_treatment_effect': float(prediction_df['individual_treatment_effect'].mean()),
                    'ipw_ate': float(prediction_df['ipw_pseudo_outcome'].mean()),
                    'aipw_ate': float(prediction_df['aipw_pseudo_outcome'].mean()),
                    'mean_abs_aipw_pseudo_outcome': float(prediction_df['aipw_pseudo_outcome'].abs().mean()),
                    'age_at_index_source': str(baseline_df['age_at_index_source'].iloc[0]),
                },
            },
        )
        prediction_df.to_parquet(output_dir / 'patient_nuisance_predictions.parquet', index=False)
        fold_metrics_df.to_parquet(output_dir / 'fold_metrics.parquet', index=False)
        return prediction_df, fold_metrics_df


    def train_xgb_crossfit_baseline(self, seed: int, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        from src.DR_Learner.xgboost_baseline import (
            TopTokenFeatureizer as XGBTopTokenFeatureizer,
            propensity_pos_weight as xgb_propensity_pos_weight,
            train_xgb_with_val_selection,
            evaluate_classifier as evaluate_xgb_classifier,
            save_json as save_xgb_json,
            set_seed as set_xgb_seed,
        )
        set_xgb_seed(seed)
        baseline_df = self.prepare_baseline_samples()
        fold_specs = self.load_crossfit_fold_specs(seed)

        fold_prediction_frames = []
        fold_metric_rows = []
        fold_artifacts = []

        for fold_spec in fold_specs:
            fold_idx = int(fold_spec['fold'])
            train_indices = list(map(int, fold_spec['train']))
            val_indices = list(map(int, fold_spec['val']))
            holdout_indices = list(map(int, fold_spec['holdout']))

            train_df = self.subset_rows(baseline_df, train_indices)
            val_df = self.subset_rows(baseline_df, val_indices)
            holdout_df = self.subset_rows(baseline_df, holdout_indices)

            propensity_featureizer = XGBTopTokenFeatureizer(self.config.XGB_TOP_K_TOKENS).fit(train_df['history_tokens'].tolist())
            propensity_pos_weight = xgb_propensity_pos_weight(
                train_df,
                disable_pos_weight=self.config.NUISANCE_DISABLE_PROPENSITY_POS_WEIGHT,
            )
            propensity_model, propensity_val_metrics, propensity_best_params = train_xgb_with_val_selection(
                train_df=train_df,
                val_df=val_df,
                featureizer=propensity_featureizer,
                label_col='treatment',
                include_treatment=False,
                n_estimators=self.config.XGB_N_ESTIMATORS,
                max_depth_options=self.config.XGB_MAX_DEPTH_OPTIONS,
                min_child_weight_options=self.config.XGB_MIN_CHILD_WEIGHT_OPTIONS,
                learning_rate=self.config.XGB_LEARNING_RATE,
                subsample=self.config.XGB_SUBSAMPLE,
                colsample_bytree=self.config.XGB_COLSAMPLE_BYTREE,
                reg_lambda=self.config.XGB_REG_LAMBDA,
                tree_method=self.config.XGB_TREE_METHOD,
                n_jobs=self.config.XGB_N_JOBS,
                seed=seed,
                pos_weight=propensity_pos_weight,
            )
            propensity_holdout_metrics = evaluate_xgb_classifier(propensity_model, propensity_featureizer.transform(holdout_df, include_treatment=False), holdout_df['treatment'].astype(int).to_numpy(), pos_weight=None)
            propensity_holdout_probs = propensity_model.predict_proba(
                propensity_featureizer.transform(holdout_df, include_treatment=False)
            )[:, 1]

            outcome_featureizer = XGBTopTokenFeatureizer(self.config.XGB_TOP_K_TOKENS).fit(train_df['history_tokens'].tolist())
            outcome_model, outcome_val_metrics, outcome_best_params = train_xgb_with_val_selection(
                train_df=train_df,
                val_df=val_df,
                featureizer=outcome_featureizer,
                label_col='outcome',
                include_treatment=True,
                n_estimators=self.config.XGB_N_ESTIMATORS,
                max_depth_options=self.config.XGB_MAX_DEPTH_OPTIONS,
                min_child_weight_options=self.config.XGB_MIN_CHILD_WEIGHT_OPTIONS,
                learning_rate=self.config.XGB_LEARNING_RATE,
                subsample=self.config.XGB_SUBSAMPLE,
                colsample_bytree=self.config.XGB_COLSAMPLE_BYTREE,
                reg_lambda=self.config.XGB_REG_LAMBDA,
                tree_method=self.config.XGB_TREE_METHOD,
                n_jobs=self.config.XGB_N_JOBS,
                seed=seed,
                pos_weight=None,
            )
            outcome_holdout_metrics = evaluate_xgb_classifier(outcome_model, outcome_featureizer.transform(holdout_df, include_treatment=True), holdout_df['outcome'].astype(int).to_numpy(), pos_weight=None)

            outcome_holdout_obs_df = holdout_df.copy()
            outcome_holdout_t0_df = holdout_df.copy()
            outcome_holdout_t0_df['treatment'] = 0
            outcome_holdout_t1_df = holdout_df.copy()
            outcome_holdout_t1_df['treatment'] = 1
            outcome_holdout_probs_obs = outcome_model.predict_proba(
                outcome_featureizer.transform(outcome_holdout_obs_df, include_treatment=True)
            )[:, 1]
            outcome_holdout_probs_t0 = outcome_model.predict_proba(
                outcome_featureizer.transform(outcome_holdout_t0_df, include_treatment=True)
            )[:, 1]
            outcome_holdout_probs_t1 = outcome_model.predict_proba(
                outcome_featureizer.transform(outcome_holdout_t1_df, include_treatment=True)
            )[:, 1]

            fold_prediction_frames.append(
                self.build_holdout_prediction_frame(
                    holdout_indices=holdout_indices,
                    holdout_df=holdout_df,
                    propensity_probs=propensity_holdout_probs,
                    outcome_probs_obs=outcome_holdout_probs_obs,
                    outcome_probs_t0=outcome_holdout_probs_t0,
                    outcome_probs_t1=outcome_holdout_probs_t1,
                    fold_idx=fold_idx,
                )
            )
            fold_metric_rows.extend([
                {
                    'fold': fold_idx,
                    'task': 'propensity',
                    'train_size': len(train_indices),
                    'val_size': len(val_indices),
                    'holdout_size': len(holdout_indices),
                    **{f'best_val/{key}': value for key, value in propensity_val_metrics.items()},
                    **{f'holdout/{key}': value for key, value in propensity_holdout_metrics.items()},
                    'best_params': str(propensity_best_params),
                },
                {
                    'fold': fold_idx,
                    'task': 'outcome',
                    'train_size': len(train_indices),
                    'val_size': len(val_indices),
                    'holdout_size': len(holdout_indices),
                    **{f'best_val/{key}': value for key, value in outcome_val_metrics.items()},
                    **{f'holdout/{key}': value for key, value in outcome_holdout_metrics.items()},
                    'best_params': str(outcome_best_params),
                },
            ])
            fold_artifacts.append({
                'fold': fold_idx,
                'propensity_best_params': propensity_best_params,
                'outcome_best_params': outcome_best_params,
            })

        prediction_df = pd.concat(fold_prediction_frames, ignore_index=True).sort_values('row_index').reset_index(drop=True)
        if prediction_df['row_index'].nunique() != len(baseline_df):
            raise RuntimeError(f'XGBoost cross-fitting for seed {seed} did not produce one holdout prediction per row.')
        prediction_df = build_pseudo_outcomes(prediction_df, clip_value=self.config.PROPENSITY_CLIP)
        fold_metrics_df = pd.DataFrame(fold_metric_rows).sort_values(['fold', 'task']).reset_index(drop=True)

        save_xgb_json(
            output_dir / 'metrics.json',
            {
                'seed': seed,
                'folds': fold_artifacts,
                'summary': {
                    'mean_propensity': float(prediction_df['propensity'].mean()),
                    'mean_individual_treatment_effect': float(prediction_df['individual_treatment_effect'].mean()),
                    'ipw_ate': float(prediction_df['ipw_pseudo_outcome'].mean()),
                    'aipw_ate': float(prediction_df['aipw_pseudo_outcome'].mean()),
                    'mean_abs_aipw_pseudo_outcome': float(prediction_df['aipw_pseudo_outcome'].abs().mean()),
                },
            },
        )
        prediction_df.to_parquet(output_dir / 'patient_nuisance_predictions.parquet', index=False)
        fold_metrics_df.to_parquet(output_dir / 'fold_metrics.parquet', index=False)
        return prediction_df, fold_metrics_df


    def cate_pseudo_outcome_label(self, pseudo_outcome_column: str) -> str:
        return pseudo_outcome_column.removesuffix('_pseudo_outcome')


    def cate_output_root_for_pseudo_outcome(self, pseudo_outcome_column: str) -> Path:
        output_root = self.config.CATE_ROOT / self.cate_pseudo_outcome_label(pseudo_outcome_column)
        output_root.mkdir(parents=True, exist_ok=True)
        return output_root


    def cate_seed_dir(self, seed: int, pseudo_outcome_column: str) -> Path:
        seed_dir = self.cate_output_root_for_pseudo_outcome(pseudo_outcome_column) / f'seed_{seed}'
        seed_dir.mkdir(parents=True, exist_ok=True)
        return seed_dir


    def prepare_dataset(self):
        self.config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
        if self.config.SAMPLES_CACHE_PATH.is_file():
            samples_df = pd.read_parquet(self.config.SAMPLES_CACHE_PATH).reset_index(drop=True)
            print(f'Loaded cached dataset: {self.config.SAMPLES_CACHE_PATH}')
        else:
            if self.config.OUTCOME_CODES is None or self.config.OUTCOME_HORIZON_DAYS is None:
                raise ValueError('Set the study outcome codes and horizon before building a new dataset.')
            self.require_files([self.config.VOCAB_PATH, self.config.TREATMENT_PKL, self.config.COMPARATOR_PKL, self.config.CONDITIONS_PKL, self.config.PERSONS_PKL])
            vocab = load_vocab(self.config.VOCAB_PATH)
            samples_df = create_treatment_effect_dataset_from_pickles(
                treatment_pkl=self.config.TREATMENT_PKL, comparator_pkl=self.config.COMPARATOR_PKL,
                conditions_pkl=self.config.CONDITIONS_PKL, persons_pkl=self.config.PERSONS_PKL, vocab=vocab,
                treatment_name=self.config.TREATMENT_NAME, comparator_name=self.config.COMPARATOR_NAME,
                outcome_horizon_days=self.config.OUTCOME_HORIZON_DAYS,
                min_history_events=self.config.MIN_HISTORY_EVENTS, max_history_events=self.config.MAX_HISTORY_EVENTS,
                use_vocab_plus_code=False, outcome_codes=self.config.OUTCOME_CODES,
            ).reset_index(drop=True)
            samples_df.insert(0, 'row_index', np.arange(len(samples_df), dtype=np.int64))
            samples_df.to_parquet(self.config.SAMPLES_CACHE_PATH, index=False)

        if self.config.TRAIN_TEST_SPLIT_PATH.is_file():
            split_payload = json.loads(self.config.TRAIN_TEST_SPLIT_PATH.read_text())
            print(f'Loaded saved split: {self.config.TRAIN_TEST_SPLIT_PATH}')
        else:
            labels = (samples_df['treatment'].astype(int) * 2 + samples_df['outcome'].astype(int)).tolist()
            splits = stratified_split_indices(labels, self.config.TOP_LEVEL_TRAIN_FRAC, 0.0, self.config.TOP_LEVEL_SPLIT_SEED)
            split_payload = {
                'train': list(map(int, splits.train)), 'test': list(map(int, splits.test)),
                'train_frac': self.config.TOP_LEVEL_TRAIN_FRAC, 'seed': self.config.TOP_LEVEL_SPLIT_SEED,
            }
            self.config.TRAIN_TEST_SPLIT_PATH.write_text(json.dumps(split_payload, indent=2), encoding='utf-8')

        self.development_samples_df = samples_df.iloc[split_payload['train']].reset_index(drop=True).copy()
        heldout_test_samples_df = samples_df.iloc[split_payload['test']].reset_index(drop=True).copy()
        print(pd.DataFrame([
            {'split': name, 'n': len(frame), 'treated_fraction': frame['treatment'].mean(), 'outcome_fraction': frame['outcome'].mean()}
            for name, frame in [('development', self.development_samples_df), ('reserved_test', heldout_test_samples_df)]
        ]))


    def train_pretrained_nuisance(self):
        self.config.NUISANCE_ROOT.mkdir(parents=True, exist_ok=True)

        for seed in self.config.SEEDS:
            output_dir = self.config.NUISANCE_ROOT / f'seed_{seed}'
            prediction_path = output_dir / 'patient_nuisance_predictions.parquet'
            if self.saved_predictions_exist(prediction_path, f'model seed {seed}'):
                print(f'Skipping nuisance seed {seed}; found {prediction_path}')
                continue

            self.require_files([self.config.TRANSFORMER_CHECKPOINT, self.config.VOCAB_PATH])
            cmd = [
                sys.executable,
                str(self.config.PROJECT_ROOT / 'src' / 'DR_Learner' / 'train_nuisance_models.py'),
                '--samples-path', str(self.config.SAMPLES_CACHE_PATH),
                '--split-indices-path', str(self.config.TRAIN_TEST_SPLIT_PATH),
                '--transformer-checkpoint', str(self.config.TRANSFORMER_CHECKPOINT),
                '--vocab-path', str(self.config.VOCAB_PATH),
                '--output-dir', str(output_dir),
                '--batch-size', str(self.config.BATCH_SIZE),
                '--epochs', str(self.config.NUISANCE_EPOCHS),
                '--transformer-lr', str(self.config.NUISANCE_MAX_TRANSFORMER_LR),
                '--head-lr', str(self.config.NUISANCE_MAX_HEAD_LR),
                '--min-transformer-lr', str(self.config.NUISANCE_MIN_TRANSFORMER_LR),
                '--max-transformer-lr', str(self.config.NUISANCE_MAX_TRANSFORMER_LR),
                '--min-head-lr', str(self.config.NUISANCE_MIN_HEAD_LR),
                '--max-head-lr', str(self.config.NUISANCE_MAX_HEAD_LR),
                '--weight-decay', str(self.config.NUISANCE_WEIGHT_DECAY),
                '--num-folds', str(self.config.NUISANCE_NUM_FOLDS),
                '--fold-val-frac', str(self.config.NUISANCE_FOLD_VAL_FRAC),
                '--outcome-loss-weight-start', str(self.config.NUISANCE_OUTCOME_LOSS_WEIGHT_START),
                '--outcome-loss-weight-end', str(self.config.NUISANCE_OUTCOME_LOSS_WEIGHT_END),
                '--propensity-clip', str(self.config.PROPENSITY_CLIP),
                '--nhead', str(self.config.NHEAD),
                '--dropout', str(self.config.DROPOUT),
                '--head-dropout', str(self.config.NUISANCE_HEAD_DROPOUT),
                '--num-workers', str(self.config.NUM_WORKERS),
                '--device', self.config.DEVICE,
                '--seed', str(seed),
                '--wandb-project', 'nuisance-models-experiment',
            ]

            if self.config.NUISANCE_DISABLE_PROPENSITY_POS_WEIGHT:
                cmd.append('--disable-propensity-pos-weight')
            if self.config.NUISANCE_FREEZE_TRANSFORMER:
                cmd.append('--freeze-transformer')


            self.run_command(cmd, workdir=self.config.PROJECT_ROOT)


    def train_scratch_nuisance(self):
        self.config.SCRATCH_TRANSFORMER_NUISANCE_ROOT.mkdir(parents=True, exist_ok=True)


        for seed in self.config.SEEDS:
            output_dir = self.config.SCRATCH_TRANSFORMER_NUISANCE_ROOT / f'seed_{seed}'
            prediction_path = output_dir / 'patient_nuisance_predictions.parquet'
            if self.saved_predictions_exist(prediction_path, f'model seed {seed}'):
                print(f'Skipping scratch transformer nuisance seed {seed}; found {prediction_path}')
                continue

            self.require_files([self.config.VOCAB_PATH])
            cmd = [
                sys.executable,
                str(self.config.PROJECT_ROOT / 'src' / 'DR_Learner' / 'train_nuisance_models.py'),
                '--samples-path', str(self.config.SAMPLES_CACHE_PATH),
                '--split-indices-path', str(self.config.TRAIN_TEST_SPLIT_PATH),
                '--random-init-transformer',
                '--vocab-path', str(self.config.VOCAB_PATH),
                '--output-dir', str(output_dir),
                '--batch-size', str(self.config.BATCH_SIZE),
                '--epochs', str(self.config.SCRATCH_NUISANCE_EPOCHS),
                '--transformer-lr', str(self.config.NUISANCE_MAX_TRANSFORMER_LR),
                '--head-lr', str(self.config.NUISANCE_MAX_HEAD_LR),
                '--min-transformer-lr', str(self.config.NUISANCE_MIN_TRANSFORMER_LR),
                '--max-transformer-lr', str(self.config.NUISANCE_MAX_TRANSFORMER_LR),
                '--min-head-lr', str(self.config.NUISANCE_MIN_HEAD_LR),
                '--max-head-lr', str(self.config.NUISANCE_MAX_HEAD_LR),
                '--weight-decay', str(self.config.NUISANCE_WEIGHT_DECAY),
                '--num-folds', str(self.config.NUISANCE_NUM_FOLDS),
                '--fold-val-frac', str(self.config.NUISANCE_FOLD_VAL_FRAC),
                '--outcome-loss-weight-start', str(self.config.NUISANCE_OUTCOME_LOSS_WEIGHT_START),
                '--outcome-loss-weight-end', str(self.config.NUISANCE_OUTCOME_LOSS_WEIGHT_END),
                '--propensity-clip', str(self.config.PROPENSITY_CLIP),
                '--d-model', str(self.config.SCRATCH_TRANSFORMER_D_MODEL),
                '--nhead', str(self.config.NHEAD),
                '--num-layers', str(self.config.SCRATCH_TRANSFORMER_NUM_LAYERS),
                '--ff-mult', str(self.config.SCRATCH_TRANSFORMER_FF_MULT),
                '--dropout', str(self.config.DROPOUT),
                '--head-dropout', str(self.config.NUISANCE_HEAD_DROPOUT),
                '--num-workers', str(self.config.NUM_WORKERS),
                '--device', self.config.DEVICE,
                '--seed', str(seed),
                '--skip-pseudo-outcomes',
                '--wandb-project', 'nuisance-scratch-transformer-experiment',
            ]

            if self.config.NUISANCE_DISABLE_PROPENSITY_POS_WEIGHT:
                cmd.append('--disable-propensity-pos-weight')
            if self.config.NUISANCE_FREEZE_TRANSFORMER:
                cmd.append('--freeze-transformer')


            self.run_command(cmd, workdir=self.config.PROJECT_ROOT)


    def train_cate(self):
        self.config.CATE_ROOT.mkdir(parents=True, exist_ok=True)

        for pseudo_outcome_column in self.config.PSEUDO_OUTCOME_COLUMNS:
            pseudo_outcome_label = self.cate_pseudo_outcome_label(pseudo_outcome_column)
            for seed in self.config.SEEDS:
                nuisance_prediction_path = self.config.NUISANCE_ROOT / f'seed_{seed}' / 'patient_nuisance_predictions.parquet'
                output_dir = self.cate_seed_dir(seed, pseudo_outcome_column)
                prediction_path = output_dir / 'patient_cate_predictions.parquet'
                if self.saved_predictions_exist(prediction_path, f'model seed {seed}'):
                    print(
                        f'Skipping CATE {pseudo_outcome_label.upper()} seed {seed}; '
                        f'found {prediction_path}'
                    )
                    continue

                self.require_files([self.config.TRANSFORMER_CHECKPOINT, self.config.VOCAB_PATH])
                cmd = [
                    sys.executable,
                    str(self.config.PROJECT_ROOT / 'src' / 'DR_Learner' / 'train_pseudo_outcome_cate_model.py'),
                    '--samples-path', str(nuisance_prediction_path),
                    '--split-indices-path', str(self.config.TRAIN_TEST_SPLIT_PATH),
                    '--pseudo-outcome-column', pseudo_outcome_column,
                    '--transformer-checkpoint', str(self.config.TRANSFORMER_CHECKPOINT),
                    '--vocab-path', str(self.config.VOCAB_PATH),
                    '--output-dir', str(output_dir),
                    '--batch-size', str(self.config.BATCH_SIZE),
                    '--epochs', str(self.config.CATE_EPOCHS),
                    '--lr', str(self.config.CATE_LR),
                    '--transformer-lr', str(self.config.CATE_TRANSFORMER_LR),
                    '--weight-decay', str(self.config.CATE_WEIGHT_DECAY),
                    '--val-frac', str(self.config.CATE_VAL_FRAC),
                    '--loss', self.config.CATE_LOSS,
                    '--nhead', str(self.config.NHEAD),
                    '--dropout', str(self.config.CATE_DROPOUT),
                    '--num-workers', str(self.config.NUM_WORKERS),
                    '--device', self.config.DEVICE,
                    '--seed', str(seed),
                    '--wandb-project', 'cate-learner-experiment',
                ]

                if self.config.CATE_FREEZE_TRANSFORMER:
                    cmd.append('--freeze-transformer')


                self.run_command(cmd, workdir=self.config.PROJECT_ROOT)


    def train_baselines(self):
        for output_root, trainer in (
            (self.config.RF_OUTPUT_DIR, self.train_rf_crossfit_baseline),
            (self.config.XGB_OUTPUT_DIR, self.train_xgb_crossfit_baseline),
            (self.config.AGE_SEX_OUTPUT_DIR, self.train_age_sex_crossfit_baseline),
        ):
            for seed in self.config.SEEDS:
                output_dir = output_root / f'seed_{seed}'
                output_dir.mkdir(parents=True, exist_ok=True)
                prediction_path = output_dir / 'patient_nuisance_predictions.parquet'
                if not self.saved_predictions_exist(prediction_path, f'{output_root.name} seed {seed}'):
                    trainer(seed, output_dir)
        self.save_mean_predictor()

    def save_mean_predictor(self):
        """Save training-fold prevalence predictions for the analysis reference."""
        for seed in self.config.SEEDS:
            output_dir = self.config.BASELINE_OUTPUT_ROOT / 'mean_predictor' / f'seed_{seed}'
            output_dir.mkdir(parents=True, exist_ok=True)
            prediction_path = output_dir / 'patient_nuisance_predictions.parquet'
            if self.saved_predictions_exist(prediction_path, f'mean predictor seed {seed}'):
                continue
            predictions = self.development_samples_df[
                ['row_index', 'person_id', 'treatment', 'outcome']
            ].copy()
            predictions['propensity'] = np.nan
            predictions['outcome_probability_observed_treatment'] = np.nan
            for fold in self.load_crossfit_fold_specs(seed):
                train = self.development_samples_df.iloc[fold['train']]
                predictions.loc[fold['holdout'], 'propensity'] = float(train['treatment'].mean())
                predictions.loc[fold['holdout'], 'outcome_probability_observed_treatment'] = float(train['outcome'].mean())
            if predictions.isna().any().any():
                raise RuntimeError('Mean-predictor folds did not cover the development population.')
            predictions.to_parquet(prediction_path, index=False)



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=EXPERIMENT_CHOICES, default='glp1_vs_dpp4')
    parser.add_argument('--seeds', type=int, nargs='+', default=[17, 24, 51, 103, 204])
    parser.add_argument('--output-dir', type=Path, help='New directory for an independent experiment; defaults to results/<experiment>/.')
    parser.add_argument('--checkpoint-dir', type=Path, help='Directory containing best.pt and vocab.json.')
    parser.add_argument('--device', default=None, help='Default: cuda when available, otherwise cpu.')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--wandb-mode', choices=('disabled', 'offline', 'online'), default='disabled')
    parser.add_argument('--stages', nargs='+', choices=('dataset', 'nuisance', 'scratch', 'baselines', 'cate'), default=['dataset', 'nuisance', 'scratch', 'baselines', 'cate'], help='Stages execute in dependency order. Later stages require saved outputs from earlier stages.')
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error('Batch size must be positive and num workers nonnegative.')
    config = build_config(experiment=args.experiment, seeds=args.seeds,
                          output_dir=args.output_dir, checkpoint_dir=args.checkpoint_dir,
                          device=args.device or ('cuda' if torch.cuda.is_available() else 'cpu'),
                          batch_size=args.batch_size, num_workers=args.num_workers,
                          wandb_mode=args.wandb_mode)
    config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Experiment: {config.EXPERIMENT}; seeds: {config.SEEDS}; device: {config.DEVICE}')
    print(f'Results: {config.RESULT_DIR}')
    runner = ExperimentRunner(config)
    # Baselines require development samples even when preparation is omitted.
    if 'dataset' in args.stages or 'baselines' in args.stages:
        if 'dataset' not in args.stages:
            runner.require_files([config.SAMPLES_CACHE_PATH, config.TRAIN_TEST_SPLIT_PATH])
        runner.prepare_dataset()
    for name, run in (('nuisance', runner.train_pretrained_nuisance),
                      ('scratch', runner.train_scratch_nuisance),
                      ('baselines', runner.train_baselines), ('cate', runner.train_cate)):
        if name in args.stages:
            run()
    print('Selected experiment stages completed. Open Treatment_effect_experiments.ipynb for analysis.')


if __name__ == '__main__':
    main()
