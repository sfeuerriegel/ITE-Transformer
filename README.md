[![CI](https://github.com/sfeuerriegel/ITE-Transformer/actions/workflows/ci.yml/badge.svg)](https://github.com/sfeuerriegel/ITE-Transformer/actions/workflows/ci.yml)
# ITE-Transformer

ITE-Transformer combines self-supervised transformer pretraining on longitudinal electronic health records (EHRs) with doubly robust causal machine learning to predict individualized treatment effects (ITEs). The predicted treatment effect are a conditional risk difference: outcome risk under treatment minus outcome risk under the comparator. For adverse outcomes, negative scores indicate lower estimated risk under treatment; positive scores indicate lower estimated risk under the comparator.

The repository includes All of Us cohort extraction notebooks, transformer pretraining, treatment-effect dataset preparation, nuisance and CATE learners, predictive baselines, and experimental analyses. Executing cohort queries is only possible in an authorized All of Us environment.

**A pretrained checkpoint and its matching vocabulary are provided in `checkpoints/`. You can skip transformer pretraining and run the downstream experiments using this pair.** Authorized access to the cohort data and downstream model fitting are still required; the pretrained checkpoint is not a fitted treatment-effect model.

All of Us participant-level data are restricted and are not distributed with this repository.

## Installation

Use Python 3.10 or newer. Run commands from the repository root. Notebooks can run from the root or their own subdirectories; they locate the repository automatically.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Weights & Biases logging is disabled by default. To enable pretraining logging, set `WANDB_MODE=offline` or `WANDB_MODE=online`. For downstream runs, pass `--wandb-mode offline` or `--wandb-mode online` to the runnable.

## Workflow

1. Extract demographics and clinical events.
2. Prepare the longitudinal event table.
3. Construct the downstream treatment cohort.
4. Load the supplied pretrained checkpoint, or optionally train a new transformer.
5. Cross-fit nuisance models, construct AIPW pseudo-outcomes, and train the CATE learner.
6. Generate predictive diagnostics.

### Repository guide

| File or directory | Purpose |
|---|---|
| `notebooks/cohort_creation/Pretraining_cohort.ipynb` | Extract demographics, diagnoses, and medications and merge clinical events. |
| `notebooks/cohort_creation/T2D_cohort.ipynb` | Construct the GLP-1 receptor agonist versus DPP-4 inhibitor cohort. |
| `notebooks/cohort_creation/Pan_cancer_cohort.ipynb` | Construct the immunotherapy versus chemotherapy cohort. |
| `notebooks/analysis/Pre_training.ipynb` | Visualise pretrained token embeddings. |
| `runnables/run_pretraining.py` | Launch transformer pretraining. |
| `notebooks/analysis/Treatment_effect_experiments.ipynb` | Load saved downstream results, aggregate metrics, and generate diagnostics and plots. |
| `runnables/run_treatment_effect_experiments.py` | Prepare datasets and run downstream models and baselines. |
| `runnables/treatment_effect_config.py` | Shared study definitions, model defaults, and output paths. |
| `src/model.py` | Transformer with token embeddings and age encoding. |
| `src/train.py` | Pretraining |
| `src/data_prep/build_sequences.py` | Vocabulary construction and sequence preparation. |
| `src/DR_Learner/prepare_treatment_effect_dataset.py` | Build indexed samples with pre-treatment histories and post-index outcomes. |
| `src/DR_Learner/train_nuisance_models.py` | Transformer propensity/outcome models. |
| `src/DR_Learner/train_pseudo_outcome_cate_model.py` | Transformer CATE regression on pseudo-outcomes. |
| `src/DR_Learner/train_linear_cate_model.py` | Linear CATE learner. |
| `src/DR_Learner/random_forest.py`, `src/DR_Learner/xgboost_baseline.py` | Tabular nuisance-models baselines. |
| `src/DR_Learner/train_age_sex_baseline.py` | Age/sex nuisance-models baseline. |
| `checkpoints/` | Supplied pretrained model and matching vocabulary. |

## Run the downstream experiments with the supplied checkpoint

### 1. Extract and prepare the shared data

Run `notebooks/cohort_creation/Pretraining_cohort.ipynb` sequentially. It writes:

- `DATA/shared/patients.pkl`: demographics;
- `DATA/shared/conditions.pkl`: diagnosis events;
- `DATA/shared/drugs.pkl`: medication events;
- `DATA/shared/events.pkl`: merged diagnosis and medication events.

If these prepared tables already exist in your authorized environment, proceed directly to cohort construction.

### 2. Construct a treatment cohort

Run `notebooks/cohort_creation/T2D_cohort.ipynb`, `notebooks/cohort_creation/Pan_cancer_cohort.ipynb`, or both. They use the shared demographic and event tables and export:

| Cohort | Output directory | Files |
|---|---|---|
| Type 2 diabetes | `DATA/cohorts/T2D/` | `glp1_drugs_df.pkl`, `dpp4_drugs_df.pkl`, `events.pkl` |
| Pan-cancer | `DATA/cohorts/PAN_CANCER/` | `immunotherapy_df.pkl`, `chemotherapy_df.pkl`, `events.pkl` |

The pan-cancer event export includes a `DEATH` token for mortality outcome construction. Downstream dataset preparation selects the pre-index history separately from the post-index outcome window.

### 3. Configure and run the experiment

Run from the repository root:

```sh
python runnables/run_treatment_effect_experiments.py --experiment glp1_vs_dpp4
# Alternatively:
python runnables/run_treatment_effect_experiments.py --experiment immunotherapy_vs_chemotherapy
```

The runnable uses the supplied `checkpoints/best.pt` and `checkpoints/vocab.json` by default. Keep the pair together: token IDs must match the vocabulary used during pretraining. Use `--checkpoint-dir` to select another pair.

The current experiment definitions are:

| Experiment | Treatment (`A=1`) | Comparator (`A=0`) | Outcome | Horizon |
|---|---|---|---|---|
| `glp1_vs_dpp4` | GLP-1 receptor agonist | DPP-4 inhibitor | Cardiovascular diagnosis proxy using I21, I22, I23, and I50 ICD codes | 1,095 days |
| `immunotherapy_vs_chemotherapy` | Immunotherapy | Chemotherapy | Recorded death (`DEATH`) | 1,095 days |

Both experiments use `DATA/shared/patients.pkl`. `runnables/treatment_effect_config.py` specifies the exact outcome-token strings, seeds, clipping threshold, model settings, and output paths. Current defaults use seeds `(17, 24, 51, 103, 204)`, five nuisance cross-fitting folds, and up to 128 clinical history events.

Command-line options include `--seeds 17 24`, `--device cpu`, `--batch-size 64`, `--num-workers 0`, and `--output-dir results/my_run`. Use `--help` for all options. `--stages` selects stages such as `dataset`, `nuisance`, `scratch`, `baselines`, and `cate`; selected stages execute in dependency order and require saved inputs from any omitted earlier stages.

**Cache behavior:** existing samples, splits, and model predictions are reused without revalidating settings. Select a new `--output-dir` when changing data, checkpoint, vocabulary, outcome definitions, or model settings.

### 4. Analyse saved results

Open `notebooks/analysis/Treatment_effect_experiments.ipynb` and set `EXPERIMENT`, `SEEDS`, and optionally `OUTPUT_DIR` to match the runnable. Execute the cells in order to aggregate results, compare predictive performance, and generate calibration and overlap figures.

## Optional: train a new transformer

```sh
python runnables/run_pretraining.py
```

This entry point uses `DATA/shared/patients.pkl` and `DATA/shared/events.pkl`, runs four GPU workers for up to 300 epochs, and saves to `new_checkpoints/`, preserving the former notebook defaults. Use `--processes 1` to run directly on CPU or GPU. Override input paths with `--person-path` and `--conditions-path` (pickle, Parquet, or CSV). All training options are available through `--help`.

Early stopping in `src/train.py` monitors validation loss and stops after 20 consecutive epochs without improvement. Set `--early-stopping-patience 0` to disable it, or change the patience and minimum loss decrease:

```sh
python runnables/run_pretraining.py --processes 1 --num-epochs 50 \
  --early-stopping-patience 10 --early-stopping-min-delta 0.001 \
  --output-dir my_checkpoints
```

Patience resets when validation loss decreases by more than the minimum delta; distributed workers use the same averaged validation loss and stop together. `--num-epochs` remains the upper limit. `best.pt` always stores the lowest validation loss, even when a smaller improvement does not reset patience; `last.pt` stores the final completed epoch.

Training saves `best.pt`, `last.pt`, and `vocab.json`. Pass `--checkpoint-dir new_checkpoints` to the downstream runnable to use that pair, and ensure `NHEAD` in the shared experiment configuration matches the model. `notebooks/analysis/Pre_training.ipynb` is now exclusively for embedding analysis and visualisation; set its `CHECKPOINT_DIR` to inspect a newly trained model.

## Data representation

| Field | Description |
|---|---|
| `person_id` | Participant identifier |
| `timestamp` | Event timestamp |
| `token_col` | Clinical code followed by its concept name |
| `type` | Event category, such as condition, drug, or death |

The supplied extraction code forms diagnosis tokens at ICD level 3 and medication tokens at ATC level 3 (`atc3_code` and `atc3_name`). Match token construction to the released vocabulary; changing granularity requires a compatible vocabulary and model.

Demographics include `person_id`, `date_of_birth`, and sex-related fields. Sequence preparation adds a sex token and aligns clinical tokens with age values. Treatment-effect samples use pre-index history and a binary outcome in the configured post-index window.

Do not distribute restricted participant histories, intermediate patient-level datasets, or patient-level predictions.

## Synthetic patients

`Synthetic patients/` contains four synthetic histories for the manuscript examples, with all events preceding treatment index age. DM01/DM02 compare a type 2 diabetes reference profile with added kidney disease (N18/N17); ONC01/ONC02 compare a lung cancer/COPD reference profile with added interstitial lung disease (J84) and corticosteroid exposure (H02A). Each paired variant adds two events.

The histories are provided as `clinical_trajectories.jsonl` (one patient per line) and `clinical_trajectories_events.csv` (one event per row, with zero-based event indices). Ages are in years. These are generated examples, not real patient records.

You can **run inference on these patients using the supplied fitted models**, without accessing All of Us data or fitting models yourself:

1. After installation, open `notebooks/analysis/Treatment_effect_experiments.ipynb`.
2. Run the imports and experiment-configuration cells in sections 0–1. Set `EXPERIMENT` to `glp1_vs_dpp4` for DM01/DM02 or `immunotherapy_vs_chemotherapy` for ONC01/ONC02.
3. Jump directly to section 7, **Synthetic patient trajectories — saved-model inference**, and run its cells. The intervening analysis sections are not required for this example.
