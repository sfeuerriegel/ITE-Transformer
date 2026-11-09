"""Shared study definitions and defaults for experiments and result analysis."""
from pathlib import Path
from types import SimpleNamespace

from src.project_paths import PROJECT_ROOT, SHARED_DATA_DIR, COHORT_DATA_DIR, RESULTS_ROOT

EXPERIMENT_CHOICES = ('glp1_vs_dpp4', 'immunotherapy_vs_chemotherapy')


def build_config(experiment='glp1_vs_dpp4', seeds=(17, 24, 51, 103, 204),
                 output_dir=None, checkpoint_dir=None, device=None, batch_size=64,
                 num_workers=0, wandb_mode='disabled'):
    if experiment not in EXPERIMENT_CHOICES:
        raise ValueError(f'Unknown experiment: {experiment}')
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Provide at least one seed, with no duplicates.')
    EXPERIMENT = experiment
    SEEDS = tuple(seeds)
    REFERENCE_SEED = SEEDS[0]
    WANDB_MODE = wandb_mode  # 'offline' or 'online' can be selected explicitly.
    DEVICE = device or 'cpu'
    BATCH_SIZE = batch_size
    NUM_WORKERS = num_workers

    CHECKPOINT_PATH = Path(checkpoint_dir).resolve() if checkpoint_dir else PROJECT_ROOT / 'checkpoints'
    TRANSFORMER_CHECKPOINT = CHECKPOINT_PATH / 'best.pt'
    VOCAB_PATH = CHECKPOINT_PATH / 'vocab.json'

    EXPERIMENTS = {
        'immunotherapy_vs_chemotherapy': {
            'treatment_name': 'immunotherapy', 'comparator_name': 'chemotherapy',
            'treatment_path': COHORT_DATA_DIR / 'PAN_CANCER' / 'immunotherapy_df.pkl',
            'comparator_path': COHORT_DATA_DIR / 'PAN_CANCER' / 'chemotherapy_df.pkl',
            'conditions_path': COHORT_DATA_DIR / 'PAN_CANCER' / 'events.pkl',
            'persons_path': SHARED_DATA_DIR / 'patients.pkl',
            'outcome_codes': ['DEATH'], 'outcome_horizon_days': 1095,
        },
        'glp1_vs_dpp4': {
            'treatment_name': 'glp1', 'comparator_name': 'dpp4',
            'treatment_path': COHORT_DATA_DIR / 'T2D' / 'glp1_drugs_df.pkl',
            'comparator_path': COHORT_DATA_DIR / 'T2D' / 'dpp4_drugs_df.pkl',
            'conditions_path': COHORT_DATA_DIR / 'T2D' / 'events.pkl',
            'persons_path': SHARED_DATA_DIR / 'patients.pkl',
            'outcome_codes': ["I21 Acute Myocardial Infarction",
                "I22 Subsequent ST elevation (STEMI) and non-ST elevation (NSTEMI) myocardial infarction",
                "I23 Certain current complications following ST elevation (STEMI) and non-ST elevation (NSTEMI) myocardial infarction (within the 28 day period)",
                "I50 Heart failure"], 'outcome_horizon_days': 1095,
        },
    }
    experiment = EXPERIMENTS[EXPERIMENT]
    TREATMENT_NAME = experiment['treatment_name']
    COMPARATOR_NAME = experiment['comparator_name']
    TREATMENT_PKL = PROJECT_ROOT / experiment['treatment_path']
    COMPARATOR_PKL = PROJECT_ROOT / experiment['comparator_path']
    CONDITIONS_PKL = PROJECT_ROOT / experiment['conditions_path']
    PERSONS_PKL = PROJECT_ROOT / experiment['persons_path']
    OUTCOME_CODES = experiment['outcome_codes']
    OUTCOME_HORIZON_DAYS = experiment['outcome_horizon_days']
    MIN_HISTORY_EVENTS = 1
    MAX_HISTORY_EVENTS = 128
    TOP_LEVEL_TRAIN_FRAC = 0.8
    TOP_LEVEL_SPLIT_SEED = 0

    RESULT_DIR = (Path(output_dir).resolve() if output_dir else
                        RESULTS_ROOT / EXPERIMENT)
    SAMPLES_CACHE_PATH = RESULT_DIR / 'treatment_effect_samples.parquet'
    TRAIN_TEST_SPLIT_PATH = RESULT_DIR / 'train_test_split_indices.json'
    FIGURE_DIR = RESULT_DIR / 'publication_figures_internal'
    TABLE_DIR = RESULT_DIR / 'publication_tables_internal'

    PSEUDO_OUTCOME_COLUMNS = ('ipw_pseudo_outcome', 'aipw_pseudo_outcome')
    PSEUDO_OUTCOME_COLUMN = 'aipw_pseudo_outcome'

    # Settings for new model fits. Existing saved results are not revalidated.
    NHEAD = 4
    DROPOUT = 0.3
    NUISANCE_HEAD_DROPOUT = 0.2
    NUM_WORKERS = num_workers
    NUISANCE_EPOCHS = 50
    NUISANCE_MAX_HEAD_LR = 0.0001
    NUISANCE_MIN_HEAD_LR = 0
    NUISANCE_MAX_TRANSFORMER_LR = 5e-06
    NUISANCE_MIN_TRANSFORMER_LR = 0
    NUISANCE_WEIGHT_DECAY = 0.1
    NUISANCE_NUM_FOLDS = 5
    NUISANCE_FOLD_VAL_FRAC = 0.15
    NUISANCE_OUTCOME_LOSS_WEIGHT_START = 1
    NUISANCE_OUTCOME_LOSS_WEIGHT_END = 1
    NUISANCE_DISABLE_PROPENSITY_POS_WEIGHT = True
    NUISANCE_FREEZE_TRANSFORMER = False
    PROPENSITY_CLIP = 0.1
    SCRATCH_TRANSFORMER_MODEL_LABEL = 'Scratch transformer'
    SCRATCH_TRANSFORMER_D_MODEL = 128
    SCRATCH_TRANSFORMER_NUM_LAYERS = 6
    SCRATCH_TRANSFORMER_FF_MULT = 6
    SCRATCH_NUISANCE_EPOCHS = 60
    RF_TOP_K_TOKENS = 512
    RF_N_ESTIMATORS = 500
    RF_MAX_DEPTH_OPTIONS = [8, 16, 24]
    RF_MIN_SAMPLES_LEAF_OPTIONS = [1, 10, 25]
    RF_MAX_FEATURES = 'sqrt'
    RF_N_JOBS = -1
    XGB_TOP_K_TOKENS = 512
    XGB_N_ESTIMATORS = 500
    XGB_MAX_DEPTH_OPTIONS = [3, 6, 9]
    XGB_MIN_CHILD_WEIGHT_OPTIONS = [5.0, 8.0]
    XGB_LEARNING_RATE = 0.05
    XGB_SUBSAMPLE = 0.8
    XGB_COLSAMPLE_BYTREE = 0.8
    XGB_REG_LAMBDA = 1.0
    XGB_TREE_METHOD = 'hist'
    XGB_N_JOBS = -1
    CATE_EPOCHS = 30
    CATE_LR = 5e-05
    CATE_TRANSFORMER_LR = 5e-06
    CATE_WEIGHT_DECAY = 0.1
    CATE_DROPOUT = 0.3
    CATE_VAL_FRAC = 0.15
    CATE_LOSS = 'huber'
    CATE_FREEZE_TRANSFORMER = True
    IN_FOLD_EFFECT_BIN_COUNTS = [4]
    IN_FOLD_CATE_PSEUDO_OUTCOME_COLUMN = 'aipw_pseudo_outcome'

    NUISANCE_ROOT = RESULT_DIR / 'nuisance_multiseed'
    SCRATCH_TRANSFORMER_NUISANCE_ROOT = RESULT_DIR / 'nuisance_scratch_transformer_multiseed'
    BASELINE_OUTPUT_ROOT = RESULT_DIR / 'nuisance_baseline_multiseed'
    RF_OUTPUT_DIR = BASELINE_OUTPUT_ROOT / 'random_forest'
    XGB_OUTPUT_DIR = BASELINE_OUTPUT_ROOT / 'xgboost'
    AGE_SEX_OUTPUT_DIR = BASELINE_OUTPUT_ROOT / 'age_sex'
    CATE_ROOT = RESULT_DIR / 'cate_multiseed'
    return SimpleNamespace(**{key: value for key, value in locals().items() if key.isupper()})
