from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset

from torch.nn.utils.rnn import pad_sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_prep.build_sequences import FEMALE_TOKEN, MALE_TOKEN, Vocab  # noqa: E402


PERSON_ID_COL = "person_id"
CONDITION_DT_COL = "condition_start_datetime"
DRUG_DT_COL = "drug_exposure_start_datetime"
DOB_COL = "date_of_birth"
GENDER_COL = "gender"
CODE_COL = "standard_concept_code"
VOCAB_COL = "standard_vocabulary"
NAME_COL = "standard_concept_name"
DEFAULT_TOKEN_COL = "token_col"
DEFAULT_TIMESTAMP_COL = "timestamp"


def load_pickle_dataframe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    with path.open("rb") as fh:
        obj = pickle.load(fh)
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"{path} did not contain a pandas DataFrame.")
    return obj


def normalize_codes(codes: Sequence[str] | None = None, file_path: str | Path | None = None) -> set[str]:
    normalized: list[str] = []
    if codes is not None:
        normalized.extend(str(code).strip() for code in codes)
    if file_path is not None:
        file_codes = Path(file_path).read_text(encoding="utf-8").splitlines()
        normalized.extend(line.strip() for line in file_codes)
    return {code for code in normalized if code}


def resolve_column_name(
    df: pd.DataFrame,
    candidates: Sequence[str],
    *,
    dataframe_name: str,
    field_name: str,
) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise KeyError(
        f"Missing required {field_name} column in {dataframe_name}. "
        f"Tried: {list(dict.fromkeys(candidates))}"
    )


def standardize_event_columns(
    df: pd.DataFrame,
    *,
    dataframe_name: str,
    token_candidates: Sequence[str],
    timestamp_candidates: Sequence[str],
    include_vocab_and_name: bool = False,
) -> pd.DataFrame:
    rename_map = {
        resolve_column_name(
            df,
            [PERSON_ID_COL],
            dataframe_name=dataframe_name,
            field_name="person id",
        ): PERSON_ID_COL,
        resolve_column_name(
            df,
            token_candidates,
            dataframe_name=dataframe_name,
            field_name="token",
        ): CODE_COL,
        resolve_column_name(
            df,
            timestamp_candidates,
            dataframe_name=dataframe_name,
            field_name="timestamp",
        ): DRUG_DT_COL if include_vocab_and_name else CONDITION_DT_COL,
    }

    standardized = df.rename(columns=rename_map).copy()
    if include_vocab_and_name:
        if VOCAB_COL not in standardized.columns:
            standardized[VOCAB_COL] = pd.NA
        if NAME_COL not in standardized.columns:
            standardized[NAME_COL] = pd.NA
    return standardized


def build_gender_map(person_df: pd.DataFrame, vocab: Vocab) -> dict[int, int]:
    # Match the existing sequence builder: prepend a learned male/female token when available.
    people = person_df[[PERSON_ID_COL, GENDER_COL]].copy()
    people = people.dropna(subset=[PERSON_ID_COL])
    people[PERSON_ID_COL] = people[PERSON_ID_COL].astype("int64")

    gender_series = people.set_index(PERSON_ID_COL)[GENDER_COL].astype("string", copy=False)
    gender_lower = gender_series.str.strip().str.lower().fillna("")

    gender_tokens = pd.Series(vocab.unk_idx, index=gender_series.index, dtype="int64")
    gender_tokens[gender_lower.str.startswith("m")] = vocab.stoi.get(MALE_TOKEN, vocab.unk_idx)
    gender_tokens[gender_lower.str.startswith("f")] = vocab.stoi.get(FEMALE_TOKEN, vocab.unk_idx)
    return gender_tokens.to_dict()


def make_history_token_series(
    conditions_df: pd.DataFrame,
    token_col: str,
    use_vocab_plus_code: bool,
    vocab_col: str | None,
) -> pd.Series:
    if use_vocab_plus_code:
        if vocab_col is None:
            raise ValueError("vocab_col must be provided when use_vocab_plus_code=True.")
        return (
            conditions_df[vocab_col].astype("string").fillna("")
            + "|"
            + conditions_df[token_col].astype("string").fillna("")
        )
    return conditions_df[token_col].astype("string").fillna("")


def earliest_exposure_per_person(drug_df: pd.DataFrame, cohort_name: str) -> pd.DataFrame:
    # Each person contributes a single candidate index event per cohort: the earliest observed exposure.
    cols = [PERSON_ID_COL, DRUG_DT_COL, CODE_COL, VOCAB_COL, NAME_COL]
    missing_cols = [col for col in cols if col not in drug_df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in drug dataframe: {missing_cols}")

    cohort = drug_df[cols].copy()
    cohort[PERSON_ID_COL] = pd.to_numeric(cohort[PERSON_ID_COL], errors="coerce")
    cohort[DRUG_DT_COL] = pd.to_datetime(cohort[DRUG_DT_COL], utc=True, errors="coerce")
    cohort = cohort.dropna(subset=[PERSON_ID_COL, DRUG_DT_COL])
    cohort[PERSON_ID_COL] = cohort[PERSON_ID_COL].astype("int64")
    cohort = cohort.sort_values([PERSON_ID_COL, DRUG_DT_COL], kind="mergesort")
    cohort = cohort.groupby(PERSON_ID_COL, sort=False, as_index=False).first()
    cohort["cohort_name"] = cohort_name
    return cohort


def select_index_events(
    treatment_df: pd.DataFrame,
    comparator_df: pd.DataFrame,
    treatment_name: str,
    comparator_name: str,
) -> pd.DataFrame:
    treat = earliest_exposure_per_person(treatment_df, treatment_name).rename(
        columns={
            DRUG_DT_COL: "treatment_start_datetime",
            CODE_COL: "treatment_code",
            VOCAB_COL: "treatment_vocabulary",
            NAME_COL: "treatment_concept_name",
        }
    )
    comp = earliest_exposure_per_person(comparator_df, comparator_name).rename(
        columns={
            DRUG_DT_COL: "comparator_start_datetime",
            CODE_COL: "comparator_code",
            VOCAB_COL: "comparator_vocabulary",
            NAME_COL: "comparator_concept_name",
        }
    )

    combined = treat.merge(comp, on=PERSON_ID_COL, how="outer")
    # If a person appears in both cohorts, define the index by whichever exposure happened first.
    treat_dt = combined["treatment_start_datetime"]
    comp_dt = combined["comparator_start_datetime"]

    use_treatment = comp_dt.isna() | (treat_dt.notna() & (treat_dt <= comp_dt))
    index = pd.DataFrame({PERSON_ID_COL: combined[PERSON_ID_COL].astype("int64")})
    index["treatment"] = use_treatment.astype("int8")
    index["treatment_name"] = treatment_name
    index["comparator_name"] = comparator_name
    index["index_datetime"] = treat_dt.where(use_treatment, comp_dt)
    index["index_code"] = combined["treatment_code"].where(use_treatment, combined["comparator_code"])
    index["index_vocabulary"] = combined["treatment_vocabulary"].where(
        use_treatment,
        combined["comparator_vocabulary"],
    )
    index["index_concept_name"] = combined["treatment_concept_name"].where(
        use_treatment,
        combined["comparator_concept_name"],
    )
    index["had_both_cohorts"] = (treat_dt.notna() & comp_dt.notna()).astype("int8")
    index = index.dropna(subset=["index_datetime"])
    return index.sort_values(PERSON_ID_COL, kind="mergesort").reset_index(drop=True)


def compute_age_years(event_dt: pd.Series, dob: pd.Timestamp | pd.NaT) -> pd.Series:
    if pd.isna(dob):
        return pd.Series(0.0, index=event_dt.index, dtype="float32")
    delta_days = (event_dt - dob).dt.total_seconds() / (3600.0 * 24.0)
    return (delta_days / 365.25).fillna(0.0).clip(lower=0.0).astype("float32")


def iter_sample_rows(
    index_df: pd.DataFrame,
    conditions_df: pd.DataFrame,
    gender_map: dict[int, int],
    dob_map: pd.Series,
    vocab: Vocab,
    token_col: str,
    use_vocab_plus_code: bool,
    vocab_col: str | None,
    outcome_codes: set[str],
    future_flag_codes: set[str],
    min_history_events: int,
    max_history_events: int | None,
    outcome_horizon_days: int | None,
) -> Iterable[dict]:
    groupby = conditions_df.groupby(PERSON_ID_COL, sort=False)

    for row in index_df.itertuples(index=False):
        person_id = int(row.person_id)
        if person_id in groupby.indices:
            person_conditions = groupby.get_group(person_id).sort_values(CONDITION_DT_COL, kind="mergesort")
        else:
            person_conditions = conditions_df.iloc[0:0].copy()

        index_dt = row.index_datetime
        # Split the trajectory at treatment time: transformer input comes from pre-index history,
        # while labels are defined from condition codes that appear on or after the index.
        pre = person_conditions[person_conditions[CONDITION_DT_COL] < index_dt].copy()
        post = person_conditions[person_conditions[CONDITION_DT_COL] >= index_dt].copy()
        if outcome_horizon_days is not None:
            outcome_horizon_dt = index_dt + pd.Timedelta(days=outcome_horizon_days)
            post = post[post[CONDITION_DT_COL] <= outcome_horizon_dt].copy()

        if max_history_events is not None and len(pre) > max_history_events:
            pre = pre.iloc[-max_history_events:].copy()

        if len(pre) < min_history_events:
            continue

        history_token_strings = make_history_token_series(pre, token_col, use_vocab_plus_code, vocab_col)
        # Keep token ids and ages aligned so they can be passed directly into the transformer encoder.
        history_token_ids = history_token_strings.map(vocab.stoi).fillna(vocab.unk_idx).astype("int64").tolist()
        dob = dob_map.get(person_id, pd.NaT)
        history_ages = compute_age_years(pre[CONDITION_DT_COL], dob).tolist()
        age_at_index = float(compute_age_years(pd.Series([index_dt]), dob).iloc[0])

        gender_token = int(gender_map.get(person_id, vocab.unk_idx))
        # The first position is reserved for gender, with a synthetic age of 0.0 just like in src/data_prep.
        full_tokens = [gender_token, *history_token_ids]
        full_ages = [0.0, *history_ages]

        post_codes = set(post[token_col].astype("string").dropna())

        yield {
            "person_id": person_id,
            "index_datetime": index_dt,
            "treatment": int(row.treatment),
            "treatment_name": row.treatment_name,
            "comparator_name": row.comparator_name,
            "index_concept_name": row.index_concept_name,
            "index_vocabulary": row.index_vocabulary,
            "index_code": row.index_code,
            "had_both_cohorts": int(row.had_both_cohorts),
            "gender_token": gender_token,
            "age_at_index": age_at_index,
            "history_tokens": full_tokens,
            "history_ages": full_ages,
            "history_token_strings": [vocab.decode(tok) for tok in full_tokens],
            "history_event_count": len(pre),
            "outcome": int(bool(outcome_codes) and any(code in post_codes for code in outcome_codes)),
            "future_icd_flag": int(bool(future_flag_codes) and any(code in post_codes for code in future_flag_codes)),
        }


def prepare_source_tables(
    treatment_df: pd.DataFrame,
    comparator_df: pd.DataFrame,
    conditions_df: pd.DataFrame,
    person_df: pd.DataFrame,
    treatment_name: str,
    comparator_name: str,
    token_col: str = CODE_COL,
    use_vocab_plus_code: bool = True,
    vocab_col: str | None = VOCAB_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    treatment_df = standardize_event_columns(
        treatment_df,
        dataframe_name="treatment dataframe",
        token_candidates=[token_col, DEFAULT_TOKEN_COL, CODE_COL],
        timestamp_candidates=[DEFAULT_TIMESTAMP_COL, DRUG_DT_COL],
        include_vocab_and_name=True,
    )
    comparator_df = standardize_event_columns(
        comparator_df,
        dataframe_name="comparator dataframe",
        token_candidates=[token_col, DEFAULT_TOKEN_COL, CODE_COL],
        timestamp_candidates=[DEFAULT_TIMESTAMP_COL, DRUG_DT_COL],
        include_vocab_and_name=True,
    )
    conditions_df = standardize_event_columns(
        conditions_df,
        dataframe_name="conditions dataframe",
        token_candidates=[token_col, DEFAULT_TOKEN_COL, CODE_COL],
        timestamp_candidates=[DEFAULT_TIMESTAMP_COL, CONDITION_DT_COL],
    )

    # Restrict heavy condition/person tables to the people who are actually in the treatment comparison.
    index_df = select_index_events(
        treatment_df=treatment_df,
        comparator_df=comparator_df,
        treatment_name=treatment_name,
        comparator_name=comparator_name,
    )
    cohort_person_ids = set(index_df[PERSON_ID_COL].tolist())

    needed_condition_cols = [PERSON_ID_COL, CONDITION_DT_COL, CODE_COL]
    if use_vocab_plus_code:
        if vocab_col is None:
            raise ValueError("vocab_col must be provided when use_vocab_plus_code=True.")
        standardized_vocab_col = resolve_column_name(
            conditions_df,
            [vocab_col, VOCAB_COL],
            dataframe_name="conditions dataframe",
            field_name="vocabulary",
        )
        if standardized_vocab_col != VOCAB_COL:
            conditions_df = conditions_df.rename(columns={standardized_vocab_col: VOCAB_COL}).copy()
        needed_condition_cols.append(VOCAB_COL)

    missing_condition_cols = [col for col in needed_condition_cols if col not in conditions_df.columns]
    if missing_condition_cols:
        raise KeyError(f"Missing required columns in conditions dataframe: {missing_condition_cols}")

    conditions_small = conditions_df[needed_condition_cols].copy()
    conditions_small[PERSON_ID_COL] = pd.to_numeric(conditions_small[PERSON_ID_COL], errors="coerce")
    conditions_small[CONDITION_DT_COL] = pd.to_datetime(
        conditions_small[CONDITION_DT_COL],
        utc=True,
        errors="coerce",
    )
    conditions_small = conditions_small.dropna(subset=[PERSON_ID_COL, CONDITION_DT_COL, CODE_COL])
    conditions_small[PERSON_ID_COL] = conditions_small[PERSON_ID_COL].astype("int64")
    conditions_small = conditions_small[conditions_small[PERSON_ID_COL].isin(cohort_person_ids)].copy()
    conditions_small = conditions_small.sort_values([PERSON_ID_COL, CONDITION_DT_COL], kind="mergesort")

    people_small = person_df[[PERSON_ID_COL, GENDER_COL, DOB_COL]].copy()
    people_small[PERSON_ID_COL] = pd.to_numeric(people_small[PERSON_ID_COL], errors="coerce")
    people_small[DOB_COL] = pd.to_datetime(people_small[DOB_COL], utc=True, errors="coerce")
    people_small = people_small.dropna(subset=[PERSON_ID_COL])
    people_small[PERSON_ID_COL] = people_small[PERSON_ID_COL].astype("int64")
    people_small = people_small.drop_duplicates(subset=[PERSON_ID_COL], keep="first")

    return index_df, conditions_small, people_small


def create_treatment_effect_dataset(
    treatment_df: pd.DataFrame,
    comparator_df: pd.DataFrame,
    conditions_df: pd.DataFrame,
    person_df: pd.DataFrame,
    vocab: Vocab,
    treatment_name: str = "treatment",
    comparator_name: str = "comparator",
    outcome_codes: Sequence[str] | None = None,
    future_flag_codes: Sequence[str] | None = None,
    outcome_codes_file: str | Path | None = None,
    future_flag_codes_file: str | Path | None = None,
    outcome_horizon_days: int | None = None,
    token_col: str = CODE_COL,
    use_vocab_plus_code: bool = True,
    vocab_col: str | None = VOCAB_COL,
    min_history_events: int = 0,
    max_history_events: int | None = None,
) -> pd.DataFrame:
    if outcome_horizon_days is not None and outcome_horizon_days < 0:
        raise ValueError("outcome_horizon_days must be non-negative or None.")

    history_token_col = CODE_COL
    history_vocab_col = VOCAB_COL if use_vocab_plus_code else vocab_col

    index_df, conditions_small, people_small = prepare_source_tables(
        treatment_df=treatment_df,
        comparator_df=comparator_df,
        conditions_df=conditions_df,
        person_df=person_df,
        treatment_name=treatment_name,
        comparator_name=comparator_name,
        token_col=token_col,
        use_vocab_plus_code=use_vocab_plus_code,
        vocab_col=vocab_col,
    )

    gender_map = build_gender_map(people_small, vocab)
    dob_map = people_small.set_index(PERSON_ID_COL)[DOB_COL]

    outcome_code_set = normalize_codes(outcome_codes, outcome_codes_file)
    future_flag_code_set = normalize_codes(future_flag_codes, future_flag_codes_file)

    # Materialize one supervised sample per person for downstream treatment-effect modeling.
    records = list(
        iter_sample_rows(
            index_df=index_df,
            conditions_df=conditions_small,
            gender_map=gender_map,
            dob_map=dob_map,
            vocab=vocab,
            token_col=history_token_col,
            use_vocab_plus_code=use_vocab_plus_code,
            vocab_col=history_vocab_col,
            outcome_codes=outcome_code_set,
            future_flag_codes=future_flag_code_set,
            min_history_events=min_history_events,
            max_history_events=max_history_events,
            outcome_horizon_days=outcome_horizon_days,
        )
    )
    return pd.DataFrame(records)


def create_treatment_effect_dataset_from_pickles(
    treatment_pkl: str | Path,
    comparator_pkl: str | Path,
    conditions_pkl: str | Path,
    persons_pkl: str | Path,
    vocab: Vocab,
    **kwargs,
) -> pd.DataFrame:
    treatment_df = load_pickle_dataframe(treatment_pkl)
    comparator_df = load_pickle_dataframe(comparator_pkl)
    conditions_df = load_pickle_dataframe(conditions_pkl)
    person_df = load_pickle_dataframe(persons_pkl)
    return create_treatment_effect_dataset(
        treatment_df=treatment_df,
        comparator_df=comparator_df,
        conditions_df=conditions_df,
        person_df=person_df,
        vocab=vocab,
        **kwargs,
    )


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        with path.open("wb") as fh:
            pickle.dump(df, fh)
        return
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        return
    raise ValueError(f"Unsupported output format for {path}. Use .pkl, .pickle, or .parquet.")


class TreatmentEffectDataset(Dataset):
    """Dataset wrapper for treatment-effect samples.

    Expected dataframe columns:
      - history_tokens: list[int]
      - history_ages: list[float]
      - treatment: int
      - outcome: int
      - future_icd_flag: int

    Optional columns such as person_id and index_datetime are passed through.
    """

    def __init__(self, samples_df: pd.DataFrame):
        required = {
            "history_tokens",
            "history_ages",
            "treatment",
            "outcome",
            "future_icd_flag",
        }
        missing = sorted(required - set(samples_df.columns))
        if missing:
            raise KeyError(f"Missing required dataset columns: {missing}")
        self.samples_df = samples_df.reset_index(drop=True).copy()
        # Materialize the columns once so DataLoader workers do not pay the
        # pandas indexing cost on every sample fetch.
        self.person_ids = self.samples_df["person_id"].tolist() if "person_id" in self.samples_df.columns else None
        self.index_datetimes = self.samples_df["index_datetime"].tolist() if "index_datetime" in self.samples_df.columns else None
        self.history_tokens = self.samples_df["history_tokens"].tolist()
        self.history_ages = self.samples_df["history_ages"].tolist()
        self.treatments = self.samples_df["treatment"].astype(int).tolist()
        self.outcomes = self.samples_df["outcome"].astype(int).tolist()
        self.future_icd_flags = self.samples_df["future_icd_flag"].astype(int).tolist()


    def __len__(self) -> int:
        return len(self.history_tokens)

    def __getitem__(self, idx: int) -> dict[str, object]:
        idx = int(idx)
        tokens = self.history_tokens[idx]
        ages = self.history_ages[idx]
        # Return tensors for model inputs/targets while keeping lightweight identifiers for analysis.
        return {
            'person_id': int(self.person_ids[idx]) if self.person_ids is not None else None,
            'index_datetime': self.index_datetimes[idx] if self.index_datetimes is not None else None,
            'history_tokens': torch.tensor(tokens, dtype=torch.long),
            'history_ages': torch.tensor(ages, dtype=torch.float32),
            'treatment': torch.tensor(self.treatments[idx], dtype=torch.float32),
            'outcome': torch.tensor(self.outcomes[idx], dtype=torch.float32),
            'future_icd_flag': torch.tensor(self.future_icd_flags[idx], dtype=torch.float32),
            'history_length': torch.tensor(len(tokens), dtype=torch.long),
        }


def collate_fn(batch):
    tokens = [item["history_tokens"] for item in batch]
    ages = [item["history_ages"] for item in batch]
    treatments = torch.tensor([item["treatment"] for item in batch], dtype=torch.float)


    tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=0)
    ages_padded = pad_sequence(ages, batch_first=True, padding_value=0)

    attention_mask = (tokens_padded != 0).long()

    return {"tokens": tokens_padded, "ages": ages_padded, "attention_mask": attention_mask, "treatment": treatments}
