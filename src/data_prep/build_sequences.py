from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Special tokens
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
MALE_TOKEN = "<male>"
FEMALE_TOKEN = "<female>"


@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]

    @property
    def pad_idx(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.stoi[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, token: str) -> int:
        return self.stoi.get(str(token), self.unk_idx)

    def decode(self, idx: int) -> str:
        return self.itos[int(idx)]


# Vocab building
def build_vocab_from_conditions(
    conditions_df: pd.DataFrame,
    min_freq: int = 5,
    token_col: str = "standard_concept_code",
    save_path: Optional[str] = None,
) -> Vocab:
    """
    Build vocab from OMOP conditions.
    Token is the code column only.
    """
    tok = conditions_df[token_col].astype("string").fillna("")

    counts = Counter(tok.tolist())

    itos = [PAD_TOKEN, UNK_TOKEN, MALE_TOKEN, FEMALE_TOKEN]
    valid = [t for t, c in counts.items() if c >= min_freq and t != ""]
    # deterministic ordering: freq desc then token
    valid_sorted = sorted(valid, key=lambda t: (-counts[t], t))
    itos.extend(valid_sorted)
    stoi = {t: i for i, t in enumerate(itos)}

    # optionally save vocab, stores `itos`
    if save_path is not None:
        with open(f"{save_path}/vocab.json", "w", encoding="utf-8") as fh:
            json.dump({"itos": itos}, fh, ensure_ascii=False, indent=2)

    return Vocab(stoi=stoi, itos=itos)


# Preprocessing: flat arrays + offsets
def prepare_patient_event_arrays(
    conditions_df: pd.DataFrame,
    person_df: pd.DataFrame,
    vocab: Vocab,
    token_col: str = "standard_concept_code",
    # filtering / limits
    min_events_per_patient: int = 2,
    max_events_per_patient: Optional[int] = None,   # truncate each patient sequence if set
    keep: str = "right",  # "right" = most recent, "left" = oldest
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, int], np.ndarray]:
    """
    Returns:
      pids_sorted: int64 array [N]   person_id per event row (sorted by pid, time)
      token_ids:   int32 array [N]
      ages:        float32 array [N]  (years, non-normalized)
      gender_map:  maps person_id to gender token
      offsets:     int64 array [P+1]  offsets into [N] arrays per patient in pid order
    """

    person_id_col = "person_id"
    dt_col = "timestamp"
    dob_col = "date_of_birth"
    gender_col = "gender"
    c = conditions_df.copy()

    # parse datetimes (vectorized)
    c[dt_col] = pd.to_datetime(c[dt_col], utc=True, errors="coerce")
    c = c.dropna(subset=[person_id_col, dt_col, token_col])
    # map date of birth via a Series map (cheaper than merge)
    dob_map = pd.to_datetime(
        person_df.set_index(person_id_col)[dob_col],
        utc=True,
        errors="coerce",
    )
    dob = c[person_id_col].map(dob_map)

    # compute ages in years (float32). unknown dob -> age 0.
    delta_days = (c[dt_col] - dob).dt.total_seconds() / (3600.0 * 24.0)
    ages = (delta_days / 365.25).fillna(0.0).clip(lower=0.0).astype("float32")

    # build token string
    tok = c[token_col].astype("string").fillna("")

    # encode tokens (vectorized map)
    token_ids = tok.map(vocab.stoi).fillna(vocab.unk_idx).astype("int32")

    # sort once globally (stable)
    c["_token_id"] = token_ids.to_numpy()
    c["_age"] = ages.to_numpy()
    c = c.sort_values([person_id_col, dt_col], kind="mergesort")

    # sizes & filter by min_events_per_patient
    sizes = c.groupby(person_id_col, sort=False).size().astype(np.int64)
    keep_pids = sizes.index[sizes >= min_events_per_patient]
    c = c[c[person_id_col].isin(keep_pids)]

    # optional per-patient truncation
    if max_events_per_patient is not None:
        sizes = c.groupby(person_id_col, sort=False).size().astype(np.int64)
        cc = c.groupby(person_id_col, sort=False).cumcount().to_numpy(np.int64)
        size_per_row = c[person_id_col].map(sizes).to_numpy(np.int64)

        if keep == "right":
            # keep last K events
            from_end = (size_per_row - 1) - cc
            c = c[from_end < max_events_per_patient]
        else:
            # keep first K events
            c = c[cc < max_events_per_patient]

        c = c.sort_values([person_id_col, dt_col], kind="mergesort")

    # build offsets
    sizes = c.groupby(person_id_col, sort=False).size().astype(np.int64)
    pids = sizes.index.to_numpy(np.int64)
    offsets = np.empty(len(pids) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(sizes.to_numpy(), out=offsets[1:])

    # flatten arrays in the same order as offsets/pids
    pids_sorted = c[person_id_col].to_numpy(np.int64)
    token_ids_arr = c["_token_id"].to_numpy(np.int32)
    ages_arr = c["_age"].to_numpy(np.float32)

    # gender tokens: create person_id -> gender mapping
    gender_series = person_df.set_index(person_id_col)[gender_col].astype("string", copy=False)
    gender_lower = gender_series.str.strip().str.lower().fillna("")
    gender_tokens = pd.Series(vocab.unk_idx, index=gender_series.index, dtype="int64")
    mask_m = gender_lower.str.startswith("m")
    mask_f = gender_lower.str.startswith("f")
    gender_tokens[mask_m] = vocab.stoi.get(MALE_TOKEN, vocab.unk_idx)
    gender_tokens[mask_f] = vocab.stoi.get(FEMALE_TOKEN, vocab.unk_idx)
    gender_map = gender_tokens.to_dict()

    return pids_sorted, token_ids_arr, ages_arr, gender_map, offsets

# Dataset for next token prediction. One sample per eligible patient per epoch.
class NextTokenDataset(Dataset):
    """
    Provides training samples:
      input_tokens: [<=max_src_tokens] token ids
      input_age:    [<=max_src_tokens] ages (float)
      next_code:    scalar token id

    Each eligible patient contributes exactly one sample.
    For short histories, the sample is the full prefix ending at the final event.
    For long histories, a window endpoint is sampled per patient and epoch.
    """

    def __init__(
        self,
        pids_sorted: np.ndarray,
        token_ids: np.ndarray,
        ages: np.ndarray,
        gender_map: Dict[int, int],
        offsets: np.ndarray,
        vocab: Vocab,
        min_src: int = 5,
        max_src_tokens: int = 50,
        seed: int = 0,
    ):
        assert token_ids.ndim == 1 and ages.ndim == 1
        assert len(token_ids) == len(ages)
        assert offsets.ndim == 1 and offsets[0] == 0

        self.pids_sorted = pids_sorted
        self.token_ids = token_ids
        self.ages = ages
        self.gender_map = gender_map
        self.offsets = offsets.astype(np.int64)
        self.vocab = vocab

        self.min_src = int(min_src)
        self.max_src_tokens = int(max_src_tokens)
        self.seed = int(seed)
        self.epoch = 0
        self.lengths = (self.offsets[1:] - self.offsets[:-1]).astype(np.int64)
        self.valid_patients = np.flatnonzero(self.lengths > self.min_src).astype(np.int64)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return int(len(self.valid_patients))

    def __getitem__(self, idx: int) -> Dict[str, object]:
        idx = int(idx)
        p = int(self.valid_patients[idx])

        start = int(self.offsets[p])
        L = int(self.lengths[p])

        # Pick a single target endpoint for this patient.
        # Short histories use the full prefix ending at the final event.
        # Long histories sample an endpoint so the context window moves across epochs.
        if self.epoch <= 0 or (L - 1) <= self.max_src_tokens:
            t = L - 1
        else:
            low = max(self.min_src, self.max_src_tokens)
            high = L - 1
            rng_seed = self.seed + self.epoch * 1_000_003 + p
            rng = np.random.default_rng(rng_seed)
            t = int(rng.integers(low, high + 1))

        # all events before t as context:
        # history window [max(0, t-max_src_tokens) .. t-1]
        ctx_start = max(0, t - self.max_src_tokens)
        s0 = start + ctx_start
        s1 = start + t

        input_tokens = self.token_ids[s0:s1].astype(np.int64)  # torch wants int64
        input_age = self.ages[s0:s1].astype(np.float32)

        # add gender token at the start of the trajectory if available
        person_id = int(self.pids_sorted[start])
        gender_token = self.gender_map.get(person_id, self.vocab.unk_idx)
        input_tokens = np.insert(input_tokens, 0, gender_token)
        input_age = np.insert(input_age, 0, 0.0)  # gender token at age 0


        next_code = int(self.token_ids[start + t])
        next_age = float(self.ages[start + t])

        return {
            "input_tokens": input_tokens,
            "input_age": input_age,
            "next_code": next_code,
            "next_age": next_age,
        }


# Collate function for DataLoader
def collate_batch(batch: List[Dict], pad_idx: int) -> Dict[str, torch.Tensor]:
    """
    Produces tensors matching the Transformer forward signature in model.py:
      - input_tokens: Long [B, L]
      - input_age:    Float [B, L]
      - key_padding_mask: Bool [B, L] True where PAD
      - token_targets: Long [B, L] shifted next-token targets
      - time_targets: Float [B, L] shifted time-to-next-event targets
      - next_code: Long [B]
    """
    B = len(batch)
    lengths = [len(x["input_tokens"]) for x in batch]
    Lmax = max(lengths) if lengths else 0

    input_tokens = torch.full((B, Lmax), pad_idx, dtype=torch.long)
    input_age = torch.zeros((B, Lmax), dtype=torch.float32)
    key_padding_mask = torch.ones((B, Lmax), dtype=torch.bool)
    token_targets = torch.full((B, Lmax), pad_idx, dtype=torch.long)
    time_targets = torch.zeros((B, Lmax), dtype=torch.float32)

    next_code = torch.zeros((B,), dtype=torch.long)

    for i, sample in enumerate(batch):
        L = lengths[i]
        if L > 0:
            input_tokens[i, :L] = torch.from_numpy(sample["input_tokens"])
            input_age[i, :L] = torch.from_numpy(sample["input_age"])
            key_padding_mask[i, :L] = False
            # Mask padding positions. Unknown tokens remain valid context.
            key_padding_mask[i, :L] |= (input_tokens[i, :L] == pad_idx)
            if L > 1:
                token_targets[i, : L - 1] = torch.from_numpy(sample["input_tokens"][1:])
                time_targets[i, : L - 1] = input_age[i, 1:L] - input_age[i, : L - 1]
            token_targets[i, L - 1] = int(sample["next_code"])
            time_targets[i, L - 1] = max(0.0, float(sample["next_age"]) - float(input_age[i, L - 1]))

        next_code[i] = int(sample["next_code"])

    return {
        "input_tokens": input_tokens,
        "input_age": input_age,
        "key_padding_mask": key_padding_mask,
        "token_targets": token_targets,
        "time_targets": time_targets,
        "next_code": next_code,
    }

# Collate function for DataLoader for neural net baseline
def collate_lag_batch(
    batch: List[Dict[str, Any]],
    num_lags: int,
    pad_idx: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    batch: list of samples with at least:
        - sample["input_tokens"]: list[int]
        - sample["next_code"]: int

    Returns:
        - lag_inputs: [B, num_lags]  (last num_lags codes, left-padded with pad_idx)
        - next_code:  [B]
    """
    B = len(batch)
    lag_inputs = torch.full((B, num_lags), pad_idx, dtype=torch.long)
    next_codes = torch.empty(B, dtype=torch.long)

    for i, sample in enumerate(batch):
        seq = sample["input_tokens"]
        next_codes[i] = sample["next_code"]

        seq_tensor = torch.tensor(seq, dtype=torch.long)
        L = len(seq_tensor)

        if L >= num_lags:
            lag_inputs[i] = seq_tensor[-num_lags:]
        else:
            # left-pad: [pad, pad, ..., seq]
            lag_inputs[i, -L:] = seq_tensor

        # If the original sequence is longer than the lag window, ensure the row's first column contains the sequence's first token(gender token).
        # For L == num_lags the first token is already present in the slice
        if L > num_lags:
            lag_inputs[i, 0] = seq_tensor[0]

    return {
        "input_tokens": lag_inputs,
        "next_code": next_codes,
    }


def save_vocab(vocab: Vocab, path: str) -> None:
    """Save Vocab to disk (writes the `itos` list).

    Args:
        vocab: Vocab instance to save.
        path: file path to write JSON to.
    """
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"itos": vocab.itos}, fh, ensure_ascii=False, indent=2)


def _build_compat_stoi(itos: List[str]) -> Dict[str, int]:
    """Build stoi and add backward-compatible aliases for expanded tokens.

    This supports vocab migrations such as:
      - `CODE` -> `CODE diagnosis name`
      - `VOCAB|CODE` -> `VOCAB|CODE|diagnosis name`

    When the alias is ambiguous across multiple vocab entries, it is skipped.
    """
    stoi = {str(token): idx for idx, token in enumerate(itos)}
    alias_to_idx: Dict[str, int] = {}
    ambiguous_aliases: set[str] = set()

    for idx, raw_token in enumerate(itos):
        token = str(raw_token)
        if token.startswith("<") and token.endswith(">"):
            continue

        candidates: set[str] = set()
        if "|" in token:
            parts = token.split("|")
            if len(parts) >= 3:
                candidates.add("|".join(parts[:-1]))
            last_part = parts[-1].strip()
            if last_part:
                candidates.add(last_part.split(maxsplit=1)[0])
        elif " " in token:
            candidates.add(token.split(maxsplit=1)[0])

        for alias in candidates:
            if not alias or alias == token or alias in stoi:
                continue
            prev_idx = alias_to_idx.get(alias)
            if prev_idx is None:
                alias_to_idx[alias] = idx
            elif prev_idx != idx:
                ambiguous_aliases.add(alias)

    for alias in ambiguous_aliases:
        alias_to_idx.pop(alias, None)
    for alias, idx in alias_to_idx.items():
        stoi.setdefault(alias, idx)
    return stoi


def load_vocab(path: str) -> Vocab:
    """Load a `Vocab` previously written with `save_vocab`.

    Returns a `Vocab` instance with `itos` and computed `stoi`.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    itos = data.get("itos")
    if itos is None:
        raise ValueError(f"Invalid vocab file: missing 'itos' key: {path}")

    stoi = _build_compat_stoi(itos)
    return Vocab(stoi=stoi, itos=itos)
