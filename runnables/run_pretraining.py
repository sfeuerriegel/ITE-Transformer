"""Pretrain from local tables, directly or with multiple torchrun workers."""

import argparse
import math
from pathlib import Path
import sys
import os
import subprocess

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.project_paths import SHARED_DATA_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-train a transformer from local patient and event tables with torchrun/DDP support."
    )
    parser.add_argument("--person-path", type=Path, default=SHARED_DATA_DIR / "patients.pkl")
    parser.add_argument("--conditions-path", type=Path, default=SHARED_DATA_DIR / "events.pkl")
    parser.add_argument("--num-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="new_checkpoints")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--ff-mult", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--min-src", type=int, default=20)
    parser.add_argument("--max-src-tokens", type=int, default=128)
    parser.add_argument("--token-col", default="token_col")
    parser.add_argument("--processes", type=int, default=4,
                        help="Local workers; 1 runs directly (CPU or GPU). Default: 4.")
    parser.add_argument("--early-stopping-patience", type=int, default=20,
                        help="Epochs without validation improvement; 0 disables stopping.")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.05,
                        help="Minimum validation-loss decrease to reset patience.")
    args = parser.parse_args()
    if args.processes < 1:
        parser.error("--processes must be positive")
    if args.early_stopping_patience < 0:
        parser.error("--early-stopping-patience must be nonnegative")
    if not math.isfinite(args.early_stopping_min_delta) or args.early_stopping_min_delta < 0:
        parser.error("--early-stopping-min-delta must be finite and nonnegative")
    return args


def read_dataframe(path: Path) -> pd.DataFrame:
    """Read a patient or event table from pickle, Parquet, or CSV."""
    if path.suffixes[-2:] == [".pkl", ".gz"] or path.suffix == ".pkl":
        return pd.read_pickle(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataframe format for {path}")


def main() -> None:
    args = parse_args()
    if args.processes > 1 and "LOCAL_RANK" not in os.environ:
        command = [sys.executable, "-m", "torch.distributed.run",
                   "--nproc_per_node", str(args.processes), str(Path(__file__).resolve()),
                   *sys.argv[1:]]
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)
        return
    # Import training dependencies after parsing so --help works independently.
    from src.train import train

    person_df = read_dataframe(args.person_path)
    conditions_df = read_dataframe(args.conditions_path)
    train(
        conditions_df=conditions_df,
        person_df=person_df,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        output_dir=args.output_dir,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        warmup_epochs=args.warmup_epochs,
        min_freq=args.min_freq,
        min_src=args.min_src,
        max_src_tokens=args.max_src_tokens,
        token_col=args.token_col,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )


if __name__ == "__main__":
    main()
