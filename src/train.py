import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from datetime import datetime
import logging
import math
import numpy as np
from typing import Optional

PROJECT_ROOT = Path(".").resolve()
sys.path.append(str(PROJECT_ROOT))

from src.model import Transformer

from src.data_prep.build_sequences import (
    build_vocab_from_conditions,
    prepare_patient_event_arrays,
    NextTokenDataset,
    collate_batch,
    collate_lag_batch,
)

from torch.utils.tensorboard import SummaryWriter

# Optional W&B; explicitly opt in with WANDB_MODE=offline or online.
os.environ.setdefault("WANDB_MODE", "disabled")

# optional wandb
try:
    import wandb
except Exception:
    wandb = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _is_main_process() -> bool:
    return (not _is_dist_initialized()) or dist.get_rank() == 0


def _distributed_mean(value: float, device: torch.device) -> float:
    if not _is_dist_initialized():
        return float(value)
    t = torch.tensor(float(value), device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return float(t.item())


def _setup_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, 1

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA GPUs.")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return True, rank, local_rank, world_size


def _cleanup_distributed() -> None:
    if _is_dist_initialized():
        dist.barrier()
        dist.destroy_process_group()


def load_data(
    conditions_df,
    person_df,
    min_freq: int = 5,
    min_src: int = 2,
    max_src_tokens: int = 50,
    batch_size: int = 32,
    num_workers: int = 0,
    train_split: float = 0.75,
    seed: int = 42,
    # OMOP-specific:
    token_col: str = "standard_concept_code",
    min_events_per_patient: int = 2,
    max_events_per_patient: int | None = 4096,
    naive: bool = False, # true for neural net baseline. this function is also used there
    num_lags: Optional[int] = None,
    save_path: Optional[Path] = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, DataLoader, DataLoader, object]:
    """
    Build dataset/vocab from OMOP tables and return train/val/test loaders + vocab.
    """

    logger.info("Building vocab from conditions...")
    vocab = build_vocab_from_conditions(
        conditions_df=conditions_df,
        min_freq=min_freq,
        token_col=token_col,
        save_path=save_path,
    )
    logger.info(f"Vocab size: {len(vocab)}")

    # Split by patient ID so histories do not cross partitions.
    # Pandas nullable integer columns return an IntegerArray here; convert to a
    # plain NumPy array before shuffling to avoid view-semantics issues.
    unique_pids = np.asarray(person_df["person_id"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_pids)

    if not 0 < train_split < 1:
        raise ValueError("train_split must be between 0 and 1")
    n_train = int(train_split * len(unique_pids))
    n_val = int((1 - train_split) / 2 * len(unique_pids))

    train_pids = set(unique_pids[:n_train])
    val_pids = set(unique_pids[n_train:n_train+n_val])
    test_pids = set(unique_pids[n_train+n_val:])

    # filter conditions dataframe by person ids
    def filter_by_pids(df, pids):
        return df[df["person_id"].isin(pids)]

    conditions_train = filter_by_pids(conditions_df, train_pids)
    conditions_val = filter_by_pids(conditions_df, val_pids)
    conditions_test = filter_by_pids(conditions_df, test_pids)

    logger.info("Preparing flat event arrays...")

    # train set
    pids_sorted_train, token_ids_train, ages_train, gender_map_train, offsets_train = prepare_patient_event_arrays(
        conditions_df=conditions_train,
        person_df=person_df,
        vocab=vocab,
        token_col=token_col,
        min_events_per_patient=min_events_per_patient,
        max_events_per_patient=max_events_per_patient,
        keep="right",
    )

    train_ds = NextTokenDataset(
        pids_sorted=pids_sorted_train,
        token_ids=token_ids_train,
        ages=ages_train,
        gender_map=gender_map_train,
        offsets=offsets_train,
        vocab=vocab,
        min_src=min_src,
        max_src_tokens=max_src_tokens,
    )
    logger.info(f"Train Dataset size: {len(train_ds)} samples")

    # validation set
    pids_sorted_val, token_ids_val, ages_val, gender_map_val, offsets_val = prepare_patient_event_arrays(
        conditions_df=conditions_val,
        person_df=person_df,
        vocab=vocab,
        token_col=token_col,
        min_events_per_patient=min_events_per_patient,
        max_events_per_patient=max_events_per_patient,
        keep="right",
    )

    val_ds = NextTokenDataset(
        pids_sorted=pids_sorted_val,
        token_ids=token_ids_val,
        ages=ages_val,
        gender_map=gender_map_val,
        offsets=offsets_val,
        vocab=vocab,
        min_src=min_src,
        max_src_tokens=max_src_tokens,
    )
    logger.info(f"Validation Dataset size: {len(val_ds)} samples")

    # test set
    pids_sorted_test, token_ids_test, ages_test, gender_map_test, offsets_test = prepare_patient_event_arrays(
        conditions_df=conditions_test,
        person_df=person_df,
        vocab=vocab,
        token_col=token_col,
        min_events_per_patient=min_events_per_patient,
        max_events_per_patient=max_events_per_patient,
        keep="right",
    )
    test_ds = NextTokenDataset(
        pids_sorted=pids_sorted_test,
        token_ids=token_ids_test,
        ages=ages_test,
        gender_map=gender_map_test,
        offsets=offsets_test,
        vocab=vocab,
        min_src=min_src,
        max_src_tokens=max_src_tokens,
    )
    logger.info(f"Validation Dataset size: {len(test_ds)} samples")


    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    collate_fn = lambda b: collate_batch(b, pad_idx=vocab.pad_idx)

    if naive:
        collate_fn = lambda b: collate_lag_batch(b, num_lags=num_lags, pad_idx=vocab.pad_idx)

    train_sampler = None
    val_sampler = None
    test_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )
        test_sampler = DistributedSampler(
            test_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, vocab

# function to get logist of last non-padded position, used for prediction
def _last_valid_logits_from_full_logits(full_logits: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
    """
    full_logits: [B, L, V]
    key_padding_mask: [B, L] True where PAD
    Returns last valid timestep logits: [B, V]
    """
    # valid lengths
    valid_lengths = (~key_padding_mask).sum(dim=1)  # [B]
    last_idx = (valid_lengths - 1).clamp(min=0)     # [B]
    B = full_logits.size(0)
    return full_logits[torch.arange(B, device=full_logits.device), last_idx, :]


def _prepare_sequence_training_tensors(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    pad_idx: int,
    unk_idx: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Flatten full-sequence predictions and targets for teacher-forced next-token training.
    """
    logits = output["logits"]
    token_targets = batch["token_targets"]

    if unk_idx is not None:
        logits = logits.clone()
        logits[..., unk_idx] = -1e9
        token_targets = token_targets.clone()
        token_targets[token_targets == unk_idx] = pad_idx

    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_token_targets = token_targets.reshape(-1)
    return flat_logits, flat_token_targets


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pad_idx: int,
    unk_idx: int | None = None,
    global_step_start: int = 0,
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    num_batches = 0
    token_criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    global_step = int(global_step_start)

    for batch_idx, batch in enumerate(train_loader):

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        seq_tokens = batch["input_tokens"]        # [B, L]
        seq_age = batch["input_age"]              # [B, L]
        seq_pad = batch["key_padding_mask"]       # [B, L]

        output = model(seq_tokens=seq_tokens, seq_age=seq_age, key_padding_mask=seq_pad)

        logits, targets = _prepare_sequence_training_tensors(
            output=output,
            batch=batch,
            pad_idx=pad_idx,
            unk_idx=unk_idx,
        )

        token_loss = token_criterion(logits, targets)
        loss = token_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % 100 == 0 and _is_main_process():
            logger.info(
                f"  Batch {batch_idx + 1}/{len(train_loader)}: "
                f"loss={loss.item():.4f}, token_loss={token_loss.item():.4f}"
            )


    new_global_step = global_step + num_batches
    return (total_loss / num_batches if num_batches > 0 else 0.0), new_global_step


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    pad_idx: int,
    unk_idx: int | None = None,
) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    token_criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            seq_tokens = batch["input_tokens"]
            seq_age = batch["input_age"]
            seq_pad = batch["key_padding_mask"]

            output = model(seq_tokens=seq_tokens, seq_age=seq_age, key_padding_mask=seq_pad)

            logits, targets = _prepare_sequence_training_tensors(
                output=output,
                batch=batch,
                pad_idx=pad_idx,
                unk_idx=unk_idx,
            )

            token_loss = token_criterion(logits, targets)
            loss = token_loss

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def train(
    conditions_df,
    person_df,
    output_dir: str = "checkpoints",
    num_epochs: int = 10,
    batch_size: int = 32,
    lr: float = 5e-4,
    warmup_epochs: int = 2,
    d_model: int = 256,
    nhead: int = 4,
    num_layers: int = 4,
    ff_mult: int = 4,
    dropout: float = 0.0,
    device:str | None = None, #str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
    # data params
    min_freq: int = 5,
    min_src: int = 2,
    max_src_tokens: int = 50,
    token_col: str = "standard_concept_code",
    early_stopping_patience: int = 20,
    early_stopping_min_delta: float = 0.0,
):
    if early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be nonnegative (0 disables it)")
    if not math.isfinite(early_stopping_min_delta) or early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta must be finite and nonnegative")
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    distributed, rank, local_rank, world_size = _setup_distributed()

    output_path = Path(output_dir)
    if _is_main_process():
        output_path.mkdir(exist_ok=True, parents=True)
    logger.info("Starting training setup")
    use_data_parallel = False
    if distributed:
        device = torch.device(f"cuda:{local_rank}")
        if rank == 0:
            logger.info(f"Training with DDP on {world_size} GPUs")
        else:
            logger.setLevel(logging.WARNING)
    else:
        logger.info(f"Training on device: {device}")
        device = torch.device(device)
        if device.type == "cuda":
            n_gpus = torch.cuda.device_count()
            if n_gpus > 1:
                use_data_parallel = True
                logger.info(
                    f"WORLD_SIZE=1, enabling DataParallel fallback on {n_gpus} GPUs. "
                    "For better throughput, launch with torchrun for DDP."
                )

    if _is_main_process():
        logger.info(f"TPopulation size: {person_df['person_id'].nunique()} individuals")

    writer = SummaryWriter(log_dir="runs") if _is_main_process() else None

    if wandb is not None and _is_main_process():
        config = dict(
            lr=lr,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            ff_mult=ff_mult,
            batch_size=batch_size,
            dropout=dropout,
            seed=seed,
            min_freq=min_freq,
            min_src=min_src,
            max_src_tokens=max_src_tokens,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
        )
        wandb.init(project="ma_transformer", config=config)
        try:
            wandb.run.name = f"transformer_{d_model}_{num_layers}layers_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        except Exception:
            pass

    data_save_path = output_path if _is_main_process() else None

    num_workers = min(8, os.cpu_count())
    train_loader, val_loader, test_loader, vocab = load_data(
        conditions_df=conditions_df,
        person_df=person_df,
        min_freq=min_freq,
        min_src=min_src,
        max_src_tokens=max_src_tokens,
        batch_size=batch_size,
        seed=seed,
        save_path=data_save_path,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        num_workers=num_workers,
        token_col=token_col,
    )

    vocab_size = len(vocab)
    pad_idx = vocab.pad_idx

    if _is_main_process():
        logger.info(f"Creating model: d_model={d_model}, nhead={nhead}, num_layers={num_layers}")
    raw_model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        ff_mult=ff_mult,
        dropout=dropout,
        pad_idx=pad_idx,
    ).to(device)

    model = raw_model
    if distributed:
        model = DDP(raw_model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
    elif use_data_parallel:
        model = nn.DataParallel(raw_model)

    n_params = sum(p.numel() for p in model.parameters())
    if _is_main_process():
        logger.info(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Learning-rate warmup followed by cosine decay.

    if warmup_epochs is None:
        warmup_epochs = 0
    warmup_epochs = int(warmup_epochs)

    min_lr = 1e-5

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0/max(1, warmup_epochs), end_factor=1.0, total_iters=warmup_epochs
    )

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, num_epochs - warmup_epochs), eta_min=min_lr
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )

    best_val_loss = float("inf")
    stopping_best_loss = float("inf")
    epochs_without_improvement = 0
    best_checkpoint_path = (output_path / "best.pt").resolve()
    last_checkpoint_path = (output_path / "last.pt").resolve()
    if _is_main_process():
        logger.info(f"Starting training for {num_epochs} epochs... dropout={dropout}")

    global_step = 0

    dataset = train_loader.dataset
    train_sampler = getattr(train_loader, "sampler", None)

    for epoch in range(1, num_epochs + 1):
        if _is_main_process():
            logger.info(f"\nEpoch {epoch}/{num_epochs}")

        # pass epoch to dataset to randomize window phases
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)

        train_loss, global_step = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            pad_idx=pad_idx,
            unk_idx=vocab.unk_idx,
            global_step_start=global_step,
        )
        train_loss = _distributed_mean(train_loss, device)
        if _is_main_process():
            logger.info(f"  Train loss: {train_loss:.4f}")

        val_loss = validate(
            model=model,
            val_loader=val_loader,
            device=device,
            pad_idx=pad_idx,
            unk_idx=vocab.unk_idx,
        )
        val_loss = _distributed_mean(val_loss, device)
        if not math.isfinite(val_loss):
            raise RuntimeError(f"Non-finite validation loss at epoch {epoch}: {val_loss}")
        # Every rank uses the same reduced loss and therefore stops together.
        if val_loss < stopping_best_loss - early_stopping_min_delta:
            stopping_best_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if _is_main_process():
            logger.info(f"  Val loss: {val_loss:.4f}")

        scheduler.step()
        if _is_main_process():
            logger.info(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        if wandb is not None and _is_main_process():
            wandb.log(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"]},
                step=global_step,
            )

        if _is_main_process():
            output_path.mkdir(parents=True, exist_ok=True)
            model_to_save = model.module if isinstance(model, (DDP, nn.DataParallel)) else model

            torch.save(model_to_save.state_dict(), last_checkpoint_path)
            logger.info(f"  Saved latest checkpoint: {last_checkpoint_path}")
            if wandb is not None:
                try:
                    wandb.save(str(last_checkpoint_path))
                except Exception:
                    pass

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model_to_save.state_dict(), best_checkpoint_path)
                logger.info(f"  Saved best checkpoint: {best_checkpoint_path}")
                if wandb is not None:
                    try:
                        wandb.save(str(best_checkpoint_path))
                    except Exception:
                        pass

        if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
            if _is_main_process():
                logger.info(f"Early stopping at epoch {epoch}: no validation improvement "
                            f"greater than {early_stopping_min_delta} for "
                            f"{early_stopping_patience} epochs.")
            break

    if writer is not None:
        writer.close()
    if wandb is not None and _is_main_process():
        try:
            wandb.finish()
        except Exception:
            pass

    if _is_main_process():
        logger.info(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")

    _cleanup_distributed()

    return test_loader, vocab
