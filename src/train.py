"""
One (strategy, source, seed) run: train a detector on one generator, evaluate it
against all seven, write `metrics.json` and the selected weights beside it.

The whole arm - 3,600 training images, 400 internal-validation images and the
4,000-image test set - is pushed to the GPU once as uint8 and indexed there.
There is no DataLoader anywhere: at ~390 MB the arm fits in T4 VRAM, so worker
processes, pinned memory and the Drive-I/O bottleneck all disappear, and the loop
is shorter for it.

Two conventions carried over from `data.py`, both load-bearing:

- **torch is imported inside the functions that need it**, `src.model` included,
  so this module imports and its selection logic tests on a laptop with numpy
  alone. func cell_accuracies is the torch-free core of func evaluate_row, the
  same way `load_arm_numpy` is the torch-free core of `load_arm`.
- **The naming trap.** `Arm.x_val` is the 10 % carved out of the *train* split and
  is the only set allowed to influence training; `Arm.x_test` comes from the split
  the dataset calls `validation` and is what every reported cell is computed from.
  Nothing here selects a checkpoint on `x_test`.

One matrix cell (source -> target) is that target generator's 500 test fakes plus
the same fixed 500 real images shared by all seven cells. Exactly balanced, so
plain accuracy is honest and the binomial SE at n=1,000 is ~1.6 pp at p=0.5.

Usage:
    python -m src.train --strategy centre_crop --source BigGAN --seed 0 \
        --cache-dir /content/drive/MyDrive/university/deep_learning/cache

Requires torch >= 2.4 for the `torch.amp` entry points used below.
"""

from __future__ import annotations

import json
import time
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING, Final

from src.data import GENERATORS, REAL_TAG, STRATEGIES, TRAIN_PER_CLASS, Strategy, load_arm

if TYPE_CHECKING:  # pragma: no cover - typing only, torch is not installed locally
    from torch import Tensor
    from torch.nn import Module

__all__ = [
    "cell_accuracies",
    "checkpoint_path",
    "evaluate_row",
    "metrics_path",
    "train_one",
]

#: Hyperparameters fixed by PLAN.md. They are reported as numbers in the report's
#: "Models and hyperparameters" section, which is generated from `config` below.
EPOCHS: Final[int] = 40
BATCH_SIZE: Final[int] = 128
LEARNING_RATE: Final[float] = 3e-4
WEIGHT_DECAY: Final[float] = 1e-4

#: Inference only, so it is bounded by VRAM rather than by optimisation.
EVAL_BATCH_SIZE: Final[int] = 512

#: The only augmentation. Every other candidate - rotation, scale jitter, colour
#: jitter, JPEG augmentation - resamples or re-encodes, which would inject the
#: exact artefact this study measures. A methodological choice, not laziness.
FLIP_PROBABILITY: Final[float] = 0.5

#: Images arrive as uint8 and are mapped to [-1, 1] by `x / 127.5 - 1`. Kept
#: character-for-character in step with `model.normalize`: training and the
#: attribution analysis must present the model with the same input scale.
UINT8_HALF_RANGE: Final[float] = 127.5

#: A logit above this means "synthetic"; the classes are balanced, so 0 (p = 0.5)
#: is the honest threshold and no tuning is done on it.
DECISION_LOGIT: Final[float] = 0.0

DEFAULT_DEVICE: Final[str] = "cuda"
DEFAULT_RESULTS_DIR: Final[Path] = Path("results/runs")
PRINT_EVERY_EPOCHS: Final[int] = 10

#: Filename of the selected `state_dict` inside a run's own output directory. Every
#: run writes one, unconditionally: the input-gradient attribution needs a trained
#: model per (strategy, source), and recovering weights we already had would mean
#: re-running the matrix. CompactCNN is 1,173,473 parameters, so a checkpoint is
#: ~4.7 MB and all 56 come to ~0.26 GB. Not committed - `.gitignore` covers `*.pt`.
CHECKPOINT_NAME: Final[str] = "model.pt"


# --------------------------------------------------------------------------- #
# Torch-free core: scoring and output layout
# --------------------------------------------------------------------------- #


def metrics_path(
    results_dir: str | Path,
    strategy: Strategy,
    source: str,
    seed: int,
) -> Path:
    """
    Locate one run's metrics file.

    Exported because the matrix driver resumes on it: the entire checkpoint story
    is `if metrics_path(...).exists(): continue`. Runs take minutes, so re-running
    a lost one is cheaper than any resume machinery.

    Args:
        results_dir: Root of the committed results tree, e.g. "results/runs".
        strategy: One of STRATEGIES.
        source: Generator the detector was trained on.
        seed: Run seed.

    Returns:
        `<results_dir>/<strategy>/<source>/seed<k>/metrics.json`.
    """
    return Path(results_dir) / strategy / source / f"seed{seed}" / "metrics.json"


def checkpoint_path(
    results_dir: str | Path,
    strategy: Strategy,
    source: str,
    seed: int,
) -> Path:
    """
    Locate one run's weights file.

    Deliberately the run's *own* directory, beside its `metrics.json`: the 56 runs
    each get a distinct (strategy, source, seed) directory, so no two can overwrite
    each other and no run needs to invent a name.

    Resume is unaffected - it keys on `metrics.json` alone, which func _main writes
    only after the checkpoint has been saved. A run interrupted mid-training leaves
    neither file behind and is simply redone.

    Args:
        results_dir: Root of the committed results tree, e.g. "results/runs".
        strategy: One of STRATEGIES.
        source: Generator the detector was trained on.
        seed: Run seed.

    Returns:
        `<results_dir>/<strategy>/<source>/seed<k>/model.pt`.
    """
    return metrics_path(results_dir, strategy, source, seed).with_name(CHECKPOINT_NAME)


def _cell_mask(gen_ids: np.ndarray, generator: str) -> np.ndarray:
    """
    Select the test rows forming one matrix cell.

    A cell is that generator's fakes plus the shared real block - not the fake
    rows alone. Accuracy over fakes alone would be a recall, would ignore the
    false-positive rate entirely, and would not be comparable across strategies.

    Args:
        gen_ids: Per-row generator tags from `Arm.gen_ids`.
        generator: Target generator for this cell.

    Returns:
        Boolean mask over the test rows.
    """
    return (gen_ids == generator) | (gen_ids == REAL_TAG)


def cell_accuracies(
    predictions: np.ndarray,
    truth: np.ndarray,
    gen_ids: np.ndarray,
) -> dict[str, float]:
    """
    Score one row of the transfer matrix from predictions alone.

    The torch-free half of func evaluate_row, so the cell arithmetic - the thing
    every reported number is made of - can be tested on a laptop with no GPU.

    Args:
        predictions: Per-row predicted labels, 1.0 synthetic and 0.0 real.
        truth: Per-row true labels, same encoding.
        gen_ids: Per-row generator tags, REAL_TAG for the real block.

    Returns:
        Generator -> accuracy over that cell's 1,000 rows, one entry per member
        of GENERATORS.

    Raises:
        ValueError: If the three arrays disagree in length, if the real block is
            missing, or if any generator has no test rows. A missing block would
            silently turn a balanced accuracy into a one-class score.
    """
    if not len(predictions) == len(truth) == len(gen_ids):
        raise ValueError(
            f"length mismatch: predictions {len(predictions)}, truth {len(truth)}, "
            f"gen_ids {len(gen_ids)}"
        )
    if not np.any(gen_ids == REAL_TAG):
        raise ValueError(f"no rows tagged {REAL_TAG!r}; every cell needs the shared real block")

    correct = predictions == truth
    accuracies: dict[str, float] = {}
    for generator in GENERATORS:
        if not np.any(gen_ids == generator):
            raise ValueError(f"no test rows for {generator}; the matrix row would be incomplete")
        accuracies[generator] = float(correct[_cell_mask(gen_ids, generator)].mean())
    return accuracies


# --------------------------------------------------------------------------- #
# Torch: batching, evaluation, the training loop
# --------------------------------------------------------------------------- #


def _device_type(device: str) -> str:
    """
    Reduce a device string to the type `torch.amp` expects.

    Args:
        device: Device string, e.g. "cuda", "cuda:0" or "cpu".

    Returns:
        "cuda" or "cpu".
    """
    return "cuda" if device.startswith("cuda") else "cpu"


def _normalise(images: Tensor) -> Tensor:
    """
    Turn a uint8 NHWC batch into a float NCHW batch in [-1, 1].

    Args:
        images: uint8 (B, 128, 128, 3), already on the target device.

    Returns:
        float32 (B, 3, 128, 128).
    """
    # `.float()` always allocates here, because the source dtype is uint8 - so the
    # in-place ops that follow cannot reach back into the cached uint8 arm, and the
    # whole mapping costs one allocation rather than one per operation.
    return images.permute(0, 3, 1, 2).float().div_(UINT8_HALF_RANGE).sub_(1.0)


def _forward_logits(model: Module, images: Tensor) -> Tensor:
    """
    Run the model and return a flat (B,) logit vector.

    `CompactCNN.forward` is specified to return (B,), but a (B, 1) return is
    accepted and squeezed: broadcasting (B, 1) against a (B,) target inside
    `binary_cross_entropy_with_logits` would silently compute a B x B loss and
    quietly ruin every run rather than failing.

    Args:
        model: The detector.
        images: float32 (B, 3, 128, 128).

    Returns:
        float (B,) logits; positive means synthetic.

    Raises:
        ValueError: If the model returns a shape that is neither (B,) nor (B, 1).
    """
    logits = model(images)
    if logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits.squeeze(1)
    if logits.ndim != 1:
        raise ValueError(f"model returned shape {tuple(logits.shape)}, expected (B,) or (B, 1)")
    return logits


def _predict(model: Module, images: Tensor, device: str, batch_size: int) -> np.ndarray:
    """
    Predict labels for a whole set, in batches, without gradients.

    Args:
        model: The detector.
        images: uint8 (N, 128, 128, 3) on `device`.
        device: Device the model and images live on.
        batch_size: Inference batch size.

    Returns:
        float32 (N,) predictions, 1.0 synthetic and 0.0 real.
    """
    import torch

    device_type = _device_type(device)
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = _normalise(images[start : start + batch_size])
            with torch.autocast(device_type, dtype=torch.float16, enabled=device_type == "cuda"):
                logits = _forward_logits(model, batch)
            predicted = (logits > DECISION_LOGIT).to(torch.float32)
            out.append(predicted.cpu().numpy())
    return np.concatenate(out)


def evaluate_row(
    model: Module,
    x_eval: Tensor,
    y_eval: Tensor,
    gen_ids: np.ndarray,
    device: str = DEFAULT_DEVICE,
    batch_size: int = EVAL_BATCH_SIZE,
) -> dict[str, float]:
    """
    Compute one row of the transfer matrix: accuracy against all seven generators.

    Args:
        model: The trained detector.
        x_eval: uint8 (N, 128, 128, 3) test images on `device`.
        y_eval: float32 (N,) true labels.
        gen_ids: Per-row generator tags, REAL_TAG for the real block.
        device: Device the model and tensors live on.
        batch_size: Inference batch size.

    Returns:
        Generator -> accuracy over that cell.
    """
    predictions = _predict(model, x_eval, device, batch_size)
    return cell_accuracies(predictions, y_eval.detach().cpu().numpy(), gen_ids)


def _validation_scores(
    model: Module,
    x_val: Tensor,
    y_val: Tensor,
    device: str,
    batch_size: int,
) -> tuple[float, float]:
    """
    Score the internal validation carve-out, for checkpoint selection only.

    Args:
        model: The detector, mid-training.
        x_val: uint8 (N, 128, 128, 3) on `device` - the 10 % held out of the
            *train* split, source generator only.
        y_val: float32 (N,).
        device: Device the model and tensors live on.
        batch_size: Inference batch size.

    Returns:
        (accuracy, mean BCE loss).
    """
    import torch
    from torch.nn import functional as F

    device_type = _device_type(device)
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for start in range(0, len(x_val), batch_size):
            batch = _normalise(x_val[start : start + batch_size])
            targets = y_val[start : start + batch_size]
            with torch.autocast(device_type, dtype=torch.float16, enabled=device_type == "cuda"):
                logits = _forward_logits(model, batch)
            logits = logits.float()
            total_loss += float(F.binary_cross_entropy_with_logits(logits, targets)) * len(targets)
            correct += int(((logits > DECISION_LOGIT).to(targets.dtype) == targets).sum())
    return correct / len(x_val), total_loss / len(x_val)


def train_one(
    cache_dir: str | Path,
    strategy: Strategy,
    source: str,
    seed: int = 0,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    train_per_class: int = TRAIN_PER_CLASS,
    device: str = DEFAULT_DEVICE,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    model_out: str | Path | None = None,
) -> dict[str, object]:
    """
    Train one detector on `source` and evaluate it against all seven generators.

    AdamW, cosine decay to zero over `epochs`, BCE-with-logits, fp16 autocast on
    CUDA, horizontal flip as the only augmentation. The best epoch is chosen by
    accuracy on the internal validation carve-out - ties broken by the lower loss,
    which matters because that set holds only ~400 images and its accuracy is
    therefore quantised to 0.25 pp. The test set is not touched until the loop is
    over.

    Args:
        cache_dir: Cache root written by `build_cache`.
        strategy: One of STRATEGIES.
        source: Generator to train on; must be in GENERATORS.
        seed: Run seed. Seeds torch, and selects the arm's training sample and its
            train/validation split. The test set is independent of it.
        epochs: Training epochs; the cosine schedule spans exactly this many.
        batch_size: Optimiser batch size.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        train_per_class: Training images per class - halve it if the calibration
            run comes in slow.
        device: "cuda" on Colab; "cpu" works for the smoke test, with AMP off.
        results_dir: Root of the results tree; only used to derive the checkpoint
            path when `model_out` is not given. It does not write `metrics.json` -
            func _main does that.
        model_out: Where to write the selected `state_dict`. Defaults to
            `checkpoint_path(results_dir, strategy, source, seed)`, so every run
            saves its weights without the caller having to name a file. Pass an
            explicit path to put one elsewhere.

    Returns:
        A JSON-serialisable dict with `strategy`, `source`, `seed`, `cells`
        (generator -> accuracy), `cell_sizes`, `in_domain`, `val_accuracy`,
        `best_epoch`, `wall_clock_s`, `load_s`, `peak_vram_mb`, `checkpoint` and
        `config`.
    """
    import torch
    from torch.nn import functional as F

    from src.model import CompactCNN

    started = time.perf_counter()
    device_type = _device_type(device)
    use_amp = device_type == "cuda"

    torch.manual_seed(seed)
    if use_amp:
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
        # Input shape is fixed at 128x128 and only the last batch differs, so
        # autotuning pays for itself within the first epoch.
        torch.backends.cudnn.benchmark = True

    arm = load_arm(cache_dir, strategy, source, seed, device=device, train_per_class=train_per_class)
    load_s = time.perf_counter() - started
    print(
        f"[train] {strategy}/{source}/seed{seed}: "
        f"{len(arm.x_train)} train, {len(arm.x_val)} val, {len(arm.x_test)} test "
        f"loaded in {load_s:.0f}s",
        flush=True,
    )

    model = CompactCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
    scaler = torch.amp.GradScaler(device_type, enabled=use_amp)

    best_state: dict[str, Tensor] = {}
    best_accuracy = -1.0
    best_loss = float("inf")
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        # On-device permutation: the arm never leaves the GPU, so neither does the
        # shuffle. Driven by the seeded CUDA RNG, hence the manual_seed_all above.
        perm = torch.randperm(len(arm.x_train), device=device)
        train_loss = 0.0
        for start in range(0, len(perm), batch_size):
            index = perm[start : start + batch_size]
            batch = _normalise(arm.x_train[index])
            # The only augmentation. dims=[3] is width in NCHW - a horizontal flip.
            flip = torch.rand(len(index), device=device) < FLIP_PROBABILITY
            batch[flip] = torch.flip(batch[flip], dims=[3])
            with torch.autocast(device_type, dtype=torch.float16, enabled=use_amp):
                loss = F.binary_cross_entropy_with_logits(
                    _forward_logits(model, batch), arm.y_train[index]
                )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss) * len(index)
        scheduler.step()

        val_accuracy, val_loss = _validation_scores(model, arm.x_val, arm.y_val, device, EVAL_BATCH_SIZE)
        # Strictly better, or an equal accuracy at a lower loss: 400 validation
        # images quantise accuracy to 0.25 pp, so ties are common and the loss is
        # what separates them.
        if val_accuracy > best_accuracy or (val_accuracy == best_accuracy and val_loss < best_loss):
            best_accuracy, best_loss, best_epoch = val_accuracy, val_loss, epoch
            # Cloned, not referenced: the live parameters keep training.
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % PRINT_EVERY_EPOCHS == 0 or epoch == epochs - 1:
            print(
                f"[train]   epoch {epoch + 1:>3}/{epochs}  "
                f"train_loss {train_loss / len(arm.x_train):.4f}  "
                f"val_acc {val_accuracy:.4f}  val_loss {val_loss:.4f}",
                flush=True,
            )

    # Every reported number comes from the selected checkpoint, never the last.
    model.load_state_dict(best_state)
    cells = evaluate_row(model, arm.x_test, arm.y_test, arm.gen_ids, device, EVAL_BATCH_SIZE)
    cell_sizes = {g: int(_cell_mask(arm.gen_ids, g).sum()) for g in GENERATORS}

    # Saved before `metrics.json` is written, which is what keeps resume honest: a
    # directory holding metrics always holds the weights they were computed from.
    saved_to = (
        Path(model_out)
        if model_out is not None
        else checkpoint_path(results_dir, strategy, source, seed)
    )
    saved_to.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, saved_to)
    print(f"[train] checkpoint -> {saved_to}", flush=True)

    wall_clock_s = time.perf_counter() - started
    off_domain = [accuracy for g, accuracy in cells.items() if g != source]
    result: dict[str, object] = {
        "strategy": strategy,
        "source": source,
        "seed": seed,
        "cells": cells,
        "cell_sizes": cell_sizes,
        "in_domain": cells[source],
        "off_domain_mean": float(np.mean(off_domain)),
        "val_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "wall_clock_s": wall_clock_s,
        "load_s": load_s,
        "peak_vram_mb": (torch.cuda.max_memory_allocated() / 1e6) if use_amp else 0.0,
        # `as_posix` so the committed JSON reads the same whether the run happened on
        # Colab or on a Windows laptop.
        "checkpoint": saved_to.as_posix(),
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR(eta_min=0)",
            "loss": "binary_cross_entropy_with_logits",
            "augmentation": "horizontal_flip_p0.5",
            "normalisation": "uint8/255 -> (x - 0.5) / 0.5",
            "train_per_class": train_per_class,
            "n_train": int(len(arm.x_train)),
            "n_val": int(len(arm.x_val)),
            "n_test": int(len(arm.x_test)),
            "selection_metric": "internal val accuracy, ties by lower loss",
            "amp": "float16" if use_amp else "off",
            "device": device,
            "torch_version": torch.__version__,
            "cache_dir": str(cache_dir),
        },
    }
    print(
        f"[train] {strategy}/{source}/seed{seed}: in-domain {cells[source]:.4f}, "
        f"off-domain mean {result['off_domain_mean']:.4f}, "
        f"best epoch {best_epoch + 1}, {wall_clock_s:.0f}s",
        flush=True,
    )
    return result


def _main() -> None:
    """
    CLI entry point: run one cell of the matrix and write its metrics.

    Example:
        python -m src.train --strategy centre_crop --source BigGAN --seed 0 \
            --cache-dir /content/drive/MyDrive/university/deep_learning/cache
    """
    import argparse

    parser = argparse.ArgumentParser(description="Train one transfer-matrix cell.")
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--strategy", required=True, choices=STRATEGIES)
    parser.add_argument("--source", required=True, choices=GENERATORS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--train-per-class", type=int, default=TRAIN_PER_CLASS)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--model-out",
        type=Path,
        default=None,
        help=f"where to save the selected weights (default: {CHECKPOINT_NAME} beside metrics.json)",
    )
    parser.add_argument("--force", action="store_true", help="re-run even if metrics.json exists")
    args = parser.parse_args()

    out = metrics_path(args.results_dir, args.strategy, args.source, args.seed)
    # The whole resume mechanism, and the same condition the matrix driver uses.
    if out.exists() and not args.force:
        print(f"[train] {out} exists, skipping (pass --force to re-run)", flush=True)
        return

    result = train_one(
        args.cache_dir,
        args.strategy,
        args.source,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_per_class=args.train_per_class,
        device=args.device,
        results_dir=args.results_dir,
        model_out=args.model_out,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[train] metrics -> {out}", flush=True)


if __name__ == "__main__":
    _main()
