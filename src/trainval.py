"""
Step 1 diagnostic: train accuracy vs validation accuracy, from the saved checkpoints.

Answers one question - is the 0.908 diagonal a variance problem or a bias problem?

    train ~ 1.00, val ~ 0.91  ->  overfitting. More capacity makes it WORSE.
    train ~ val ~ 0.91        ->  underfitting. Widen the model.

    python -m src.trainval --cache-dir CACHE --ckpt-root CKPT --owner ido

Writes `train_val_gap_<owner>[_<tag>].json` into `--out-dir` (default `results/figures`) and
prints the same numbers as a table.

**`--tag` is what makes this reusable.** Every path is a parameter, so pointing
`--results-dir` and `--ckpt-root` at a second grid and passing `--tag d4` scores that grid
without touching the first file. That is the intended use: if the D4 ablation runs, the
before/after on this gap is the measurement that says whether augmentation closed it, and it
must not overwrite the flip-only numbers it is being compared against.

No retraining. Every run saved `model.pt`, and `load_arm` is seed-deterministic, so reloading
with the same seed reproduces byte-identical training rows. Scoring reuses `train._predict`,
so these numbers sit on exactly the same footing as the reported ones - and the recomputed
validation accuracy must reproduce what training recorded, which is checked rather than
assumed.

Conventions inherited from `analyze.py` and `train.py`:

- **torch is imported inside the functions that need it**, so the summaries and the verdict
  run on a laptop that has never had torch installed. Only func run_train_val touches a GPU.
- **`--ckpt-root` is not `--results-dir`.** `*.pt` is gitignored and /content is wiped on a
  runtime recycle, so the weights live only in each account's own Drive backup; the two trees
  have the same shape and are addressed by the same `checkpoint_path` helper.

**Both accounts run this, each over its own half.** Each commits a small per-owner JSON,
exactly the way `metrics.json` already merges the two halves of the matrix.
"""

from __future__ import annotations

import json
import time
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING, Final

from src.train import DEFAULT_DEVICE, DEFAULT_RESULTS_DIR, EVAL_BATCH_SIZE, _predict, checkpoint_path
from src.data import load_arm

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

__all__ = [
    "accuracy",
    "check_val_reproduction",
    "format_report",
    "run_train_val",
    "verdict",
]

DEFAULT_FIGURE_DIR: Final[Path] = Path("results/figures")

#: The recomputed validation accuracy must reproduce what training recorded - same seed,
#: same rows, same eval path. Anything above this means the arm being reloaded is not the
#: arm that was trained on, and the train accuracy beside it describes different images.
REPRODUCTION_TOLERANCE: Final[float] = 1e-9

#: Above this the network is effectively memorizing its training set, and widening it can
#: only push train toward 1.0 without moving val.
TRAIN_MEMORIZATION: Final[float] = 0.99

#: Gap thresholds in accuracy points. Above OVERFIT the variance channel is the constraint;
#: below UNDERFIT train barely beats val and the model is too small.
OVERFIT_GAP: Final[float] = 0.05
UNDERFIT_GAP: Final[float] = 0.02


# --------------------------------------------------------------------------- #
# Torch-free core: the reproduction guard, summaries, the verdict
# --------------------------------------------------------------------------- #


def check_val_reproduction(rows: list[dict]) -> float:
    """
    Verify that the recomputed validation accuracy matches what training recorded.

    The sibling of `rotation.check_reproduction`, on a different field: there the control
    is the k = 0 pass, here it is the validation set the checkpoint was selected on.

    Args:
        rows: Per-checkpoint result dicts as assembled by func run_train_val.

    Returns:
        The largest absolute drift observed.

    Raises:
        ValueError: If `rows` is empty, or if any row drifts beyond REPRODUCTION_TOLERANCE.
            A drift means the reloaded arm is not the arm that was trained on, so the train
            accuracy beside it was measured on different images and means nothing.
    """
    if not rows:
        raise ValueError("no rows to check")

    drift = max(abs(r["val_accuracy"] - r["val_recorded"]) for r in rows)
    if drift >= REPRODUCTION_TOLERANCE:
        raise ValueError(
            f"validation drifts {drift:.2e} from what training recorded - the train "
            f"accuracies are meaningless until this is explained. Check --cache-dir, "
            f"--ckpt-root and the seed."
        )
    return drift


def format_report(rows: list[dict]) -> str:
    """
    Render the summary tables printed after a run.

    Args:
        rows: Per-checkpoint result dicts.

    Returns:
        The per-strategy table and the widest per-source gap, ready to print.
    """
    lines = [f"{'arm':<13}{'train':>9}{'val':>9}{'gap':>9}{'n':>5}"]
    for strategy in sorted({r["strategy"] for r in rows}):
        subset = [r for r in rows if r["strategy"] == strategy]
        train = float(np.mean([r["train_accuracy"] for r in subset]))
        val = float(np.mean([r["val_accuracy"] for r in subset]))
        lines.append(f"{strategy:<13}{train:>9.4f}{val:>9.4f}{train - val:>+9.4f}{len(subset):>5}")

    # Per-source spread is large and non-uniform - Midjourney was the worst on both axes in
    # the flip-only grid - so the widest source is worth naming rather than averaging away.
    widest = max(rows, key=lambda r: r["gap"])
    lines.append("")
    lines.append(
        f"widest gap: {widest['strategy']}/{widest['source']}/seed{widest['seed']}  "
        f"train {widest['train_accuracy']:.4f}  val {widest['val_accuracy']:.4f}  "
        f"gap {widest['gap']:+.4f}"
    )
    return "\n".join(lines)


def verdict(rows: list[dict]) -> str:
    """
    Turn the measured train accuracy and gap into the step-1 decision.

    Args:
        rows: Per-checkpoint result dicts.

    Returns:
        Human-readable verdict naming the next diagnostic to run.
    """
    train_mean = float(np.mean([r["train_accuracy"] for r in rows]))
    gap_mean = float(np.mean([r["gap"] for r in rows]))

    if train_mean > TRAIN_MEMORIZATION:
        return (
            f"VARIANCE-LIMITED (memorizing the training set; train {train_mean:.4f}).\n"
            "   A bigger model makes this worse. The levers are D4 augmentation and\n"
            "   regularization - run Gate 0 (`python -m src.rotation`) before spending\n"
            "   GPU-hours on the ablation, then the learning curve (step 2)."
        )
    if gap_mean > OVERFIT_GAP:
        return (
            f"VARIANCE-LIMITED (clear overfitting, not yet total; gap {gap_mean:+.4f}).\n"
            "   Capacity is not the constraint. Run Gate 0 (`python -m src.rotation`)\n"
            "   first: it decides whether D4 has anything to give, for free."
        )
    if gap_mean < UNDERFIT_GAP:
        return (
            f"BIAS-LIMITED (train barely beats val; gap {gap_mean:+.4f}).\n"
            "   The model is too small. Run the width sweep (step 3) and widen.\n"
            "   Watch VRAM: peak is already 9.9 GB of ~15 GB at width x1.0."
        )
    return (
        f"MIXED - a moderate gap with room left on train (gap {gap_mean:+.4f}).\n"
        "   Run both step 2 (learning curve) and step 3 (width sweep);\n"
        "   whichever curve is still climbing is the real constraint."
    )


# --------------------------------------------------------------------------- #
# Torch: the scoring pass
# --------------------------------------------------------------------------- #


def accuracy(
    model: Module,
    images: Tensor,
    labels: Tensor,
    device: str = DEFAULT_DEVICE,
    batch_size: int = EVAL_BATCH_SIZE,
) -> float:
    """
    Fraction correct, using the same eval path as every reported cell.

    Args:
        model: The trained detector, already on `device`.
        images: uint8 (N, 128, 128, 3) on `device`.
        labels: float32 (N,) true labels, 1.0 synthetic and 0.0 real.
        device: Device the model and tensors live on.
        batch_size: Inference batch size.

    Returns:
        Accuracy over all N rows.
    """
    predictions = _predict(model, images, device, batch_size)
    return float((predictions == labels.detach().cpu().numpy()).mean())


def run_train_val(
    cache_dir: str | Path,
    ckpt_root: str | Path,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    device: str = DEFAULT_DEVICE,
) -> list[dict]:
    """
    Score every checkpoint this account holds on its own training set and its own val set.

    Runs on whichever checkpoints are present: `--results-dir` comes from git and lists the
    whole grid, while only this account's half has weights in its own Drive backup. The rest
    are skipped, which is expected rather than an error - the partner covers them.

    Args:
        cache_dir: Cache root the arms are loaded from.
        ckpt_root: Drive backup root holding `<strategy>/<source>/seed<k>/model.pt`.
        results_dir: Committed results tree, read for the recorded metrics.
        device: Device to score on.

    Returns:
        One dict per checkpoint, carrying the run's identity, both accuracies, the recorded
        validation accuracy for the reproduction check, and the gap.

    Raises:
        FileNotFoundError: If no checkpoint under `ckpt_root` matches any run.
        ValueError: If the recomputed validation accuracy does not reproduce the recorded one.
    """
    import torch
    from src.model import CompactCNN

    runs = sorted(Path(results_dir).glob("*/*/seed*/metrics.json"))
    print(f"metrics: {len(runs)} runs from {results_dir}")
    print(f"weights: {len(sorted(Path(ckpt_root).glob('*/*/seed*/model.pt')))} under {ckpt_root}\n")

    rows: list[dict] = []
    started = time.perf_counter()

    for path in runs:
        recorded = json.loads(path.read_text())
        strategy, source, seed = recorded["strategy"], recorded["source"], recorded["seed"]

        # Same tree shape as the results dir, so the same helper addresses it - there is
        # deliberately no second path convention to keep in step.
        checkpoint = checkpoint_path(ckpt_root, strategy, source, seed)
        if not checkpoint.exists():
            print(f"  skip {strategy}/{source}/seed{seed} - not this account's arm")
            continue

        arm = load_arm(cache_dir, strategy, source, seed, device=device)
        model = CompactCNN().to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))

        train_accuracy = accuracy(model, arm.x_train, arm.y_train, device)
        val_accuracy = accuracy(model, arm.x_val, arm.y_val, device)

        rows.append({
            "strategy": strategy,
            "source": source,
            "seed": seed,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "val_recorded": recorded["val_accuracy"],
            "in_domain": recorded["in_domain"],
            "gap": train_accuracy - val_accuracy,
            "n_train": int(len(arm.x_train)),
        })
        print(
            f"  {strategy:<12} {source:<11} seed{seed}  "
            f"train {train_accuracy:.4f}  val {val_accuracy:.4f}  "
            f"gap {train_accuracy - val_accuracy:+.4f}"
        )

        del arm, model
        torch.cuda.empty_cache()

    if not rows:
        raise FileNotFoundError(
            f"no model.pt under {ckpt_root}\n"
            f"  - is --owner right, and is Drive mounted on THIS account?\n"
            f"  - the weights are the BACKUP from 01_run_matrix cell 17, not the repo tree"
        )

    drift = check_val_reproduction(rows)
    print(f"\nval reproduction check: max drift {drift:.2e}  OK")
    print(f"{len(rows)} checkpoints ({time.perf_counter() - started:.0f}s)")
    return rows


def _main() -> None:
    """
    CLI entry point.

    Example:
        python -m src.trainval --cache-dir /content/cache --ckpt-root "$BACKUP/runs" --owner ido
        python -m src.trainval --cache-dir /content/cache --ckpt-root "$BACKUP/runs_d4" \\
            --results-dir results/runs_d4 --owner ido --tag d4
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument(
        "--ckpt-root",
        required=True,
        type=Path,
        help="Drive backup holding <strategy>/<source>/seed<k>/model.pt; NOT --results-dir, "
             "which carries only metrics.json back through git",
    )
    parser.add_argument(
        "--owner",
        required=True,
        help="names the output file, so the two accounts cannot collide when both commit",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="suffix for the output filename, e.g. `d4`. Use it whenever --results-dir points "
             "at a second grid, so the comparison is not overwritten by the thing it compares to",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    args = parser.parse_args()

    rows = run_train_val(args.cache_dir, args.ckpt_root, args.results_dir, args.device)

    suffix = f"_{args.tag}" if args.tag else ""
    out_path = args.out_dir / f"train_val_gap_{args.owner}{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))

    print()
    print(format_report(rows))
    print(f"\n-> {out_path}\n")
    print("VERDICT:", verdict(rows))


if __name__ == "__main__":
    _main()
