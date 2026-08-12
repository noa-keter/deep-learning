"""
Gate 0: is the trained detector orientation-sensitive? Zero training, seconds per checkpoint.

Decides whether the D4 augmentation ablation is worth 28 GPU-runs, before spending them.

    python -m src.rotation --cache-dir CACHE --ckpt-root CKPT --owner ido

Writes `rotation_sensitivity_<owner>.json` into `--out-dir` (default `results/figures`) and
prints the same numbers as a table, so nothing has to be read off a screenshot to be quoted.

**The physics question needs no experiment and must not be given one.** k x 90 deg is an exact
permutation of the pixel grid, the radially averaged spectrum is invariant under it to 1e-12
because (u, v) -> (-v, u) preserves radius, and CACHE_SIZE_PX % 8 == 0 so the JPEG 8x8 lattice
maps onto itself. Any spectral or high-pass-residual test of "does the fingerprint survive
rotation?" returns "invariant" by construction and carries no information.

**The open question is about the network, not the signal.** A CNN is equivariant to translation
and to nothing else, so an oriented filter bank has to learn each orientation separately. That
is measurable for free by scoring the checkpoints we already have on rotated test sets:

    rotated ~ upright          -> already orientation-invariant. D4 buys nothing. Stop.
    rotated collapses to ~0.5  -> ~7/8 of capacity is spent relearning one texture. Run it.
    tnr falls further than tpr -> the REAL class carries the orientation-dependent cue, which
                                  is the one mechanism by which D4 could move the
                                  cross-generator number rather than only the diagonal.

Conventions inherited from `analyze.py` and `train.py`, and not to be broken here:

- **torch is imported inside the functions that need it**, so the scoring arithmetic, the
  summaries and the verdict run on a laptop that has never had torch installed. Only
  func run_gate0 and func rotate touch a GPU.
- **The k = 0 pass must reproduce the recorded `in_domain` and `off_domain_mean` exactly.**
  `load_arm` is seed-deterministic and this reuses `train._predict`, so a drift means the arm
  being reloaded is not the arm that was trained on - and every rotated number computed beside
  it would be meaningless. Checked in func check_reproduction, never assumed.
- **Cell accuracies are recomputed here rather than taken from `cell_accuracies`**, because the
  whole point of this diagnostic is the tpr/tnr split that the cell view averages away. The cell
  arithmetic is duplicated in exactly one place, func score_predictions, and is checked against
  the recorded matrix on every run by the k = 0 pass.

**Both accounts run this, each over its own half.** `*.pt` is gitignored and /content is wiped on
a runtime recycle, so the weights live only in each account's own Drive backup - hence
`--ckpt-root`, which is a different tree from `--results-dir` but has the same shape.
"""

from __future__ import annotations

import json
import time
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING, Final

from src.train import DEFAULT_DEVICE, DEFAULT_RESULTS_DIR, EVAL_BATCH_SIZE, _predict, checkpoint_path
from src.data import GENERATORS, REAL_TAG, load_arm

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

__all__ = [
    "ROTATIONS",
    "check_reproduction",
    "format_report",
    "masked_accuracy",
    "rotate",
    "rotation_drop",
    "run_gate0",
    "score_checkpoint",
    "score_predictions",
    "score_rotation",
    "summarize",
    "verdict",
]

DEFAULT_FIGURE_DIR: Final[Path] = Path("results/figures")

#: The four elements of C4. k = 0 is the control and reproduces the recorded matrix.
#: Combined with the horizontal flip already in `train.py`, these generate the full D4
#: group - which is what the proposed augmentation would draw from uniformly.
ROTATIONS: Final[tuple[int, ...]] = (0, 1, 2, 3)

#: `Arm.x_test` is uint8 NHWC, so the spatial axes are 1 and 2. Rotating any other pair
#: would silently mix batch or channel into the rotation and produce garbage.
SPATIAL_DIMS: Final[tuple[int, int]] = (1, 2)

#: k = 0 must reproduce training's recorded numbers bit for bit - same seed, same rows,
#: same eval path.
REPRODUCTION_TOLERANCE: Final[float] = 1e-9

#: Verdict thresholds, in accuracy points, applied to the drop from k = 0 to the mean of
#: k = 1, 2, 3. Below FLAT the model is already orientation-invariant and D4 has nothing to
#: give; above LARGE there is enough headroom to justify the 28-run ablation.
ORIENTATION_FLAT_DROP: Final[float] = 0.02
ORIENTATION_LARGE_DROP: Final[float] = 0.10

#: How much further tnr must fall than tpr before the real class is called the carrier of
#: the orientation-dependent cue. Below this the asymmetry is not worth reporting.
CLASS_ASYMMETRY_DROP: Final[float] = 0.05


# --------------------------------------------------------------------------- #
# Torch-free core: scoring arithmetic, summaries, the verdict
# --------------------------------------------------------------------------- #


def masked_accuracy(predictions: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    """
    Fraction correct over a subset of the test rows.

    Args:
        predictions: float32 (N,) predicted labels, 1.0 synthetic and 0.0 real.
        truth: float32 (N,) true labels, same encoding.
        mask: Boolean (N,) selecting the rows to score.

    Returns:
        Accuracy over the masked rows.

    Raises:
        ValueError: If the mask selects nothing. An empty block would make the rate NaN
            and propagate quietly into the summary rather than failing here.
    """
    if not mask.any():
        raise ValueError("empty mask; a class block is missing from the test set")
    return float((predictions[mask] == truth[mask]).mean())


def score_predictions(
    predictions: np.ndarray,
    truth: np.ndarray,
    gen_ids: np.ndarray,
    source: str,
) -> dict[str, float]:
    """
    Turn one prediction vector into the five numbers this diagnostic reports.

    The torch-free half of func score_rotation, so the arithmetic every reported figure
    is made of can be tested on a laptop with no GPU - the split `train.py` keeps between
    `cell_accuracies` and `evaluate_row`.

    Args:
        predictions: float32 (N,) predicted labels, 1.0 synthetic and 0.0 real.
        truth: float32 (N,) true labels, same encoding.
        gen_ids: Per-row generator tags, REAL_TAG for the shared real block.
        source: Generator this checkpoint was trained on.

    Returns:
        Keys `in_domain`, `off_domain_mean`, `tnr`, `tpr_in_domain`, `tpr_off_domain`.
        The first two are cell accuracies - a cell is that generator's fakes plus the
        shared real block - and so are directly comparable to the recorded matrix. The
        last three are the mechanism probe.

    Raises:
        ValueError: If `source` is not one of GENERATORS, or if any generator block or
            the real block is missing from the test rows.
    """
    if source not in GENERATORS:
        raise ValueError(f"unknown source {source!r}; expected one of {GENERATORS}")

    is_real = gen_ids == REAL_TAG
    cells = {
        generator: masked_accuracy(predictions, truth, (gen_ids == generator) | is_real)
        for generator in GENERATORS
    }
    off_domain = [accuracy for name, accuracy in cells.items() if name != source]

    return {
        "in_domain": cells[source],
        "off_domain_mean": float(np.mean(off_domain)),
        # The real block is the same 500 images in every cell of every matrix, so tnr is
        # comparable across every checkpoint scored here without further qualification.
        "tnr": masked_accuracy(predictions, truth, is_real),
        "tpr_in_domain": masked_accuracy(predictions, truth, gen_ids == source),
        "tpr_off_domain": masked_accuracy(predictions, truth, ~is_real & (gen_ids != source)),
    }


def check_reproduction(rows: list[dict]) -> float:
    """
    Verify that the k = 0 pass reproduces what training recorded.

    Args:
        rows: Per-(checkpoint, rotation) result dicts, as assembled by func run_gate0.

    Returns:
        The largest absolute drift observed at k = 0.

    Raises:
        ValueError: If no k = 0 rows are present, or if any drifts beyond
            REPRODUCTION_TOLERANCE. A drift means the reloaded arm is not the arm that
            was trained on, which invalidates every rotated number computed beside it.
    """
    upright = [r for r in rows if r["k"] == 0]
    if not upright:
        raise ValueError("no k = 0 rows; the control pass is what makes the rest meaningful")

    drift = max(
        max(abs(r["in_domain"] - r["in_domain_recorded"]),
            abs(r["off_domain_mean"] - r["off_domain_recorded"]))
        for r in upright
    )
    if drift >= REPRODUCTION_TOLERANCE:
        raise ValueError(
            f"k = 0 drifts {drift:.2e} from the recorded matrix - the rotated numbers are "
            f"meaningless until this is explained. Check --owner, --cache-dir and the seed."
        )
    return drift


def summarize(rows: list[dict], key: str) -> dict[int, float]:
    """
    Mean of one metric across all checkpoints, per rotation.

    Args:
        rows: Per-(checkpoint, rotation) result dicts.
        key: Metric name present in every row.

    Returns:
        Rotation k -> mean over checkpoints.
    """
    return {
        k: float(np.mean([r[key] for r in rows if r["k"] == k]))
        for k in ROTATIONS
        if any(r["k"] == k for r in rows)
    }


def rotation_drop(rows: list[dict], key: str) -> float:
    """
    How far one metric falls from upright to the mean of the three turned orientations.

    Args:
        rows: Per-(checkpoint, rotation) result dicts.
        key: Metric name present in every row.

    Returns:
        Positive means the metric is worse when the test set is rotated.
    """
    per_k = summarize(rows, key)
    turned = [per_k[k] for k in ROTATIONS if k != 0 and k in per_k]
    return per_k[0] - float(np.mean(turned))


def verdict(rows: list[dict]) -> str:
    """
    Turn the measured drops into the Gate 0 decision.

    Args:
        rows: Per-(checkpoint, rotation) result dicts.

    Returns:
        Human-readable verdict naming the next action, with the class-asymmetry
        finding appended - that second half is what distinguishes "D4 lifts the
        diagonal" from "D4 could move the cross-generator number".
    """
    drop = rotation_drop(rows, "in_domain")
    tnr_drop = rotation_drop(rows, "tnr")
    tpr_drop = rotation_drop(rows, "tpr_in_domain")

    if drop < ORIENTATION_FLAT_DROP:
        head = (
            f"ORIENTATION-INVARIANT ALREADY (in-domain drop {drop:+.4f}).\n"
            "   The model does not need D4 - it has learned the texture at every\n"
            "   orientation from flips and the isotropy of the fingerprint alone.\n"
            "   DO NOT run the 28-run ablation. Report this as a one-line negative\n"
            "   finding: it forecloses 'why not just augment?' for free."
        )
    elif drop >= ORIENTATION_LARGE_DROP:
        head = (
            f"STRONGLY ORIENTATION-SENSITIVE (in-domain drop {drop:+.4f}).\n"
            "   Capacity is being spent relearning one texture per orientation.\n"
            "   RUN Gate 1: center_crop + rescale, 7 sources, 2 seeds, EPOCHS=80,\n"
            "   uniform D4 (k~U{0,1,2,3} x flip p=0.5). 28 runs, ~1.8 GPU-hours.\n"
            "   Keep it an ablation - the reported matrices stay flip-only."
        )
    else:
        head = (
            f"MILDLY ORIENTATION-SENSITIVE (in-domain drop {drop:+.4f}).\n"
            "   Real but small headroom. Run Gate 1 only if the report, spectra and\n"
            "   attribution are already on track; this is not worth slipping them."
        )

    if tnr_drop - tpr_drop >= CLASS_ASYMMETRY_DROP:
        tail = (
            f"\n   ASYMMETRY: tnr falls {tnr_drop:+.4f} vs tpr {tpr_drop:+.4f}.\n"
            "   The REAL class carries the orientation-dependent cue - consistent with\n"
            "   anisotropic-resampling and scene-orientation shortcuts. This is the one\n"
            "   mechanism by which D4 could move the CROSS-GENERATOR number, not just\n"
            "   the diagonal. Expect the effect in `rescale` first if it is real."
        )
    else:
        tail = (
            f"\n   No class asymmetry (tnr {tnr_drop:+.4f} vs tpr {tpr_drop:+.4f}).\n"
            "   Whatever D4 buys should land on the diagonal only."
        )
    return head + tail


def format_report(rows: list[dict]) -> str:
    """
    Render the two summary tables printed after a run.

    Args:
        rows: Per-(checkpoint, rotation) result dicts.

    Returns:
        The per-rotation table and the per-strategy drop table, ready to print.
    """
    lines = [f"{'k':<4}{'in-domain':>11}{'off-domain':>12}{'tnr':>9}{'tpr-in':>9}{'tpr-off':>10}"]
    for k in ROTATIONS:
        subset = [r for r in rows if r["k"] == k]
        if not subset:
            continue
        lines.append(
            f"{k:<4}"
            f"{np.mean([r['in_domain'] for r in subset]):>11.4f}"
            f"{np.mean([r['off_domain_mean'] for r in subset]):>12.4f}"
            f"{np.mean([r['tnr'] for r in subset]):>9.4f}"
            f"{np.mean([r['tpr_in_domain'] for r in subset]):>9.4f}"
            f"{np.mean([r['tpr_off_domain'] for r in subset]):>10.4f}"
        )

    # Per strategy, because the mechanism is arm-specific: `rescale` is the arm where real
    # images are aspect-distorted and generated ones are not, so it is where an
    # orientation-signed real-class cue would show up first.
    lines.append("")
    lines.append(f"{'arm':<13}{'k=0':>9}{'k=1,2,3':>10}{'drop':>9}")
    for strategy in sorted({r["strategy"] for r in rows}):
        subset = [r for r in rows if r["strategy"] == strategy]
        lines.append(f"{strategy:<13}"
                     f"{summarize(subset, 'in_domain')[0]:>9.4f}"
                     f"{summarize(subset, 'in_domain')[0] - rotation_drop(subset, 'in_domain'):>10.4f}"
                     f"{rotation_drop(subset, 'in_domain'):>+9.4f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Torch: rotation and the scoring pass
# --------------------------------------------------------------------------- #


def rotate(images: Tensor, k: int) -> Tensor:
    """
    Rotate a uint8 NHWC batch by k x 90 degrees counter-clockwise.

    Exact: a permutation of the pixel grid with no interpolation and no clipping, so the
    high-frequency residual that carries the generator fingerprint is preserved byte for
    byte. This is the property that makes D4 available at zero methodological cost.

    Args:
        images: uint8 (N, H, W, C) on any device.
        k: Number of quarter turns. k = 0 returns the input unchanged.

    Returns:
        uint8 (N, H, W, C). Only ever called on the square 128 x 128 cache, so the shape
        is always preserved.
    """
    import torch

    if k == 0:
        return images
    return torch.rot90(images, k, dims=SPATIAL_DIMS)


def score_rotation(
    model: Module,
    images: Tensor,
    truth: np.ndarray,
    gen_ids: np.ndarray,
    source: str,
    device: str = DEFAULT_DEVICE,
    batch_size: int = EVAL_BATCH_SIZE,
) -> dict[str, float]:
    """
    Score one checkpoint against one orientation of the test set.

    Args:
        model: The trained detector, already on `device`.
        images: uint8 (N, 128, 128, 3) test images on `device`, possibly rotated.
        truth: float32 (N,) true labels.
        gen_ids: Per-row generator tags, REAL_TAG for the shared real block.
        source: Generator this checkpoint was trained on.
        device: Device the model and tensors live on.
        batch_size: Inference batch size.

    Returns:
        As func score_predictions.
    """
    predictions = _predict(model, images, device, batch_size)
    return score_predictions(predictions, truth, gen_ids, source)


def score_checkpoint(
    model: Module,
    arm,
    source: str,
    device: str = DEFAULT_DEVICE,
) -> list[dict[str, float]]:
    """
    Score one checkpoint against all four orientations.

    Args:
        model: The trained detector, already on `device` with weights loaded.
        arm: The `Arm` this checkpoint was trained on; only its test fields are read.
        source: Generator this checkpoint was trained on.
        device: Device the model and tensors live on.

    Returns:
        One dict per rotation, each carrying its own `k`, in ROTATIONS order.
    """
    import torch

    truth = arm.y_test.detach().cpu().numpy()
    scores = []
    for k in ROTATIONS:
        images = rotate(arm.x_test, k)
        scores.append({"k": k, **score_rotation(model, images, truth, arm.gen_ids, source, device)})
        if k != 0:
            # 4000 x 128 x 128 x 3 uint8 is ~196 MB per rotated copy, on top of an arm
            # holding ~390 MB. Peak VRAM was 9.9 GB of ~15 GB during training, so this is
            # not tight - but there is no reason to stack four of them.
            del images
            torch.cuda.empty_cache()
    return scores


def run_gate0(
    cache_dir: str | Path,
    ckpt_root: str | Path,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    device: str = DEFAULT_DEVICE,
) -> list[dict]:
    """
    Score every checkpoint this account holds against all four orientations.

    Runs on whichever checkpoints are present: `--results-dir` comes from git and lists all
    56 runs, while only this account's 28 have weights in its own Drive backup. The rest are
    skipped, which is expected rather than an error - the partner covers them.

    Args:
        cache_dir: Cache root the arms are loaded from.
        ckpt_root: Drive backup root holding `<strategy>/<source>/seed<k>/model.pt`.
        results_dir: Committed results tree, read for the recorded metrics.
        device: Device to score on.

    Returns:
        One dict per (checkpoint, rotation), carrying the run's identity, the recorded
        values for the k = 0 check, and the five scores.

    Raises:
        FileNotFoundError: If no checkpoint under `ckpt_root` matches any run.
        ValueError: If the k = 0 pass does not reproduce the recorded matrix.
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

        scores = score_checkpoint(model, arm, source, device)
        for score in scores:
            rows.append({
                "strategy": strategy,
                "source": source,
                "seed": seed,
                "in_domain_recorded": recorded["in_domain"],
                "off_domain_recorded": recorded["off_domain_mean"],
                **score,
            })
        print(f"  {strategy:<12} {source:<11} seed{seed}  in-domain by k: "
              + "  ".join(f"{s['in_domain']:.3f}" for s in scores))

        del arm, model
        torch.cuda.empty_cache()

    if not rows:
        raise FileNotFoundError(
            f"no model.pt under {ckpt_root}\n"
            f"  - is --owner right, and is Drive mounted on THIS account?\n"
            f"  - the weights are the BACKUP from 01_run_matrix cell 17, not the repo tree"
        )

    drift = check_reproduction(rows)
    print(f"\nk=0 reproduction check: max drift {drift:.2e}  OK")
    print(f"{len(rows) // len(ROTATIONS)} checkpoints x {len(ROTATIONS)} orientations "
          f"({time.perf_counter() - started:.0f}s)")
    return rows


def _main() -> None:
    """
    CLI entry point.

    Example:
        python -m src.rotation --cache-dir /content/cache --ckpt-root "$BACKUP/runs" --owner ido
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
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    args = parser.parse_args()

    rows = run_gate0(args.cache_dir, args.ckpt_root, args.results_dir, args.device)

    out_path = args.out_dir / f"rotation_sensitivity_{args.owner}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))

    print()
    print(format_report(rows))
    print(f"\n-> {out_path}\n")
    print("VERDICT:", verdict(rows))


if __name__ == "__main__":
    _main()
