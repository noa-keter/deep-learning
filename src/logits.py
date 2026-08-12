"""
Save the raw decision scores of every trained checkpoint, over the full D4 orbit.

    python -m src.logits --cache-dir CACHE --ckpt-root CKPT --owner ido

Writes `logits_<owner>.npz` into `--out-dir` (default `results/figures`). One pass, no
training, minutes on a T4.

**Why this exists.** `train._predict` applies `> DECISION_LOGIT` internally and returns hard
labels, and `metrics.json` keeps only cell accuracies, so the model's confidence is discarded
at the point it is produced. Every way of combining models needs it back:

- **Across arms** - the headline says `center_crop` and `rescale` rank the generators with
  rho = 0.00, i.e. they fail on different generators, which is the textbook precondition for
  averaging to beat either one.
- **Across orientations** - `rotation.py` found off-domain accuracy varies by 0.009 over the
  whole orbit, which is exactly when test-time averaging helps.
- **Across seeds** - two checkpoints per cell already exist.

Averaging *labels* cannot do any of this: two models disagreeing 1-1 is a tie, and the
confidence that would have broken it has already been thrown away.

The eight orientations are the dihedral group: four quarter turns x {identity, mirror}. They
are exact permutations of the pixel grid - no interpolation, nothing resampled - which is what
makes them admissible in a study about resampling.

Conventions, all inherited and load-bearing:

- **torch is imported inside the functions that need it**, so `src.ensemble` can read and
  combine this file on a laptop with no GPU. Only func dump touches CUDA.
- **The upright, unmirrored pass must reproduce the recorded matrix.** Checked here against
  each run's `metrics.json`, in flipped rows and with the same fp16 budget `rotation.py` uses;
  a large drift means these scores did not come from the model that produced the report.
- **`--ckpt-root` is not `--results-dir`.** Weights live only in each account's Drive backup.

Ido's two arms, `center_crop` and `rescale`, are the pair the headline rests on, so cross-arm
combination needs nothing from Noa. Her 28 would add `random_crop` as a third view.
"""

from __future__ import annotations

import json
import time
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple

from src.train import DEFAULT_DEVICE, DEFAULT_RESULTS_DIR, EVAL_BATCH_SIZE, checkpoint_path
from src.rotation import CELL_ROWS, MAX_FLIPPED_PREDICTIONS, OFF_DOMAIN_ROWS, ROTATIONS
from src.data import GENERATORS, REAL_TAG, load_arm

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

__all__ = ["Orientation", "ORIENTATIONS", "dump", "run_key", "split_key"]

DEFAULT_FIGURE_DIR: Final[Path] = Path("results/figures")

#: Separator inside `.npz` member names. Underscores appear in no strategy or generator
#: name, so `split_key` can invert `run_key` unambiguously.
KEY_SEP: Final[str] = "__"


class Orientation(NamedTuple):
    """
    One element of D4: `k` quarter turns, then a horizontal mirror if `flip`.

    Fields:
        k: Quarter turns counter-clockwise, 0-3.
        flip: Whether to mirror horizontally after rotating.
    """

    k: int
    flip: bool

    @property
    def name(self) -> str:
        """Short label used in reports and as the row meaning inside the `.npz`."""
        return f"r{self.k}{'m' if self.flip else ''}"


#: The full dihedral group, upright-unmirrored first so index 0 is always the control pass.
#: Rotation is applied before the mirror; the two orders generate the same eight elements,
#: so this only fixes which label goes with which array row.
ORIENTATIONS: Final[tuple[Orientation, ...]] = tuple(
    Orientation(k, flip) for flip in (False, True) for k in ROTATIONS
)


def run_key(strategy: str, source: str, seed: int) -> str:
    """
    Name one run's array inside the `.npz`.

    Args:
        strategy: One of STRATEGIES.
        source: Generator the detector was trained on.
        seed: Run seed.

    Returns:
        e.g. `center_crop__ADM__0`.
    """
    return f"{strategy}{KEY_SEP}{source}{KEY_SEP}{seed}"


def split_key(key: str) -> tuple[str, str, int]:
    """
    Invert func run_key.

    Args:
        key: A member name written by func dump.

    Returns:
        (strategy, source, seed).

    Raises:
        ValueError: If the key is not in the three-part form, which would mean the file
            was written by something other than this module.
    """
    parts = key.split(KEY_SEP)
    if len(parts) != 3:
        raise ValueError(f"not a run key: {key!r}")
    return parts[0], parts[1], int(parts[2])


def _reproduction_rows(scores: np.ndarray, truth: np.ndarray, gen_ids: np.ndarray,
                       source: str, recorded: dict) -> tuple[int, int]:
    """
    Compare the upright pass against the recorded matrix, in flipped test rows.

    Args:
        scores: float32 (n_test,) raw logits from the upright, unmirrored pass.
        truth: float32 (n_test,) true labels, 1.0 synthetic.
        gen_ids: Per-row generator tags, REAL_TAG for the shared real block.
        source: Generator this checkpoint was trained on.
        recorded: The run's parsed `metrics.json`.

    Returns:
        (rows differing in the source cell, rows differing elsewhere), by the same
        residual accounting `rotation.flipped_predictions` uses - the real block is shared
        by all seven cells, so an in-domain move is not double-counted off-domain.
    """
    predictions = (scores > 0.0).astype(np.float32)
    correct = predictions == truth
    is_real = gen_ids == REAL_TAG
    cells = {g: float(correct[(gen_ids == g) | is_real].mean()) for g in GENERATORS}

    in_drift = abs(cells[source] - recorded["in_domain"])
    off = float(np.mean([a for g, a in cells.items() if g != source]))
    off_drift = abs(off - recorded["off_domain_mean"])
    return round(in_drift * CELL_ROWS), round(max(0.0, off_drift - in_drift) * OFF_DOMAIN_ROWS)


def _orient(images: Tensor, orientation: Orientation) -> Tensor:
    """
    Apply one D4 element to a uint8 NHWC batch.

    Args:
        images: uint8 (N, H, W, C) on any device.
        orientation: Which element to apply.

    Returns:
        uint8 (N, H, W, C). Exact - a permutation of the pixel grid, no interpolation.
    """
    import torch

    out = images if orientation.k == 0 else torch.rot90(images, orientation.k, dims=(1, 2))
    # dims=2 is width in NHWC. `rotation.rotate` mirrors nothing, so the two modules
    # deliberately do not share this helper.
    return torch.flip(out, dims=(2,)) if orientation.flip else out


def _score(model: Module, images: Tensor, device: str, batch_size: int) -> np.ndarray:
    """
    Raw logits for a whole set, in batches, without gradients.

    A near-copy of `train._predict` with the threshold removed - deliberately not a
    refactor of it. `_predict` is on the path that produced every reported number, and
    changing its shape to serve an analysis module would put that at risk for no gain.

    Args:
        model: The detector, already on `device`.
        images: uint8 (N, 128, 128, 3) on `device`.
        device: Device the model and images live on.
        batch_size: Inference batch size.

    Returns:
        float32 (N,) logits; positive means synthetic.
    """
    import torch

    from src.train import _device_type, _forward_logits, _normalize

    device_type = _device_type(device)
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = _normalize(images[start : start + batch_size])
            with torch.autocast(device_type, dtype=torch.float16, enabled=device_type == "cuda"):
                logits = _forward_logits(model, batch)
            # float32 on the way out: the scores are averaged across models downstream,
            # and fp16 has ~3 decimal digits, which is not enough to average safely.
            out.append(logits.float().cpu().numpy())
    return np.concatenate(out)


def dump(
    cache_dir: str | Path,
    ckpt_root: str | Path,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    device: str = DEFAULT_DEVICE,
) -> dict[str, np.ndarray]:
    """
    Score every checkpoint this account holds, over all eight orientations.

    Args:
        cache_dir: Cache root the arms are loaded from.
        ckpt_root: Drive backup holding `<strategy>/<source>/seed<k>/model.pt`.
        results_dir: Committed results tree, read for the recorded metrics.
        device: Device to score on.

    Returns:
        Arrays ready for `np.savez_compressed`: one float32 (8, n_test) per run under its
        func run_key, plus `y_test`, `gen_ids` and `orientations`.

    Raises:
        FileNotFoundError: If no checkpoint under `ckpt_root` matches any run.
        ValueError: If the test rows differ between arms, or if the upright pass does not
            reproduce the recorded matrix within MAX_FLIPPED_PREDICTIONS rows.
    """
    import torch

    from src.model import CompactCNN

    runs = sorted(Path(results_dir).glob("*/*/seed*/metrics.json"))
    print(f"metrics: {len(runs)} runs from {results_dir}")
    print(f"weights: {len(sorted(Path(ckpt_root).glob('*/*/seed*/model.pt')))} under {ckpt_root}\n")

    arrays: dict[str, np.ndarray] = {}
    truth: np.ndarray | None = None
    gen_ids: np.ndarray | None = None
    worst = (0, 0, "")
    started = time.perf_counter()

    for path in runs:
        recorded = json.loads(path.read_text())
        strategy, source, seed = recorded["strategy"], recorded["source"], recorded["seed"]

        checkpoint = checkpoint_path(ckpt_root, strategy, source, seed)
        if not checkpoint.exists():
            print(f"  skip {strategy}/{source}/seed{seed} - not this account's arm")
            continue

        arm = load_arm(cache_dir, strategy, source, seed, device=device)
        model = CompactCNN().to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))

        run_truth = arm.y_test.detach().cpu().numpy()
        if truth is None:
            truth, gen_ids = run_truth, arm.gen_ids
        # The whole point of combining models is that they are scored on the same images.
        # The test pool is drawn with a fixed seed and is independent of strategy, source
        # and run seed - but "independent by construction" is exactly the kind of claim
        # that stops being true after a refactor, so it is checked on every run.
        elif not (np.array_equal(run_truth, truth) and np.array_equal(arm.gen_ids, gen_ids)):
            raise ValueError(
                f"{strategy}/{source}/seed{seed} has different test rows from the first run "
                f"scored. Averaging across models would be comparing different images."
            )

        scores = np.stack([
            _score(model, _orient(arm.x_test, o), device, EVAL_BATCH_SIZE) for o in ORIENTATIONS
        ]).astype(np.float32)
        arrays[run_key(strategy, source, seed)] = scores

        in_rows, off_rows = _reproduction_rows(scores[0], run_truth, arm.gen_ids, source, recorded)
        if in_rows + off_rows > sum(worst[:2]):
            worst = (in_rows, off_rows, f"{strategy}/{source}/seed{seed}")
        if in_rows + off_rows > MAX_FLIPPED_PREDICTIONS:
            raise ValueError(
                f"{strategy}/{source}/seed{seed}: the upright pass differs from the recorded "
                f"matrix by {in_rows + off_rows} test rows, over the "
                f"{MAX_FLIPPED_PREDICTIONS}-row fp16 budget. These scores did not come from "
                f"the model that produced the report - check --cache-dir and --ckpt-root."
            )

        print(f"  {strategy:<12} {source:<11} seed{seed}  "
              f"logits [{scores.min():+.1f}, {scores.max():+.1f}]  "
              f"upright drift {in_rows + off_rows} row(s)")

        del arm, model
        torch.cuda.empty_cache()

    if not arrays:
        raise FileNotFoundError(
            f"no model.pt under {ckpt_root}\n"
            f"  - is --owner right, and is Drive mounted on THIS account?\n"
            f"  - the weights are the BACKUP from 01_run_matrix cell 17, not the repo tree"
        )

    print(f"\nupright reproduction: worst {sum(worst[:2])} row(s)"
          f"{' at ' + worst[2] if worst[2] else ''}, budget {MAX_FLIPPED_PREDICTIONS}")
    print(f"{len(arrays)} runs x {len(ORIENTATIONS)} orientations "
          f"({time.perf_counter() - started:.0f}s)")

    return {
        **arrays,
        "y_test": truth,
        "gen_ids": np.asarray(gen_ids),
        "orientations": np.asarray([o.name for o in ORIENTATIONS]),
    }


def _main() -> None:
    """
    CLI entry point.

    Example:
        python -m src.logits --cache-dir /content/cache --ckpt-root "$BACKUP/runs" --owner ido
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
        help="Drive backup holding <strategy>/<source>/seed<k>/model.pt; NOT --results-dir",
    )
    parser.add_argument("--owner", required=True, help="names the output file")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    args = parser.parse_args()

    arrays = dump(args.cache_dir, args.ckpt_root, args.results_dir, args.device)

    out_path = args.out_dir / f"logits_{args.owner}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    print(f"\n-> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print("Combine on the laptop with: python -m src.ensemble --logits " + str(out_path))


if __name__ == "__main__":
    _main()
