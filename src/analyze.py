"""
Analysis and figures: transfer matrices, the generator ranking, input-gradient
attribution, radially averaged spectra.

Four commands, run in that order, plus `all` which runs them in sequence. Only the
last two need the cache and a GPU; the first two read `metrics.json` alone and run
on a laptop with numpy + matplotlib.

    python -m src.analyze matrices    --results-dir results/runs --baseline-dir results/baseline
    python -m src.analyze ranking     --results-dir results/runs
    python -m src.analyze attribution --cache-dir CACHE --results-dir results/runs
    python -m src.analyze spectra     --cache-dir CACHE
    python -m src.analyze all         --cache-dir CACHE

Each command writes `<name>.png` and `<name>.json` into `--out-dir` (default
`results/figures`) and prints the same numbers as a table, so a figure never has
to be read off a screenshot to be quoted in the report.

Three conventions this module inherits and must not break:

- **torch is imported inside the functions that need it**, `src.model` included,
  so the matrix and ranking half of this file runs on a machine that has never
  had torch installed - the same split `data.py` and `train.py` keep.
- **One name addresses everything.** A strategy directory under `--results-dir` is
  named for whatever `train.py` was invoked with, and that one string addresses
  three things at once - the run directory, the cache directory, and the
  `strategy` field inside `metrics.json`. There is deliberately no alias table: a
  tree spelled some other way was built beside a cache spelled the same other way,
  so translating one without the other only moves the failure. func run_strategies
  refuses a name STRATEGIES does not contain, which says "rename the tree" rather
  than failing later at the cache lookup.
- **The test pool is loaded through `load_arm_numpy`, never re-derived.** Its test
  block is drawn with the fixed REAL_POOL_SEED and depends on neither the source
  generator nor the run seed, so attribution and spectra see byte-identical images
  to the ones every reported cell was scored on. Re-implementing that draw here
  would be one refactor away from silently analyzing different images.

Colors follow the `dataviz` skill's reference palette: a single-hue blue ramp for
magnitude (accuracy heatmaps, saliency maps), the blue-to-red diverging pair over a
neutral midpoint for the signed Spearman panel, and the fixed categorical order for
the seven generators - assigned by generator, never by rank, so a generator keeps
its color across every figure.
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from itertools import permutations
from typing import TYPE_CHECKING, Final, NamedTuple

from src.train import CHECKPOINT_NAME, DEFAULT_DEVICE, checkpoint_path
from src.data import (
    CACHE_SIZE_PX,
    GENERATORS,
    REAL_TAG,
    STRATEGIES,
    TEST_PER_GENERATOR,
    TEST_REAL_N,
    Strategy,
    load_arm_numpy,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, torch is not installed locally
    from torch import Tensor
    from torch.nn import Module

__all__ = [
    "Attribution",
    "MatrixSet",
    "attribution_for_arm",
    "border_mass",
    "discover_labels",
    "generator_color",
    "load_matrices",
    "load_test_pool",
    "matrix_summary",
    "mean_attribution",
    "off_diagonal_means",
    "plot_attribution",
    "plot_border_mass",
    "plot_matrices",
    "plot_ranking",
    "plot_spectra",
    "radial_profile",
    "ranking_table",
    "run_strategies",
    "saliency_maps",
    "spearman_permutation_p",
    "spearman_rho",
    "spectra_for_strategy",
    "spectral_difference",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Every tag a test row can carry: the seven generators, then the shared real block.
TAGS: Final[tuple[str, ...]] = (*GENERATORS, REAL_TAG)

DEFAULT_RESULTS_DIR: Final[Path] = Path("results/runs")
DEFAULT_BASELINE_DIR: Final[Path] = Path("results/baseline")
DEFAULT_FIGURE_DIR: Final[Path] = Path("results/figures")

#: `load_arm_numpy` always builds the training half, and analysis never uses it.
#: Asking for one image per class makes that half free while leaving the test pool -
#: the part that must match `train.py` exactly - untouched.
MIN_TRAIN_PER_CLASS: Final[int] = 1

#: Width of the ring PLAN.md defines as "the border". The `pad` arm zero-pads a
#: non-square image to a square, so if that model is reading its own padding rather
#: than the picture, its saliency concentrates here.
BORDER_PX: Final[int] = 16

#: Border mass of a saliency map that is spatially uniform - the null value the
#: measured numbers are read against. 1 - (96/128)^2 = 0.4375.
UNIFORM_BORDER_MASS: Final[float] = 1.0 - ((CACHE_SIZE_PX - 2 * BORDER_PX) / CACHE_SIZE_PX) ** 2

#: PLAN.md: "averaged over 200 images per generator per strategy". The cut ladder
#: allows 100 if time runs short; it is a CLI flag for that reason.
ATTRIBUTION_IMAGES: Final[int] = 200
ATTRIBUTION_BATCH: Final[int] = 64

#: The whole test block per tag - 500 images. The spectra are numpy-only and cost
#: seconds, so there is no reason to subsample them.
SPECTRUM_IMAGES: Final[int] = 500
SPECTRUM_BATCH: Final[int] = 64

#: One ring per pixel of radius at 128x128. Rings beyond CACHE_SIZE_PX / 2 fall in
#: the corners of the shifted spectrum, where the sampling is not isotropic, and are
#: dropped rather than averaged into a misleading tail.
RADIAL_BINS: Final[int] = 64

#: ITU-R BT.601 luminance, the same weighting PIL's "L" mode uses. Stated as a
#: constant because a different grayscale would change every spectrum by a scale
#: factor and nothing would look wrong.
LUMINANCE_WEIGHTS: Final[tuple[float, float, float]] = (0.299, 0.587, 0.114)

#: Exhaustive permutation tests are exact but factorial. Seven generators is 5,040
#: permutations, which is instant; beyond this the p-value is reported as NaN rather
#: than silently switching to an approximation nobody asked for.
MAX_EXACT_PERMUTATION_N: Final[int] = 8

#: Guards log10 of a zero spectral bin (the DC bin, which is zero by construction
#: once the image mean is subtracted) and division by a zero saliency sum.
EPSILON: Final[float] = 1e-12

#: The first radial bin is the DC term and is zero after mean subtraction; every
#: plotted and quoted curve starts one bin later.
FIRST_PLOTTED_BIN: Final[int] = 1

FIGURE_DPI: Final[int] = 160

#: Panels per row in the matrix figure. Four is the number of strategies, so the
#: arms fill row one and any baseline panels wrap below them.
MAX_PANEL_COLUMNS: Final[int] = 4

#: Inches of vertical padding between wrapped panel rows.
PANEL_ROW_PAD_IN: Final[float] = 0.22

#: Smallest gap between two direct labels on the slopegraph, as a fraction of the
#: y-range. About one line height at the sizes this figure is drawn at.
LABEL_GAP_FRACTION: Final[float] = 0.035

# dataviz reference palette. Sequential single hue (blue, steps 100 -> 700) for
# magnitude; the categorical order for series identity, assigned to GENERATORS by
# index so a generator keeps its color in every figure.
BLUE_RAMP: Final[tuple[str, ...]] = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)
CATEGORICAL: Final[tuple[str, ...]] = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)
DIVERGING: Final[tuple[str, str, str]] = ("#2a78d6", "#f0efec", "#e34948")
SURFACE: Final[str] = "#fcfcfb"
TEXT_PRIMARY: Final[str] = "#0b0b0b"
TEXT_SECONDARY: Final[str] = "#52514e"
GRID_COLOR: Final[str] = "#e6e5e1"

#: Seven series share four dash patterns, so color is never the only cue to
#: identity - the pairing (color, dash) is unique.
DASHES: Final[tuple[str, ...]] = ("-", "--", "-.", ":")


class MatrixSet(NamedTuple):
    """
    One 7x7 transfer matrix, averaged over the seeds that were found.

    Fields:
        label: Directory name the runs were read from.
        accuracy: (7, 7) float, rows = source, cols = target, mean over seeds.
            NaN where no run exists.
        spread: (7, 7) float, max minus min over seeds; 0.0 with one seed, NaN
            with none. Not a standard error - with two seeds a range is the
            honest statistic, and the binomial SE per cell is a separate number.
        n_seeds: (7, 7) int, how many runs each cell averages.
        seeds: The seeds present anywhere in this arm.
    """

    label: str
    accuracy: np.ndarray
    spread: np.ndarray
    n_seeds: np.ndarray
    seeds: tuple[int, ...]


class Attribution(NamedTuple):
    """
    Input-gradient saliency for one (label, source, seed) checkpoint.

    Fields:
        strategy: The arm the checkpoint came from.
        source: Generator the detector was trained on.
        seed: Run seed.
        maps: Tag -> (128, 128) mean saliency. Each image's saliency is
            normalized to sum to 1 before averaging, so the mean is a
            distribution over pixels and no single high-gradient image dominates.
        border_mass: Tag -> fraction of that mean map inside the outer
            BORDER_PX ring. Because the maps are per-image distributions, this
            equals the mean of the per-image border fractions.
        n_images: Images averaged per tag.
    """

    strategy: str
    source: str
    seed: int
    maps: dict[str, np.ndarray]
    border_mass: dict[str, float]
    n_images: int


# --------------------------------------------------------------------------- #
# Reading the run tree
# --------------------------------------------------------------------------- #


def _label_order(label: str) -> tuple[int, str]:
    """
    Sort key putting known strategies in STRATEGIES order and the rest after them.

    Keeps the four arms in the order `data.py` declares them - which is the order
    the report and the deck use - and lets a baseline rule directory, whose name is
    not a strategy at all, sort after them instead of raising.

    Args:
        label: Directory name.

    Returns:
        Sort key.
    """
    return (STRATEGIES.index(label), "") if label in STRATEGIES else (len(STRATEGIES), label)


def discover_labels(results_dir: str | Path, marker: str = "metrics.json") -> list[str]:
    """
    List the strategy directories that actually hold runs.

    Args:
        results_dir: Root of a results tree, e.g. "results/runs". The baseline
            tree has the same shape, so its rule directories are found the same
            way.
        marker: The file a directory must contain somewhere below it to count.
            The two analyses need different evidence - a matrix needs
            `metrics.json`, attribution needs the weights - and the two do not
            always arrive together: `metrics.json` is committed through git while
            `*.pt` is gitignored and copied through Drive, so either can be
            present on a given machine without the other.

    Returns:
        Directory names holding at least one `marker`, known strategies first in
        STRATEGIES order.
    """
    root = Path(results_dir)
    if not root.is_dir():
        return []
    labels = [child.name for child in root.iterdir() if child.is_dir() and any(child.rglob(marker))]
    return sorted(labels, key=_label_order)


def _check_strategy(name: str) -> Strategy:
    """
    Accept a strategy name only if `data.py` defines it.

    Refused rather than translated. One string addresses the run directory, the
    cache directory and the `strategy` field inside `metrics.json`, so a tree
    spelled some other way came with a cache spelled the same other way, and
    silently rewriting the name would move the failure to the cache lookup where
    it is harder to read.

    Args:
        name: Directory name, e.g. "center_crop".

    Returns:
        The same name, typed as a Strategy.

    Raises:
        KeyError: If it is not one of STRATEGIES.
    """
    if name not in STRATEGIES:
        raise KeyError(f"{name!r} is not a known strategy; expected one of {list(STRATEGIES)}")
    return name  # type: ignore[return-value]


def run_strategies(
    results_dir: str | Path,
    names: list[str] | None = None,
    marker: str = CHECKPOINT_NAME,
) -> list[Strategy]:
    """
    List the strategies whose runs are present, in STRATEGIES order.

    Args:
        results_dir: Root of the results tree.
        names: Restrict to these; None takes every one found.
        marker: Passed to func discover_labels; defaults to the checkpoint, since
            the commands that call this need weights rather than metrics.

    Returns:
        The usable strategies.

    Raises:
        KeyError: If an explicitly requested name is not a known strategy.
    """
    found = discover_labels(results_dir, marker)
    if names is None:
        return [_check_strategy(name) for name in found if name in STRATEGIES]

    strategies: list[Strategy] = []
    for name in names:
        # Checked before anything expensive runs: a typo here would otherwise
        # surface as an empty figure rather than as an error.
        strategy = _check_strategy(name)
        if name in found:
            strategies.append(strategy)
        else:
            print(f"[analyze] no runs under {Path(results_dir) / name}, skipping")
    return strategies


def _cell_accuracy(cell: object) -> float:
    """
    Read one cell's accuracy from either metrics format.

    `train.py` writes generator -> float; `baseline.py` writes generator -> dict
    with an "accuracy" key alongside TPR and TNR. Both are loaded through here so
    the CNN arms and the baseline rules land in one figure without special-casing.

    Args:
        cell: The value stored under `cells[<generator>]`.

    Returns:
        Accuracy as a float.

    Raises:
        TypeError: If the cell is neither a number nor a dict carrying "accuracy".
    """
    if isinstance(cell, dict):
        return float(cell["accuracy"])
    if isinstance(cell, (int, float)):
        return float(cell)
    raise TypeError(f"cannot read an accuracy out of {type(cell).__name__}: {cell!r}")


def load_matrices(
    results_dir: str | Path,
    labels: list[str] | None = None,
    seeds: list[int] | None = None,
) -> list[MatrixSet]:
    """
    Load every strategy's 7x7 matrix from a results tree.

    Args:
        results_dir: Root of the results tree.
        labels: Restrict to these directory names; None takes every one found.
        seeds: Restrict to these seeds; None takes every seed present.

    Returns:
        One MatrixSet per directory, known strategies first.
    """
    root = Path(results_dir)
    available = discover_labels(root)
    found = available if labels is None else [label for label in labels if label in available]

    out: list[MatrixSet] = []
    for label in found:
        # One (7, 7) layer per seed, filled in as runs are met, so the walk order
        # of the directories cannot affect which layer a run lands in.
        per_seed: dict[int, np.ndarray] = {}
        partial = False
        for seed_dir in sorted((root / label).glob("*/seed*")):
            suffix = seed_dir.name.removeprefix("seed")
            source = seed_dir.parent.name
            path = seed_dir / "metrics.json"
            if not suffix.isdigit() or source not in GENERATORS or not path.exists():
                continue
            seed = int(suffix)
            if seeds is not None and seed not in seeds:
                continue

            run = json.loads(path.read_text(encoding="utf-8"))
            # `baseline.py` writes this when it scored a pilot cache: smaller pools,
            # composition check skipped, and in its own words "not comparable to the
            # CNN runs". Such a panel is kept but renamed, because the one thing that
            # must not happen is it sitting unmarked beside panels it cannot be
            # compared with.
            partial = partial or bool(run.get("partial", False))
            layer = per_seed.setdefault(seed, np.full((len(GENERATORS), len(GENERATORS)), np.nan))
            row = GENERATORS.index(source)
            for column, target in enumerate(GENERATORS):
                layer[row, column] = _cell_accuracy(run["cells"][target])

        if not per_seed:
            continue
        if partial:
            print(f"[analyze] {label}: scored on a partial cache - not comparable to the CNN runs")
            label = f"{label} (partial)"
        seeds_present = tuple(sorted(per_seed))
        values = np.stack([per_seed[seed] for seed in seeds_present], axis=-1)

        # Written out rather than left to nanmean/nanmax, which warn on an
        # all-missing cell - and a missing cell is expected while half the matrix
        # is still running on the other account.
        finite = np.isfinite(values)
        counts = finite.sum(axis=-1)
        present = counts > 0
        totals = np.where(finite, values, 0.0).sum(axis=-1)
        highest = np.where(finite, values, -np.inf).max(axis=-1)
        lowest = np.where(finite, values, np.inf).min(axis=-1)
        out.append(
            MatrixSet(
                label=label,
                accuracy=np.where(present, totals / np.maximum(counts, 1), np.nan),
                spread=np.where(present, highest - lowest, np.nan),
                n_seeds=counts,
                seeds=seeds_present,
            )
        )
    return out


def _mean_present(values: np.ndarray) -> float:
    """
    Mean of the finite entries, NaN if there are none.

    `np.nanmean` is right except that it warns on an all-missing slice, and a
    half-finished matrix has those by design.

    Args:
        values: Any array.

    Returns:
        The mean of the finite entries.
    """
    finite = np.isfinite(values)
    return float(values[finite].mean()) if finite.any() else float("nan")


def matrix_summary(accuracy: np.ndarray) -> dict[str, float]:
    """
    Diagonal mean, off-diagonal mean, and the gap between them.

    That gap is the cross-generator failure the project is about, so it is
    computed once here and quoted everywhere rather than re-derived per figure.

    Args:
        accuracy: (7, 7) matrix, rows = source, cols = target. NaNs are skipped.

    Returns:
        {"diagonal_mean", "off_diagonal_mean", "gap"}.
    """
    diagonal_mean = _mean_present(np.diagonal(accuracy))
    off_mean = _mean_present(accuracy[~np.eye(len(GENERATORS), dtype=bool)])
    return {
        "diagonal_mean": diagonal_mean,
        "off_diagonal_mean": off_mean,
        "gap": diagonal_mean - off_mean,
    }


def off_diagonal_means(accuracy: np.ndarray, axis: str = "source") -> np.ndarray:
    """
    Per-generator off-diagonal mean, along either axis.

    The two answer different questions and PLAN.md's "rank the seven generators by
    mean off-diagonal accuracy" does not say which, so both are computed and the
    JSON carries both:

    - "source": row mean excluding the diagonal - how well a detector trained on
      this generator transfers to the other six. Matches `off_domain_mean` in each
      run's `metrics.json`.
    - "target": column mean excluding the diagonal - how detectable this generator
      is to detectors trained on the other six. The quantity a cross-generator
      table in the literature usually reports.

    Args:
        accuracy: (7, 7) matrix, rows = source, cols = target.
        axis: "source" or "target".

    Returns:
        (7,) means in GENERATORS order.

    Raises:
        ValueError: If axis is neither "source" nor "target".
    """
    if axis not in ("source", "target"):
        raise ValueError(f"axis must be 'source' or 'target', got {axis!r}")
    off_diagonal = ~np.eye(len(GENERATORS), dtype=bool)
    return np.asarray(
        [
            _mean_present(accuracy[index][off_diagonal[index]])
            if axis == "source"
            else _mean_present(accuracy[:, index][off_diagonal[:, index]])
            for index in range(len(GENERATORS))
        ]
    )


# --------------------------------------------------------------------------- #
# Ranking and rank correlation
# --------------------------------------------------------------------------- #


def _rankdata(values: np.ndarray) -> np.ndarray:
    """
    Rank values ascending, averaging tied ranks.

    Written out rather than taken from scipy so the ranking half of this module
    keeps the same dependency footprint as the rest of the laptop-side code.

    Args:
        values: (n,) finite values.

    Returns:
        (n,) float ranks, 1-based.
    """
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    # Flattened explicitly: numpy 2.0 made return_inverse keep the input's shape,
    # and this has to behave the same on either side of that change.
    inverse = np.reshape(inverse, -1)
    return (np.bincount(inverse, weights=ranks) / counts)[inverse]


def _centered_ranks(values: np.ndarray) -> np.ndarray:
    """
    Mean-centered ranks, normalized to unit length.

    Two of these dotted together is Spearman's rho, which is what makes the
    permutation test below a loop over dot products.

    Args:
        values: (n,) finite values.

    Returns:
        (n,) unit vector, or a zero vector if every value ties.
    """
    ranks = _rankdata(values)
    centered = ranks - ranks.mean()
    norm = float(np.linalg.norm(centered))
    return centered / norm if norm > EPSILON else centered


def spearman_rho(first: np.ndarray, second: np.ndarray) -> float:
    """
    Spearman rank correlation between two rankings of the same generators.

    Args:
        first: (n,) values.
        second: (n,) values, aligned with `first`.

    Returns:
        Rho in [-1, 1], or NaN if either side is entirely tied.
    """
    a, b = _centered_ranks(np.asarray(first, float)), _centered_ranks(np.asarray(second, float))
    if np.linalg.norm(a) < EPSILON or np.linalg.norm(b) < EPSILON:
        return float("nan")
    return float(np.dot(a, b))


def spearman_permutation_p(first: np.ndarray, second: np.ndarray) -> float:
    """
    Exact two-sided p-value for Spearman's rho, by enumerating permutations.

    With seven generators there are 5,040 orderings, so the exact null is cheaper
    than any approximation and needs no distributional assumption - which matters
    here, because n = 7 is far too small for the asymptotic t-approximation that
    the usual implementations fall back on.

    Args:
        first: (n,) values.
        second: (n,) values, aligned with `first`.

    Returns:
        Fraction of permutations whose |rho| is at least the observed |rho|, or
        NaN if n exceeds MAX_EXACT_PERMUTATION_N or either side is entirely tied.
    """
    a, b = _centered_ranks(np.asarray(first, float)), _centered_ranks(np.asarray(second, float))
    if len(a) > MAX_EXACT_PERMUTATION_N or np.linalg.norm(a) < EPSILON or np.linalg.norm(b) < EPSILON:
        return float("nan")

    observed = abs(float(np.dot(a, b)))
    hits = 0
    total = 0
    for order in permutations(range(len(a))):
        # Floating-point slack, so a permutation that reproduces the observed
        # ordering exactly is not excluded by a 1e-16 shortfall.
        hits += abs(float(np.dot(a[list(order)], b))) >= observed - 1e-9
        total += 1
    return hits / total


def ranking_table(matrices: list[MatrixSet], axis: str = "source") -> dict[str, object]:
    """
    Rank the generators within each arm and correlate the arms' rankings.

    A low rho between two strategies is the headline finding: it says a published
    cross-generator ranking depends on a preprocessing choice that papers do not
    report. A high rho everywhere is a null result and is reported as one.

    Args:
        matrices: One MatrixSet per arm.
        axis: Passed to func off_diagonal_means.

    Returns:
        {"axis", "generators", "values" (label -> list), "ranks" (label -> list,
        1 = highest accuracy), "spearman" (list of pairwise records)}.
    """
    values = {matrix.label: off_diagonal_means(matrix.accuracy, axis) for matrix in matrices}
    # Negated so rank 1 is the highest accuracy, which is how the table reads. A
    # generator with no runs has no rank: `argsort` would otherwise sort its NaN
    # to the end and hand it last place, which is a claim about it.
    ranks = {
        label: np.where(np.isfinite(value), _rankdata(-np.asarray(value)), np.nan)
        for label, value in values.items()
    }

    pairs: list[dict[str, object]] = []
    labels = list(values)
    for index, left in enumerate(labels):
        for right in labels[index + 1 :]:
            usable = np.isfinite(values[left]) & np.isfinite(values[right])
            if usable.sum() < 3:
                pairs.append({"a": left, "b": right, "rho": float("nan"), "p": float("nan"), "n": int(usable.sum())})
                continue
            left_values, right_values = values[left][usable], values[right][usable]
            pairs.append(
                {
                    "a": left,
                    "b": right,
                    "rho": spearman_rho(left_values, right_values),
                    "p": spearman_permutation_p(left_values, right_values),
                    "n": int(usable.sum()),
                }
            )

    return {
        "axis": axis,
        "generators": list(GENERATORS),
        "values": {label: [float(v) for v in value] for label, value in values.items()},
        "ranks": {label: [float(r) for r in rank] for label, rank in ranks.items()},
        "spearman": pairs,
    }


# --------------------------------------------------------------------------- #
# Input-gradient attribution (torch)
# --------------------------------------------------------------------------- #


def load_test_pool(cache_dir: str | Path, strategy: Strategy) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the 4,000-image test pool for one strategy, without its training half.

    Both splits must be cached: the arm loader reads the train metadata even when
    almost none of the training half is asked for.

    The pool is drawn inside `load_arm_numpy` with the fixed REAL_POOL_SEED and
    depends on neither the source generator nor the run seed, so any (source,
    seed) returns the identical images - which is exactly why this reuses that
    function instead of re-deriving the draw. `train_per_class` is set to
    MIN_TRAIN_PER_CLASS because the training half is unused here and would
    otherwise cost ~390 MB and a Drive read per call.

    Args:
        cache_dir: Cache root.
        strategy: Canonical strategy name.

    Returns:
        (images, gen_ids): uint8 (4000, 128, 128, 3) and the per-row tags.
    """
    *_, images, _, gen_ids = load_arm_numpy(
        cache_dir,
        strategy,
        GENERATORS[0],
        seed=0,
        train_per_class=MIN_TRAIN_PER_CLASS,
    )
    return images, gen_ids


def _load_model(path: str | Path, device: str) -> Module:
    """
    Rebuild CompactCNN from a saved `state_dict` and put it in eval mode.

    Args:
        path: Checkpoint written by `train.train_one`.
        device: Device to place the model on.

    Returns:
        The model, in eval mode.

    Raises:
        FileNotFoundError: If the checkpoint is missing.
    """
    import torch

    from src.model import CompactCNN

    weights = Path(path)
    if not weights.exists():
        raise FileNotFoundError(f"no checkpoint at {weights}; run that cell before analyzing it")

    model = CompactCNN()
    # weights_only: the file holds nothing but tensors, and the loader should not
    # be able to execute anything if that ever stops being true.
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    return model.to(device).eval()


def border_mass(saliency: np.ndarray) -> float:
    """
    Fraction of a saliency map lying in the outer BORDER_PX ring.

    Args:
        saliency: (128, 128) non-negative map.

    Returns:
        Ring sum divided by total sum; UNIFORM_BORDER_MASS for a flat map.
    """
    ring = np.ones(saliency.shape, dtype=bool)
    ring[BORDER_PX:-BORDER_PX, BORDER_PX:-BORDER_PX] = False
    total = float(saliency.sum())
    return float(saliency[ring].sum() / total) if total > EPSILON else float("nan")


def saliency_maps(
    model: Module,
    images: np.ndarray,
    device: str = DEFAULT_DEVICE,
    batch_size: int = ATTRIBUTION_BATCH,
) -> np.ndarray:
    """
    Input-gradient saliency for a set of images: `max_channel |df/dx|`.

    The method taught in Lecture 10, slides 53-57 - the gradient of the raw logit
    with respect to the input pixels - and deliberately not Grad-CAM, which would
    read a 8x8 feature map and could not resolve a 16-pixel border at all.

    No autocast: fp16 gradients would quantize the small values that make up most
    of a saliency map, and one backward pass over 200 images costs nothing.

    Args:
        model: A CompactCNN in eval mode on `device`.
        images: uint8 (N, 128, 128, 3) on the host.
        device: Device the model lives on.
        batch_size: Images per backward pass.

    Returns:
        float32 (N, 128, 128), each image normalized to sum to 1. An image whose
        gradient vanishes entirely comes back as all-NaN rather than as a
        silently uniform map.
    """
    import torch

    from src.model import normalize

    out = np.empty((len(images), CACHE_SIZE_PX, CACHE_SIZE_PX), dtype=np.float32)
    for start in range(0, len(images), batch_size):
        chunk = np.ascontiguousarray(images[start : start + batch_size])
        batch = normalize(torch.from_numpy(chunk).to(device)).requires_grad_(True)
        logits = model(batch)
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        # The sum over the batch is a sum of independent terms, so each image's
        # gradient is its own - one backward pass gives the whole batch.
        gradient: Tensor = torch.autograd.grad(logits.sum(), batch)[0]
        saliency = gradient.abs().amax(dim=1)
        totals = saliency.sum(dim=(1, 2), keepdim=True)
        out[start : start + len(chunk)] = (saliency / totals).detach().cpu().numpy()
    return out


def attribution_for_arm(
    cache_dir: str | Path,
    results_dir: str | Path,
    strategy: Strategy,
    source: str,
    seed: int = 0,
    n_images: int = ATTRIBUTION_IMAGES,
    device: str = DEFAULT_DEVICE,
    batch_size: int = ATTRIBUTION_BATCH,
    pool: tuple[np.ndarray, np.ndarray] | None = None,
) -> Attribution:
    """
    Mean saliency and border mass for one checkpoint, per test tag.

    Args:
        cache_dir: Cache root.
        results_dir: Root of the results tree holding the checkpoint.
        strategy: One of STRATEGIES; names both the run directory and the cache.
        source: Generator the detector was trained on.
        seed: Run seed.
        n_images: Images per tag; the first `n_images` of each tag's block, which
            is a fixed subset because the block itself is drawn deterministically.
        device: Device to run on.
        batch_size: Images per backward pass.
        pool: Output of func load_test_pool, to avoid reloading it per source.

    Returns:
        An Attribution.
    """
    images, gen_ids = load_test_pool(cache_dir, strategy) if pool is None else pool
    model = _load_model(checkpoint_path(results_dir, strategy, source, seed), device)

    maps: dict[str, np.ndarray] = {}
    masses: dict[str, float] = {}
    for tag in TAGS:
        rows = np.flatnonzero(gen_ids == tag)[:n_images]
        if not len(rows):
            continue
        saliency = saliency_maps(model, images[rows], device, batch_size)
        # An image whose gradient vanished comes back all-NaN; dropped rather than
        # averaged in, and reported, because a silent drop would change what the
        # printed image count means.
        usable = np.isfinite(saliency).all(axis=(1, 2))
        if not usable.all():
            print(f"[analyze]   {tag}: {int((~usable).sum())} of {len(rows)} images had no gradient")
        if not usable.any():
            continue
        mean_map = saliency[usable].mean(axis=0)
        maps[tag] = mean_map
        masses[tag] = border_mass(mean_map)

    return Attribution(
        strategy=strategy,
        source=source,
        seed=seed,
        maps=maps,
        border_mass=masses,
        n_images=n_images,
    )


def mean_attribution(attributions: list[Attribution]) -> dict[str, dict[str, np.ndarray]]:
    """
    Average the per-source saliency maps within each arm.

    PLAN.md asks for a map "per generator per strategy", and an arm has seven
    trained detectors. Averaging over the source models answers that question
    about the strategy rather than about one arbitrarily chosen run.

    Args:
        attributions: Per-checkpoint results, any mix of arms.

    Returns:
        label -> tag -> (128, 128) mean map.
    """
    out: dict[str, dict[str, list[np.ndarray]]] = {}
    for item in attributions:
        for tag, saliency in item.maps.items():
            out.setdefault(item.strategy, {}).setdefault(tag, []).append(saliency)
    return {
        label: {tag: np.mean(stack, axis=0) for tag, stack in tags.items()}
        for label, tags in out.items()
    }


# --------------------------------------------------------------------------- #
# Radially averaged spectra (numpy only)
# --------------------------------------------------------------------------- #


def _radial_bins(size_px: int = CACHE_SIZE_PX, n_bins: int = RADIAL_BINS) -> tuple[np.ndarray, np.ndarray]:
    """
    Ring index of every pixel of a shifted spectrum, and each ring's pixel count.

    Args:
        size_px: Spectrum side length.
        n_bins: Number of rings spanning radius 0 to size_px / 2.

    Returns:
        (index, counts): int (size_px, size_px) with -1 outside the last ring -
        the corners, which are not isotropically sampled - and (n_bins,) counts.
    """
    center = size_px // 2
    rows, columns = np.indices((size_px, size_px))
    radius = np.hypot(rows - center, columns - center)
    index = np.floor(radius * (2 * n_bins / size_px)).astype(np.int64)
    index[index >= n_bins] = -1
    counts = np.bincount(index[index >= 0], minlength=n_bins).astype(np.float64)
    return index, counts


def radial_profile(magnitude: np.ndarray, n_bins: int = RADIAL_BINS) -> np.ndarray:
    """
    Average a shifted magnitude spectrum over concentric rings.

    Args:
        magnitude: (size, size) non-negative, already `fftshift`ed.
        n_bins: Number of rings.

    Returns:
        (n_bins,) mean magnitude per ring, ring 0 being DC.
    """
    index, counts = _radial_bins(magnitude.shape[0], n_bins)
    inside = index >= 0
    totals = np.bincount(index[inside], weights=magnitude[inside], minlength=n_bins)
    return totals / np.maximum(counts, 1.0)


def _mean_magnitude(images: np.ndarray, batch_size: int = SPECTRUM_BATCH) -> np.ndarray:
    """
    Mean magnitude spectrum of a set of images.

    Grayscale by BT.601 luminance, per-image mean subtracted, `fft2`, `fftshift`,
    magnitude. No window function: the `pad` arm's zero border is a real feature
    of that arm and a window would taper exactly the region under test. That has
    to be said in the figure caption, since a windowed spectrum is the usual
    default and the difference is invisible in the plot.

    Args:
        images: uint8 (N, 128, 128, 3).
        batch_size: Images per FFT call.

    Returns:
        (128, 128) mean magnitude, shifted so DC is at the center.
    """
    weights = np.asarray(LUMINANCE_WEIGHTS, dtype=np.float64)
    total = np.zeros(images.shape[1:3], dtype=np.float64)
    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size].astype(np.float64) @ weights
        chunk -= chunk.mean(axis=(1, 2), keepdims=True)
        total += np.abs(np.fft.fftshift(np.fft.fft2(chunk, axes=(1, 2)), axes=(1, 2))).sum(axis=0)
    return total / len(images)


def spectra_for_strategy(
    cache_dir: str | Path,
    strategy: Strategy,
    n_images: int = SPECTRUM_IMAGES,
    n_bins: int = RADIAL_BINS,
    pool: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """
    Radially averaged spectrum per class for one strategy.

    Args:
        cache_dir: Cache root.
        strategy: Canonical strategy name.
        n_images: Images per tag.
        n_bins: Radial rings.
        pool: Output of func load_test_pool, to avoid reloading it.

    Returns:
        tag -> (n_bins,) mean magnitude profile, including REAL_TAG.
    """
    images, gen_ids = load_test_pool(cache_dir, strategy) if pool is None else pool
    out: dict[str, np.ndarray] = {}
    for tag in TAGS:
        rows = np.flatnonzero(gen_ids == tag)[:n_images]
        if len(rows):
            out[tag] = radial_profile(_mean_magnitude(images[rows]), n_bins)
    return out


def spectral_difference(profiles: dict[str, np.ndarray], generator: str) -> np.ndarray:
    """
    Real minus fake, in log magnitude - the informative curve.

    Both classes share the same scene statistics at low frequency, so the raw
    curves sit on top of each other and only their difference is readable.

    Args:
        profiles: Output of func spectra_for_strategy.
        generator: Which fake class to subtract.

    Returns:
        (n_bins,) log10(real) - log10(fake).
    """
    real = np.maximum(profiles[REAL_TAG], EPSILON)
    fake = np.maximum(profiles[generator], EPSILON)
    return np.log10(real) - np.log10(fake)


def _frequency_axis(n_bins: int = RADIAL_BINS) -> np.ndarray:
    """
    Ring centers in cycles per pixel, 0 to 0.5 at Nyquist.

    The rings span radius 0 to size/2 by construction, so the axis depends on the
    ring count alone and not on the image size.

    Args:
        n_bins: Radial rings.

    Returns:
        (n_bins,) frequencies.
    """
    return (np.arange(n_bins) + 0.5) * (0.5 / n_bins)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def _pyplot():
    """
    Import pyplot with the non-interactive backend and this project's styling.

    Returns:
        The `matplotlib.pyplot` module.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "text.color": TEXT_PRIMARY,
            "axes.labelcolor": TEXT_SECONDARY,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "axes.edgecolor": GRID_COLOR,
            "grid.color": GRID_COLOR,
            "font.size": 8,
            "axes.titlesize": 9,
            "figure.dpi": FIGURE_DPI,
        }
    )
    return plt


def _blue_colormap():
    """
    The dataviz sequential blue ramp as a matplotlib colormap.

    Returns:
        A LinearSegmentedColormap over BLUE_RAMP.
    """
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("dataviz_blue", BLUE_RAMP)


def _diverging_colormap():
    """
    The dataviz diverging pair, blue to a neutral gray to red.

    Returns:
        A LinearSegmentedColormap over DIVERGING.
    """
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("dataviz_diverging", DIVERGING)


def generator_color(generator: str) -> str:
    """
    The fixed color of one generator.

    Assigned by identity, never by rank, so a generator keeps its color when a
    figure reorders the series - which the ranking figure does by construction.

    Args:
        generator: Member of GENERATORS.

    Returns:
        A hex color.
    """
    return CATEGORICAL[GENERATORS.index(generator) % len(CATEGORICAL)]


def _save(fig, out_png: str | Path) -> Path:
    """
    Write a figure and report where it went.

    Args:
        fig: The figure.
        out_png: Destination path; parents are created.

    Returns:
        The written path.
    """
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"[analyze] figure -> {out}")
    return out


def plot_matrices(matrices: list[MatrixSet], out_png: str | Path) -> Path:
    """
    The four transfer matrices, plus any baseline panels, on one color scale.

    Claim: the diagonal is near-perfect in every arm and the off-diagonal is not,
    and how far apart they sit depends on which equalization was applied.

    Args:
        matrices: Panels, in the order they should appear.
        out_png: Destination path.

    Returns:
        The written path.

    Raises:
        ValueError: If there is nothing to plot.
    """
    if not matrices:
        raise ValueError("no matrices to plot; check --results-dir")
    plt = _pyplot()

    finite = np.concatenate([matrix.accuracy[np.isfinite(matrix.accuracy)] for matrix in matrices])
    # Floored to a 5-point step below the worst cell so the scale is round, and
    # capped at chance so the low end always means "no better than guessing".
    vmin = min(0.5, np.floor(finite.min() * 20) / 20)

    # Wrapped rather than laid out in one row. Six panels in a row is 2,500 px
    # wide, and scaled into a report column that leaves the cell numbers - which
    # are the actual data here - too small to read. Four per row also splits the
    # panels where the meaning splits: the arms on one row, the baselines below.
    n_columns = min(len(matrices), MAX_PANEL_COLUMNS)
    n_rows = -(-len(matrices) // n_columns)
    # Constrained layout rather than tight_layout: the shared colorbar spans every
    # panel, and with two rows of rotated tick labels the default spacing puts the
    # second row's titles on top of the first row's x labels.
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(2.9 * n_columns + 1.1, 3.6 * n_rows),
        squeeze=False,
        layout="constrained",
    )
    if n_rows > 1:
        # Extra inches between rows: the rotated tick labels hanging below one row
        # otherwise graze the titles of the row beneath.
        fig.get_layout_engine().set(h_pad=PANEL_ROW_PAD_IN)
    image = None
    for index, matrix in enumerate(matrices):
        ax = axes[index // n_columns][index % n_columns]
        column = index % n_columns
        image = ax.imshow(
            matrix.accuracy, cmap=_blue_colormap(), vmin=vmin, vmax=1.0, interpolation="nearest"
        )
        summary = matrix_summary(matrix.accuracy)
        ax.set_title(
            f"{matrix.label}\ndiag {100 * summary['diagonal_mean']:.1f}  "
            f"off {100 * summary['off_diagonal_mean']:.1f}  "
            f"gap {100 * summary['gap']:+.1f} pp",
            color=TEXT_PRIMARY,
        )
        ax.set_xticks(range(len(GENERATORS)), GENERATORS, rotation=45, ha="right")
        ax.set_yticks(range(len(GENERATORS)), GENERATORS if column == 0 else [""] * len(GENERATORS))
        for row in range(len(GENERATORS)):
            for col in range(len(GENERATORS)):
                value = matrix.accuracy[row, col]
                if not np.isfinite(value):
                    ax.text(col, row, "-", ha="center", va="center", color=TEXT_SECONDARY)
                    continue
                # The number is the datum; the color is the overview. White ink
                # once the fill is dark enough to swallow black.
                shade = (value - vmin) / (1.0 - vmin)
                ax.text(
                    col,
                    row,
                    f"{100 * value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=SURFACE if shade > 0.55 else TEXT_PRIMARY,
                )

    for index in range(len(matrices), n_rows * n_columns):
        axes[index // n_columns][index % n_columns].axis("off")

    bar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    bar.set_label("cell accuracy (%)")
    # Figure-level rather than per-panel: with the panels wrapped onto two rows, a
    # per-panel xlabel lands directly on the next row's title.
    fig.supxlabel("evaluated on", color=TEXT_SECONDARY)
    fig.supylabel("trained on", color=TEXT_SECONDARY)
    bar.set_ticks(np.linspace(vmin, 1.0, 6), labels=[f"{100 * t:.0f}" for t in np.linspace(vmin, 1.0, 6)])
    fig.suptitle(
        "Cross-generator transfer, per equalization strategy (cell = "
        f"{TEST_PER_GENERATOR} target fakes + the shared {TEST_REAL_N} reals, "
        "binomial SE ~1.6 pp)",
        color=TEXT_PRIMARY,
    )
    return _save(fig, out_png)


def _spread_labels(values: np.ndarray, min_gap: float) -> np.ndarray:
    """
    Nudge label positions apart so none overlaps its neighbor, keeping their order.

    A slopegraph's whole point is where the lines end up, so two arms landing a
    tenth of a point apart is information - but their two labels then print on top
    of each other. Only the label moves; the marker stays on the datum.

    Args:
        values: (n,) label positions in data coordinates.
        min_gap: Smallest acceptable separation, same units.

    Returns:
        (n,) adjusted positions, in the input order.
    """
    adjusted = np.asarray(values, dtype=float).copy()
    order = np.argsort(adjusted)
    for previous, current in zip(order, order[1:]):
        if adjusted[current] - adjusted[previous] < min_gap:
            adjusted[current] = adjusted[previous] + min_gap
    return adjusted


def plot_ranking(table: dict[str, object], out_png: str | Path) -> Path:
    """
    The generator ranking per strategy, and the rank correlation between them.

    Claim: if the lines cross, the ordering of generators by cross-generator
    accuracy is a function of the preprocessing and not of the generators.

    Args:
        table: Output of func ranking_table.
        out_png: Destination path.

    Returns:
        The written path.
    """
    plt = _pyplot()
    values: dict[str, list[float]] = table["values"]  # type: ignore[assignment]
    labels = list(values)
    pairs: list[dict[str, object]] = table["spearman"]  # type: ignore[assignment]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.55, 1.0]})

    slope = axes[0]
    x = np.arange(len(labels))
    ends: dict[str, float] = {}
    for index, generator in enumerate(GENERATORS):
        series = np.asarray([values[label][index] for label in labels], dtype=float)
        slope.plot(
            x,
            100 * series,
            DASHES[index % len(DASHES)],
            color=generator_color(generator),
            marker="o",
            markersize=5,
            linewidth=2.0,
            label=generator,
        )
        if np.isfinite(series[-1]):
            ends[generator] = 100 * float(series[-1])

    # Direct labels at the right end - seven series is past the point where a
    # legend alone keeps identity readable - pushed apart where two arms land
    # within a line-height of each other, which on real data they do.
    low, high = slope.get_ylim()
    positions = _spread_labels(np.asarray(list(ends.values())), LABEL_GAP_FRACTION * (high - low))
    for (generator, value), position in zip(ends.items(), positions):
        slope.annotate(
            generator,
            (x[-1], value),
            xytext=(x[-1] + 0.06, position),
            textcoords="data",
            va="center",
            fontsize=7,
            color=TEXT_SECONDARY,
        )
    slope.set_xticks(x, labels)
    slope.set_xlim(-0.25, len(labels) - 0.55)
    # Direct labels carry identity on this panel; the legend is here because a
    # figure with more than one series should never need the reader to trust that.
    slope.legend(frameon=False, fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    slope.set_ylabel(f"mean off-diagonal accuracy (%), by {table['axis']}")
    slope.set_title("Generator ranking per strategy", color=TEXT_PRIMARY)
    slope.grid(axis="y", linewidth=0.6)
    slope.set_axisbelow(True)
    for spine in ("top", "right"):
        slope.spines[spine].set_visible(False)

    grid = axes[1]
    # The diagonal is 1 by definition and would otherwise take the strongest
    # color in the panel, drawing the eye to the one number that says nothing.
    rho = np.full((len(labels), len(labels)), np.nan)
    lookup = {(str(pair["a"]), str(pair["b"])): pair for pair in pairs}
    for row, left in enumerate(labels):
        for column, right in enumerate(labels):
            pair = lookup.get((left, right)) or lookup.get((right, left))
            if pair is not None:
                rho[row, column] = float(pair["rho"])  # type: ignore[arg-type]
    colormap = _diverging_colormap()
    colormap.set_bad(SURFACE)
    grid.imshow(rho, cmap=colormap, vmin=-1.0, vmax=1.0, interpolation="nearest")
    grid.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    grid.set_yticks(range(len(labels)), labels)
    for row, left in enumerate(labels):
        for column, right in enumerate(labels):
            pair = None if row == column else (lookup.get((left, right)) or lookup.get((right, left)))
            if pair is None:
                continue
            p_value = float(pair["p"])  # type: ignore[arg-type]
            grid.text(
                column,
                row,
                # `+ 0.0` so a rho that rounds to zero from below prints "0.00"
                # rather than "-0.00", which reads as a direction it does not have.
                f"{round(float(pair['rho']), 2) + 0.0:.2f}\n"  # type: ignore[arg-type]
                + ("p<0.001" if p_value < 1e-3 else f"p={p_value:.3f}"),
                ha="center",
                va="center",
                fontsize=7,
                color=TEXT_PRIMARY,
            )
    grid.set_title("Spearman rho between strategy rankings\n(exact permutation p, n=7)", color=TEXT_PRIMARY)

    fig.tight_layout()
    return _save(fig, out_png)


def plot_attribution(maps: dict[str, dict[str, np.ndarray]], out_png: str | Path) -> Path:
    """
    Mean input-gradient saliency per strategy and class, with border mass.

    Claim: a strategy whose border mass exceeds the uniform value is reading the
    frame rather than the picture - which is the specific charge against `pad`.

    Each map is scaled to its own maximum, because arms differ in gradient
    magnitude by orders of magnitude and a shared scale would render most panels
    blank. The quantitative comparison is the border-mass number in each title and
    the companion bar chart, not the shading.

    Args:
        maps: label -> tag -> (128, 128) mean map, from func mean_attribution.
        out_png: Destination path.

    Returns:
        The written path.

    Raises:
        ValueError: If there is nothing to plot.
    """
    if not maps:
        raise ValueError("no attribution maps to plot")
    plt = _pyplot()
    from matplotlib.patches import Rectangle

    labels = list(maps)
    tags = [tag for tag in TAGS if any(tag in maps[label] for label in labels)]
    fig, axes = plt.subplots(
        len(labels), len(tags), figsize=(1.35 * len(tags), 1.95 * len(labels)), squeeze=False
    )
    for row, label in enumerate(labels):
        for column, tag in enumerate(tags):
            ax = axes[row][column]
            ax.set_xticks([])
            ax.set_yticks([])
            saliency = maps[label].get(tag)
            if saliency is None:
                ax.axis("off")
                continue
            ax.imshow(saliency, cmap=_blue_colormap(), vmin=0.0, vmax=float(saliency.max()))
            # The ring the border mass is measured over, drawn so the number and
            # the picture cannot be read as describing different regions.
            ax.add_patch(
                Rectangle(
                    (BORDER_PX - 0.5, BORDER_PX - 0.5),
                    CACHE_SIZE_PX - 2 * BORDER_PX,
                    CACHE_SIZE_PX - 2 * BORDER_PX,
                    fill=False,
                    edgecolor=CATEGORICAL[1],
                    linewidth=0.8,
                )
            )
            ax.set_title(f"{tag}\nborder {border_mass(saliency):.3f}", fontsize=7, color=TEXT_PRIMARY)
            if column == 0:
                ax.set_ylabel(label, fontsize=8, color=TEXT_SECONDARY)
    fig.suptitle(
        "Mean input-gradient saliency (L10 slides 53-57). "
        f"Uniform border mass = {UNIFORM_BORDER_MASS:.3f}; above it means the frame carries the evidence.",
        y=1.02,
        color=TEXT_PRIMARY,
    )
    # Each panel carries a two-line title, so the rows need more room between them
    # than the default pad leaves.
    fig.tight_layout(h_pad=1.8)
    return _save(fig, out_png)


def plot_border_mass(maps: dict[str, dict[str, np.ndarray]], out_png: str | Path) -> Path:
    """
    Border mass per strategy and class, against the uniform-saliency reference.

    Claim: the whole comparison in one panel - which arms read their own frame.

    Args:
        maps: label -> tag -> (128, 128) mean map.
        out_png: Destination path.

    Returns:
        The written path.
    """
    plt = _pyplot()
    labels = list(maps)
    tags = [tag for tag in TAGS if any(tag in maps[label] for label in labels)]

    fig, ax = plt.subplots(figsize=(1.1 * len(tags) + 2.2, 3.4))
    width = 0.8 / max(len(labels), 1)
    x = np.arange(len(tags))
    for index, label in enumerate(labels):
        heights = [
            border_mass(maps[label][tag]) if tag in maps[label] else np.nan for tag in tags
        ]
        ax.bar(
            x + index * width - 0.4 + width / 2,
            heights,
            width * 0.9,  # a surface gap between adjacent bars
            label=label,
            color=CATEGORICAL[index % len(CATEGORICAL)],
        )
    ax.axhline(UNIFORM_BORDER_MASS, color=TEXT_SECONDARY, linewidth=1.0, linestyle="--")
    ax.annotate(
        f"uniform saliency = {UNIFORM_BORDER_MASS:.3f}",
        (-0.45, UNIFORM_BORDER_MASS),
        xytext=(2, 4),
        textcoords="offset points",
        ha="left",
        fontsize=7,
        color=TEXT_SECONDARY,
    )
    ax.set_xticks(x, tags, rotation=45, ha="right")
    ax.set_ylabel(f"saliency in the outer {BORDER_PX}px ring")
    ax.set_title("Border mass per strategy", color=TEXT_PRIMARY)
    ax.grid(axis="y", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(
        frameon=False,
        fontsize=7,
        ncol=len(labels),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
    )
    fig.tight_layout()
    return _save(fig, out_png)


def plot_spectra(spectra: dict[str, dict[str, np.ndarray]], out_png: str | Path) -> Path:
    """
    Real minus fake log-magnitude spectra, one panel per strategy.

    Claim: the frequency band where real and generated images differ moves with
    the equalization, so a detector trained under one strategy is not looking at
    the same evidence as one trained under another.

    Args:
        spectra: label -> tag -> radial profile, including REAL_TAG.
        out_png: Destination path.

    Returns:
        The written path.

    Raises:
        ValueError: If there is nothing to plot.
    """
    if not spectra:
        raise ValueError("no spectra to plot")
    plt = _pyplot()

    labels = list(spectra)
    frequency = _frequency_axis()[FIRST_PLOTTED_BIN:]
    fig, axes = plt.subplots(
        1, len(labels), figsize=(3.3 * len(labels) + 0.6, 3.4), squeeze=False, sharey=True
    )
    for column, label in enumerate(labels):
        ax = axes[0][column]
        profiles = spectra[label]
        for index, generator in enumerate(GENERATORS):
            if generator not in profiles or REAL_TAG not in profiles:
                continue
            ax.plot(
                frequency,
                spectral_difference(profiles, generator)[FIRST_PLOTTED_BIN:],
                DASHES[index % len(DASHES)],
                color=generator_color(generator),
                linewidth=1.8,
                label=generator,
            )
        ax.axhline(0.0, color=TEXT_SECONDARY, linewidth=0.8)
        ax.set_title(label, color=TEXT_PRIMARY)
        ax.set_xlabel("frequency (cycles/pixel)")
        ax.grid(linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if column == 0:
            ax.set_ylabel("log10 magnitude, real minus fake")

    handles, names = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=len(GENERATORS), frameon=False, fontsize=7)
    fig.suptitle(
        "Radially averaged spectra, real minus generated. "
        "No window function: the pad arm's zero border is a real feature of that arm.",
        y=1.03,
        color=TEXT_PRIMARY,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    return _save(fig, out_png)


# --------------------------------------------------------------------------- #
# Printing and writing
# --------------------------------------------------------------------------- #


def _print_matrix(matrix: MatrixSet) -> None:
    """
    Print one 7x7 matrix in the same layout `baseline.py` uses.

    Args:
        matrix: The matrix to print.
    """
    print(f"\n{matrix.label}  seeds {list(matrix.seeds)}  (rows = trained on, cols = evaluated on, %)")
    print("            " + "".join(f"{target[:10]:>11}" for target in GENERATORS))
    for row, source in enumerate(GENERATORS):
        cells = "".join(
            "          -" if not np.isfinite(matrix.accuracy[row, column])
            else f"{100 * matrix.accuracy[row, column]:>11.1f}"
            for column in range(len(GENERATORS))
        )
        print(f"{source[:11]:<12}{cells}")
    summary = matrix_summary(matrix.accuracy)
    missing = int(np.sum(matrix.n_seeds == 0))
    print(
        f"  diag {100 * summary['diagonal_mean']:.1f}%   "
        f"off-diag {100 * summary['off_diagonal_mean']:.1f}%   "
        f"gap {100 * summary['gap']:+.1f} pp"
        + (f"   [{missing} cells missing]" if missing else "")
    )


def _json_safe(payload: object) -> object:
    """
    Make a result tree valid JSON: numpy scalars to Python, non-finite to null.

    Two problems, one walk. `json.dumps` cannot serialize `np.float64` at all, and
    it writes NaN as the bare token `NaN`, which is readable by Python and by
    nothing else - so a missing cell would arrive in the report generator as a
    parse error. Null is also the more honest encoding: the cell is absent, not
    zero and not undefined-in-place.

    Args:
        payload: Any nesting of dicts, sequences, arrays and scalars.

    Returns:
        The same tree, JSON-serializable.
    """
    if isinstance(payload, dict):
        return {str(key): _json_safe(value) for key, value in payload.items()}
    if isinstance(payload, np.ndarray):
        return _json_safe(payload.tolist())
    if isinstance(payload, (list, tuple)):
        return [_json_safe(value) for value in payload]
    if isinstance(payload, (np.floating, float)):
        return float(payload) if np.isfinite(payload) else None
    if isinstance(payload, (np.integer, int)):
        return int(payload)
    return payload


def _write_json(payload: object, out_json: str | Path) -> Path:
    """
    Write one command's numbers beside its figure.

    Args:
        payload: Result tree; passed through func _json_safe first.
        out_json: Destination path; parents are created.

    Returns:
        The written path.
    """
    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    print(f"[analyze] numbers -> {out}")
    return out


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def command_matrices(args) -> list[MatrixSet]:
    """
    Build the transfer-matrix figure and its JSON.

    Args:
        args: Parsed CLI arguments.

    Returns:
        The panels that were plotted, so func command_all can reuse them.
    """
    matrices = load_matrices(args.results_dir, args.strategies, args.seeds)
    if not matrices:
        raise SystemExit(f"no runs under {args.results_dir}; nothing to analyse")
    baselines = load_matrices(args.baseline_dir) if args.baseline_dir else []

    for matrix in (*matrices, *baselines):
        _print_matrix(matrix)

    plot_matrices([*matrices, *baselines], Path(args.out_dir) / "transfer_matrices.png")
    _write_json(
        {
            "generators": list(GENERATORS),
            "arms": [
                {
                    "label": matrix.label,
                    "seeds": list(matrix.seeds),
                    "accuracy": matrix.accuracy,
                    "seed_spread": matrix.spread,
                    "n_seeds": matrix.n_seeds,
                    "summary": matrix_summary(matrix.accuracy),
                }
                for matrix in (*matrices, *baselines)
            ],
        },
        Path(args.out_dir) / "transfer_matrices.json",
    )
    return matrices


def command_ranking(args, matrices: list[MatrixSet] | None = None) -> None:
    """
    Build the ranking figure and its JSON, for both off-diagonal axes.

    Args:
        args: Parsed CLI arguments.
        matrices: Panels already loaded by func command_matrices, if any.
    """
    matrices = matrices or load_matrices(args.results_dir, args.strategies, args.seeds)
    if len(matrices) < 2:
        print("[analyze] fewer than two arms present - a ranking comparison needs at least two")
        if not matrices:
            raise SystemExit(f"no runs under {args.results_dir}")

    both = {axis: ranking_table(matrices, axis) for axis in ("source", "target")}
    for axis, table in both.items():
        print(f"\nranking by {axis} off-diagonal mean (%)")
        print("            " + "".join(f"{label[:10]:>13}" for label in table["values"]))
        for index, generator in enumerate(GENERATORS):
            cells = ""
            for label in table["values"]:
                value, rank = table["values"][label][index], table["ranks"][label][index]
                cells += "            -" if not np.isfinite(value) else f"{100 * value:>9.1f} #{int(rank)}"
            print(f"{generator[:11]:<12}{cells}")
        for pair in table["spearman"]:
            print(f"  rho({pair['a']}, {pair['b']}) = {pair['rho']:+.3f}   p = {pair['p']:.4f}")

    plot_ranking(both[args.axis], Path(args.out_dir) / "ranking.png")
    _write_json(both, Path(args.out_dir) / "ranking.json")


def command_attribution(args) -> None:
    """
    Compute input-gradient attribution per arm and write both of its figures.

    Args:
        args: Parsed CLI arguments.
    """
    strategies = run_strategies(args.results_dir, args.strategies)
    if not strategies:
        raise SystemExit(f"no checkpoints under {args.results_dir}")
    sources = args.sources or list(GENERATORS)

    results: list[Attribution] = []
    for strategy in strategies:
        pool = load_test_pool(args.cache_dir, strategy)
        for source in sources:
            weights = checkpoint_path(args.results_dir, strategy, source, args.seed)
            if not weights.exists():
                print(f"[analyze] no checkpoint at {weights}, skipping")
                continue
            attribution = attribution_for_arm(
                args.cache_dir,
                args.results_dir,
                strategy,
                source,
                seed=args.seed,
                n_images=args.n_images,
                device=args.device,
                pool=pool,
            )
            results.append(attribution)
            print(
                f"[analyze] {strategy}/{source}/seed{args.seed}: border mass "
                + "  ".join(f"{tag} {value:.3f}" for tag, value in attribution.border_mass.items())
            )

    if not results:
        raise SystemExit("no checkpoints found; attribution needs the weights saved beside metrics.json")

    maps = mean_attribution(results)
    plot_attribution(maps, Path(args.out_dir) / "attribution.png")
    plot_border_mass(maps, Path(args.out_dir) / "attribution_border_mass.png")
    _write_json(
        {
            "border_px": BORDER_PX,
            "uniform_border_mass": UNIFORM_BORDER_MASS,
            "n_images_per_tag": args.n_images,
            "seed": args.seed,
            "per_run": [
                {
                    "strategy": item.strategy,
                    "source": item.source,
                    "seed": item.seed,
                    "border_mass": item.border_mass,
                }
                for item in results
            ],
            "per_strategy": {
                label: {tag: border_mass(saliency) for tag, saliency in tags.items()}
                for label, tags in maps.items()
            },
        },
        Path(args.out_dir) / "attribution.json",
    )


def command_spectra(args) -> None:
    """
    Compute radially averaged spectra per arm and write the figure.

    Args:
        args: Parsed CLI arguments.
    """
    # Spectra need the cache and nothing else, so a named strategy is taken at
    # face value: this command is runnable before any run has finished.
    strategies = (
        [_check_strategy(name) for name in args.strategies]
        if args.strategies
        else run_strategies(args.results_dir)
    )
    if not strategies:
        raise SystemExit(
            f"no runs under {args.results_dir}; pass --strategies to profile the cache "
            "before any run exists"
        )

    spectra = {
        strategy: spectra_for_strategy(args.cache_dir, strategy, args.n_images)
        for strategy in strategies
    }
    for strategy, profiles in spectra.items():
        missing = [tag for tag in TAGS if tag not in profiles]
        print(
            f"[analyze] {strategy}: {len(profiles)} classes profiled"
            + (f", missing {missing}" if missing else "")
        )

    plot_spectra(spectra, Path(args.out_dir) / "spectra.png")
    _write_json(
        {
            "frequency_cycles_per_px": _frequency_axis().tolist(),
            "first_plotted_bin": FIRST_PLOTTED_BIN,
            "n_images_per_tag": args.n_images,
            "window": "none",
            "profiles": {
                label: {tag: profile.tolist() for tag, profile in profiles.items()}
                for label, profiles in spectra.items()
            },
            "real_minus_fake_log10": {
                label: {
                    generator: spectral_difference(profiles, generator).tolist()
                    for generator in GENERATORS
                    if generator in profiles and REAL_TAG in profiles
                }
                for label, profiles in spectra.items()
            },
        },
        Path(args.out_dir) / "spectra.json",
    )


def command_all(args) -> None:
    """
    Run every command in order, reusing the loaded matrices between the first two.

    Args:
        args: Parsed CLI arguments.
    """
    matrices = command_matrices(args)
    command_ranking(args, matrices)
    command_attribution(args)
    # The two commands want different counts - 200 images is PLAN.md's attribution
    # budget, while the spectra are numpy and can afford the whole test block - so
    # `all` carries both flags and hands the second one over here.
    args.n_images = args.spectrum_images
    command_spectra(args)


def _main() -> None:
    """
    CLI entry point.

    Example:
        python -m src.analyze matrices --results-dir results/runs
        python -m src.analyze attribution --cache-dir /content/cache --device cuda
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    common.add_argument("--out-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    common.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help="strategy directory names under --results-dir; default is every one found. "
             "`spectra` accepts names with no runs yet, since it reads only the cache",
    )

    # One parent per flag group, so no command advertises a flag it ignores.
    seeds = argparse.ArgumentParser(add_help=False)
    seeds.add_argument("--seeds", nargs="+", type=int, default=None, help="default: every seed present")

    axis = argparse.ArgumentParser(add_help=False)
    axis.add_argument(
        "--axis",
        default="source",
        choices=("source", "target"),
        help="which off-diagonal mean the ranking figure plots; both are always written to JSON",
    )

    panels = argparse.ArgumentParser(add_help=False)
    panels.add_argument(
        "--baseline-dir",
        type=Path,
        default=None,
        help=f"add the baseline rules as extra panels, e.g. {DEFAULT_BASELINE_DIR}",
    )

    cache = argparse.ArgumentParser(add_help=False)
    cache.add_argument("--cache-dir", required=True, type=Path)

    weights = argparse.ArgumentParser(add_help=False)
    weights.add_argument("--seed", type=int, default=0, help="which seed's checkpoints to attribute")
    weights.add_argument("--device", default=DEFAULT_DEVICE)
    weights.add_argument("--sources", nargs="+", default=None, choices=GENERATORS)

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("matrices", parents=[common, seeds, panels], help="the four 7x7 heatmaps")
    sub.add_parser("ranking", parents=[common, seeds, axis], help="generator ranking and Spearman rho")
    attribution = sub.add_parser(
        "attribution", parents=[common, cache, weights], help="input-gradient saliency and border mass"
    )
    attribution.add_argument("--n-images", type=int, default=ATTRIBUTION_IMAGES)
    spectra = sub.add_parser("spectra", parents=[common, cache], help="radially averaged spectra")
    spectra.add_argument("--n-images", type=int, default=SPECTRUM_IMAGES)
    everything = sub.add_parser(
        "all",
        parents=[common, seeds, axis, panels, cache, weights],
        help="every command, in order",
    )
    everything.add_argument("--n-images", type=int, default=ATTRIBUTION_IMAGES, help="attribution")
    everything.add_argument("--spectrum-images", type=int, default=SPECTRUM_IMAGES, help="spectra")

    args = parser.parse_args()
    {
        "matrices": command_matrices,
        "ranking": command_ranking,
        "attribution": command_attribution,
        "spectra": command_spectra,
        "all": command_all,
    }[args.command](args)


if __name__ == "__main__":
    _main()
