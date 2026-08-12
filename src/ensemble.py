"""
Combine trained detectors by averaging their raw scores, and score the result.

    python -m src.ensemble --logits results/figures/logits_ido.npz

Reads the `.npz` written by `src.logits`. **No GPU, no torch, seconds on a laptop** - which
is the point: every combination worth trying can be tried without another Colab session.

Three combinations, all decided in advance:

- **across arms** - `center_crop` + `rescale`. The headline reports rho = 0.00 between their
  generator rankings, i.e. they fail on *different* generators. That is the precondition for
  averaging to beat either member, so this is a prediction the headline makes about itself.
- **across orientations** - the D4 orbit, i.e. test-time augmentation. `rotation.py` measured
  off-domain accuracy varying by 0.009 across the whole orbit; near-equal views are exactly
  when averaging helps.
- **across seeds** - the two checkpoints per cell. **Reported separately and never folded
  into the matrices**, because the seed spread *is* the error bar: measured at 0.0167 mean
  and 0.0347 max on off-domain, it is what says whether a ranking difference is real.

Two rules that keep this honest, both fixed before any number was seen:

- **The rule is an unweighted mean of logits.** No fitted weights, no tuned threshold. A
  weighting chosen to make the output look good would be the test set leaking into the
  method - the same objection that retired threshold calibration on 2026-08-12.
- **`pad` is excluded from every ensemble**, on the stated ground that it is a border
  detector (diagonal 0.9776 against a 0.977 pure-border ceiling), not because of how its
  numbers come out. Averaging it in would inflate the result with the contamination the
  study exists to expose. `--arms` will refuse it.

The single-arm rows must reproduce the recorded matrix. That is checked, not assumed: if
`center_crop` alone does not come back at its recorded diagonal and off-diagonal, the file
is not describing the reported models and no ensemble number in it means anything.
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Final, NamedTuple

from src.logits import split_key
from src.rotation import CELL_ROWS
from src.train import DECISION_LOGIT, DEFAULT_RESULTS_DIR
from src.data import GENERATORS, REAL_TAG

__all__ = [
    "Scored",
    "cell_accuracies",
    "combine",
    "is_run_key",
    "score",
    "seeds_in",
    "select",
    "verify_single_arms",
]

#: Never ensembled. `pad`'s model reads the black border that real photos get and the
#: generators never do, so including it would raise the number using the giveaway that
#: finding #2 exists to expose. A methodological exclusion, fixed in advance.
EXCLUDED_ARMS: Final[frozenset[str]] = frozenset({"pad"})

#: Arms combined when `--arms` is not given: the pair the headline rests on, and the two
#: this account holds checkpoints for.
DEFAULT_ARMS: Final[tuple[str, ...]] = ("center_crop", "rescale")

#: Tolerance for the single-arm reproduction check, in test rows out of 1,000 per cell.
#: Matches `rotation.MAX_FLIPPED_PREDICTIONS` and exists for the same reason: the scores
#: were produced under fp16 autocast, so a logit at the threshold is GPU-dependent.
MAX_FLIPPED_PREDICTIONS: Final[int] = 4


class Scored(NamedTuple):
    """
    One combination's transfer numbers.

    Fields:
        label: What was combined, e.g. "center_crop+rescale, D4 TTA".
        in_domain: Mean accuracy on the source generator's own cell.
        off_domain: Mean accuracy over the six other cells.
        tnr: Accuracy on the shared real block.
        tpr_off: Accuracy on the fakes of generators other than the source.
        n_sources: How many source generators the means are over.
    """

    label: str
    in_domain: float
    off_domain: float
    tnr: float
    tpr_off: float
    n_sources: int


def is_run_key(key: str) -> bool:
    """
    Whether an `.npz` member is a run rather than metadata.

    Args:
        key: Member name.

    Returns:
        True for `strategy__source__seed`, False for `y_test`, `gen_ids`, `orientations`.
    """
    try:
        split_key(key)
    except ValueError:
        return False
    return True


def select(keys, arms=None, seeds=None) -> dict[str, list[str]]:
    """
    Group the run keys in a dump by source generator.

    Args:
        keys: Member names from the `.npz`, including the non-run metadata entries.
        arms: Strategies to keep; `None` keeps DEFAULT_ARMS.
        seeds: Seeds to keep; `None` keeps every seed present.

    Returns:
        source generator -> its matching run keys.

    Raises:
        ValueError: If `arms` names an excluded arm.
    """
    wanted = tuple(DEFAULT_ARMS if arms is None else arms)
    forbidden = EXCLUDED_ARMS.intersection(wanted)
    if forbidden:
        raise ValueError(
            f"{sorted(forbidden)} cannot be ensembled: a border detector averaged into the "
            f"result would inflate it with the contamination finding #2 exists to expose"
        )

    grouped: dict[str, list[str]] = {}
    for key in keys:
        try:
            strategy, source, seed = split_key(key)
        except ValueError:
            continue  # y_test, gen_ids, orientations
        if strategy in wanted and (seeds is None or seed in seeds):
            grouped.setdefault(source, []).append(key)
    return grouped


def combine(dump, keys, orientations=(0,)) -> np.ndarray:
    """
    Average raw scores across runs and orientations.

    The mean of logits, unweighted - see the module docstring. Averaging in logit space
    rather than after a sigmoid keeps a confidently-wrong member from being clipped into
    near-agreement, which is what would blunt the very disagreement being exploited.

    Args:
        dump: The loaded `.npz` (or any mapping key -> (n_orient, n_test) array).
        keys: Run keys to average over.
        orientations: Row indices into the orientation axis. `(0,)` is the upright,
            unmirrored pass alone; `range(8)` is full D4 test-time augmentation.

    Returns:
        float64 (n_test,) combined scores.

    Raises:
        ValueError: If `keys` is empty.
    """
    if not keys:
        raise ValueError("nothing to combine")
    stacked = [np.asarray(dump[k], dtype=np.float64)[list(orientations)] for k in keys]
    return np.mean(np.concatenate(stacked, axis=0), axis=0)


def seeds_in(dump) -> tuple[int, ...]:
    """
    Which seeds a dump holds runs for.

    Args:
        dump: The loaded `.npz` (or any mapping).

    Returns:
        The seeds present, ascending.
    """
    keys = dump.files if hasattr(dump, "files") else dump.keys()
    return tuple(sorted({split_key(k)[2] for k in keys if is_run_key(k)}))


def cell_accuracies(scores, truth, gen_ids) -> dict[str, float]:
    """
    Accuracy on each generator's cell: its fakes plus the shared real block.

    Args:
        scores: (n_test,) combined logits.
        truth: (n_test,) true labels, 1.0 synthetic.
        gen_ids: (n_test,) generator tags, REAL_TAG for the shared real block.

    Returns:
        generator -> accuracy over CELL_ROWS rows.

    Raises:
        ValueError: If a generator block is missing.
    """
    correct = (scores > DECISION_LOGIT).astype(np.float64) == truth
    is_real = gen_ids == REAL_TAG

    accuracies = {}
    for generator in GENERATORS:
        mask = (gen_ids == generator) | is_real
        if not mask.any():
            raise ValueError(f"no test rows for {generator}")
        accuracies[generator] = float(correct[mask].mean())
    return accuracies


def score(scores, truth, gen_ids, source, label="") -> Scored:
    """
    Turn combined scores into one source generator's transfer numbers.

    Args:
        scores: (n_test,) combined logits.
        truth: (n_test,) true labels, 1.0 synthetic.
        gen_ids: (n_test,) generator tags, REAL_TAG for the shared real block.
        source: Generator these detectors were trained on.
        label: Free text describing the combination, carried into the result.

    Returns:
        The four rates plus `n_sources = 1`.

    Raises:
        ValueError: If `source` is unknown or a generator block is missing.
    """
    if source not in GENERATORS:
        raise ValueError(f"unknown source {source!r}")

    cells = cell_accuracies(scores, truth, gen_ids)
    correct = (scores > DECISION_LOGIT).astype(np.float64) == truth
    is_real = gen_ids == REAL_TAG
    off_mask = ~is_real & (gen_ids != source)
    return Scored(
        label=label,
        in_domain=cells[source],
        off_domain=float(np.mean([a for g, a in cells.items() if g != source])),
        tnr=float(correct[is_real].mean()),
        tpr_off=float(correct[off_mask].mean()),
        n_sources=1,
    )


def average(results, label) -> Scored:
    """
    Average per-source results into one row of the summary table.

    Args:
        results: One `Scored` per source generator.
        label: Name for the combined row.

    Returns:
        A `Scored` whose `n_sources` counts the inputs.
    """
    return Scored(
        label=label,
        in_domain=float(np.mean([r.in_domain for r in results])),
        off_domain=float(np.mean([r.off_domain for r in results])),
        tnr=float(np.mean([r.tnr for r in results])),
        tpr_off=float(np.mean([r.tpr_off for r in results])),
        n_sources=len(results),
    )


def evaluate(dump, arms, orientations, label, seeds=None) -> Scored:
    """
    Score one combination across every source generator present.

    **Seeds are averaged as accuracies, one seed at a time - never as logits.** A cell holds
    two checkpoints, and pooling their scores into one mean logit would silently make every
    row of the table a two-model ensemble: the `center_crop alone` baseline would be a
    seed-ensemble, and `center_crop + rescale` a four-model one, so the table would compare
    four models against two and call the difference a cross-arm effect. The seed spread is
    the error bar and stays outside the matrices, exactly as the module docstring says.

    Args:
        dump: The loaded `.npz`.
        arms: Strategies to average over, within a seed.
        orientations: Orientation rows to average over.
        label: Name for the resulting row.
        seeds: Seeds to report over; `None` uses every seed present. Pass a single seed to
            get that one model's row, which is what the error-bar block wants.

    Returns:
        The averaged row.

    Raises:
        ValueError: If no run matches the selection.
    """
    truth = np.asarray(dump["y_test"], dtype=np.float64)
    gen_ids = np.asarray(dump["gen_ids"]).astype(str)
    keys = dump.files if hasattr(dump, "files") else dump.keys()

    per_seed = []
    for seed in seeds_in(dump) if seeds is None else tuple(seeds):
        grouped = select(keys, arms, (seed,))
        if not grouped:
            continue
        per_source = [
            score(combine(dump, run_keys, orientations), truth, gen_ids, source)
            for source, run_keys in sorted(grouped.items())
        ]
        per_seed.append(average(per_source, label))

    if not per_seed:
        raise ValueError(f"no runs in the dump for arms={arms}, seeds={seeds}")

    folded = average(per_seed, label)
    return folded._replace(n_sources=per_seed[0].n_sources)


def recorded_cells(results_dir, strategy, source, seed) -> dict[str, float]:
    """
    The per-cell accuracies a run is recorded as having produced.

    Args:
        results_dir: Root holding `<strategy>/<source>/seed<seed>/metrics.json`.
        strategy: Arm name.
        source: Generator trained on.
        seed: Run seed.

    Returns:
        generator -> recorded accuracy.

    Raises:
        FileNotFoundError: If the run has no `metrics.json`.
    """
    path = Path(results_dir) / strategy / source / f"seed{seed}" / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def verify_single_arms(dump, arms, results_dir=DEFAULT_RESULTS_DIR) -> str:
    """
    Check that each arm alone reproduces the matrix it is recorded as producing.

    This is the guard that makes every other number in the file meaningful: an ensemble
    of models that are not the reported models is not a result about this project. Each
    checkpoint's upright, unmirrored pass is compared against its own `metrics.json`,
    cell by cell - not arm-level means, which can agree while individual cells do not.

    Args:
        dump: The loaded `.npz`.
        arms: Strategies to check, each scored on its own.
        results_dir: Where the recorded `metrics.json` files live.

    Returns:
        A one-line-per-arm summary for printing.

    Raises:
        ValueError: If any cell is further than MAX_FLIPPED_PREDICTIONS rows from its
            recorded accuracy.
    """
    truth = np.asarray(dump["y_test"], dtype=np.float64)
    gen_ids = np.asarray(dump["gen_ids"]).astype(str)
    tolerance = MAX_FLIPPED_PREDICTIONS / CELL_ROWS

    drifted = []
    lines = []
    for arm in arms:
        keys = dump.files if hasattr(dump, "files") else dump.keys()
        for source, run_keys in sorted(select(keys, (arm,), None).items()):
            for key in sorted(run_keys):
                _, _, seed = split_key(key)
                found = cell_accuracies(combine(dump, [key], (0,)), truth, gen_ids)
                for generator, expected in recorded_cells(results_dir, arm, source, seed).items():
                    drift = abs(found[generator] - expected)
                    if drift > tolerance:
                        drifted.append(
                            f"    {arm}/{source}/seed{seed} cell {generator}: "
                            f"{found[generator]:.4f} vs recorded {expected:.4f} "
                            f"({round(drift * CELL_ROWS)} rows)"
                        )

        single = evaluate(dump, (arm,), (0,), arm)
        lines.append(
            f"  {arm:<13}in-domain {single.in_domain:.4f}  off-domain {single.off_domain:.4f}  "
            f"({single.n_sources} sources)"
        )

    if drifted:
        raise ValueError(
            f"{len(drifted)} cell(s) drifted more than {MAX_FLIPPED_PREDICTIONS} rows from the "
            f"recorded matrix - these scores are not about the reported models:\n"
            + "\n".join(drifted)
        )
    return "\n".join(lines)


def format_table(rows) -> str:
    """
    Render the comparison table.

    Args:
        rows: `Scored` entries, baselines first.

    Returns:
        The table, with each row's change from the first row.
    """
    base = rows[0]
    out = [f"{'combination':<34}{'in-dom':>9}{'off-dom':>9}{'d off':>8}{'tnr':>8}{'tpr-off':>9}"]
    for row in rows:
        out.append(
            f"{row.label:<34}{row.in_domain:>9.4f}{row.off_domain:>9.4f}"
            f"{row.off_domain - base.off_domain:>+8.4f}{row.tnr:>8.4f}{row.tpr_off:>9.4f}"
        )
    return "\n".join(out)


def _main() -> None:
    """
    CLI entry point.

    Example:
        python -m src.ensemble --logits results/figures/logits_ido.npz
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--logits", required=True, type=Path, help="written by src.logits")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="recorded metrics.json tree, checked against the upright pass",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=list(DEFAULT_ARMS),
        help=f"strategies to ensemble (default {' '.join(DEFAULT_ARMS)}). "
             f"{sorted(EXCLUDED_ARMS)} is refused",
    )
    args = parser.parse_args()

    dump = np.load(args.logits, allow_pickle=False)
    arms = tuple(args.arms)
    n_orient = len(dump["orientations"])
    seeds = seeds_in(dump)

    print(f"{args.logits}: {len(seeds)} seed(s) {list(seeds)}, {n_orient} orientations\n")
    print("single-arm reproduction (checked against the recorded matrix, cell by cell):")
    print(verify_single_arms(dump, arms, args.results_dir))
    print()

    rows = [evaluate(dump, (arm,), (0,), f"{arm} alone") for arm in arms]
    rows.append(evaluate(dump, arms, (0,), " + ".join(arms)))
    rows.append(evaluate(dump, arms, range(n_orient), " + ".join(arms) + ", D4 TTA"))
    for arm in arms:
        rows.append(evaluate(dump, (arm,), range(n_orient), f"{arm}, D4 TTA"))
    print(format_table(rows))

    # Seeds last and labelled, so the line that costs the error bar cannot be mistaken for
    # one of the rows above. Both seeds are already inside every row above; this one asks
    # what a single ensembled model per cell would score.
    if len(seeds) > 1:
        print(f"\nper-seed, for the error bar (never fold these into the matrices):")
        for seed in seeds:
            single = evaluate(dump, arms, (0,), f"{' + '.join(arms)}, seed {seed}", seeds=(seed,))
            print(f"  {single.label:<34}in-dom {single.in_domain:.4f}  "
                  f"off-dom {single.off_domain:.4f}")


if __name__ == "__main__":
    _main()
