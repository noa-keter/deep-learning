"""Size-confound baselines: how much of "AI-image detection" is solved by image size alone.

Both rules use only the cached metadata (h, w, generator) -- no pixels, no training, no GPU.
They measure how far a detector gets by exploiting the fact that generated images are square
while real photographs are not, which is the confound the four equalization strategies are
meant to remove.

  square rule       predict synthetic iff h == w. Zero parameters, nothing fitted.
  size-lookup rule  fit the (h, w) pairs a generator emits in train; predict synthetic
                    iff the test pair is in that set.

Each rule produces a 7x7 source->target matrix in the same layout and file format as the
CNN runs, at <out>/{rule}/{source}/seed0/metrics.json, scored on the same images: the cache
layout, the generator vocabulary and the evaluation pools all come from data.py.

Run:  python src/baseline.py --cache /path/to/aidet/cache --out results/baseline
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Imported rather than restated: sharing these constants is what puts the baseline on the
# same evaluation images as the CNN runs, which is the only reason the panels compare.
from data import (
    CACHE_SIZE_PX,
    GENERATORS,
    REAL_POOL_SEED,
    TEST_PER_GENERATOR,
    TEST_REAL_N,
    TRAIN_PER_CLASS,
    load_meta,
)

# The split the dataset calls `validation` is what this project reports on. The
# model-selection split is carved out of `train` and never appears here.
TEST_SPLIT = "validation"
TRAIN_SPLIT = "train"

# Images per class in a reported cell. Overridden only by --partial, for the pilot.
FULL_POOL_SIZES = (TEST_PER_GENERATOR, TEST_REAL_N)

# Expected split sizes, derived so they track data.py rather than drifting from it. Each
# split holds one real image per fake, which is why reals and fakes share a count.
EXPECTED_TEST_FAKES = len(GENERATORS) * TEST_PER_GENERATOR
EXPECTED_TEST_REALS = EXPECTED_TEST_FAKES
EXPECTED_TEST_ROWS = EXPECTED_TEST_FAKES + EXPECTED_TEST_REALS
EXPECTED_TRAIN_ROWS = 2 * len(GENERATORS) * TRAIN_PER_CLASS

# The square rule should be near-perfect: every generated image is square, and only the few
# real photos that happen to be square cost anything. Well below this means bad metadata.
GATE_MIN_ACCURACY = 0.97


def _real_mask(generator):
    """True for real rows. Real is whatever tag is not a known generator.

    Found by exclusion because the dataset stores its own name for the real class verbatim.
    Exactly one unknown tag is allowed: more than one means a generator was renamed, and
    counting it as real would corrupt every cell.
    """
    known = set(GENERATORS)
    tags = set(np.unique(generator).tolist())
    unknown_tags = sorted(tags - known)
    if not tags & known:
        raise ValueError("no known generator tags in cache; found {}".format(sorted(tags)))
    if len(unknown_tags) != 1:
        raise ValueError(
            "expected exactly one real tag besides the generators, found {}".format(unknown_tags)
        )
    return generator == unknown_tags[0]


def _choose(pool, n, seed, what):
    """n rows from pool without replacement, deterministically, returned sorted.

    Mirrors the draw the training pipeline performs on the same pools with the same seed,
    so both score identical images. Raises rather than returning a short selection: a
    silent shortfall would unbalance the classes and invalidate the results.
    """
    if len(pool) < n:
        raise ValueError("{}: need {} rows, cache has {}".format(what, n, len(pool)))
    return np.sort(np.random.default_rng(seed).choice(pool, size=n, replace=False))


def select_real_block(meta, n_real=TEST_REAL_N):
    """The real half of every cell, drawn once. Returns sorted indices.

    Drawn from REAL_POOL_SEED rather than a run seed, and computed once per run rather than
    per cell: passing this block down is what makes two cells physically unable to score
    different real images, which no amount of checking afterwards could guarantee.
    """
    is_real = _real_mask(np.asarray(meta["generator"]))
    return _choose(np.flatnonzero(is_real), n_real, REAL_POOL_SEED, "test real block")


def select_fake_block(meta, generator, n_fake=TEST_PER_GENERATOR):
    """The fake half of one cell: `generator`'s test rows. Returns sorted indices."""
    gen = np.asarray(meta["generator"])
    return _choose(
        np.flatnonzero(gen == generator),
        n_fake,
        REAL_POOL_SEED,
        "test fakes for {}".format(generator),
    )


def select_eval_indices(meta, generator, n_fake=TEST_PER_GENERATOR, n_real=TEST_REAL_N):
    """Rows for one (source -> target) cell: the target's fakes plus the shared real block.

    Returns (fake_idx, real_idx), each sorted. The counts are arguments only so the pilot
    can score a partial cache; a reported run uses the defaults, which are the pools
    train.py draws from.
    """
    return select_fake_block(meta, generator, n_fake), select_real_block(meta, n_real)


def _pilot_pool_sizes(test_meta):
    """Largest per-class pool a partial cache can support, as (n_fake, n_real).

    The smallest generator sets the fake count so every cell scores the same number of
    images, and the real block matches it to keep the classes balanced.
    """
    gen = np.asarray(test_meta["generator"])
    n_fake = min(int((gen == g).sum()) for g in GENERATORS)
    n_real = int(_real_mask(gen).sum())
    if n_fake == 0:
        raise ValueError("at least one generator has no test rows; nothing to score")
    return min(n_fake, TEST_PER_GENERATOR), min(n_fake, n_real, TEST_REAL_N)


def real_selection_hash(meta, real_idx, length=8):
    """Digest of the shared real block's (h, w) pairs, for confirming two runs agree.

    Hashes the sizes rather than the row indices so it also catches a cache built in a
    different shard order, which an index-level comparison would miss.
    """
    pairs = np.stack([np.asarray(meta["h"])[real_idx], np.asarray(meta["w"])[real_idx]], axis=1)
    pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    return hashlib.sha256(pairs.astype(np.int64).tobytes()).hexdigest()[:length]


def _cell_scores(pred, truth):
    """Accuracy plus TPR/TNR for one cell. pred/truth are bool arrays, True = synthetic.

    TPR and TNR are kept because accuracy alone cannot distinguish a cell that gets every
    real right and every fake wrong from one that is guessing -- both read as 50%, and the
    size-lookup matrix produces the first kind often.
    """
    is_fake, is_real = truth, ~truth
    return {
        "accuracy": float((pred == truth).mean()),
        "tpr": float(pred[is_fake].mean()) if is_fake.any() else float("nan"),
        "tnr": float((~pred[is_real]).mean()) if is_real.any() else float("nan"),
        "n": int(truth.size),
    }


def _eval_cell(meta, target, rule, real_idx, n_fake=TEST_PER_GENERATOR):
    """Apply a fitted rule to one (source -> target) cell: the target's fakes + shared reals.

    Takes the real block rather than drawing it, so every cell in the grid provably scores
    the same real images.
    """
    fake_idx = select_fake_block(meta, target, n_fake)
    idx = np.concatenate([fake_idx, real_idx])

    hw = np.stack([meta["h"][idx], meta["w"][idx]], axis=1)
    truth = np.concatenate([np.ones(len(fake_idx), bool), np.zeros(len(real_idx), bool)])
    return _cell_scores(rule(hw), truth)


def _eval_cell_full(meta, target, rule):
    """Same rule, scored on every test fake of `target` and every test real.

    A tighter estimate than the sampled cell, which the baseline can afford and the CNN
    cannot. Reports balanced accuracy because reals outnumber fakes here by roughly seven
    to one, and plain accuracy over that would mostly measure the real class.
    """
    gen = meta["generator"]
    fake_idx = np.flatnonzero(gen == target)
    real_idx = np.flatnonzero(_real_mask(gen))
    idx = np.concatenate([fake_idx, real_idx])

    hw = np.stack([meta["h"][idx], meta["w"][idx]], axis=1)
    truth = np.concatenate([np.ones(len(fake_idx), bool), np.zeros(len(real_idx), bool)])
    scores = _cell_scores(rule(hw), truth)
    scores["balanced_accuracy"] = 0.5 * (scores["tpr"] + scores["tnr"])
    scores["n_fake"], scores["n_real"] = int(len(fake_idx)), int(len(real_idx))
    return scores


def square_rule_matrix(test_meta, real_idx, n_fake=TEST_PER_GENERATOR):
    """Predict synthetic iff h == w. Returns (cells, full_pool), each a 7x7 source->target dict.

    Nothing is fitted, so the source axis has no effect and all seven rows come out
    identical. The full 7x7 is emitted anyway so this drops into the figure grid beside
    the CNN matrices without special-casing.
    """
    def rule(hw):
        return hw[:, 0] == hw[:, 1]

    shared_row = {t: _eval_cell(test_meta, t, rule, real_idx, n_fake) for t in GENERATORS}
    shared_full = {t: _eval_cell_full(test_meta, t, rule) for t in GENERATORS}
    matrix = {s: {t: dict(shared_row[t]) for t in GENERATORS} for s in GENERATORS}
    full = {s: {t: dict(shared_full[t]) for t in GENERATORS} for s in GENERATORS}
    return matrix, full


def size_lookup_matrix(train_meta, test_meta, real_idx, n_fake=TEST_PER_GENERATOR):
    """Fit per source: the (h, w) pairs that generator emits in train; in-set = synthetic.

    Returns (cells, full_pool), each a 7x7 source->target dict. Cell (source, target) reads
    "knows source's resolutions, tested on target" -- fitted on train, scored on test, the
    same protocol the CNN follows. Expect the matrix to block by resolution rather than by
    generator, since generators sharing a native size will score well on each other.
    """
    gen = train_meta["generator"]
    matrix, full = {}, {}

    for source in GENERATORS:
        source_rows = gen == source
        known_sizes = set(
            zip(train_meta["h"][source_rows].tolist(), train_meta["w"][source_rows].tolist())
        )
        if not known_sizes:
            raise ValueError("no train rows for source {!r}".format(source))

        # Bound as a default argument so each source's rule keeps its own size set rather
        # than closing over the loop variable.
        def rule(hw, sizes=known_sizes):
            return np.array([(int(h), int(w)) in sizes for h, w in hw], bool)

        matrix[source] = {
            t: _eval_cell(test_meta, t, rule, real_idx, n_fake) for t in GENERATORS
        }
        full[source] = {t: _eval_cell_full(test_meta, t, rule) for t in GENERATORS}

    return matrix, full


def _summarize(matrix):
    """Diagonal mean, off-diagonal mean, and the gap -- the same three numbers as the CNN arms."""
    diagonal = [matrix[gen][gen]["accuracy"] for gen in GENERATORS]
    off_diagonal = [
        matrix[source][target]["accuracy"]
        for source in GENERATORS
        for target in GENERATORS
        if source != target
    ]
    return {
        "diagonal_mean": float(np.mean(diagonal)),
        "off_diagonal_mean": float(np.mean(off_diagonal)),
        "gap": float(np.mean(diagonal) - np.mean(off_diagonal)),
    }


def _check_composition(test_meta, train_meta):
    """Stop unless the cache holds the full expected dataset.

    These are exact population counts, so any mismatch means the build dropped, duplicated
    or mis-split rows. Worth failing on, because a wrong cache still yields a complete and
    plausible-looking matrix. Skipped under --partial, which exists for the pilot.
    """
    n_test, n_train = len(test_meta["h"]), len(train_meta["h"])
    if (n_test, n_train) != (EXPECTED_TEST_ROWS, EXPECTED_TRAIN_ROWS):
        raise AssertionError(
            "expected {:,} test / {:,} train rows, got {:,} / {:,}".format(
                EXPECTED_TEST_ROWS, EXPECTED_TRAIN_ROWS, n_test, n_train
            )
        )

    gen = test_meta["generator"]
    counts = {g: int((gen == g).sum()) for g in GENERATORS}
    mismatched = {g: n for g, n in counts.items() if n != TEST_PER_GENERATOR}
    if mismatched:
        raise AssertionError(
            "expected {} test fakes per generator, got {}".format(
                TEST_PER_GENERATOR, mismatched
            )
        )

    n_real = int(_real_mask(gen).sum())
    if n_real != EXPECTED_TEST_REALS:
        raise AssertionError(
            "expected {:,} test reals, got {:,}".format(EXPECTED_TEST_REALS, n_real)
        )


def _report(name, matrix):
    print("\n{}  (rows = source, cols = target, accuracy %)".format(name))
    print("            " + "".join("{:>11}".format(target[:10]) for target in GENERATORS))
    for source in GENERATORS:
        row = "".join(
            "{:>11.1f}".format(100 * matrix[source][target]["accuracy"]) for target in GENERATORS
        )
        print("{:<12}{}".format(source[:11], row))

    summary = _summarize(matrix)
    print(
        "  diag {:.1f}%   off-diag {:.1f}%   gap {:+.1f} pp".format(
            100 * summary["diagonal_mean"],
            100 * summary["off_diagonal_mean"],
            100 * summary["gap"],
        )
    )


def main():
    # Raw formatter: the default one re-wraps the docstring into a single paragraph,
    # which runs the two rule definitions together and splits "size-lookup" at its hyphen.
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cache", required=True, help="cache root holding meta/")
    ap.add_argument("--out", default="results/baseline")
    ap.add_argument(
        "--partial",
        action="store_true",
        help="score a partial cache (pilot only); shrinks the pools and skips the "
             "composition check. Results are not comparable to the CNN runs.",
    )
    args = ap.parse_args()

    # No --seed: these rules fit nothing and the evaluation pools are drawn with a fixed
    # seed, so there is no run-to-run variation for a seed to control.
    test_meta = load_meta(args.cache, TEST_SPLIT)
    train_meta = load_meta(args.cache, TRAIN_SPLIT)
    print("loaded {} test rows, {} train rows".format(len(test_meta["h"]), len(train_meta["h"])))

    # Reported because these rows cannot be cropped to 128x128 without upscaling, which
    # resamples them -- the thing the crop strategies exist to avoid. Sizes are all this
    # script reads, so it is unaffected either way.
    is_real = _real_mask(test_meta["generator"])
    short_side = np.minimum(test_meta["h"], test_meta["w"])[is_real]
    undersized = int((short_side < CACHE_SIZE_PX).sum())
    print("  test reals with min(h,w) < {}: {} of {}".format(
        CACHE_SIZE_PX, undersized, int(is_real.sum())))

    # The pilot scores two shards to confirm the cache is sane before the full build, so a
    # partial cache has to be runnable. Only the pool sizes and the population check give
    # way; every other check below still applies.
    if args.partial:
        n_fake, n_real = _pilot_pool_sizes(test_meta)
        print("\n  *** --partial: {} fakes + {} reals per cell, composition check skipped."
              "\n  *** Pilot output only -- not comparable to the CNN runs. ***".format(
                  n_fake, n_real))
    else:
        n_fake, n_real = FULL_POOL_SIZES
        _check_composition(test_meta, train_meta)

    # SD14 is a category in the source dataset with no rows, so there are seven generators
    # and not eight. Raised rather than asserted, like the other checks that guard whether
    # the results mean anything, so `python -O` cannot strip it.
    present_generators = set(test_meta["generator"][~is_real].tolist())
    if "SD14" in present_generators:
        raise AssertionError("SD14 has rows in this cache, so it is not the expected dataset")
    if present_generators != set(GENERATORS):
        raise AssertionError("generators in cache: {}".format(sorted(present_generators)))

    # Drawn once and handed to both rules: every cell scores this exact block, so no two
    # cells can differ by which reals they saw.
    real_idx = select_real_block(test_meta, n_real)

    square_cells, square_full = square_rule_matrix(test_meta, real_idx, n_fake)
    lookup_cells, lookup_full = size_lookup_matrix(train_meta, test_meta, real_idx, n_fake)
    matrices = {
        "square_rule": (square_cells, square_full),
        "size_lookup": (lookup_cells, lookup_full),
    }

    for name, (cells, _) in matrices.items():
        _report(name, cells)

    # Lets anyone scoring against this cache confirm they drew the same real block.
    print("\nreal selection hash: {}".format(real_selection_hash(test_meta, real_idx)))

    # The gate. Runs before anything is written so a failure leaves no results behind, and
    # reads the full pool so the number does not depend on which reals were drawn.
    gate_cell = square_full[GENERATORS[0]][GENERATORS[0]]
    gate_accuracy = gate_cell["balanced_accuracy"]
    print("\nGATE: square-rule balanced accuracy over the full test pool {:.1%}"
          "  (TPR {:.1%}, TNR {:.1%}, {} fakes vs {} reals)".format(
              gate_accuracy, gate_cell["tpr"], gate_cell["tnr"],
              gate_cell["n_fake"], gate_cell["n_real"]))
    if gate_accuracy < GATE_MIN_ACCURACY:
        print("  FAIL -- expected at least {:.0%}. Stop and re-check the cache build "
              "before writing anything.".format(GATE_MIN_ACCURACY))
        return 1
    print("  OK -- the size confound alone nearly solves the task.")

    # Same directory shape and filename the CNN runs use, so both kinds of panel load
    # through one code path. seed0 is a fixed path component rather than a run seed --
    # neither rule has any seed-dependent behavior to vary.
    for name, (cells, full) in matrices.items():
        summary = _summarize(cells)
        for source in GENERATORS:
            run_dir = os.path.join(args.out, name, source, "seed0")
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "metrics.json"), "w") as f:
                json.dump(
                    {
                        "strategy": name,
                        "source": source,
                        "seed": 0,
                        # Marks pilot output on disk so it cannot be mistaken for a
                        # reportable run.
                        "partial": args.partial,
                        # cells matches the CNN's protocol: the target's fakes plus the
                        # shared real block.
                        "cells": cells[source],
                        # full_pool has no CNN counterpart -- the whole test population,
                        # which only the baseline is cheap enough to score.
                        "full_pool": full[source],
                        "summary": summary,
                    },
                    f,
                    indent=2,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
