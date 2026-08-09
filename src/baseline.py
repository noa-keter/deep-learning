"""Size-confound baselines: how much of "AI-image detection" is solved by image size alone.

Both rules use only the cached metadata (h, w, label, generator) -- no pixels, no training,
no GPU. They measure how far a detector gets by exploiting the fact that generated images
are square while real photographs are not, which is the confound the four equalization
strategies are meant to remove.

  square rule       predict synthetic iff h == w. Zero parameters, nothing fitted.
  size-lookup rule  fit the (h, w) pairs a generator emits in train; predict synthetic
                    iff the test pair is in that set.

Each rule produces a 7x7 source->target matrix in the same layout and file format as the
CNN runs, at <out>/{rule}/{source}/seed{n}/metrics.json.

Run:  python src/baseline.py --cache /path/to/aidet/cache --out results/baseline
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_select import real_selection_hash, select_eval_indices

GENERATORS = ["ADM", "BigGAN", "GLIDE", "Midjourney", "SD15", "VQDM", "Wukong"]

# The dataset's own category order, needed when `generator` arrives as integer codes.
# Index 5 is SD14, which contributes no rows: decoding through GENERATORS above would read
# SD15 as SD14 and shift every name after it, giving a correctly-shaped matrix with the
# wrong row labels.
CLASS_LABEL_NAMES = [
    "Real", "ADM", "BigGAN", "GLIDE", "Midjourney", "SD14", "SD15", "VQDM", "Wukong",
]

# Column names vary with how the metadata was written; both spellings map to the same field.
KEY_ALIASES = {"height": "h", "width": "w", "class": "label"}

# The validation split is written either way depending on the export.
SPLIT_PREFIXES = {"val": ("val", "validation"), "train": ("train",)}


def _shard_number(fname):
    """Trailing integer of a <split>-<n>.npz filename, for ordering shards numerically."""
    stem = os.path.splitext(fname)[0]
    return int(stem.rsplit("-", 1)[1])


def load_meta(cache_dir, split):
    """Concatenate every meta/<split>-<shard>.npz into one dict of flat arrays.

    Returns {'h', 'w', 'label', 'generator'}, all length N, ordered shard-then-row to match
    how the .npy pixel shards concatenate, so one index addresses both.
    """
    meta_dir = os.path.join(cache_dir, "meta")
    available = [f for f in os.listdir(meta_dir) if f.endswith(".npz")]

    shards = []
    for prefix in SPLIT_PREFIXES.get(split, (split,)):
        shards = [f for f in available if f.startswith(prefix + "-")]
        if shards:
            break
    if not shards:
        raise FileNotFoundError("no {} shards in {}".format(split, meta_dir))

    # Numeric, not lexicographic: with unpadded numbers "val-10" would sort before "val-2"
    # and the metadata would line up against the wrong pixel rows. Nothing raises when that
    # happens -- the matrix still prints, built on mismatched rows.
    shards.sort(key=_shard_number)

    columns = {name: [] for name in ("h", "w", "label", "generator")}
    for fname in shards:
        with np.load(os.path.join(meta_dir, fname), allow_pickle=False) as shard:
            stored_name = {KEY_ALIASES.get(key, key): key for key in shard.files}
            for name in columns:
                if name not in stored_name:
                    raise KeyError(
                        "{}: no {!r} column (has {})".format(fname, name, sorted(shard.files))
                    )
                columns[name].append(shard[stored_name[name]])

    meta = {name: np.concatenate(chunks) for name, chunks in columns.items()}
    meta["generator"] = _decode_generator(meta["generator"])
    return meta


def _decode_generator(column):
    """Generator column -> array of str names, whether it arrives as ints, bytes, or str."""
    if column.dtype.kind in "iu":
        if column.max() >= len(CLASS_LABEL_NAMES):
            raise ValueError(
                "generator code {} exceeds the {} known class names".format(
                    column.max(), len(CLASS_LABEL_NAMES)
                )
            )
        return np.asarray(CLASS_LABEL_NAMES, dtype="U")[column]
    # np.savez stores str arrays as fixed-width bytes on some numpy versions, so a
    # comparison against a str generator name would silently match nothing.
    if column.dtype.kind == "S":
        return column.astype("U")
    return column


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


def _eval_cell(meta, target, rule, seed):
    """Apply a fitted rule to one (source -> target) cell's 500 fakes + fixed 500 reals."""
    fake_idx, real_idx = select_eval_indices(meta, target, seed=seed)
    idx = np.concatenate([fake_idx, real_idx])

    hw = np.stack([meta["h"][idx], meta["w"][idx]], axis=1)
    truth = np.concatenate([np.ones(len(fake_idx), bool), np.zeros(len(real_idx), bool)])
    return _cell_scores(rule(hw), truth)


def _eval_cell_full(meta, target, rule):
    """Same rule, scored on every val fake of `target` and every val real.

    The 500+500 cell matches the CNN's protocol; this one uses the whole population
    because the baseline is cheap enough to afford it, giving a ~2.6x tighter estimate
    that does not depend on which reals were sampled.

    Balanced accuracy (mean of TPR and TNR) rather than plain accuracy: the pool is ~500
    fakes against ~3,500 reals, so plain accuracy would mostly measure the real class.
    """
    gen, label = meta["generator"], meta["label"]
    fake_idx = np.flatnonzero((label == 1) & (gen == target))
    real_idx = np.flatnonzero(label == 0)
    idx = np.concatenate([fake_idx, real_idx])

    hw = np.stack([meta["h"][idx], meta["w"][idx]], axis=1)
    truth = np.concatenate([np.ones(len(fake_idx), bool), np.zeros(len(real_idx), bool)])
    scores = _cell_scores(rule(hw), truth)
    scores["balanced_accuracy"] = 0.5 * (scores["tpr"] + scores["tnr"])
    scores["n_fake"], scores["n_real"] = int(len(fake_idx)), int(len(real_idx))
    return scores


def square_rule_matrix(val_meta, seed=0):
    """Predict synthetic iff h == w. Returns (cells, full_pool), each a 7x7 source->target dict.

    Nothing is fitted, so the source axis has no effect and all seven rows come out
    identical. The full 7x7 is emitted anyway so this drops into the figure grid beside
    the CNN matrices without special-casing.
    """
    rule = lambda hw: hw[:, 0] == hw[:, 1]
    row = {t: _eval_cell(val_meta, t, rule, seed) for t in GENERATORS}
    full = {t: _eval_cell_full(val_meta, t, rule) for t in GENERATORS}
    matrix = {s: {t: dict(row[t]) for t in GENERATORS} for s in GENERATORS}
    return matrix, {s: {t: dict(full[t]) for t in GENERATORS} for s in GENERATORS}


def size_lookup_matrix(train_meta, val_meta, seed=0):
    """Fit per source: the (h, w) pairs that generator emits in train; in-set = synthetic.

    Returns (cells, full_pool), each a 7x7 source->target dict. Cell (s, t) means "knows
    s's resolutions, tested on t", fitted on train and scored on val -- the same protocol
    the CNN follows, which is what makes the two matrices comparable.

    Generators sharing a native resolution will score well on each other, so the matrix is
    expected to come out blocked by resolution rather than by generator identity.
    """
    gen = train_meta["generator"]
    matrix, full = {}, {}

    for source in GENERATORS:
        # The label term is redundant here (reals carry "Real" in the generator column) but
        # states the intent -- fit on this generator's fakes -- and survives a re-export
        # that tags reals differently.
        fitted = (gen == source) & (train_meta["label"] == 1)
        known = set(zip(train_meta["h"][fitted].tolist(), train_meta["w"][fitted].tolist()))
        if not known:
            raise ValueError("no train rows for source {!r}".format(source))

        # Closure over `known` by default-arg so each row's rule keeps its own fitted set.
        rule = lambda hw, k=known: np.array([(int(a), int(b)) in k for a, b in hw], bool)
        matrix[source] = {t: _eval_cell(val_meta, t, rule, seed) for t in GENERATORS}
        full[source] = {t: _eval_cell_full(val_meta, t, rule) for t in GENERATORS}

    return matrix, full


def _summarize(matrix):
    """Diagonal mean, off-diagonal mean, and the gap -- the same three numbers as the CNN arms."""
    diagonal = [matrix[gen][gen]["accuracy"] for gen in GENERATORS]
    off_diagonal = [matrix[src][tgt]["accuracy"] for src in GENERATORS for tgt in GENERATORS if src != tgt]
    return {
        "diagonal_mean": float(np.mean(diagonal)),
        "off_diagonal_mean": float(np.mean(off_diagonal)),
        "gap": float(np.mean(diagonal) - np.mean(off_diagonal)),
    }


def _check_composition(val_meta, train_meta):
    """Stop unless the cache holds the expected dataset.

    These are exact population counts, so any mismatch means the build dropped, duplicated
    or mis-split rows. Worth failing on, because a wrong cache still yields a complete and
    plausible-looking matrix.
    """
    n_val, n_train = len(val_meta["h"]), len(train_meta["h"])
    if (n_val, n_train) != (7000, 28000):
        raise AssertionError(
            "expected 7,000 val / 28,000 train rows, got {:,} / {:,}".format(n_val, n_train)
        )

    gen, label = val_meta["generator"], val_meta["label"]
    counts = {g: int(((gen == g) & (label == 1)).sum()) for g in GENERATORS}
    wrong = {g: n for g, n in counts.items() if n != 500}
    if wrong:
        raise AssertionError("expected 500 val fakes per generator, got {}".format(wrong))

    n_real = int((label == 0).sum())
    if n_real != 3500:
        raise AssertionError("expected 3,500 val reals, got {:,}".format(n_real))


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
    ap.add_argument("--cache", required=True, help="aidet/cache dir holding meta/")
    ap.add_argument("--out", default="results/baseline")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    val_meta = load_meta(args.cache, "val")
    train_meta = load_meta(args.cache, "train")
    print("loaded {} val rows, {} train rows".format(len(val_meta["h"]), len(train_meta["h"])))

    # Reported because these rows cannot be cropped to 128x128 without upscaling, which
    # resamples them -- the thing the crop strategies exist to avoid. Sizes are all this
    # script reads, so it is unaffected either way.
    real = val_meta["label"] == 0
    undersized = int((np.minimum(val_meta["h"], val_meta["w"])[real] < 128).sum())
    print("  val reals with min(h,w) < 128: {} of {}".format(undersized, int(real.sum())))

    _check_composition(val_meta, train_meta)

    # SD14 is a category in the source dataset with no rows, so there are seven generators
    # and not eight. Checked here so the count is never taken on trust.
    present = set(val_meta["generator"][val_meta["label"] == 1].tolist())
    assert "SD14" not in present, "SD14 has rows in this cache, so it is not the expected dataset"
    assert present == set(GENERATORS), "generators in cache: {}".format(sorted(present))

    square_cells, square_full = square_rule_matrix(val_meta, args.seed)
    lookup_cells, lookup_full = size_lookup_matrix(train_meta, val_meta, args.seed)
    matrices = {
        "square_rule": (square_cells, square_full),
        "size_lookup": (lookup_cells, lookup_full),
    }

    for name, (cells, _) in matrices.items():
        _report(name, cells)

    # Lets anyone scoring against this cache confirm they drew the same reals.
    print("\nreal selection hash (seed {}): {}".format(
        args.seed, real_selection_hash(val_meta, seed=args.seed)))

    # Nothing is fitted in the square rule and the real half is fixed, so the source axis
    # cannot affect any cell -- all 49 must be identical. If they are not, the real
    # selection is changing between calls and no cell is comparable to any other.
    distinct = {round(square_cells[src][tgt]["accuracy"], 12) for src in GENERATORS for tgt in GENERATORS}
    assert len(distinct) == 1, "square-rule cells differ across the grid: {}".format(sorted(distinct))

    # The gate: every generated image is square, so this rule should be near-perfect. A low
    # score means the metadata is wrong, and everything downstream would be built on it.
    # Checked before any file is written so a failure leaves no results behind, and taken
    # from the full pool rather than the 500-sample so it does not depend on the draw.
    gate_cell = square_full[GENERATORS[0]][GENERATORS[0]]
    gate_accuracy = gate_cell["balanced_accuracy"]
    print("\nGATE: square-rule balanced accuracy over the full val pool {:.1%}"
          "  (TPR {:.1%}, TNR {:.1%}, {} fakes vs {} reals)".format(
              gate_accuracy, gate_cell["tpr"], gate_cell["tnr"],
              gate_cell["n_fake"], gate_cell["n_real"]))
    if gate_accuracy < 0.97:
        print("  FAIL -- expected 97-99%. Stop and re-check the cache build before writing anything.")
        return 1
    print("  OK -- the size confound alone nearly solves the task.")

    # Written as {rule}/{source}/seed{n}/metrics.json -- the same directory shape and the
    # same filename the CNN runs use, so the baseline panels load through exactly the code
    # path the four strategy panels do instead of needing a special case.
    for name, (cells, full) in matrices.items():
        for source in GENERATORS:
            run_dir = os.path.join(args.out, name, source, "seed{}".format(args.seed))
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "metrics.json"), "w") as f:
                json.dump(
                    {
                        "strategy": name,
                        "source": source,
                        "seed": args.seed,
                        # cells matches the CNN's protocol: 500 fakes + the fixed 500 reals.
                        "cells": cells[source],
                        # full_pool has no CNN counterpart -- it is the whole val population,
                        # which only the baseline is cheap enough to score.
                        "full_pool": full[source],
                        "summary": _summarize(cells),
                    },
                    f,
                    indent=2,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
