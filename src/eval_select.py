"""Evaluation-set selection, shared by the baseline and the CNN runs.

Every reported cell is scored on one target generator's fakes plus a fixed sample of real
validation images. The real sample depends only on `seed`, so it is the same for every
generator and every equalization strategy -- cells in a row then differ only by their fake
half, which is the comparison the transfer matrices rest on.
"""

import hashlib

import numpy as np

N_EVAL = 500


def select_eval_indices(meta, generator, n=N_EVAL, seed=0):
    """Indices into the val split for one (source -> target) cell.

    meta: mapping with 'label' (1 = synthetic) and 'generator', both length-N arrays.
    Returns (fake_idx, real_idx), each int64, length n, sorted ascending.
    """
    label = np.asarray(meta["label"])
    gen = np.asarray(meta["generator"])

    # np.savez stores str arrays as fixed-width bytes on some numpy versions, so a
    # comparison against a str generator name would silently match nothing.
    if gen.dtype.kind == "S":
        gen = gen.astype("U")

    # Undecoded categorical codes would compare unequal to every name and surface below as
    # "val holds 0 fakes", which points at the split rather than at the encoding. Callers
    # loading metadata by hand are the ones who hit this.
    if gen.dtype.kind in "iu":
        raise TypeError(
            "generator column holds integer codes; decode it to names before selecting"
        )

    fake_pool = np.flatnonzero((label == 1) & (gen == generator))
    real_pool = np.flatnonzero(label == 0)

    # The fake half is not sampled: each generator contributes 2,500 images of which 2,000
    # go to train, so val holds exactly the 500 this asks for and taking the whole filtered
    # set is both simpler and exact. Asserting rather than slicing means a changed split
    # fraction fails here instead of quietly scoring a subsample of a differently-sized pool.
    if len(fake_pool) != n:
        raise ValueError(
            "generator {!r}: val holds {} fakes, expected exactly {} -- "
            "has the train/val split changed?".format(generator, len(fake_pool), n)
        )
    if len(real_pool) < n:
        raise ValueError("only {} val reals, need {}".format(len(real_pool), n))

    real_idx = np.random.default_rng(seed).choice(real_pool, n, replace=False)
    return fake_pool, np.sort(real_idx)


def real_selection_hash(meta, n=N_EVAL, seed=0, length=8):
    """Short digest of the selected reals' (h, w) pairs, for cross-checking between machines.

    Two people running different code paths can compare eight characters instead of
    assuming their fixed real half matches. Hashing the sizes rather than the indices
    means it also catches a cache built in a different row order, which is the failure
    that would otherwise survive an index-level comparison.
    """
    _, real_idx = select_eval_indices(meta, _any_generator(meta), n=n, seed=seed)
    pairs = np.stack([np.asarray(meta["h"])[real_idx], np.asarray(meta["w"])[real_idx]], axis=1)
    pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    return hashlib.sha256(pairs.astype(np.int64).tobytes()).hexdigest()[:length]


def _any_generator(meta):
    """The real draw ignores the generator, but select_eval_indices still needs one."""
    gen = np.asarray(meta["generator"])
    if gen.dtype.kind == "S":
        gen = gen.astype("U")
    return gen[np.asarray(meta["label"]) == 1][0]