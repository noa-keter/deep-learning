"""
Pilot verification for the equalisation cache - run it after two shards, before
committing the full build.

PLAN.md: "A colour-channel swap or transposed axis costs 5 minutes to catch here,
and 28 ruined runs if you catch it later." This module does the four mechanical
checks and renders the contact sheet; the fifth check is Ido actually looking at
the sheet.

Usage:
    python -m src.pilot --cache-dir /content/drive/MyDrive/aidet/cache --split validation
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from collections import Counter

from src.data import GENERATORS, REAL_TAG, STRATEGIES, Strategy, load_meta, shard_paths

__all__ = ["check_cache", "native_sizes", "render_contact_sheet", "sd14_is_declared_and_empty"]

#: Native sizes recorded in PROJECT_STATE.md, verified against the dataset in
#: 2026-08. A mismatch here means the decode path is wrong, not that the note is.
EXPECTED_NATIVE_SIZES: dict[str, tuple[int, int] | None] = {
    "ADM": (256, 256),
    "BigGAN": (128, 128),
    "GLIDE": (256, 256),
    "Midjourney": (1024, 1024),
    "SD15": (512, 512),
    "VQDM": (256, 256),
    "Wukong": (512, 512),
    REAL_TAG: None,  # real photographs vary; 500x375, 375x500, 500x333, ...
}

EMPTY_CLASS_NAME = "SD14"
SAMPLES_PER_GENERATOR = 3
CACHE_SIZE_PX = 128


def _tagged_generators(meta: dict[str, np.ndarray]) -> np.ndarray:
    """
    Map each row's generator to a tag, collapsing all real rows onto REAL_TAG.

    Args:
        meta: Output of `load_meta`.

    Returns:
        Per-row tag array.
    """
    tags = meta["generator"].copy()
    tags[~np.isin(tags, GENERATORS)] = REAL_TAG
    return tags


def native_sizes(meta: dict[str, np.ndarray]) -> dict[str, Counter]:
    """
    Count the native (w, h) pairs each generator emits.

    Args:
        meta: Output of `load_meta`.

    Returns:
        Tag -> Counter over (width, height) pairs.
    """
    tags = _tagged_generators(meta)
    out: dict[str, Counter] = {}
    for tag in sorted(set(tags.tolist())):
        rows = tags == tag
        out[tag] = Counter(zip(meta["w"][rows].tolist(), meta["h"][rows].tolist()))
    return out


def sd14_is_declared_and_empty(cache_dir: str | Path, meta: dict[str, np.ndarray]) -> bool:
    """
    Assert SD14 is a declared class name that carries zero rows.

    The claim "seven generators, not eight" rests on this. The name is read from
    the parquet schema metadata that `build_cache` copied into
    `meta/class_names.json`; the row count is read from the cache itself.

    Args:
        cache_dir: Cache root.
        meta: Output of `load_meta`.

    Returns:
        True if SD14 is declared and has zero rows.

    Raises:
        AssertionError: If SD14 carries rows, or is not declared anywhere and the
            declaration file is missing entirely.
        FileNotFoundError: If class_names.json was never written.
    """
    rows = int(np.sum(meta["generator"] == EMPTY_CLASS_NAME))
    assert rows == 0, f"{EMPTY_CLASS_NAME} carries {rows} rows - the seven-generator claim is wrong"

    path = Path(cache_dir) / "meta" / "class_names.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; rebuild a shard so the class names get recorded")
    declared = json.loads(path.read_text(encoding="utf-8"))
    names = [n for value in declared.values() if value for n in value]
    assert EMPTY_CLASS_NAME in names, (
        f"{EMPTY_CLASS_NAME} is not among the declared class names {names}; "
        "the zero-row claim cannot be made from this cache"
    )
    return True


def check_cache(cache_dir: str | Path, split: str = "validation") -> dict[str, object]:
    """
    Run every mechanical pilot check and print a report.

    Checks: shard/row counts agree across all four strategies and the metadata;
    array shape and dtype; per-generator row counts; native sizes against
    EXPECTED_NATIVE_SIZES; the SD14 zero-row assertion; and how many images were
    smaller than the cache size on either axis (expected: zero).

    Args:
        cache_dir: Cache root.
        split: Split to check.

    Returns:
        A summary dict, also printed.

    Raises:
        AssertionError: If any check fails.
    """
    root = Path(cache_dir)
    meta = load_meta(root, split)
    n_rows = len(meta["h"])
    tags = _tagged_generators(meta)

    per_strategy: dict[str, int] = {}
    for strategy in STRATEGIES:
        total = 0
        for path in shard_paths(root, strategy, split):
            array = np.load(path, mmap_mode="r")
            assert array.dtype == np.uint8, f"{path.name}: dtype {array.dtype}, expected uint8"
            assert array.shape[1:] == (CACHE_SIZE_PX, CACHE_SIZE_PX, 3), (
                f"{path.name}: shape {array.shape}, expected (n, 128, 128, 3) - "
                "a transposed axis would show up here"
            )
            total += array.shape[0]
        per_strategy[strategy] = total
        assert total == n_rows, f"{strategy}: {total} rows, metadata says {n_rows}"

    counts = {tag: int(np.sum(tags == tag)) for tag in sorted(set(tags.tolist()))}
    sizes = native_sizes(meta)

    size_report: dict[str, str] = {}
    for tag, expected in EXPECTED_NATIVE_SIZES.items():
        if tag not in sizes:
            size_report[tag] = "ABSENT from this split"
            continue
        common = sizes[tag].most_common(3)
        if expected is None:
            size_report[tag] = f"varies, top: {common}"
        else:
            modal = common[0][0]
            off = sum(n for size, n in sizes[tag].items() if size != expected)
            # Hard-assert the modal size: a wrong one means the decode is wrong.
            # A handful of off-size rows is reported loudly but does not abort the
            # pilot at 1 a.m. - it is a finding about the dataset, not a bug.
            assert modal == expected, f"{tag}: modal native size {modal}, expected {expected}"
            size_report[tag] = f"{modal} ok" if off == 0 else f"{modal} ok, but {off} rows differ: {common[1:]}"

    undersized = int(np.sum((meta["w"] < CACHE_SIZE_PX) | (meta["h"] < CACHE_SIZE_PX)))
    sd14 = sd14_is_declared_and_empty(root, meta)

    print(f"[pilot] split={split} rows={n_rows}")
    print(f"[pilot] rows per strategy: {per_strategy}")
    print(f"[pilot] rows per generator: {counts}")
    for tag, line in size_report.items():
        print(f"[pilot]   {tag:<11} {line}")
    print(f"[pilot] images smaller than {CACHE_SIZE_PX}px on an axis: {undersized} (expected 0)")
    print(f"[pilot] SD14 declared with zero rows: {sd14}")
    return {
        "split": split,
        "rows": n_rows,
        "per_strategy": per_strategy,
        "per_generator": counts,
        "native_sizes": size_report,
        "undersized": undersized,
        "sd14_declared_empty": sd14,
    }


def render_contact_sheet(
    cache_dir: str | Path,
    out_png: str | Path,
    split: str = "validation",
    samples: int = SAMPLES_PER_GENERATOR,
    seed: int = 0,
) -> Path:
    """
    Render `samples` images per generator per strategy into one PNG, and look at it.

    Rows are generators plus the real class; column groups are the four
    strategies. This is the check that catches a BGR swap or a transposed axis,
    neither of which any assertion above would notice.

    Args:
        cache_dir: Cache root.
        out_png: Destination PNG path.
        split: Split to sample from.
        samples: Images per generator per strategy.
        seed: Sampling seed.

    Returns:
        The written path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(cache_dir)
    meta = load_meta(root, split)
    tags = _tagged_generators(meta)
    rng = np.random.default_rng(seed)

    order = [tag for tag in (*GENERATORS, REAL_TAG) if np.any(tags == tag)]
    picks = {tag: rng.choice(np.flatnonzero(tags == tag), size=samples, replace=False) for tag in order}

    n_cols = len(STRATEGIES) * samples
    fig, axes = plt.subplots(
        len(order), n_cols, figsize=(1.15 * n_cols, 1.15 * len(order)), squeeze=False
    )
    for col, strategy in enumerate(STRATEGIES):
        arrays = [np.load(p, mmap_mode="r") for p in shard_paths(root, strategy, split)]
        offsets = np.zeros(len(arrays), np.int64)
        offsets[1:] = np.cumsum([a.shape[0] for a in arrays])[:-1]
        for row, tag in enumerate(order):
            for k, global_row in enumerate(picks[tag]):
                shard = int(meta["shard"][global_row])
                image = np.asarray(arrays[shard][global_row - offsets[shard]])
                ax = axes[row][col * samples + k]
                ax.imshow(image)
                ax.set_xticks([])
                ax.set_yticks([])
                if k == 0:
                    ax.spines["left"].set_linewidth(2.0)
                if row == 0 and k == samples // 2:
                    ax.set_title(strategy, fontsize=8)
                if col == 0 and k == 0:
                    ax.set_ylabel(tag, fontsize=7, rotation=0, ha="right", va="center")

    fig.suptitle(f"Tiny-GenImage cache pilot - {split}", fontsize=10)
    fig.tight_layout()
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[pilot] contact sheet -> {out}  (open it and actually look at it)")
    return out


def _main() -> None:
    """
    CLI entry point: run the checks, then render the contact sheet.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Verify a pilot cache build.")
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--out-png", type=Path, default=Path("results/figures/pilot_contact.png"))
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    check_cache(args.cache_dir, args.split)
    if not args.no_render:
        render_contact_sheet(args.cache_dir, args.out_png, args.split)


if __name__ == "__main__":
    _main()
