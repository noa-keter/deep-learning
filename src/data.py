"""
Data layer for the resolution-bias study: the four equalisation transforms, the
shard-by-shard cache builder, and the per-run arm loader.

This is the only module that touches image bytes. It is importable on a laptop
with **numpy + Pillow only**: `pyarrow`, `huggingface_hub` and `torch` are
imported lazily inside the functions that need them, so `equalize` and the
metadata helpers can be unit-tested without a GPU stack.

Cache layout written by `build_cache`:

    <out_dir>/<strategy>/<split>-<shard>.npy      uint8 (n, 128, 128, 3)
    <out_dir>/meta/<split>-<shard>.npz            h, w, label, generator [, native_*]
    <out_dir>/meta/class_names.json               declared class names per column
    <out_dir>/shards/<split>-<shard>.parquet      the source file, kept as-is
    <out_dir>/native/<split>-<shard>.bin          originals, verbatim, concatenated
    <out_dir>/markers/<split>-<shard>.done        resume marker, written last

The middle two are independent ways to keep the originals, neither re-encoding
anything. `shards/` (on by default) keeps the upstream parquet untouched: the most
faithful copy, and the only one that can rebuild the cache offline, but reading one
image needs pyarrow. `native/` (off by default) stores one blob per shard rather
than one file per image - a Drive FUSE mount is far slower per-file than per-byte -
and `open_native` seeks into it by the offsets in the metadata, needing only
Pillow. Enabling both stores the originals twice.
"""

from __future__ import annotations

import io
import os
import json
import time
import zlib
import shutil
import numpy as np
from PIL import Image
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, NamedTuple

if TYPE_CHECKING:  # pragma: no cover - typing only, torch is not installed locally
    from torch import Tensor

__all__ = [
    "Arm",
    "GENERATORS",
    "REAL_TAG",
    "STRATEGIES",
    "Strategy",
    "build_cache",
    "equalize",
    "export_natives",
    "load_arm",
    "load_arm_numpy",
    "load_meta",
    "native_bytes",
    "open_native",
    "row_rng",
    "shard_paths",
]

Strategy = Literal["centre_crop", "random_crop", "rescale", "pad"]
STRATEGIES: Final[tuple[Strategy, ...]] = ("centre_crop", "random_crop", "rescale", "pad")

#: Largest resampling-free common size the dataset admits: BigGAN is 128x128 native.
CACHE_SIZE_PX: Final[int] = 128

#: The seven generators with rows. `SD14` is a declared class name with zero rows
#: and is deliberately absent here - never claim eight generators.
GENERATORS: Final[tuple[str, ...]] = (
    "ADM",
    "BigGAN",
    "GLIDE",
    "Midjourney",
    "SD15",
    "VQDM",
    "Wukong",
)

#: Tag this module writes for real rows in `gen_ids`. It is our own label, not a
#: claim about the dataset: whatever string the dataset uses for real rows is
#: preserved verbatim in the metadata `.npz`.
REAL_TAG: Final[str] = "REAL"

DEFAULT_REPO_ID: Final[str] = "TheKernel01/Tiny-GenImage"

#: The **test** pools are drawn with this fixed seed, never the run seed, so every
#: cell of every matrix sees byte-identical images and a difference between cells
#: is attributable to the generator alone. Training pools use the run seed instead.
REAL_POOL_SEED: Final[int] = 20260809

TRAIN_PER_CLASS: Final[int] = 2_000
#: Standard three-way split, with one naming hazard: the *dataset's* split named
#: `validation` is what we use as the **test** set. Our validation set is carved
#: out of `train`. So `x_val` never comes from the `validation-*.npy` files.
VAL_FRACTION: Final[float] = 0.10
TEST_PER_GENERATOR: Final[int] = 500
TEST_REAL_N: Final[int] = 500

DECODE_BATCH_ROWS: Final[int] = 64
IMAGE_COLUMN: Final[str] = "image"
LABEL_COLUMN: Final[str] = "label"
GENERATOR_COLUMN: Final[str] = "generator"

_BILINEAR: Final = Image.Resampling.BILINEAR

#: Set at module import, which only helps if this module is imported before
#: huggingface_hub - see the check in `build_cache`.
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")


class Arm(NamedTuple):
    """
    One (strategy, source, seed) run's tensors.

    Standard train/val/test, but read the names carefully: `x_val` is carved out
    of the *train* split and is the only set allowed to influence training, while
    `x_test` comes from the split the dataset calls `validation`. See VAL_FRACTION.

    Fields:
        x_train: uint8 (n_train, 128, 128, 3) - 90 % of the arm's training set.
        y_train: float32 (n_train,) - 1.0 synthetic, 0.0 real.
        x_val: uint8 - the internal 10 %, source generator only, used for
            checkpoint selection and nothing else.
        y_val: float32.
        x_test: uint8 (4000, 128, 128, 3) - seven 500-image generator blocks from
            the dataset's `validation` split, then the fixed 500-image real block.
            Never seen until training is over; every reported number comes from here.
        y_test: float32.
        gen_ids: str array (4000,) - the generator of each test row, REAL_TAG for
            the real block. One matrix cell is `gen_ids == g` plus `gen_ids == REAL_TAG`.
    """

    x_train: Tensor
    y_train: Tensor
    x_val: Tensor
    y_val: Tensor
    x_test: Tensor
    y_test: Tensor
    gen_ids: np.ndarray


# --------------------------------------------------------------------------- #
# The four equalisation transforms
# --------------------------------------------------------------------------- #


def equalize(
    img: Image.Image,
    strategy: Strategy,
    size_px: int = CACHE_SIZE_PX,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Reduce an image to a fixed size_px x size_px RGB array by one strategy.

    Behaviour on images whose native size is smaller than `size_px` on one or
    both axes (expected count on Tiny-GenImage: zero, since the smallest native
    size is BigGAN at 128x128 - the pilot counts and reports any):

    - `centre_crop`: the crop window is centred and extends past the image edge;
      the deficit is zero-padded symmetrically. It is **not** upsampled, because
      upsampling would inject the very resampling artefact the crop arms exist to
      avoid.
    - `random_crop`: the window origin is drawn from `[min(w - size, 0),
      max(w - size, 0)]`, so an undersized axis randomises how the zero padding
      is split between the two sides. Same no-upsampling rationale.
    - `rescale`, `pad`: resample as usual, upscaling if needed. Both arms already
      resample by definition, so there is nothing to protect.

    Args:
        img: Source image at native resolution, any mode.
        strategy: One of STRATEGIES.
        size_px: Output side length in pixels.
        rng: Required for "random_crop"; seed it per row index for reproducibility.

    Returns:
        A (size_px, size_px, 3) uint8 array.

    Raises:
        ValueError: If strategy is unknown, or rng is missing for "random_crop".
    """
    img = img.convert("RGB")
    width_px, height_px = img.size

    if strategy == "centre_crop":
        left = (width_px - size_px) // 2
        top = (height_px - size_px) // 2
        out = img.crop((left, top, left + size_px, top + size_px))

    elif strategy == "random_crop":
        if rng is None:
            raise ValueError("random_crop requires a seeded rng")
        left = int(rng.integers(min(width_px - size_px, 0), max(width_px - size_px, 0) + 1))
        top = int(rng.integers(min(height_px - size_px, 0), max(height_px - size_px, 0) + 1))
        out = img.crop((left, top, left + size_px, top + size_px))

    elif strategy == "rescale":
        out = img.resize((size_px, size_px), _BILINEAR)

    elif strategy == "pad":
        scale = size_px / max(width_px, height_px)
        new_wh = (max(round(width_px * scale), 1), max(round(height_px * scale), 1))
        small = img.resize(new_wh, _BILINEAR)
        out = Image.new("RGB", (size_px, size_px), (0, 0, 0))
        out.paste(small, ((size_px - new_wh[0]) // 2, (size_px - new_wh[1]) // 2))

    else:
        raise ValueError(f"unknown strategy: {strategy}")

    return np.asarray(out, dtype=np.uint8)


def row_rng(seed: int, split: str, shard_ordinal: int, row_index: int) -> np.random.Generator:
    """
    Deterministic per-image RNG for `random_crop`.

    Keyed by (seed, split, shard ordinal, row within shard) rather than a global
    row index, so a shard can be rebuilt in isolation after a disconnect without
    knowing how many rows preceded it. `zlib.crc32` is used instead of `hash()`
    because it is stable across processes and machines.

    Args:
        seed: Build seed.
        split: Split name, e.g. "validation".
        shard_ordinal: Index of the shard within its split.
        row_index: Row index within the shard.

    Returns:
        A seeded numpy Generator.
    """
    return np.random.default_rng([seed, zlib.crc32(split.encode()), shard_ordinal, row_index])


# --------------------------------------------------------------------------- #
# Cache build
# --------------------------------------------------------------------------- #


def _shard_ordinal(stem: str, fallback: int) -> int:
    """
    Parse the shard ordinal out of a HuggingFace parquet stem.

    Args:
        stem: File stem, e.g. "validation-00000-of-00004".
        fallback: Value to use when the stem does not carry an ordinal.

    Returns:
        The ordinal, or `fallback`.
    """
    # From the name, not the loop counter: --max-shards and resume both shift the
    # counter, and this ordinal keys the random_crop RNG.
    parts = stem.split("-")
    for i, part in enumerate(parts):
        if part == "of" and i > 0 and parts[i - 1].isdigit():
            return int(parts[i - 1])
    return fallback


def _class_names(schema, column: str) -> list[str] | None:
    """
    Recover a column's declared ClassLabel names from parquet schema metadata.

    This is how `SD14` is shown to be a *declared* class with zero rows: the name
    is in the schema, no row carries it.

    Args:
        schema: A `pyarrow.Schema`.
        column: Column name to look up.

    Returns:
        The list of class names, or None if the column is not a ClassLabel.
    """
    # Parquet sees a ClassLabel as int64; the names live only in HF's JSON blob in
    # the schema metadata - the only way to learn SD14 exists.
    raw = (schema.metadata or {}).get(b"huggingface")
    if raw is None:
        return None
    try:
        features = json.loads(raw.decode("utf-8")).get("info", {}).get("features", {})
    except (ValueError, AttributeError):
        # An unreadable blob must not kill a 3 h build; fall back to raw values.
        return None
    feature = features.get(column)
    if isinstance(feature, dict) and feature.get("_type") == "ClassLabel":
        names = feature.get("names")
        if isinstance(names, list):
            return [str(n) for n in names]
    return None


def _to_strings(values: list, names: list[str] | None) -> list[str]:
    """
    Render a column's raw values as strings, decoding ClassLabel ints if needed.

    Args:
        values: Raw python values from the parquet column.
        names: ClassLabel names, or None.

    Returns:
        One string per value.
    """
    # Names, not ints: the .npz outlives the parquet schema that explains them.
    if names is None:
        return [str(v) for v in values]
    out: list[str] = []
    for v in values:
        # Non-int or out-of-range values pass through, surfacing in the pilot's tags.
        out.append(names[v] if isinstance(v, int) and 0 <= v < len(names) else str(v))
    return out


def _cell_bytes(cell: object) -> bytes:
    """
    Recover one parquet image cell's encoded file bytes, verbatim.

    These are the bytes `build_cache` archives when `keep_native` is on. They are
    never decoded and re-encoded on the way to disk: a JPEG round-trip would add
    compression artefacts, which is precisely the class of signal this project
    measures.

    Args:
        cell: A HuggingFace Image feature value - a dict with "bytes"/"path", or
            raw encoded bytes, or a filesystem path.

    Returns:
        The encoded image file's bytes.

    Raises:
        TypeError: If the cell shape is not recognised.
    """
    # HF Image struct; bytes inline is this dataset's case - an assumption the pilot
    # confirms. `path` is used when the dataset references files on disk.
    if isinstance(cell, dict):
        if cell.get("bytes") is not None:
            return bytes(cell["bytes"])
        if cell.get("path"):
            return Path(cell["path"]).read_bytes()
        raise TypeError("image cell has neither bytes nor path")
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    if isinstance(cell, str):
        return Path(cell).read_bytes()
    raise TypeError(f"unrecognised image cell type: {type(cell)!r}")


def _open_image(cell: object) -> Image.Image:
    """
    Open one parquet image cell as a PIL image.

    Args:
        cell: A HuggingFace Image feature value.

    Returns:
        An open PIL image (not yet converted to RGB).
    """
    return Image.open(io.BytesIO(_cell_bytes(cell)))


def build_cache(
    split: str,
    out_dir: str | Path,
    repo_id: str = DEFAULT_REPO_ID,
    size_px: int = CACHE_SIZE_PX,
    seed: int = 0,
    max_shards: int | None = None,
    keep_native: bool = False,
    keep_shards: bool = True,
) -> None:
    """
    Download one split of Tiny-GenImage shard by shard and write four uint8 caches.

    Each shard is downloaded, decoded **once per row** into all four strategies,
    written out, then deleted before the next shard is fetched - so peak local
    disk is one shard (~440 MB) regardless of split size. Resumable: a shard whose
    `.done` marker exists is skipped, so a Colab disconnect costs at most one shard.
    The marker is written last, after every array is on disk.

    Two independent ways to retain the originals, both storing exactly the bytes
    upstream shipped and neither re-encoding anything:

    - `keep_shards` (default) preserves the downloaded `.parquet` under `shards/`
      instead of deleting it. The most faithful copy of the source, and the only
      one that can rebuild the whole cache with no network - but reading a single
      image needs pyarrow and a row-group decompress.
    - `keep_native` writes `native/<stem>.bin`, one append-only blob per shard
      (not one file per image - a Drive FUSE mount is far slower per-file than
      per-byte). Random access by byte offset, readable with Pillow alone, so it
      works on a laptop with no pyarrow. Read it with func open_native.

    Enabling both stores the originals twice. Prefer one.

    Peak *local* disk stays one shard either way; both options grow the output
    directory by the full source size.

    Args:
        split: Split name as it appears in the parquet filenames, e.g.
            "validation" or "train". Build validation first - it is where every
            reported cell comes from.
        out_dir: Cache root, e.g. "/content/drive/MyDrive/university/deep_learning/cache".
        repo_id: HuggingFace dataset id.
        size_px: Output side length.
        seed: Build seed; feeds the per-row `random_crop` RNG.
        max_shards: Stop after this many shards - used for the 2-shard pilot.
        keep_native: Archive the original encoded images as a per-shard blob.
        keep_shards: Keep the source .parquet files under `shards/`.

    Raises:
        FileNotFoundError: If the split matches no parquet files in the repo.
    """
    # Before the import, not after: huggingface_hub latches this into its constants
    # at import time, so assigning it afterwards is a no-op.
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

    # Lazy: the laptop imports this module with numpy + Pillow only.
    import pyarrow.parquet as pq
    import huggingface_hub as hub
    from importlib.util import find_spec
    from huggingface_hub import HfApi, hf_hub_download

    # Roughly 20 min versus 3 h, so report which is about to happen. The repo is
    # Xet-backed, so the fast path is hf_xet; without it the download falls back to
    # plain HTTPS. `hf_transfer` was the pre-Xet mechanism and is no longer consulted.
    # A notebook that imported huggingface_hub first latched the old value; only a
    # restart clears it.
    if find_spec("hf_xet") is None:
        print("[build_cache] WARNING: hf_xet not installed, download will be slow", flush=True)
    elif getattr(getattr(hub, "constants", None), "HF_XET_HIGH_PERFORMANCE", None) is False:
        print("[build_cache] WARNING: HF_XET_HIGH_PERFORMANCE off, download will be slower", flush=True)

    out_root = Path(out_dir)
    optional_dirs = (*(("native",) if keep_native else ()), *(("shards",) if keep_shards else ()))
    for sub in (*STRATEGIES, "meta", "markers", *optional_dirs):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    # Metadata-only listing, not load_dataset: that would fetch the whole 8.4 GB
    # split up front and lose per-shard resume.
    files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    # Prefix match assumes the standard HF layout - on the pilot's list. sorted()
    # fixes the order load_meta and shard_paths use to align meta with the .npy.
    shards = sorted(f for f in files if f.endswith(".parquet") and Path(f).stem.startswith(split))
    if not shards:
        raise FileNotFoundError(f"no parquet shards for split {split!r} in {repo_id}")
    # After the sort, so the full run skips the pilot's shards by their markers.
    if max_shards is not None:
        shards = shards[:max_shards]
    print(f"[build_cache] {split}: {len(shards)} shard(s) -> {out_root}", flush=True)

    for ordinal, shard in enumerate(shards):
        started = time.perf_counter()
        stem = Path(shard).stem
        if not stem.startswith(split):
            stem = f"{split}-{stem}"
        marker = out_root / "markers" / f"{stem}.done"
        # The resume mechanism; the marker is written last, so it means "complete".
        if marker.exists():
            print(f"[build_cache] {stem}: done, skipping", flush=True)
            continue

        local = hf_hub_download(repo_id, shard, repo_type="dataset")
        native_tmp = out_root / "native" / f"{stem}.bin.tmp"
        # Bound before the try: if open() raised, an unbound name in the finally
        # would mask the real exception.
        blob = None
        try:
            parquet = pq.ParquetFile(local)
            try:
                # Header value, trusted only as far as the written != n_rows check.
                n_rows = parquet.metadata.num_rows
                names = {
                    col: _class_names(parquet.schema_arrow, col)
                    for col in (LABEL_COLUMN, GENERATOR_COLUMN)
                }
                _write_class_names(out_root, names)

                # Preallocated and filled by index: stacking at the end would hold the
                # rows twice, and these are ~344 MB for a 1,750-row shard.
                stacks = {s: np.empty((n_rows, size_px, size_px, 3), np.uint8) for s in STRATEGIES}
                heights = np.empty(n_rows, np.int32)
                widths = np.empty(n_rows, np.int32)
                labels: list[str] = []
                generators: list[str] = []
                # Shard-local offsets into this shard's .bin - hence load_meta's "shard".
                offsets = np.zeros(n_rows, np.int64)
                nbytes = np.zeros(n_rows, np.int64)
                formats: list[str] = []

                # Streamed, not accumulated: the originals are as large as the shard.
                if keep_native:
                    blob = native_tmp.open("wb")
                cursor = 0

                written = 0
                # A row group at a time; read_table would load all ~440 MB at once.
                for batch in parquet.iter_batches(batch_size=DECODE_BATCH_ROWS):
                    block = batch.to_pydict()
                    labels.extend(_to_strings(block[LABEL_COLUMN], names[LABEL_COLUMN]))
                    generators.extend(_to_strings(block[GENERATOR_COLUMN], names[GENERATOR_COLUMN]))
                    for cell in block[IMAGE_COLUMN]:
                        raw = _cell_bytes(cell)
                        if blob is not None:
                            # Straight from the cell, before any decode: upstream's bytes.
                            blob.write(raw)
                            offsets[written] = cursor
                            nbytes[written] = len(raw)
                            cursor += len(raw)
                        with Image.open(io.BytesIO(raw)) as img:
                            # Decode once for all four strategies; PIL is lazy, so
                            # force it here rather than four times below.
                            img.load()
                            widths[written], heights[written] = img.size
                            # None once detached from the source bytes.
                            formats.append(img.format or "")
                            # Consumed only by random_crop. Keyed shard-locally so a
                            # lone rebuilt shard reproduces byte for byte.
                            rng = row_rng(
                                seed=seed,
                                split=split,
                                shard_ordinal=_shard_ordinal(stem, ordinal),
                                row_index=written,
                            )
                            for strategy in STRATEGIES:
                                stacks[strategy][written] = equalize(
                                    img, strategy, size_px=size_px, rng=rng
                                )
                        written += 1
            finally:
                # Both close before anything unlinks or renames what they point at.
                if blob is not None:
                    blob.close()
                # Close before unlinking: Windows refuses to delete an open file, and
                # on Linux a leaked handle keeps the shard's bytes on disk.
                parquet.close()

            # Preallocated, so a short read leaves zeroed rows under real labels -
            # black images that train fine and report nonsense. Fail instead.
            if written != n_rows:
                raise ValueError(f"{stem}: decoded {written} rows, header says {n_rows}")

            for strategy in STRATEGIES:
                _atomic_save(out_root / strategy / f"{stem}.npy", stacks[strategy])

            extra: dict[str, np.ndarray] = {}
            if keep_native:
                # Renamed only once the shard is known sound, so a visible .bin is
                # complete and its offsets reach the metadata below.
                native_tmp.replace(out_root / "native" / f"{stem}.bin")
                extra = {
                    "native_offset": offsets,
                    "native_nbytes": nbytes,
                    "native_format": np.asarray(formats, dtype=np.str_),
                }
            # asarray sizes the width from the data; np.full(n, tag, dtype=np.str_)
            # would allocate <U1 and truncate. Same trap as in _tag_array.
            np.savez_compressed(
                out_root / "meta" / f"{stem}.npz",
                h=heights,
                w=widths,
                label=np.asarray(labels, dtype=np.str_),
                generator=np.asarray(generators, dtype=np.str_),
                **extra,
            )
            # Free ~344 MB before the next iteration allocates its own.
            del stacks

            if keep_shards:
                _keep_shard(Path(local), out_root / "shards" / Path(shard).name)
        finally:
            # In a finally: the one-shard disk ceiling must hold even on failure.
            Path(local).unlink(missing_ok=True)
            native_tmp.unlink(missing_ok=True)  # no-op once renamed; clears a failed shard

        # Last, and outside the try: any earlier and a half-built shard would be
        # permanently skippable.
        marker.touch()
        elapsed = time.perf_counter() - started
        notes = ""
        if keep_native:
            notes += f", {cursor / 1e6:.0f} MB archived"
        if keep_shards:
            notes += f", {(out_root / 'shards' / Path(shard).name).stat().st_size / 1e6:.0f} MB kept"
        print(
            f"[build_cache] {stem}: {n_rows} rows written in {elapsed:.0f}s{notes}",
            flush=True,
        )


def _keep_shard(local: Path, destination: Path) -> None:
    """
    Copy a downloaded shard into the cache, resolving the hub's symlink first.

    `hf_hub_download` returns a path that is usually a symlink into the hub's
    blob store, and the caller unlinks it immediately afterwards - so the link
    has to be followed here or the kept file would dangle. Written via a
    temporary name so a disconnect cannot leave a short .parquet that looks
    complete.

    Args:
        local: Path returned by `hf_hub_download`.
        destination: Where to keep it.
    """
    # A resume would otherwise rewrite an identical ~440 MB over a Drive mount.
    if destination.exists():
        return
    tmp = destination.with_name(destination.name + ".tmp")
    # resolve() is load-bearing: the caller unlinks `local` moments later.
    shutil.copyfile(local.resolve(), tmp)
    tmp.replace(destination)


def _atomic_save(path: Path, array: np.ndarray) -> None:
    """
    Save a .npy via a temporary file and a rename.

    The `.done` marker already covers the resume, but the cache is also globbed
    while the build runs - by the pilot, and from the other Colab account - so a
    partial array must never be visible under its final name.

    Args:
        path: Destination path.
        array: Array to save.
    """
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array)  # write to the handle: np.save would append ".npy" to a path
    tmp.replace(path)


def _write_class_names(out_root: Path, names: dict[str, list[str] | None]) -> None:
    """
    Record the declared class names once, for the SD14 zero-row assertion.

    Args:
        out_root: Cache root.
        names: Column name -> declared class names (or None).
    """
    path = out_root / "meta" / "class_names.json"
    if path.exists():
        return
    path.write_text(json.dumps(names, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Cache reading
# --------------------------------------------------------------------------- #


def shard_paths(cache_dir: str | Path, strategy: Strategy, split: str) -> list[Path]:
    """
    List a split's cached shards for one strategy, in stable order.

    Args:
        cache_dir: Cache root.
        strategy: One of STRATEGIES.
        split: Split name.

    Returns:
        Sorted list of .npy paths.

    Raises:
        FileNotFoundError: If nothing is cached for that split and strategy.
    """
    paths = sorted((Path(cache_dir) / strategy).glob(f"{split}-*.npy"))
    if not paths:
        raise FileNotFoundError(f"no cached shards for {split}/{strategy} under {cache_dir}")
    return paths


def load_meta(cache_dir: str | Path, split: str) -> dict[str, np.ndarray]:
    """
    Concatenate a split's metadata across shards.

    This is all `baseline.py` needs - the dimension rule never touches pixels.

    Args:
        cache_dir: Cache root.
        split: Split name.

    Returns:
        {"h", "w", "label", "generator", "shard"} as parallel arrays, row order
        matching the cached .npy files concatenated in the same shard order. If
        the cache was built with `keep_native`, also "native_offset",
        "native_nbytes" and "native_format" - shard-local byte offsets into
        `native/<shard>.bin`, which is why "shard" is needed to read them.

    Raises:
        FileNotFoundError: If no metadata shards exist for that split.
    """
    paths = sorted((Path(cache_dir) / "meta").glob(f"{split}-*.npz"))
    if not paths:
        raise FileNotFoundError(f"no metadata shards for {split} under {cache_dir}")
    required = ("h", "w", "label", "generator")
    optional = ("native_offset", "native_nbytes", "native_format")
    parts: dict[str, list[np.ndarray]] = {k: [] for k in (*required, "shard")}
    for index, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as npz:
            for key in required:
                parts[key].append(npz[key])
            # Present only for caches built with keep_native; all shards agree,
            # since a split is always built by one call.
            for key in optional:
                if key in npz.files:
                    parts.setdefault(key, []).append(npz[key])
            parts["shard"].append(np.full(len(npz["h"]), index, np.int32))
    return {key: np.concatenate(value) for key, value in parts.items()}


def native_bytes(cache_dir: str | Path, split: str, row: int, meta: dict | None = None) -> bytes:
    """
    Read one row's original encoded image back out of the native archive.

    The bytes are exactly what the dataset shipped - the same file, not a
    re-encode - so hashes, EXIF and compression structure all survive.

    Args:
        cache_dir: Cache root.
        split: Split name.
        row: Global row index within the split, matching func load_meta order.
        meta: Output of func load_meta, to avoid re-reading it in a loop.

    Returns:
        The encoded image file's bytes.

    Raises:
        FileNotFoundError: If the cache was built without `keep_native`.
    """
    root = Path(cache_dir)
    meta = load_meta(root, split) if meta is None else meta
    if "native_offset" not in meta:
        raise FileNotFoundError(
            f"no native archive for {split} under {root} - built with keep_native=False"
        )
    shard = int(meta["shard"][row])
    blobs = sorted((root / "native").glob(f"{split}-*.bin"))
    with blobs[shard].open("rb") as handle:
        handle.seek(int(meta["native_offset"][row]))
        return handle.read(int(meta["native_nbytes"][row]))


def open_native(cache_dir: str | Path, split: str, row: int, meta: dict | None = None):
    """
    Open one row's original image at native resolution.

    Use this for figures that need the real thing - the deck's "the size alone
    tells you" slide, or any side-by-side against a 128x128 cached crop.

    Args:
        cache_dir: Cache root.
        split: Split name.
        row: Global row index within the split.
        meta: Output of func load_meta, to avoid re-reading it in a loop.

    Returns:
        An open PIL image at its original size.
    """
    return Image.open(io.BytesIO(native_bytes(cache_dir, split, row, meta)))


def export_natives(
    cache_dir: str | Path,
    split: str,
    out_dir: str | Path,
    rows: np.ndarray | list[int],
) -> list[Path]:
    """
    Write selected originals out as ordinary image files, for browsing.

    The archive is one blob per shard, which is right for Drive but not for
    flicking through in a file manager. This unpacks a chosen subset.

    Args:
        cache_dir: Cache root.
        split: Split name.
        out_dir: Directory to write into; created if absent.
        rows: Global row indices to export.

    Returns:
        The written paths, in the order given.
    """
    root = Path(cache_dir)
    meta = load_meta(root, split)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for row in rows:
        row = int(row)
        suffix = str(meta["native_format"][row]).lower() or "bin"
        path = target / f"{split}-{row:06d}-{meta['generator'][row]}.{suffix}"
        path.write_bytes(native_bytes(root, split, row, meta))
        written.append(path)
    return written


def _real_mask(generator: np.ndarray) -> np.ndarray:
    """
    Split rows into real and fake by generator tag, refusing to guess.

    Fake rows are those whose generator tag is in GENERATORS. Everything else is
    real - but only if there is exactly one such tag. More than one unknown tag
    means the tags are not what this module expects (a renamed generator would
    otherwise be silently counted as real, which would corrupt every cell), so it
    raises instead.

    Args:
        generator: Per-row generator tags.

    Returns:
        Boolean mask, True for real rows.

    Raises:
        ValueError: If the tag vocabulary is not GENERATORS plus exactly one
            real tag, or if no generator tag is present at all.
    """
    tags = set(np.unique(generator).tolist())
    fake_tags = tags & set(GENERATORS)
    other_tags = sorted(tags - set(GENERATORS))
    if not fake_tags:
        raise ValueError(f"no known generator tags in cache; found {sorted(tags)}")
    if len(other_tags) != 1:
        raise ValueError(
            f"expected exactly one real tag besides {GENERATORS}, found {other_tags}"
        )
    return generator == other_tags[0]


def _check_label_agrees(label: np.ndarray, is_real: np.ndarray) -> None:
    """
    Verify the stored `label` column is a function of real-vs-fake.

    Cheap integrity check on metadata only; catches a swapped or misread column
    before 28 runs are committed to it.

    Args:
        label: Per-row label strings.
        is_real: Boolean mask from func _real_mask.

    Raises:
        ValueError: If either class carries more than one label value, or both
            classes carry the same value.
    """
    real_labels = set(np.unique(label[is_real]).tolist())
    fake_labels = set(np.unique(label[~is_real]).tolist())
    if len(real_labels) != 1 or len(fake_labels) != 1:
        raise ValueError(f"label is not constant per class: real={real_labels}, fake={fake_labels}")
    if real_labels == fake_labels:
        raise ValueError(f"real and fake share label {real_labels}; the column is uninformative")


def _gather(cache_dir: Path, strategy: Strategy, split: str, meta: dict, rows: np.ndarray) -> np.ndarray:
    """
    Read selected rows out of a split's cached shards without loading all of them.

    Shards are memory-mapped and only the requested rows are materialised - an
    arm is 4,000 of 28,000 rows, so this reads ~200 MB instead of 1.4 GB.

    Args:
        cache_dir: Cache root.
        strategy: One of STRATEGIES.
        split: Split name.
        meta: Output of func load_meta for that split.
        rows: Global row indices to gather, in the order wanted.

    Returns:
        uint8 (len(rows), 128, 128, 3).
    """
    paths = shard_paths(cache_dir, strategy, split)
    shard_of = meta["shard"]
    starts = np.zeros(len(paths), np.int64)
    counts = np.bincount(shard_of, minlength=len(paths))
    starts[1:] = np.cumsum(counts)[:-1]

    out: np.ndarray | None = None
    for shard_index, path in enumerate(paths):
        wanted = np.flatnonzero(shard_of[rows] == shard_index)
        if wanted.size == 0:
            continue
        memmap = np.load(path, mmap_mode="r")
        local_rows = rows[wanted] - starts[shard_index]
        block = np.asarray(memmap[local_rows])
        if out is None:
            out = np.empty((len(rows), *block.shape[1:]), np.uint8)
        out[wanted] = block
        del memmap
    if out is None:
        raise ValueError("no rows selected")
    return out


def _choose(pool: np.ndarray, n: int, seed: int, what: str) -> np.ndarray:
    """
    Pick n rows from a pool without replacement, deterministic for a given seed.

    Args:
        pool: Candidate global row indices.
        n: How many to take.
        seed: RNG seed.
        what: Human-readable description, used in the error message.

    Returns:
        Sorted selection of n indices.

    Raises:
        ValueError: If the pool is smaller than n. A silent shortfall would
            unbalance the classes and quietly invalidate accuracy.
    """
    if len(pool) < n:
        raise ValueError(f"{what}: need {n} rows, cache has {len(pool)}")
    picked = np.random.default_rng(seed).choice(pool, size=n, replace=False)
    return np.sort(picked)


def _tag_array(tag: str, n: int) -> np.ndarray:
    """
    Build an n-long array of one repeated tag, at full string width.

    Deliberately not `np.full(n, tag, dtype=np.str_)`: that allocates `<U1` from
    the dtype before looking at the fill value and silently truncates every tag
    to its first character, which would mark the whole eval set as synthetic.

    Args:
        tag: The string to repeat.
        n: Length.

    Returns:
        A unicode array of width len(tag).
    """
    return np.array([tag] * n, dtype=np.str_)


def load_arm_numpy(
    cache_dir: str | Path,
    strategy: Strategy,
    source: str,
    seed: int = 0,
    train_per_class: int = TRAIN_PER_CLASS,
    train_split: str = "train",
    test_split: str = "validation",
) -> tuple[np.ndarray, ...]:
    """
    Build one arm as numpy arrays - the torch-free core of func load_arm.

    Train: `train_per_class` fakes from `source` plus the same number of real
    images, both pools drawn with the run `seed`, so the training sample moves
    between seeds and the seed-to-seed spread reflects sampling variability and
    not weight init alone. Note this bites only where a pool is larger than the
    request: at the default `train_per_class` the train split holds exactly 2,000
    fakes per generator, so the fake half is exhaustive and identical across seeds,
    and in practice only the real half (2,000 of 14,000) moves. VAL_FRACTION is
    then split off, stratified by class and shuffled with the same seed, for
    checkpoint selection only. Selection is keyed by seed and generator and never
    by strategy, so within one seed all four strategies train on the same rows.

    Test: TEST_PER_GENERATOR fakes for each of the seven generators, then a fixed
    TEST_REAL_N real block shared by all seven cells, both drawn with
    REAL_POOL_SEED so every cell of every matrix sees byte-identical images. The
    test set never influences model selection - that is what the validation
    carve-out above is for - and never moves with the seed.

    Args:
        cache_dir: Cache root.
        strategy: One of STRATEGIES.
        source: Generator the detector is trained on; must be in GENERATORS.
        seed: Run seed; selects which training images are drawn and controls the
            train/val split and its shuffle. The test set is independent of it.
        train_per_class: Training images per class - halve it if the calibration
            run comes in slow.
        train_split: Dataset split to train from.
        test_split: Dataset split the reported numbers come from. Its default is
            the string `"validation"` because that is what this dataset calls its
            held-out split; it is our test set, not our validation set.

    Returns:
        (x_train, y_train, x_val, y_val, x_test, y_test, gen_ids).

    Raises:
        ValueError: If `source` is not a known generator, or the cache's tags or
            labels fail their consistency checks.
    """
    if source not in GENERATORS:
        raise ValueError(f"unknown source {source!r}; expected one of {GENERATORS}")
    root = Path(cache_dir)

    train_meta = load_meta(root, train_split)
    train_real = _real_mask(train_meta["generator"])
    _check_label_agrees(train_meta["label"], train_real)

    # Keyed by seed and generator, never by strategy: within one seed all four
    # strategies draw the same rows, so the strategy comparison stays paired.
    fake_rows = _choose(
        np.flatnonzero(train_meta["generator"] == source),
        train_per_class,
        seed,
        f"train fakes for {source}",
    )
    real_rows = _choose(np.flatnonzero(train_real), train_per_class, seed, "train real pool")

    x_fit = _gather(root, strategy, train_split, train_meta, np.concatenate([fake_rows, real_rows]))
    y_fit = np.concatenate(
        [np.ones(len(fake_rows), np.float32), np.zeros(len(real_rows), np.float32)]
    )

    rng = np.random.default_rng(seed)
    keep: list[np.ndarray] = []
    held: list[np.ndarray] = []
    for value in (1.0, 0.0):
        idx = np.flatnonzero(y_fit == value)
        rng.shuffle(idx)
        cut = int(round(len(idx) * VAL_FRACTION))
        held.append(idx[:cut])
        keep.append(idx[cut:])
    keep_idx = np.concatenate(keep)
    val_idx = np.concatenate(held)
    rng.shuffle(keep_idx)
    rng.shuffle(val_idx)

    test_meta = load_meta(root, test_split)
    test_real = _real_mask(test_meta["generator"])
    _check_label_agrees(test_meta["label"], test_real)

    test_rows: list[np.ndarray] = []
    gen_ids: list[np.ndarray] = []
    for generator in GENERATORS:
        rows = _choose(
            np.flatnonzero(test_meta["generator"] == generator),
            TEST_PER_GENERATOR,
            REAL_POOL_SEED,
            f"test fakes for {generator}",
        )
        test_rows.append(rows)
        gen_ids.append(_tag_array(generator, len(rows)))
    real_test = _choose(np.flatnonzero(test_real), TEST_REAL_N, REAL_POOL_SEED, "test real block")
    test_rows.append(real_test)
    gen_ids.append(_tag_array(REAL_TAG, len(real_test)))

    test_index = np.concatenate(test_rows)
    x_test = _gather(root, strategy, test_split, test_meta, test_index)
    ids = np.concatenate(gen_ids)
    y_test = (ids != REAL_TAG).astype(np.float32)

    return (
        x_fit[keep_idx],
        y_fit[keep_idx],
        x_fit[val_idx],
        y_fit[val_idx],
        x_test,
        y_test,
        ids,
    )


def load_arm(
    cache_dir: str | Path,
    strategy: Strategy,
    source: str,
    seed: int = 0,
    device: str = "cpu",
    train_per_class: int = TRAIN_PER_CLASS,
) -> Arm:
    """
    Load one (strategy, source, seed) arm as torch tensors.

    Images stay uint8 and NHWC; `train.py` moves them to the GPU once and does the
    permute/normalise per batch. The whole arm is ~390 MB, small enough to live in
    T4 VRAM, which is why there is no DataLoader anywhere in this project.

    torch is imported inside this function on purpose: every other function in
    this module runs on the laptop with numpy + Pillow only.

    Args:
        cache_dir: Cache root.
        strategy: One of STRATEGIES.
        source: Generator to train on.
        seed: Run seed; selects the training sample and the train/val split.
        device: Destination device for the tensors.
        train_per_class: Training images per class.

    Returns:
        An Arm.
    """
    import torch

    x_train, y_train, x_val, y_val, x_test, y_test, gen_ids = load_arm_numpy(
        cache_dir, strategy, source, seed, train_per_class=train_per_class
    )
    def to_device(array: np.ndarray) -> Tensor:
        """
        Move one array to the target device without copying twice.

        Args:
            array: Source array.

        Returns:
            The tensor on `device`.
        """
        return torch.from_numpy(np.ascontiguousarray(array)).to(device)

    return Arm(
        x_train=to_device(x_train),
        y_train=to_device(y_train),
        x_val=to_device(x_val),
        y_val=to_device(y_val),
        x_test=to_device(x_test),
        y_test=to_device(y_test),
        gen_ids=gen_ids,
    )


def _main() -> None:
    """
    CLI entry point: build one split's cache.

    Example:
        python -m src.data --split validation --out-dir /content/drive/MyDrive/university/deep_learning/cache
    """
    import argparse

    parser = argparse.ArgumentParser(description="Build the Tiny-GenImage equalisation cache.")
    parser.add_argument("--split", default="validation", help="validation first, then train")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-shards", type=int, default=None, help="pilot: stop after N shards")
    parser.add_argument(
        "--no-shards",
        action="store_true",
        help="delete each source .parquet after use; the cache alone trains and reports",
    )
    parser.add_argument(
        "--keep-native",
        action="store_true",
        help="also archive the originals as a per-shard blob readable with Pillow alone",
    )
    args = parser.parse_args()

    build_cache(
        args.split,
        args.out_dir,
        repo_id=args.repo_id,
        seed=args.seed,
        max_shards=args.max_shards,
        keep_native=args.keep_native,
        keep_shards=not args.no_shards,
    )


if __name__ == "__main__":
    _main()
