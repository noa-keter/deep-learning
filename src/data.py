"""
Data layer for the resolution-bias study: the four equalisation transforms and
the shard-by-shard cache builder.

This is the only module that touches image bytes. It is importable on a laptop
with **numpy + Pillow only**: `pyarrow` and `huggingface_hub` are imported lazily
inside the functions that need them, so `equalize` can be unit-tested without a
GPU stack.

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
and the offsets recorded in the metadata seek into it. Enabling both stores the
originals twice.
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
from typing import Final, Literal

__all__ = [
    "STRATEGIES",
    "Strategy",
    "build_cache",
    "equalize",
    "row_rng",
]

Strategy = Literal["centre_crop", "random_crop", "rescale", "pad"]
STRATEGIES: Final[tuple[Strategy, ...]] = ("centre_crop", "random_crop", "rescale", "pad")

#: Largest resampling-free common size the dataset admits: BigGAN is 128x128 native.
CACHE_SIZE_PX: Final[int] = 128

DEFAULT_REPO_ID: Final[str] = "TheKernel01/Tiny-GenImage"

DECODE_BATCH_ROWS: Final[int] = 64
IMAGE_COLUMN: Final[str] = "image"
LABEL_COLUMN: Final[str] = "label"
GENERATOR_COLUMN: Final[str] = "generator"

_BILINEAR: Final = Image.Resampling.BILINEAR

#: Set at module import, which only helps if this module is imported before
#: huggingface_hub - see the check in `build_cache`.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


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
      works on a laptop with no pyarrow. Read it with :func:`open_native`.

    Enabling both stores the originals twice. Prefer one.

    Peak *local* disk stays one shard either way; both options grow the output
    directory by the full source size.

    Args:
        split: Split name as it appears in the parquet filenames, e.g.
            "validation" or "train". Build validation first - it is where every
            reported cell comes from.
        out_dir: Cache root, e.g. "/content/drive/MyDrive/aidet/cache".
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
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    # Lazy: the laptop imports this module with numpy + Pillow only.
    import pyarrow.parquet as pq
    import huggingface_hub as hub
    from importlib.util import find_spec
    from huggingface_hub import HfApi, hf_hub_download

    # Roughly 20 min versus 3 h, so report which is about to happen. A notebook that
    # imported huggingface_hub first latched the old value; only a restart clears it.
    enabled = getattr(getattr(hub, "constants", None), "HF_HUB_ENABLE_HF_TRANSFER", None)
    if enabled is False or (enabled and not find_spec("hf_transfer")):
        print("[build_cache] WARNING: hf_transfer inactive, download will be slow", flush=True)

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


def _main() -> None:
    """
    CLI entry point: build one split's cache.

    Example:
        python -m src.data --split validation --out-dir /content/drive/MyDrive/aidet/cache
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