"""
Regenerate the two presentation-only figures for the 15-minute project talk.

The report's committed figures are sized for a portrait A4 page and do not fit a
16:9 slide, so this script re-renders the transfer matrices in a wide 1x4 layout
and draws the size-equalization diagram that the report describes in prose but
never plots.

Both outputs are derived from artifacts already committed under results/figures/,
so a grader can reproduce them from a clean clone:

    python docs/slide_figures.py

Nothing here touches torch, a GPU or the image cache.
"""

from __future__ import annotations

import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Rectangle

mpl.use("Agg")

__all__ = ["build_strategy_diagram", "build_wide_transfer_matrices", "main"]

# ── Constants ─────────────────────────────────────────────────────────────────

TARGET_PX = 128
"""Network input side, in pixels. Every strategy maps a native image to this."""

GENERATED_NATIVE_PX = 512
"""Native side of an SD 1.5 / Wukong image, used as the square exemplar."""

REAL_NATIVE_WH_PX = (500, 375)
"""Native (width, height) of a typical real ImageNet photograph in this dataset."""

RANDOM_CROP_SEED = 7
"""Fixes the random_crop offset so the diagram is byte-reproducible."""

DISC_CENTRE_FRACTION = 0.42
DISC_RADIUS_FRACTION = 0.13
"""Disc placement in the synthetic scene, as fractions of the frame and short side."""

COLOR_SCALE_MIN_PCT = 45.0
COLOR_SCALE_MAX_PCT = 100.0
"""Shared color limits for the transfer panels, matching the report's figure."""

ARM_ORDER = ("center_crop", "random_crop", "rescale", "pad")
"""Left-to-right panel order: the two non-resampling arms first, then the rest."""

INK = "#141A2E"
ACCENT = "#D9484E"
MUTED = "#6B7691"

FIGURES_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"
OUTPUT_DIR = Path(__file__).resolve().parent / "slide_assets"


# ── Synthetic exemplars for the strategy diagram ──────────────────────────────


def _make_scene(width_px: int, height_px: int) -> np.ndarray:
    """
    Draw a synthetic RGB scene with structure at several spatial scales.

    The grid makes resampling visible (cell size changes), the disc makes aspect
    distortion visible (it becomes an ellipse), and the fine checker stands in for
    the high-frequency content that carries the generative fingerprint.

    Args:
        width_px: Scene width in pixels.
        height_px: Scene height in pixels.

    Returns:
        A (height_px, width_px, 3) uint8 array.
    """
    yy, xx = np.mgrid[0:height_px, 0:width_px]
    short_side = min(width_px, height_px)

    coarse_cell = short_side // 8
    fine_cell = max(short_side // 48, 1)
    coarse = ((xx // coarse_cell) + (yy // coarse_cell)) % 2
    fine = ((xx // fine_cell) + (yy // fine_cell)) % 2

    scene = np.zeros((height_px, width_px, 3), dtype=np.float64)
    scene[..., 0] = 0.22 + 0.30 * coarse
    scene[..., 1] = 0.34 + 0.32 * coarse
    scene[..., 2] = 0.52 + 0.30 * coarse
    scene += 0.10 * (fine[..., None] - 0.5)

    # Off-centre and small, so the 128px centre window shows grid plus part of the
    # disc rather than a flat patch of its interior.
    centre_x = DISC_CENTRE_FRACTION * width_px
    centre_y = DISC_CENTRE_FRACTION * height_px
    radius = DISC_RADIUS_FRACTION * short_side
    distance = np.sqrt((xx - centre_x) ** 2 + (yy - centre_y) ** 2)
    scene[distance <= radius] = np.array([0.94, 0.66, 0.25])
    scene[np.abs(distance - radius) < (0.016 * short_side)] = np.array([0.08, 0.10, 0.18])

    return (np.clip(scene, 0.0, 1.0) * 255).astype(np.uint8)


def _resize_bilinear(image: np.ndarray, out_h_px: int, out_w_px: int) -> np.ndarray:
    """
    Resize an RGB image with bilinear interpolation.

    Implemented with numpy so the diagram has no dependency the rest of the
    laptop-side pipeline does not already have.

    Args:
        image: (H, W, 3) uint8 array.
        out_h_px: Output height in pixels.
        out_w_px: Output width in pixels.

    Returns:
        A (out_h_px, out_w_px, 3) uint8 array.
    """
    in_h, in_w = image.shape[:2]
    src_y = (np.arange(out_h_px) + 0.5) * in_h / out_h_px - 0.5
    src_x = (np.arange(out_w_px) + 0.5) * in_w / out_w_px - 0.5
    src_y = np.clip(src_y, 0, in_h - 1)
    src_x = np.clip(src_x, 0, in_w - 1)

    y0 = np.floor(src_y).astype(int)
    x0 = np.floor(src_x).astype(int)
    y1 = np.minimum(y0 + 1, in_h - 1)
    x1 = np.minimum(x0 + 1, in_w - 1)
    wy = (src_y - y0)[:, None, None]
    wx = (src_x - x0)[None, :, None]

    img = image.astype(np.float64)
    top = img[y0][:, x0] * (1 - wx) + img[y0][:, x1] * wx
    bottom = img[y1][:, x0] * (1 - wx) + img[y1][:, x1] * wx
    return np.clip(top * (1 - wy) + bottom * wy, 0, 255).astype(np.uint8)


def _center_crop(image: np.ndarray) -> np.ndarray:
    """
    Take the central TARGET_PX window at native resolution.

    Args:
        image: (H, W, 3) uint8 array, both sides at least TARGET_PX.

    Returns:
        A (TARGET_PX, TARGET_PX, 3) uint8 array.
    """
    h, w = image.shape[:2]
    top = (h - TARGET_PX) // 2
    left = (w - TARGET_PX) // 2
    return image[top : top + TARGET_PX, left : left + TARGET_PX]


def _random_crop(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Take a TARGET_PX window at a random offset, at native resolution.

    Args:
        image: (H, W, 3) uint8 array, both sides at least TARGET_PX.
        rng: Seeded generator, so the diagram reproduces exactly.

    Returns:
        A (TARGET_PX, TARGET_PX, 3) uint8 array.
    """
    h, w = image.shape[:2]
    top = int(rng.integers(0, h - TARGET_PX + 1))
    left = int(rng.integers(0, w - TARGET_PX + 1))
    return image[top : top + TARGET_PX, left : left + TARGET_PX]


def _rescale(image: np.ndarray) -> np.ndarray:
    """
    Resize the whole image to TARGET_PX square, ignoring aspect ratio.

    Args:
        image: (H, W, 3) uint8 array.

    Returns:
        A (TARGET_PX, TARGET_PX, 3) uint8 array.
    """
    return _resize_bilinear(image, TARGET_PX, TARGET_PX)


def _pad(image: np.ndarray) -> np.ndarray:
    """
    Scale the long side to TARGET_PX preserving aspect, then zero-pad to square.

    Args:
        image: (H, W, 3) uint8 array.

    Returns:
        A (TARGET_PX, TARGET_PX, 3) uint8 array.
    """
    h, w = image.shape[:2]
    scale = TARGET_PX / max(h, w)
    new_h = max(int(round(h * scale)), 1)
    new_w = max(int(round(w * scale)), 1)
    resized = _resize_bilinear(image, new_h, new_w)

    canvas = np.zeros((TARGET_PX, TARGET_PX, 3), dtype=np.uint8)
    top = (TARGET_PX - new_h) // 2
    left = (TARGET_PX - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def build_strategy_diagram(out_path: Path) -> Path:
    """
    Draw the four size-equalization strategies applied to a square and a non-square image.

    The figure exists to make one asymmetry visible: every generator emits square
    images, so pad and rescale coincide on them, and only the real class receives
    a black border. That is the mechanism behind the pad contamination finding.

    Args:
        out_path: Destination PNG path.

    Returns:
        The path written.
    """
    rng = np.random.default_rng(RANDOM_CROP_SEED)
    real_w, real_h = REAL_NATIVE_WH_PX

    rows = (
        (
            "Generated  (SD 1.5)",
            f"{GENERATED_NATIVE_PX}x{GENERATED_NATIVE_PX}  square",
            _make_scene(GENERATED_NATIVE_PX, GENERATED_NATIVE_PX),
        ),
        (
            "Real photo  (ImageNet)",
            f"{real_w}x{real_h}  variable aspect",
            _make_scene(real_w, real_h),
        ),
    )

    # 17.5 x 7.4 is chosen so the figure lands at 16:9-friendly proportions: any
    # taller and it cannot fit under a slide title without shrinking the panels.
    fig, axes = plt.subplots(2, 5, figsize=(17.5, 7.4))
    fig.patch.set_facecolor("white")

    for row_idx, (row_name, row_sub, native) in enumerate(rows):
        outputs = (
            ("native", native),
            ("center_crop", _center_crop(native)),
            ("random_crop", _random_crop(native, rng)),
            ("rescale", _rescale(native)),
            ("pad", _pad(native)),
        )
        for col_idx, (label, img) in enumerate(outputs):
            ax = axes[row_idx, col_idx]
            ax.imshow(img, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])

            is_native = col_idx == 0
            edge = MUTED if is_native else INK
            for spine in ax.spines.values():
                spine.set_edgecolor(edge)
                spine.set_linewidth(2.4 if not is_native else 1.6)

            if row_idx == 0:
                ax.set_title(
                    label,
                    fontsize=19,
                    fontweight="bold",
                    color=INK if not is_native else MUTED,
                    pad=14,
                )
            size_note = row_sub if is_native else f"{TARGET_PX}x{TARGET_PX}"
            ax.set_xlabel(size_note, fontsize=13, color=MUTED, labelpad=7)

        axes[row_idx, 0].set_ylabel(
            row_name, fontsize=16, fontweight="bold", color=INK, labelpad=14
        )

    # The two callouts are the whole point of the figure.
    axes[0, 4].add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            transform=axes[0, 4].transAxes,
            fill=False,
            edgecolor=ACCENT,
            linewidth=5,
            zorder=5,
        )
    )
    axes[0, 4].text(
        0.5,
        -0.32,
        "identical to rescale\n(every generator is square)",
        transform=axes[0, 4].transAxes,
        ha="center",
        va="top",
        fontsize=14,
        color=ACCENT,
        fontweight="bold",
    )
    axes[1, 4].add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            transform=axes[1, 4].transAxes,
            fill=False,
            edgecolor=ACCENT,
            linewidth=5,
            zorder=5,
        )
    )
    axes[1, 4].text(
        0.5,
        -0.32,
        "black bars, on real photos only\nborder  <=>  real",
        transform=axes[1, 4].transAxes,
        ha="center",
        va="top",
        fontsize=14,
        color=ACCENT,
        fontweight="bold",
    )

    fig.subplots_adjust(
        left=0.055, right=0.99, top=0.915, bottom=0.17, wspace=0.10, hspace=0.55
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    return out_path


# ── Wide transfer matrices ────────────────────────────────────────────────────


def build_wide_transfer_matrices(matrices_json: Path, out_path: Path) -> Path:
    """
    Re-render the four CNN transfer matrices as a single wide 1x4 strip for a 16:9 slide.

    Reads the same committed JSON the report's figure is built from, so the cells
    are the identical seed-averaged numbers.

    Args:
        matrices_json: Path to results/figures/transfer_matrices.json.
        out_path: Destination PNG path.

    Returns:
        The path written.

    Raises:
        KeyError: If an expected arm is absent from the JSON.
    """
    with open(matrices_json, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    generators = payload["generators"]
    by_label = {arm["label"]: arm for arm in payload["arms"]}
    missing = [name for name in ARM_ORDER if name not in by_label]
    if missing:
        raise KeyError(f"transfer_matrices.json is missing arms: {missing}")

    n = len(generators)
    off_mask = ~np.eye(n, dtype=bool)

    fig, axes = plt.subplots(1, len(ARM_ORDER), figsize=(19.5, 5.9))
    fig.patch.set_facecolor("white")
    mesh = None

    for ax, name in zip(axes, ARM_ORDER):
        acc = np.asarray(by_label[name]["accuracy"], dtype=float) * 100.0
        diag = float(np.mean(np.diag(acc)))
        off = float(np.mean(acc[off_mask]))

        mesh = ax.imshow(
            acc,
            cmap="Blues",
            vmin=COLOR_SCALE_MIN_PCT,
            vmax=COLOR_SCALE_MAX_PCT,
            interpolation="nearest",
        )
        for i in range(n):
            for j in range(n):
                ax.text(
                    j,
                    i,
                    f"{acc[i, j]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=12.5,
                    color="white" if acc[i, j] > 78 else INK,
                )

        highlight = ACCENT if name in ("rescale", "pad") else INK
        # Three short lines rather than one long one: at this panel pitch a single
        # stats line is wider than the panel and collides with its neighbour.
        ax.set_title(
            f"{name}\ndiag {diag:.1f}    off {off:.1f}\ngap {diag - off:+.1f} pp",
            fontsize=15,
            fontweight="bold",
            color=highlight,
            pad=10,
            linespacing=1.5,
        )
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(generators, rotation=45, ha="right", fontsize=11.5, color=INK)
        ax.set_yticklabels(generators if ax is axes[0] else [], fontsize=11.5, color=INK)
        ax.set_xlabel("evaluated on", fontsize=12.5, color=MUTED, labelpad=6)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)

    axes[0].set_ylabel("trained on", fontsize=13.5, fontweight="bold", color=INK, labelpad=8)

    fig.subplots_adjust(left=0.062, right=0.935, top=0.755, bottom=0.20, wspace=0.15)
    cbar_ax = fig.add_axes([0.947, 0.20, 0.011, 0.555])
    cbar = fig.colorbar(mesh, cax=cbar_ax)
    cbar.set_label("cell accuracy (%)", fontsize=12, color=MUTED)
    cbar.ax.tick_params(labelsize=11, colors=MUTED, length=0)
    cbar.outline.set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    return out_path


def main() -> None:
    """
    Build both slide figures into docs/slide_assets/ and report their paths.
    """
    diagram = build_strategy_diagram(OUTPUT_DIR / "strategies_diagram.png")
    matrices = build_wide_transfer_matrices(
        FIGURES_DIR / "transfer_matrices.json",
        OUTPUT_DIR / "transfer_matrices_wide.png",
    )
    for path in (diagram, matrices):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
