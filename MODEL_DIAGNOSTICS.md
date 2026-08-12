# How to find out what limits the model

Diagonal accuracy is 0.908. Before changing anything, find out **why** it isn't higher. Guessing
wrong wastes hours: the fix for "not enough capacity" makes "too much capacity" worse.

Three experiments, ~1 GPU-hour total. Each one produces a sentence for the report whichever way it
comes out.

---

## Step 1 — Train accuracy (30 min) — this is the fork

Every run saved `model.pt`. Load each checkpoint, score it on **its own training set**, compare to
the val accuracy already recorded.

Run in Colab (torch isn't installed locally). You have 28 checkpoints locally — `center_crop` and
`rescale`, which is exactly the pair the headline rests on. Noa has the other 28; `.pt` is
gitignored so they were never pushed.

**Implemented:** `src/trainval.py`, driven by `notebooks/02_train_val_gap.ipynb`.

```bash
python -m src.trainval --cache-dir CACHE --ckpt-root "$BACKUP/runs" --owner ido
```

Every path is a parameter, so a second grid is scored by pointing `--results-dir` and
`--ckpt-root` at it and passing `--tag d4` — which is how the before/after on this gap gets
measured without overwriting the numbers it is being compared against.

Also add train accuracy to `train.py` permanently — it belongs in the report's hyperparameters
section either way.

| Result | Meaning | What helps |
|---|---|---|
| Train ≈ 100%, val ≈ 91% | **Overfitting** (variance-limited) | More data, more augmentation. **A bigger model makes it worse.** |
| Train ≈ 92%, val ≈ 91% | **Underfitting** (bias-limited) | More capacity — wider, longer |

Expected: overfitting. 1.17M parameters on 3,600 images with flip-only augmentation, and
`best_epoch` averages 33/40 so it isn't short of training. But measure it — don't assume.

---

## Step 2 — Learning curve (15 min) — does more data help?

Fix one strategy and one source. Train with **500 / 1000 / 2000** fakes per class. Plot val accuracy
against training-set size.

3 runs × ~2 min. Use `center_crop` (the clean arm) and one mid-difficulty source.

| Curve at 2000 | Meaning |
|---|---|
| Still rising steeply | Data-limited. More data would help — but you're capped (see below), so this is a stated limitation, not a fix. |
| Flat | Data-saturated. More data buys nothing, and you can say so with a figure. |

**The cap:** Tiny-GenImage has 2,500 images per generator, 500 held out for val → 2,000 max, and
balanced training caps each run at 4,000 images. You cannot buy more without changing dataset.

---

## Step 3 — Width sweep (30 min) — does more capacity help?

Same cell, width multiplier **×0.5, ×1.0, ×1.5** (widths 16/32/64/128 · 32/64/128/256 · 48/96/192/384).
×1.0 already exists, so this is 2 new runs.

| Result | Meaning |
|---|---|
| ×1.5 ≈ ×1.0 | Capacity is **not** the constraint. This is the figure that proves 0.908 isn't undertraining. |
| ×1.5 clearly better | Capacity **is** the constraint — widen, and re-run the grid (read the rules below first). |

**Widen, don't deepen.** A fifth block takes the spatial map to 4×4 and erodes the global-average-pool
support, and it contradicts the "nothing downsamples before block 1" rationale that the report
depends on.

**Watch VRAM.** Peak is 9,853 MB of a ~15 GB T4 — 65% used, because block 1 runs two stride-1 convs
at full 128². ×2.0 width will likely OOM at batch 128, and lowering the batch size changes the
effective LR schedule and breaks comparability with the 56 runs you already have.

---

## If the answer is "overfitting" — the one lever available

**Rotations by 90° do not resample.** k×90° is an exact permutation of the pixel grid — a transpose
and a reversal, byte-exact, no interpolation. With vertical flip that's the full D4 group: **8×
augmentation at zero methodological cost**, and it's currently unused.

**Run Gate 0 first — it is free.** `src/trainval.py` says the model is variance-limited; it does not
say D4 can help, because a CNN is equivariant to translation and to nothing else. Scoring the
existing checkpoints on rotated test sets answers that with **zero training**:

```bash
python -m src.rotation --cache-dir CACHE --ckpt-root "$BACKUP/runs" --owner ido
```

Flat under rotation → the model is already orientation-invariant and D4 has nothing to give. Collapse
toward 0.5 → real headroom. And if **tnr** falls harder than **tpr**, the real class carries the
orientation cue, which is the one mechanism by which D4 could move the *cross-generator* number
rather than only the diagonal. Notebook: `notebooks/03_rotation_sensitivity.ipynb`.

~~Caveat: under 90° rotation a padded portrait's black bars move from top/bottom to left/right~~ —
**corrected 2026-08-12, measured.** A padded *portrait* starts with **left/right** bars, and the
padded black fraction is **0.250, a permutation invariant**; the `AdaptiveAvgPool2d(1)` head cannot
read bar position anyway. **D4 does not touch the `pad` contamination.** Run D4 as an **ablation**
rather than a silent replacement for a different reason: rho at n = 7 is fragile, the reported
matrices must stay flip-only, and the 56 completed runs are the control arm.

Also measured that day: `rot90` **commutes** with `center_crop`, `rescale` and `pad` (byte-exact for
the square generators under `center_crop`; elsewhere a 1-px window-parity shift and ±1 LSB), so D4
needs no cache rebuild. And any *spectral* test of "does the fingerprint survive rotation?" is
uninformative by construction — the radially averaged spectrum is invariant under k×90° to 1e-12.
In `random_crop`, rotate the cached crop; re-drawing from the native does **not** commute and would
convert the arm to per-epoch cropping.

Rejected, with reasons worth a line in the report:
- **Cutout** — stamps black rectangles. Hard edges are broadband high-frequency artifacts, and in a
  study whose `pad` arm is contaminated *by a black border*, teaching the model to ignore black-region
  boundaries is actively destructive.
- **Mixup** — doesn't resample, but averaging two images halves the variance of the high-frequency
  residual that *is* the generator fingerprint, and it invents images belonging to no generator.
- **Rotation by arbitrary angles, scale jitter, JPEG augmentation** — all resample or re-encode, i.e.
  they inject the exact artifact under study.

---

## Rules for any change

1. **Applies identically to all four arms**, or the comparison breaks.
2. **Input stays 128×128.** BigGAN's native size is 128²; anything larger needs upsampling, which
   injects the artifact under study.
3. **From scratch stays.** The real class is ImageNet, so a pretrained backbone is contaminated.
4. **A re-run is an addition, never a replacement.** Write this down *before* starting one. Once two
   grids exist, reporting only the nicer one is an integrity problem, and reporting both without
   having committed in advance looks like you picked.
5. **Preview before committing hours.** 2–3 sources × 4 strategies × 1 seed ≈ 25 min tells you
   whether cells move by more than ~2× the ±1.6 pp binomial SE.

## What is already safe

- **The `pad` finding.** It's a ceiling effect — a pure border detector maxes at 0.977 and pad is at
  0.9776. Capacity can't push past a ceiling using the same cue.
- **Not undertrained.** `best_epoch` averages 33/40 and only 7 of 56 runs peaked at ≥38.

## What is fragile

- **The ρ = 0.96 vs ρ = 0.00 contrast**, which is the headline. Both are over n = 7, so ρ = 0.00 with
  p = 1.0 is *absence of evidence*, not evidence of absence. A different model could land it at ±0.4.
  The qualitative story likely survives; the quotable number may not.
