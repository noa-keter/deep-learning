# Integration notes — where my code meets Ido's

Ido and I write in parallel and our files only meet once, late. The goal is that when they do, the
integration costs **no code changes** — so every mismatch below is resolved now, in advance, by one
of two moves:

- **Defend** — make my code accept either shape. Costs a few lines, needs nobody's cooperation, and
  is the default answer whenever it's possible.
- **Agree** — a one-line message to Ido. Reserved for the things tolerance genuinely cannot fix:
  channel order, the sub-128 policy, input scaling.

Defend beats agree wherever both are available, because a defence can't be forgotten by someone else.

The ordering below is by **how the failure announces itself**, because that is what decides how much
care each one deserves. A crash is cheap: it stops the run and points at the line. A wrong number is
expensive: it produces a full, plausible-looking 7×7 matrix that goes into the report.

---

## Silent failures — wrong numbers, no error

### S1. Shard ordering — the dangerous one

`load_meta` collects shards with `sorted(...)`, which is lexicographic. If Ido writes `val-0.npz …
val-10.npz`, that sorts `val-10` **before** `val-2`. The metadata then concatenates in a different
order than the `.npy` pixel shards, so every index addresses the wrong image — labels, sizes and
pixels all silently offset from each other. Nothing raises. The baseline still prints a full matrix.

**Defend:** sort by the integer in the filename rather than the string, and assert the metadata row
count equals the `.npy` row count for the same split. Both are cheap and neither depends on his
naming.

### S2. `train.py` scaling its own way

`normalize()` lives in `model.py` so that training and attribution share one input scale. If Ido
writes `x / 255.0` in the training loop instead of importing it, the network trains on `[0, 1]` while
`analyze.py` attributes on `[-1, 1]`. Gradients then get taken at an operating point the model never
saw, and border mass still returns a perfectly reasonable-looking number.

**Agree**, then **defend**: ask him to import it; also have `analyze.py` assert its input range
before the backward pass, so a mismatch shows up as a failed assertion rather than a plausible figure.

### S3. `h` and `w` swapped

`PIL.Image.size` is `(width, height)`; `np.array(img).shape` is `(height, width, 3)`. Mixing the two
transposes every pair. This does **not** affect either baseline rule — the square rule tests `h == w`,
and the size-lookup rule compares pairs against pairs from the same source, so a consistent swap
cancels. It does corrupt the native-size table in the Data section and any aspect-ratio claim.

**Defend:** check a known asymmetric case against the dataset viewer — a real image listed as 500×376
must come back as `h=376, w=500` if `h` really is height.

### S4. Channel order

RGB versus BGR is invisible to shapes and to the baseline, and only mildly perturbs attribution — but
it shifts the luminance-weighted grayscale conversion the spectra depend on. The pilot's
"render three images per generator per strategy and look at them" is the only thing that catches it.

**Agree** — it's a property of `data.py` and cannot be recovered downstream.

### S5. Different 500 reals — settled, listed so it isn't re-opened

If Ido's eval set differs from mine, the square rule's TNR moves by about 1 pp, roughly 0.5 pp on
balanced accuracy, against a ±1.6 pp per-cell noise floor and a baseline-vs-CNN gap of 30+ points.
Immaterial. `real_selection_hash()` exists so we can compare eight characters and footnote it if they
differ. **No action.**

---

## Loud failures — a crash, at a named line

These are listed so I recognise them in one second instead of ten minutes. All are cheap.

| # | Mismatch | How it surfaces | Move |
|---|---|---|---|
| L1 | `generator` arrives as ClassLabel ints 0–8, not strings | assert at `baseline.py` generator check | **Defend** — decode via the nine-name list |
| L2 | Split written as `validation-*.npz`, not `val-*` | `FileNotFoundError` in `load_meta` | **Defend** — accept both prefixes |
| L3 | Metadata keys named `height`/`width`/`class` | `KeyError` in `load_meta` | Defend with an alias map |
| L4 | `centre_crop` spelled `center_crop` | `KeyError` on the strategy | Agree — one word, and it's in `PLAN.md` already |
| L5 | `.npy` stored as `(N,3,128,128)` CHW | `RuntimeError`: Conv2d expects 3 channels, gets 128 | Defend — detect the axis |
| L6 | Checkpoint is a pickled model, or has a `module.` prefix | `load_state_dict` KeyError | Defend in `analyze.py`, which I own |
| L7 | `metrics.json` keys differ from the baseline's | `KeyError` in `analyze.py` | Defend — three-line adapter |

**The ClassLabel decode has a trap worth stating separately.** The canonical order is

```
0 Real · 1 ADM · 2 BigGAN · 3 GLIDE · 4 Midjourney · 5 SD14 · 6 SD15 · 7 VQDM · 8 Wukong
```

**Index 5 is SD14** — the class with zero rows. Decoding through my seven-name `GENERATORS` list
would relabel SD15 as SD14, VQDM as SD15, and shift everything after. The decode must use the full
nine-name list, and the result of that mistake is a matrix with the right shape and the wrong row
labels, which is a silent failure wearing a loud failure's clothes.

---

## Verified against the live dataset — do not re-check

From the HF dataset server, `TheKernel01/Tiny-GenImage`:

| Fact | Value |
|---|---|
| Splits | `train` 28,000 · `validation` 7,000 · total 35,000 |
| Val composition | Real 3,500 · 500 each for ADM, BigGAN, GLIDE, Midjourney, SD15, VQDM, Wukong |
| SD14 | **0 rows** — seven generators, never eight |
| `label` | ClassLabel `["real", "fake"]` → 0 / 1 |
| `generator` | ClassLabel, nine names, `Real` is class 0 — **no real image carries a generator's name** |
| Native sizes | ADM/GLIDE/VQDM 256² · BigGAN 128² · SD15/Wukong 512² · Midjourney 1024² |
| Real dimensions | 62 to 3072, median 500 |

The val composition is a complete-population count, not a sample, so `len(fake_pool) == 500` in
`eval_select.py` is exactly right.

**One number not to quote:** the statistics endpoint's *train* scan covered 22,000 of the 28,000
rows, so its per-generator train counts are partial. The true figures follow from the split sizes —
14,000 real and 2,000 per generator.

---

## Paste to Ido

> A few conventions so our files join without either of us rewriting anything at the end.
>
> 1. **Zero-pad the shard numbers** — `val-00.npz`, not `val-0.npz`. Otherwise shard 10 sorts before
>    shard 2 and the metadata stops lining up with the pixels, silently.
> 2. **Metadata keys** `h`, `w`, `label`, `generator`, with `h` = height. `generator` decoded to the
>    string names — it comes out of HF as a ClassLabel int, and index 5 is SD14 with zero rows, so a
>    naive decode shifts every name after it.
> 3. **Split prefix** — `val` or `validation`? Either is fine, just tell me which.
> 4. **Strategy directory names** exactly `centre_crop`, `random_crop`, `rescale`, `pad`.
> 5. **`.npy` shape** `(N, 128, 128, 3)` uint8, **RGB**.
> 6. **Import `normalize` from `model.py`** in the training loop rather than scaling inline — it
>    keeps training and my attribution on the same input scale, which is not recoverable afterwards.
> 7. **Real images smaller than 128 px exist** (val runs down to 62). They can't be cropped to 128²
>    without upscaling, which would resample them — and not resampling is the entire reason the crop
>    arms are in the study. I suggest dropping reals with `min(h,w) < 128` from **all four** arms so
>    the arms keep a shared evaluation set. Needs deciding in the pilot.
> 8. **Checkpoints** as `state_dict` at `results/runs/{strategy}/{source}/seed{n}/best.pt`.
> 9. `select_eval_indices` already exists in `src/eval_select.py` — import it rather than writing a
>    second one.

Items 6 and 7 are the two that can't be fixed after the fact. The rest I can absorb on my side if
they come back different.
