# Plan — Resolution Bias and Cross-Generator Detection of AI-Generated Images

Train a small CNN from scratch to tell real photos from AI-generated ones, and measure how the
answer changes depending on how you equalize image size. Four strategies × 7 source generators,
evaluated on all 7 → four 7×7 transfer matrices.

**Target: Tue 11 Aug 2026.** Self-imposed; there is no course deadline. The project is 80 % of the
grade, so **if something has to give, the date gives — never a requirement.**

Full detail (code snippets, report structure): `OneDrive\...\חומרים לnotebookLM\PROJECT_PLAN.md`.

---

## Who owns what

Split by file, so neither person waits on the other. Both people write PyTorch, and both touch the
experiment and the analysis — the spec requires each member to answer questions on the whole project.

| | **Ido** | **Noa** |
|---|---|---|
| **Code** | `data.py` (cache + 4 transforms) · `train.py` | `model.py` (CompactCNN) · `baseline.py` |
| **Runs** | `center_crop`, `rescale` — 14/seed | `pad`, `random_crop` — 14/seed |
| **Figures** | Transfer matrices · the ranking result | Attribution · spectra |
| **Report** | Motivation · Models & hyperparameters · Results · repo link | Related work · Data · Analysis · Discussion · References |
| **Also** | README + reproducibility | Contribution table |

Roughly even in hours. Before writing the report, each reads the other's two files — 20 minutes, and
you both need it for the presentation Q&A.

---

## Requirements check

Against `הנחיות לפרויקט בקורס למידה עמוקה 2026ב`. All satisfied by the plan as written.

| Requirement | Delivered by |
|---|---|
| PyTorch | `CompactCNN`, trained from scratch |
| Substantive training | 56 training runs |
| Clear baseline | Dimension rule, reported per 7×7 cell |
| Appropriate metrics | Per-cell accuracy (classes balanced) + binomial SE ±1.6 pp |
| `השוואה בין שיטות/מודלים` | Four equalization **methods** across the grid. The slash is "or" — one architecture is correct here, since holding the model fixed is what isolates the preprocessing effect |
| Deep analysis | Ranking + Spearman ρ + attribution + spectra, ≥2 seeds |
| Reproducibility | README, fixed seeds, one-command reproduction |
| Public repo link | This repo |
| Contribution breakdown | Report section |

> When quoting a requirement in the report, read the Hebrew off the PDF. Don't copy from
> `_kb/course/project-spec.md` — that's an English paraphrase, and its invented "R1–R9" numbers
> appear nowhere in the real document.

---

## Critical path

```
data.py  →  cache build  →  56 runs  →  figures  →  report
   2h        3h (idle)       4h         3h         5h
```

Only `data.py` blocks the build. Everything else runs alongside it or after.

---

# 🌙 Tonight — Sun 9 Aug (~4 h)

**The goal is one thing: have the cache build running before you close the laptop.**

### Setup — 30 min, both

1. Check Drive free space on both accounts — **need ≥ 17 GB** for the defaults we use (6.88 GB cache
   + 8.36 GB retained source + 1.74 GB native validation), or ≥ 8 GB with `--no-shards`.
2. Open Colab on both accounts, confirm **a T4 is actually assigned**. Free-tier availability varies;
   if one account is CPU-only, the run split changes tonight, not Monday afternoon.
3. Ido pushes the skeleton, makes the repo **public**, adds Noa.

```
src/{data,model,train,baseline,analyze}.py
notebooks/{00_build_cache,01_run_matrix}.ipynb
results/{runs,figures}/.gitkeep
docs/                 # the submitted proposal
requirements.txt  README.md  PLAN.md
```

`requirements.txt`: `torch numpy pillow pyarrow huggingface_hub hf_xet matplotlib scipy`

> **Don't install torch locally.** Python here is 3.13.3 and the wheel may not resolve. Try once,
> then move on — the laptop only needs `numpy` and `Pillow` for the baseline and every figure. All
> torch work happens in Colab.

### Ido — `src/data.py`, 2 h

Three functions.

**`equalize(img, strategy) -> uint8[128,128,3]`** — the four strategies. Get these exactly right;
everything downstream depends on them.

| Strategy | Definition | Resamples? |
|---|---|---|
| `center_crop` | Central 128×128 window of the native image | No |
| `random_crop` | 128×128 window at a random top-left, drawn **once per image**, RNG seeded by row index | No |
| `rescale` | `img.resize((128,128), BILINEAR)` | Yes |
| `pad` | Long side → 128 keeping aspect, then zero-pad to square | Yes, plus a border |

**`build_cache(split, out_dir)`** — one shard at a time: download → decode each row **once** → write
all four strategies from that single decode → save four `.npy` + one `.npz` of metadata
(`h`, `w`, `label`, `generator`) → delete the shard → touch a `.done` marker. Re-running skips
finished shards. Peak local disk ~440 MB.

**`load_arm(cache_dir, strategy, source, seed)`** — returns train/eval tensors for one run.

Set `HF_XET_HIGH_PERFORMANCE=1` and `pip install -q hf_xet`. (The dataset is Xet-backed;
`hf_transfer` was the pre-Xet mechanism and is now deprecated and ignored.)

### Noa — `src/model.py`, 1.5 h

Needs no data — test it with random tensors.

`CompactCNN`: 4 blocks of `[Conv3×3 → BN → ReLU]×2 → MaxPool2`, widths 32/64/128/256, then global
average pool → dropout(0.3) → `Linear(256,1)`. About 1.2 M parameters.

**Stride-1 convs from layer 1, first pool only after block 1.** The signal that distinguishes real
from generated is high-frequency, so downsampling early would throw it away before the network sees it.

### Ido — pilot 2 shards, then verify — 30 min. **Do not skip.**

A color-channel swap or transposed axis costs 5 minutes to catch here, and 28 ruined runs if you
catch it later.

- Row counts and split sizes match the shard header
- Native sizes match: ADM/GLIDE/VQDM 256², BigGAN 128², SD15/Wukong 512², Midjourney 1024², real varies
- `SD14` appears as a label with **zero** rows — assert it; never claim eight generators
- **Render 3 images per generator per strategy and look at them**

### Ido — 🚀 launch the full build, then sleep (2–3 h, unattended)

**Validation split first, then train.** Val is where every reported cell comes from and it carries
the metadata Noa needs, so finishing it first lets her start in the morning.

Writes to `/content/drive/MyDrive/university/deep_learning/cache/<strategy>/<split>-<shard>.npy`
and `.../meta/<split>-<shard>.npz`. Output: four uint8 arrays of `(35000,128,128,3)`, 1.72 GB each,
**6.88 GB total**.

`build_cache` keeps the source parquets by default (`keep_shards=True`), which adds the full
**8.36 GB** of source to that figure — pass `--no-shards` to drop it. We keep them: Drive has 5 TB,
and retaining them means a rebuild costs no re-download. Validation is also built with
`--keep-native`, a further 1.74 GB, so a native-resolution original can be shown beside its 128×128
crop in the report. Do **not** pass `--keep-native` on train.

Keep the tab open and the laptop awake. If it disconnects, re-run the cell — finished shards are
skipped, so a disconnect costs one shard.

**If the download stalls:** same loop, but fetch rows through the datasets-server `/rows` API,
100 per request. That route is already proven to work on this dataset. Slower (1–3 h), same code.
**Never** `snapshot_download` the whole repo — that already failed once, and a timeout throws away
the entire transfer.

**Noa's copy:** share `/MyDrive/university/deep_learning/` and use *Add shortcut to Drive*. Costs no
quota. Never build it twice.

---

# 📅 Mon 10 Aug — code, calibrate, run (~10 h)

### Morning, in parallel

**Ido — `src/train.py`, 2.5 h.** AdamW, lr 3e-4, weight decay 1e-4, cosine to 0, batch 128,
40 epochs, AMP, BCE-with-logits. **Horizontal flip only** — every other augmentation resamples or
re-encodes, which would inject the very artifact under study. That's a methodological point; put it
in the report.

**No DataLoader.** An arm is ~390 MB of uint8: push it to the GPU once and index it. Faster, and
less code.

Each cell (source→target) = 500 fakes from the target generator + the **same fixed 500 real** val
images. Hold out 10 % of the training set internally for checkpoint selection; the reported cells
come from the official val split and never influence model selection.

**Noa — `src/baseline.py`, 1.5 h.** Laptop, no GPU. Needs sizes only, never pixels — the val
metadata landed last night.

- **Square rule** — predict synthetic iff `h == w`. Zero parameters.
- **Size-lookup rule** — collect the `(h,w)` pairs fakes emit in train; predict synthetic iff the
  test pair is in that set.

Report both per 7×7 cell, so they sit alongside the CNN matrices.

> **Gate:** if the baseline is *not* near-perfect, stop and recheck the metadata before writing
> anything else. The whole framing rests on this number.

### Midday — calibration run, 30 min, Ido

One real run: `center_crop` / `BigGAN` / seed 0.

| Time per run | Then |
|---|---|
| ≤ 4 min | Proceed, keep 40 epochs |
| 4–8 min | Drop to 25 epochs |
| > 8 min | Also halve the training set (4,000 → 2,000) |

Diagonal accuracy must be **> 95 %** — that's what confirms the pipeline learns at all, before you
commit 28 runs to it.

### Afternoon — seed 0, 28 runs, ~2 h, both in parallel

Ido runs `center_crop` + `rescale`; Noa runs `pad` + `random_crop`. 14 each.

`notebooks/01_run_matrix.ipynb`: clone → mount Drive → loop over the seven sources. The entire resume
mechanism is `if metrics.json exists: continue`.

**Syncing two Google accounts:** each person commits their `metrics.json` files (~2 KB each) to the
repo. That's it — no large transfers between accounts, ever.

### Evening — figures, ~3 h, split

**Ido — matrices and the ranking.** Four 7×7 heatmaps on a shared color scale, plus the baseline as
a fifth panel. Report per-arm diagonal mean, off-diagonal mean, and **the gap between them** — that
gap is the cross-generator failure the project is about.

Then the main result: rank the seven generators by mean off-diagonal accuracy, once per strategy.
Four rankings side by side, Spearman ρ between each pair. **A low ρ is the finding** — it means
published cross-generator numbers depend on a preprocessing choice nobody reports.

**Noa — attribution and spectra.**
*Input-gradient attribution* (the method taught in L10, slides 53–57 — **not Grad-CAM**):
`g = ∂f(x)/∂x`, `saliency = max_channel |g|`, averaged over 200 images per generator per strategy.
Report **border mass** — the fraction of saliency in the outer 16-pixel ring, which tests directly
whether the `pad` model is reading its own zero border.

*Radially averaged spectra*: per (class, generator, strategy) — grayscale, subtract the mean, `fft2`,
`fftshift`, magnitude, average over 64 radial rings, plot log-magnitude vs frequency. The informative
curve is **real minus fake**. No window function: the `pad` border is a real feature of that arm and
a window would hide it. Say so in the caption.

Use the `dataviz` skill. Every figure gets a one-sentence claim attached.

**🎯 Monday is done when all 28 seed-0 cells exist and the figures are drafted.**

---

# 📅 Tue 11 Aug — write and ship (~10 h)

### First thing: start seed 1 in the background

28 runs, ~1 h per account, unattended. **Two seeds is required** — the headline claim is a *ranking
change*, and a ranking from one seed can't survive ±1.6 pp per-cell noise. Seed 2 only if the
machines are idle anyway.

### Morning–midday — the report, ≤5 pages, 5–6 h

The spec names **ten required components**. Use them as the section list so a grader can tick each
one off. **Count the rendered PDF, not the Markdown**, and recount after every edit.

| # | Component | Pages | Who |
|---|---|---|---|
| 1 | Motivation and problem definition | 0.5 | I |
| 2 | Related work | 0.4 | N |
| 3 | Data | 0.4 | N |
| 4 | **Models and hyperparameters** — as numbers, not prose | 0.7 | I |
| 5 | Results — four matrices, baseline, ranking table | 1.4 | I |
| 6 | Analysis — attribution + spectra | 0.9 | N |
| 7 | Discussion — insights, **limitations, future directions** | 0.4 | N |
| 8 | References | 0.2 | N |
| 9 | Contribution breakdown | 0.05 | N |
| 10 | Repo link | — | I |

Items 4, 7 and 8 are the ones that vanish under time pressure. Each is explicitly named in the spec.

**Include these — they're what separates "good" from "excellent":**
- `pad` and `rescale` are identical for all seven generators, since every generator emits square
  images. They differ **only on the real class**, the one class with variable aspect ratio. That's
  not a flaw — it makes the pair a clean intervention on real photographs alone.
- **BigGAN's whole row is a control.** At native 128², all four strategies are the identity on it, so
  any variation across strategies in that row comes purely from the real class.
- The ±1.6 pp binomial SE per cell, and which ranking differences survive it.
- How many seeds each cell carries.
- Grommelt et al. (2024), arXiv 2403.17608, already established this bias. **Our claim isn't the
  bias — it's that which correction you choose changes the conclusion.** Say it outright; it
  protects you from the grader who has read that paper.

### Afternoon — README, 1.5 h, Ido

Description · `pip install -r requirements.txt` · the cache-build command · the exact `train.py`
command for one cell · the loop that reproduces all 28 · where metrics land · results table inline.
Check that `results/` is committed and `cache/` is not. Verify a clean clone runs.

### Evening — deck, as far as it gets

12–14 slides: title · **the shortcut** (a real photo beside a 512² fake — "the size alone tells you")
· the baseline number, which should land as a shock · the question: does the fix change the answer? ·
data · the four strategies, one visual row each · model and protocol · the matrices ·
**the ranking table, the money slide** · attribution · spectra · limitations · contributions + repo.

The deck may finish after 11 Aug — the presentation date is set by the course and is separate.
Rehearse against a clock before the real thing; overrunning 15 minutes is the easiest way to lose
points on a good talk.

**🎯 Tuesday is done when the report PDF is ≤5 pages, covers all ten components, and is pushed.**

---

## Run budget

| Block | Runs | Est. | When |
|---|---|---|---|
| Seed 0 | 28 | 2 h | Mon afternoon |
| Seed 1 — required | 28 | 2 h | Tue, background |
| Seed 2 — bonus | 28 | 2 h | only if idle |
| **Required** | **56** | **~4 h** | **~2 h per account** |

Comfortably inside the free-tier allowance. Runs finish in minutes each, so a 90-minute idle
disconnect costs at most one run.

## Risks

| When | Risk | Already handled by |
|---|---|---|
| Tonight | Drive full, build dies at hour 3 | The 2-minute quota check |
| Tonight | An account has no T4 | Checked before runs are scheduled |
| Tonight | Download stalls | datasets-server fallback, same loop |
| Tonight | Bad cache ruins every run | 2-shard pilot + look at the images |
| Mon | Runs slower than estimated | Calibration gate cuts epochs, then data |
| Tue | **All four rankings agree — a null result** | Write it as a reportable finding, not a rewrite. The baseline number and the four matrices stand on their own. |

## If you run out of time

Cut top-down, stop when it fits. Everything here is execution quality, never a requirement.

1. **Let the date slip.** 11 Aug is self-imposed; the project is 80 % of the grade.
2. Let the deck finish after 11 Aug.
3. Seeds 3 → 2. **Never below 2.**
4. Epochs 40 → 25.
5. Halve each arm's training set.
6. Attribution over 100 images per cell instead of 200.

**Never cut:** any of the four strategies · spectra or attribution · training from scratch · the
baseline · two seeds · the contribution table, repo link, or README. The first three are promised in
the submitted proposal; the rest are graded requirements.
