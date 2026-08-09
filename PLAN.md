# Schedule — Resolution Bias and Cross-Generator Detection of AI-Generated Images

**Requirements come first. The date comes second.** Target is Tue 11 Aug 2026 inclusive — but it is
self-imposed, there is no formal course deadline, and the project is **80 % of the grade**. So the
rule is: *if something has to give, the date gives, not a requirement.*

Reference plan (full code snippets, exact transform definitions, report structure):
`OneDrive\...\למידה עמוקה\חומרים לnotebookLM\PROJECT_PLAN.md`. This file is the running order.

**I** = Ido, **N** = Noa. Two Colab accounts.

---

## Requirements audit — what the spec demands and where this plan delivers it

Spec: `_kb/course/project-spec.md`, from `הנחיות לפרויקט בקורס למידה עמוקה 2026ב`.
**The R-numbers below are internal labels for this table only. Never cite them in the report, the
deck, or to a grader — the instructions PDF has no numbered requirements.** Quote the Hebrew.

| Requirement | Delivered by | Status |
|---|---|---|
| PyTorch | `CompactCNN`, trained from scratch | ✅ |
| Substantive training of a deep model | 56–70 training runs from scratch | ✅ |
| Clear baseline | Dimension rule, two variants, reported per 7×7 cell | ✅ |
| Appropriate metrics | Per-cell accuracy (classes exactly balanced) + binomial SE ±1.6 pp | ✅ |
| `השוואה בין שיטות/מודלים` — comparison between methods **or** models | Four equalisation strategies, compared across the full 7×7 grid, plus the baseline | ✅ |
| Deep analysis | Ranking + Spearman ρ + attribution + spectra, **≥2 seeds** | ✅ |
| Reproducibility: organised code, run instructions | README phase, fixed seeds, one-command reproduction | ✅ |
| Public repo link in the report | This repo, public | ✅ |
| Per-member contribution breakdown | Report section, drafted as work happens | ✅ |

### On the comparison bullet — one architecture is enough

The Hebrew is `השוואה בין שיטות/מודלים` — a **slash, not "and"**. Comparing methods *or* models
satisfies it. The four equalisation strategies, compared across the full 7×7 grid against a common
baseline, are a comparison between methods. **No second architecture is needed, and none is planned.**

The single architecture is also the methodologically correct choice here: the study's claim is about
what preprocessing does to cross-generator transfer, and holding the model fixed is what isolates
that. Adding a second architecture would spend GPU hours widening the design rather than deepening it.

### Two decisions that follow from "requirements first"

1. **Two seeds minimum, not one.** This rests on the *separate* `ניתוח מעמיק` bullet, not on the
   comparison bullet, so it stands regardless of the above. The headline claim is a *ranking change
   across strategies*, and a ranking from a single seed is not defensible against ±1.6 pp per-cell
   noise. Third seed if time.
2. **The build launches tonight.** It is the only multi-hour unattended step; a day spent
   not-building cannot be recovered.

Never cut, in any scenario: the four equalisation arms · spectra + attribution · training from
scratch · the baseline · ≥2 seeds · the contribution breakdown · the repo and README.

---

## Critical path

```
data.py  →  cache build  →  56 runs  →  figures  →  report
   2h          3h (idle)      4h         3h         5h
```

Only `data.py` blocks the build. Everything else runs alongside it or after.

---

# 🌙 TONIGHT — Sun 9 Aug (~4 h) — the whole point is to launch the build before sleeping

### 0. Setup (30 min, both) — do this before opening an editor

| # | Do | Who |
|---|---|---|
| 1 | Check Google Drive free space, both accounts — **need ≥ 8 GB** | I + N |
| 2 | Open Colab on both accounts, confirm a **T4 is actually assigned** | I + N |
| 3 | Push the repo skeleton, make it **public**, add Noa as collaborator | I |

Item 2 is not optional: free-tier T4 availability varies, and if one account is CPU-only the run
split has to change tonight, not on Monday afternoon.

Skeleton (`C:\GitHub\TAU_repos\deep-learning`):

```
src/{data,model,train,baseline,analyze}.py
notebooks/{00_build_cache,01_run_matrix}.ipynb
results/{runs,figures}/.gitkeep
docs/                          # copy of the submitted proposal PDF + md
requirements.txt  README.md  PLAN.md
```

`requirements.txt`: `torch numpy pillow pyarrow huggingface_hub hf_transfer matplotlib scipy`
`.gitignore` already has `cache/ *.npy *.npz *.pt *.pth`.

> **Skip local torch.** Try `pip install torch --index-url https://download.pytorch.org/whl/cpu` once;
> Python here is 3.13.3 and if no wheel resolves, **do not spend deadline hours on it.** The laptop
> needs only `numpy` + `Pillow` (already installed) for the baseline, the spectra and every figure.
> All torch work happens in Colab.

### 1. `src/data.py` (2 h, I) — write nothing else

**`equalize(img, strategy) -> uint8[128,128,3]`**

| Strategy | Definition | Resampling? |
|---|---|---|
| `centre_crop` | Central 128×128 window of the native image | **No** |
| `random_crop` | 128×128 window at a random top-left, drawn **once per image**, RNG seeded by row index | **No** |
| `rescale` | `img.resize((128,128), BILINEAR)` — aspect-distorting | Yes |
| `pad` | Long side → 128 preserving aspect (BILINEAR), zero-pad symmetrically | Yes, plus a hard border |

**`build_cache(split, out_dir)`** — shard loop: `hf_hub_download` one parquet shard → decode each row
**once** → emit all four strategies from that single decode → write four per-shard `.npy` + one
`.npz` of metadata (`h`, `w`, `label`, `generator`) → `os.remove(shard)` → touch `.done`.
Re-running skips shards with a marker. ~440 MB peak local disk, resumable at shard granularity.

**`load_arm(cache_dir, strategy, source, seed)`** — train/eval uint8 tensors for one run.

Set `HF_HUB_ENABLE_HF_TRANSFER=1`, `pip install -q hf_transfer`.

### 2. Pilot: 2 shards, then verify (30 min, I) — **do not skip**

A BGR swap or transposed axis costs 5 minutes here and 3 hours plus 28 poisoned runs if found later.

- Row counts and split sizes match the shard header
- Native sizes match: ADM/GLIDE/VQDM 256², BigGAN 128², SD15/Wukong 512², Midjourney 1024², real varies
- `SD14` present as a label with **zero** rows — assert it, never claim eight generators
- **Spot-render 3 images per generator per strategy in a grid and look at them**

### 3. 🚀 LAUNCH THE FULL BUILD, then go to sleep (2–3 h wall, unattended)

**Validation split first (4 shards, ~25 % of the work), then train.** Val is where every reported
cell comes from and it carries the metadata the baseline needs — finishing it first lets Noa start
in the morning without waiting.

Writes to `/content/drive/MyDrive/aidet/cache/<strategy>/<split>-<shard>.npy` and
`/content/drive/MyDrive/aidet/meta/<split>-<shard>.npz`.
Output: four uint8 arrays of `(35000,128,128,3)` = 1.72 GB each, **6.88 GB total**. The 8.4 GB source
never lands on Drive or the laptop.

Keep the browser tab open and the laptop awake. If it disconnects, re-run the cell — done shards are
skipped, so a disconnect costs one shard.

**Fallback A** (if `hf_hub_download` stalls): same loop, rows via the datasets-server `/rows` API,
100/request — the route already proven to work on this dataset. 1–3 h.
**Never** `snapshot_download` the whole repo: that is what already failed, and one timeout discards
the entire transfer.

**Noa's copy:** share `/MyDrive/aidet/`, she uses *Add shortcut to Drive*. Costs no quota. Never
build it twice.

**🎯 Tonight is a success if the build is running when you close the laptop.** Nothing else tonight
matters as much.

---

# 📅 MON 10 AUG — code, calibrate, run the matrix (~10 h, both)

### Morning · in parallel

**Ido — `src/model.py` + `src/train.py` (3 h).** Needs no data; smoke-test against **random tensors**
of the right shape, so a failed build costs nothing here.

`CompactCNN`: 4 blocks of `[Conv3×3 → BN → ReLU]×2 → MaxPool2`, widths 32/64/128/256, GAP →
dropout(0.3) → `Linear(256,1)`. ~1.2 M params. **Stride-1 3×3 convs from layer 1, first pool only
after block 1** — the cue is high-frequency; aggressive early downsampling destroys it before the
network sees it.

**One architecture only.** The spec's comparison bullet is `שיטות/מודלים` — methods *or* models — and
the four equalisation strategies are the comparison. Holding the model fixed is also what isolates
the preprocessing effect, which is the whole claim.

Training: AdamW, lr 3e-4, wd 1e-4, cosine to 0, batch 128, **epochs 40 (drop to 25 if calibration
says so)**, AMP, BCE-with-logits. **Horizontal flip only** — every other augmentation is a resampling
or re-encoding op and would inject the artefact under study. Methodology, not laziness; it goes in
the report.

**No DataLoader.** An arm is ~390 MB of uint8 — push it to VRAM once and index it. Faster, and less code.

Split discipline: 10 % of each arm's train set is an internal val set for checkpoint selection. Every
reported cell comes from the official val split and is never used for model selection. Each cell
(s→t) = 500 fakes from generator *t* + the **same fixed 500 real** val images.

**Noa — `src/baseline.py` (1.5 h, laptop, no GPU).** Needs sizes only, never pixels; the val metadata
landed last night.
- **Square rule** — predict synthetic iff `h == w`. Zero parameters.
- **Size-lookup rule** — collect `(h,w)` pairs emitted by fakes in train; predict synthetic iff the
  test pair is in that set.

Report both **per 7×7 cell**, same shape as the CNN matrices.

> **Gate.** If the baseline is not near-perfect, stop and re-check the metadata before writing another
> line of code. The whole framing rests on this number.

### Midday · calibration run (30 min, I) — **decision gate**

One real run: `centre_crop` / `BigGAN` / seed 0. Record wall clock, peak VRAM, diagonal accuracy.

| Measured | Then |
|---|---|
| ≤ 4 min | Run the 28 as planned, keep 40 epochs |
| 4–8 min | Drop to **25 epochs** and re-time |
| > 8 min | Halve the training set (4,000 → 2,000) as well |

Diagonal accuracy must be **> 95 %** — that confirms the pipeline learns at all before 28 runs are
committed.

### Afternoon · seed 0 matrix — 28 runs, ~2 h, both in parallel

| Account | Arms | Runs |
|---|---|---|
| **Ido** | `centre_crop`, `rescale` | 14 |
| **Noa** | `pad`, `random_crop` | 14 |

`notebooks/01_run_matrix.ipynb`: clone → mount Drive → nested loop over the seven sources. Resume
story is one line: `if metrics.json exists: continue`.

**Sync across two Google accounts:** each person commits their `metrics.json` (~2 KB each) to the
shared GitHub repo. That is the entire mechanism — no large transfer between accounts, ever.

### Evening · start the figures (2–3 h, split)

**Ido — transfer matrices + the ranking result.** Four 7×7 heatmaps on a shared scale plus the
baseline as a fifth panel. Headline numbers: per-arm diagonal mean, off-diagonal mean, and **the gap**
between them — that gap is the cross-generator failure the project is about.

Then **the result**: rank the seven generators by mean off-diagonal accuracy, once per strategy. Four
rankings side by side, Spearman ρ between each pair. **A low ρ is the finding** — it means published
cross-generator numbers depend on a preprocessing choice nobody reports.

**Noa — attribution + spectra.**
*Input-gradient attribution*, the taught method (L10 slides 53–57), **not Grad-CAM**:
`g = ∂f(x)/∂x`, `saliency = max_channel |g|`, averaged over 200 images per generator per strategy.
Two summary numbers: **border mass** (fraction of saliency in the outer 16-px ring — tests directly
whether the `pad` model reads its own zero border) and the radial spectrum of the saliency map.

*Radially averaged spectra*: per (class, generator, strategy) — grayscale, subtract mean, `fft2`,
`fftshift`, magnitude, average over 64 radial annuli, log-magnitude vs normalised frequency. The
informative curve is **real minus fake**. No window function — the `pad` border is a real feature of
that arm and windowing would hide it; note the choice in the caption.

Use the `dataviz` skill. Every figure gets a one-sentence claim attached to it.

**🎯 Monday is a success if all 28 seed-0 cells exist and the figures are drafted.**

### Total run budget

| Block | Runs | Est. | When |
|---|---|---|---|
| Seed 0, four arms | 28 | 2 h | Mon afternoon |
| Seed 1, four arms — **required** | 28 | 2 h | Tue, background |
| Seed 2 — bonus | 28 | 2 h | only if idle |
| **Required total** | **56** | **~4 h** | **~2 h per account** |

Inside the free-tier allowance for both accounts. Individual runs finish in minutes, so a 90-minute
idle disconnect costs at most one run.

---

# 📅 TUE 11 AUG — write, render, ship (~10 h, both)

### First thing: launch the remaining GPU work in the background (unattended)

Both accounts start these **before opening the report**. They run for hours without supervision
while the writing happens — GPU time and writing time are free to overlap.

| Priority | What | Runs | Why this order |
|---|---|---|---|
| 1 | **Seed 1**, all four arms | 28 | Required — a ranking claim from one seed is not defensible |
| 2 | Seed 2, all four arms | 28 | Bonus. Tightens the spread; drop it first |

Priority 1 ≈ **1 h of GPU per account** — it finishes early in the writing block. Priority 2 goes
only if the machines are idle anyway.

The report states exactly which cells carry how many seeds, whatever the count ends up being.

### Morning–midday · the report, ≤5 pages PDF (5–6 h, both, split by section)

**Count the rendered PDF, not the Markdown**, and re-count after every edit.

The spec lists **ten required components** for the report. Use them as the section plan so a grader
can tick every one off — none is optional:

| # | Required component | Pages | Who |
|---|---|---|---|
| 1 | Motivation and problem definition | 0.5 | I |
| 2 | Review of related work | 0.4 | N |
| 3 | Description of the data | 0.4 | N |
| 4 | **Models and hyperparameters** — both architectures, optimiser, lr, schedule, batch, epochs, seeds | 0.7 | I |
| 5 | Results — four matrices, baseline matrix, ranking table | 1.4 | I |
| 6 | Analysis of results — attribution + spectra | 0.9 | N |
| 7 | Discussion: insights, **limitations, future directions** | 0.4 | N |
| 8 | References | 0.2 | N |
| 9 | Per-member contribution breakdown | 0.05 | both |
| 10 | Code repository link | — | I |

Items 4, 7 and 8 are the ones most easily lost under time pressure and are each explicitly named in
the spec. **Hyperparameters must be stated as numbers, not prose** — that is what makes item 4
gradeable.

**Put these in — they are what separates "good" from "excellent":**
- `pad` ≡ `rescale` for all seven generators (all emit square images). They differ **only on the real
  class**, the only one with variable aspect. Not a defect — a clean single-class intervention.
- **BigGAN's entire row is a control**: at native 128², all four strategies are the identity on it,
  so any variation across strategies in that row comes purely from the real class.
- Binomial SE **±1.6 pp** per cell at n=1000, and which ranking differences survive it.
- How many seeds each cell carries, and which ranking differences survive the spread.
- Whether the spectra and the attribution agree: does accuracy drop in the arms where the spectra
  show the high-frequency band attenuated? That link between two independent analyses is exactly the
  "insight" the grading criteria reward.
- Grommelt et al. (2024), arXiv 2403.17608, already established the bias. **Our claim is not the
  bias — it is that which correction you choose changes the conclusion.** Say it explicitly; it
  protects against the grader who has read that paper.

### Afternoon · README + reproducibility (1.5 h, I)

One-paragraph description · `pip install -r requirements.txt` · the cache-build command · the exact
`train.py` command for one cell · the loop reproducing all 28 · where metrics land · results table
inline. Confirm `results/runs/**/metrics.json` and `results/figures/*.png` are committed and `cache/`
is not. Verify a clean clone runs.

### Evening · deck, as far as it gets (may slip past 11 Aug)

12–14 slides: (1) title · (2) the shortcut — a real photo beside a 512² fake, "the size alone tells
you" · (3) the baseline number, which should shock · (4) the question: does the fix change the
answer? · (5) data · (6) the four strategies, one visual row each · (7) model and protocol ·
(8–9) the matrices · (10) **the ranking table — the money slide** · (11) attribution · (12) spectra ·
(13) limitations · (14) contributions + repo.

Rehearse against a clock before the actual presentation. Overrunning 15 minutes is the most common
way to lose points on a talk that is otherwise fine.

**🎯 Tuesday is a success if the report PDF is ≤5 pages, covers all ten required components, and is
pushed.** If it does not, Wednesday exists — see the cut ladder. The date is the thing that gives.

---

## Risks, in the order they can bite

| When | Risk | Mitigation, already scheduled |
|---|---|---|
| Tonight | Drive full → build dies at hour 3 | 2-minute quota check first |
| Tonight | An account has no T4 | Checked before any run is scheduled |
| Tonight | HF download stalls again | Fallback A shares the same loop body |
| Tonight | Corrupt cache poisons every run | 2-shard pilot + visual grid **before** the full build |
| Mon midday | Runs slower than estimated | Calibration gate cuts epochs, then training-set size |
| Mon evening | Matrix incomplete at end of day | Runs are resumable per cell; finish Tuesday morning alongside the seeds |
| Tue | **All four rankings agree — null result** | Write the report so a null is a reportable finding, not a rewrite. The baseline number and the four matrices stand on their own. |

## Cut ladder — requirements first, date second

Everything on this ladder is **quality of execution, never a requirement**. Apply top-down and stop
as soon as it fits.

1. **Let the date slip.** 11 Aug is self-imposed and there is no course deadline. A day late with the
   requirements met beats on-time with a requirement missing — the project is 80 % of the grade.
2. Let the **deck** finish after 11 Aug. The presentation date is set by the course and is separate.
3. Seeds 3 → 2. (**Never below 2** — see the seeds decision above.)
4. Epochs 40 → 25.
5. Halve each arm's training set (4,000 → 2,000).
6. Attribution over 100 images per cell instead of 200.
7. Drop the saliency radial spectrum — keep the image spectra, which are what the proposal promised.

**Never cut, at any deadline pressure:**

| | Why |
|---|---|
| Any of the four equalisation arms | Promised in the submitted proposal; the comparison *is* the result |
| Spectra + attribution | Promised in the submitted proposal |
| Training from scratch | Promised, and a pretrained backbone is contaminated by ImageNet |
| The dimension-rule baseline | Spec requires a clear baseline |
| Two seeds | Spec grades depth of analysis; a one-seed ranking claim fails it |
| Contribution breakdown, repo link, README | Each explicitly required and explicitly graded |
