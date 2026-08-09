# Schedule — Resolution Bias and Cross-Generator Detection of AI-Generated Images

Ordered by **time**, not by topic. Reference plan (full code snippets, exact transform definitions,
report structure) lives in the course folder:
`OneDrive\...\למידה עמוקה\חומרים לnotebookLM\PROJECT_PLAN.md`. This file is the running order.

`T+` figures are **cumulative hands-on hours** for the pair. Deadlines are unknown, so nothing is
dated. Two people, two Colab accounts: **I** = Ido, **N** = Noa.

---

## The critical path, in one line

```
data.py  →  cache build (long, unattended)  →  84 GPU runs  →  analysis  →  report
```

**Only `data.py` blocks the long pole.** Everything else — model, training loop, baseline, figures —
either runs in parallel with the build or comes after it. So the schedule below writes `data.py`
first and writes nothing else until the build is launched.

---

# T+0 — Fire the slow things now (30 min, both, today)

These are zero-effort and long-latency. Every hour they are not started is an hour added to the end
of the project. Do them before opening an editor.

| # | Do | Who | Why now |
|---|---|---|---|
| 1 | **Ask the course staff for the report and presentation deadlines.** | I | Latency is days, and the answer can change the whole plan (see cut ladder). Oldest unanswered question on the project. |
| 2 | **Check free Google Drive space on both accounts — need ≥ 8 GB.** | I + N | 2-minute check that prevents the cache build dying at hour 3 with the disk full. |
| 3 | **Open Colab on both accounts, confirm a T4 is actually assigned.** | I + N | Free-tier T4 availability varies. If an account only gets CPU, the run split has to change *before* runs are scheduled, not during. |
| 4 | **Push the repo skeleton, make it public, add Noa as collaborator.** | I | Gates *all* of Noa's work. Until this exists she cannot start anything. |

Skeleton to push (`C:\GitHub\TAU_repos\deep-learning`):

```
src/{data,model,train,baseline,analyze}.py   # empty stubs
notebooks/{00_build_cache,01_run_matrix}.ipynb
results/{runs,figures}/.gitkeep
docs/                                        # copy of the submitted proposal PDF + md
requirements.txt   README.md   PLAN.md
```

Append to `.gitignore`: `cache/`, `*.npy`, `*.pt`, `*.pth`

`requirements.txt`: `torch numpy pillow pyarrow huggingface_hub hf_transfer matplotlib scipy`

> **Local torch is optional.** Try `pip install torch --index-url https://download.pytorch.org/whl/cpu`
> once. Python here is 3.13.3; if no wheel resolves, do not fight it. The laptop only needs `numpy`
> and `Pillow` for the baseline, the spectra and every figure. All torch work happens in Colab.

**Done when:** Noa can clone the repo, both accounts show a T4, both have ≥ 8 GB Drive free, and the
deadline question has been sent.

---

# T+0.5 → T+2.5 — `src/data.py` — the only thing on the critical path (2 h, I)

Write nothing else during this block. Every hour of delay here is an hour the cache is not building.

Three pieces:

1. **`equalize(img, strategy) -> uint8[128,128,3]`** — the four strategies. This is the load-bearing
   part of the whole project; get the definitions written down before any other code.

   | Strategy | Definition | Resampling? |
   |---|---|---|
   | `centre_crop` | Central 128×128 window of the native image. | **No** |
   | `random_crop` | 128×128 window at a random top-left, drawn **once per image**, RNG seeded by row index. | **No** |
   | `rescale` | `img.resize((128,128), BILINEAR)` — aspect-distorting. | Yes |
   | `pad` | Long side → 128 preserving aspect (BILINEAR), then zero-pad symmetrically. | Yes, plus a hard border |

2. **`build_cache(split, out_dir)`** — shard loop: `hf_hub_download` one parquet shard → decode each
   row **once** → emit all four strategies from that single decode → write four per-shard `.npy` +
   one `.npz` of metadata (`h`, `w`, `label`, `generator`) → `os.remove(shard)` → touch `.done`.
   Re-running skips shards with a marker. Resumable at ~440 MB granularity, ~440 MB peak local disk.

3. **`load_arm(cache_dir, strategy, source, seed)`** — returns the train/eval uint8 tensors for one
   run. Called by `train.py`.

Set `HF_HUB_ENABLE_HF_TRANSFER=1` and `pip install -q hf_transfer` in the notebook.

**Done when:** `equalize` has unit-checkable output shapes for a 128², a 256², a 1024² and a
500×375 input, and the shard loop runs end-to-end on **one** shard.

---

# T+2.5 → T+3.0 — Pilot build: 2 shards, then verify (30 min, I, Colab)

**Do not launch the full build yet.** Build two shards, then run the verification gate below. A BGR
swap or a transposed axis costs 5 minutes here and 3 hours plus 84 poisoned runs if it is found later.

Verification (all five, plus the eyeball — this project has already been burned once by
characterising a dataset from a prefix):

- Row count and split sizes match the shard header.
- Labels present in both classes.
- Native sizes match the known map: ADM/GLIDE/VQDM 256², BigGAN 128², SD15/Wukong 512²,
  Midjourney 1024², real variable.
- `SD14` appears as a label with **zero** rows. Assert it. Never claim eight generators.
- **Spot-render 3 images per generator per strategy in a grid and look at them.**

**Done when:** the grid looks like photographs, right way up, correct colour.

---

# T+3.0 — 🚀 LAUNCH THE FULL CACHE BUILD, then walk away (2–3 h wall, unattended)

This is the long pole. Once it is running, it needs no attention, and everything below happens
while it runs.

Order matters: **build `validation` first (4 shards, ~25 % of the work), then `train`.**
The val split is where every reported cell comes from, and it carries the metadata the baseline
needs — so finishing it first unblocks Noa roughly an hour earlier.

Writes to `/content/drive/MyDrive/aidet/cache/<strategy>/<split>-<shard>.npy` and
`/content/drive/MyDrive/aidet/meta/<split>-<shard>.npz`.

Expected output: four `uint8` arrays of `(35000, 128, 128, 3)` = 1.72 GB each, **6.88 GB total**.
The 8.4 GB source never lands on Drive or the laptop.

**Fallback A** (if `hf_hub_download` stalls): identical loop, fetch rows through the datasets-server
`/rows` API, 100 rows/request. Proven to work on this dataset. 1–3 h.
**Fallback B** (both Colab routes fail): laptop overnight `huggingface-cli download --resume-download`,
build locally, upload the 6.88 GB cache — hours, last resort.
**Never** `snapshot_download` the whole repo — that is what already failed, and one timeout discards
the entire transfer.

**Noa's copy:** share `/MyDrive/aidet/` and have her use *Add shortcut to Drive*. A shortcut costs no
quota. Do not build it twice.

---

## While the build runs — two parallel tracks

### Track I (Ido) · T+3 → T+6 · `src/model.py` + `src/train.py` (3 h)

Needs no data. Smoke-test against **random tensors** of the right shape — so if the build fails
entirely, nothing here is lost.

`CompactCNN`: 4 blocks of `[Conv3×3 → BN → ReLU]×2 → MaxPool2`, widths 32/64/128/256, GAP →
dropout(0.3) → `Linear(256,1)`. ~1.2 M params. **Stride-1 3×3 convs from layer 1, first pool only
after block 1** — the cue is high-frequency, aggressive early downsampling throws it away before the
network sees it.

Training: AdamW, lr 3e-4, wd 1e-4, cosine to 0, batch 128, 40 epochs, AMP, BCE-with-logits.
**Horizontal flip only** — every other augmentation is a resampling or re-encoding op and would
inject the artefact under study. This is methodology, not laziness; it goes in the report.

**No DataLoader.** An arm is ~390 MB of uint8 — push it to VRAM once and index it. Faster, and
*less* code.

Split discipline: 10 % of each arm's train set is an internal val set for checkpoint selection. All
reported cells come from the official val split and are never used for model selection. Each cell
(s→t) = 500 fakes from generator *t* + the **same fixed 500 real** val images.

**Done when:** `python -m src.train --strategy centre_crop --source BigGAN --seed 0` writes a
`metrics.json` with seven cells.

### Track N (Noa) · T+3 → T+4.5 · `src/baseline.py` (1.5 h, laptop, no GPU)

Can start as soon as the **first val metadata shard** lands (~10 min into the build). Needs sizes
only, never pixels.

- **Square rule** — predict synthetic iff `h == w`. Zero parameters.
- **Size-lookup rule** — collect `(h,w)` pairs emitted by fakes in train; predict synthetic iff the
  test pair is in that set.

Report both **per 7×7 cell**, same shape as the CNN matrices, so they can sit side by side.

> **Gate.** If the baseline is *not* near-perfect, stop and re-check the metadata before writing
> another line of code. The entire framing of the project rests on this number.

---

# T+6.5 — Calibration run (30 min, I, Colab T4) — **decision gate**

One real run: `centre_crop` / `BigGAN` / seed 0. Record wall clock, peak VRAM, in-domain accuracy.

| Measured | Then |
|---|---|
| ≤ 5 min | Launch all 84 runs as planned |
| 5–10 min | Drop to 2 seeds → 56 runs |
| > 10 min | Go to the cut ladder |

**Done when:** the measured minutes-per-run is written into the table below, and diagonal accuracy
is > 95 % — confirming the pipeline learns at all before 84 runs are committed.

---

# T+7 → T+9 — Seed 0 matrix · 28 runs, ~2 h GPU · **both in parallel**

| Account | Arms | Runs |
|---|---|---|
| **Ido** | `centre_crop`, `rescale` | 14 |
| **Noa** | `pad`, `random_crop` | 14 |

`notebooks/01_run_matrix.ipynb`: clone repo → mount Drive → nested loop over the seven sources.
Resume story is one line: `if metrics.json exists: continue`.

**Sync between two Google accounts:** each person commits their `metrics.json` files (~2 KB each) to
the shared GitHub repo. That is the entire mechanism — no large transfer between accounts, ever.

**Done when:** 28 `metrics.json` on `main` and four complete 7×7 matrices with no missing cells.

---

# T+9 → T+13 — Seeds 1 and 2 · 56 runs, ~4 h GPU · both in parallel

Same arm split. Report mean ± range over three seeds.

If time is tight: **finish seed 1 for all 28 first**, then seed 2. Two seeds everywhere beats three
seeds on half the grid, because the claim is about the whole grid. State in the report exactly which
cells are single-seed.

### Run budget

| Arm | Runs/seed | Seeds | Total | @4 min | Account |
|---|---|---|---|---|---|
| `centre_crop` | 7 | 3 | 21 | 1.4 h | Ido |
| `rescale` | 7 | 3 | 21 | 1.4 h | Ido |
| `pad` | 7 | 3 | 21 | 1.4 h | Noa |
| `random_crop` | 7 | 3 | 21 | 1.4 h | Noa |
| **Total** | 28 | 3 | **84** | **5.6 h** | **2.8 h/account** |

2.8 h per account ≈ two Colab sessions each, inside the free-tier allowance. Individual runs finish
in minutes, so a 90-minute idle disconnect costs at most one run.

---

# T+13 → T+19 — Analysis · ~6 h · split three ways · all CPU, all laptop

**8a — Transfer matrices (I).** Four 7×7 heatmaps on a shared scale + the baseline as a fifth panel.
Headline numbers: per-arm diagonal mean, off-diagonal mean, and **the gap** — that gap is the
cross-generator failure the project is about.

**8b — The ranking result, i.e. THE result (I).** Rank the seven generators by mean off-diagonal
accuracy, once per strategy. Four rankings side by side, Spearman ρ between each pair. **A low ρ is
the finding**: published cross-generator numbers depend on a preprocessing choice nobody reports.
Overlay the three-seed spread so the reordering is visibly not noise.

**8c — Input-gradient attribution (N).** The taught method, L10 slides 53–57 — **not Grad-CAM**.
`g = ∂f(x)/∂x`, `saliency = max_channel |g|`, averaged over 200 images per generator per strategy.
Two summary numbers: **border mass** (fraction of saliency in the outer 16-px ring — tests directly
whether the `pad` model reads its own zero border) and the **radial spectrum of the saliency map**.

**8d — Radially averaged spectra (N).** Per (class, generator, strategy): grayscale, subtract mean,
`fft2`, `fftshift`, magnitude, average over 64 radial annuli, log-magnitude vs normalised frequency.
The informative curve is **real minus fake**. No window function — the `pad` border is a real feature
of that arm and windowing would hide it; note the choice in the caption.

**Done when:** every figure the report will use is a PNG in `results/figures/`, each with a
one-sentence claim attached. Use the `dataviz` skill.

---

# T+19 → T+27 — Report, ≤5 pages PDF · ~8 h · both

**Count the rendered PDF, not the Markdown**, and re-count after every edit.

Allocation: problem + confound 0.5 p · data 0.5 p · method (model, four strategies, baseline) 1 p ·
results (four matrices + ranking table) 1.5 p · analysis (attribution + spectra) 1 p · limitations
and conclusion 0.5 p.

Mandatory from the spec: **per-member contribution breakdown** (draft it as you go), repo link,
fixed seeds and run commands.

**Put these in — they are what separates "good" from "excellent":**
- `pad` ≡ `rescale` for all seven generators (all emit square images). They differ **only on the real
  class**, the only one with variable aspect. Not a defect — a clean single-class intervention.
- **BigGAN's entire row is a control**: at native 128², all four strategies are the identity on it,
  so any variation across strategies in that row comes purely from the real class.
- Binomial SE ±1.6 pp per cell at n=1000, and which ranking differences survive it.
- Which cells are single-seed.
- Grommelt et al. (2024), arXiv 2403.17608, already established the bias. **Our claim is not the
  bias — it is that which correction you choose changes the conclusion.** Say so explicitly; it
  protects against the grader who has read that paper.

---

# T+27 → T+31 — Presentation, 15 min · ~4 h · both

12–14 slides, ~7 min each. (1) title · (2) the shortcut: a real photo beside a 512² fake, "the size
alone tells you" · (3) the baseline number, which should shock · (4) the question: does the fix
change the answer? · (5) data · (6) the four strategies, one visual row each · (7) model and protocol ·
(8–9) the matrices · (10) **the ranking table — the money slide** · (11) attribution · (12) spectra ·
(13) limitations · (14) contributions + repo.

Rehearse once against a clock. Overrunning 15 minutes is the most common way to lose points on a
talk that is otherwise fine.

---

# T+31 → T+33 — Reproducibility pass · ~2 h · I

`README.md`: one-paragraph description · `pip install -r requirements.txt` · the cache-build command ·
the exact `train.py` command for one cell · the loop reproducing all 28 · where metrics land ·
results table inline. Confirm `results/runs/**/metrics.json` and `results/figures/*.png` are
committed and `cache/` is not. Verify a clean clone runs.

**Done when:** a stranger can clone and reproduce one cell without asking a question.

---

## Risks, in the order they can bite

| When | Risk | Mitigation, already scheduled |
|---|---|---|
| T+0 | Deadline is sooner than assumed | Question fires at T+0; cut ladder below |
| T+0 | Drive full → build dies at hour 3 | 2-minute quota check at T+0 |
| T+3 | Corrupt cache poisons all 84 runs | 2-shard pilot + visual grid **before** the full build |
| T+3 | HF download stalls again | Fallback A (datasets-server) shares the same loop body |
| T+6.5 | Runs 4× slower than estimated | Calibration gate decides run count before committing |
| T+13 | **All four rankings agree — null result** | Write the report so a null is a reportable finding, not a rewrite. The baseline number and the four matrices stand on their own. |

## Cut ladder — apply top-down, stop as soon as it fits

1. Seeds 3 → 2 → 1
2. Epochs 40 → 25
3. Halve each arm's training set (4,000 → 2,000)
4. Drop the saliency radial spectrum (keep the image spectra)

**Never cut:** one of the four equalisation arms · the spectra or attribution · training from
scratch. All three are promised in the submitted proposal.
