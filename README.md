# Resolution Bias and Cross-Generator Detection of AI-Generated Images

Deep Learning, Tel Aviv University, 2026b. Noa Keter and Ido Josephsberg.

Detectors of AI-generated images score almost perfectly on the generator they trained on and
degrade sharply on unseen generators. In GenImage-style benchmarks each generator emits one
fixed image size while real photographs vary, so image dimensions alone identify both the
class and the generator. On this data a rule with **zero parameters** ("predict synthetic if
the image is square") reaches **0.977** accuracy on every source/target pair, beating three of
our four trained detectors cross-generator.

Removing that cue is not neutral. This project makes **the correction itself the experimental
variable**: one architecture, one dataset, one protocol, and four ways of equalizing image
size. The headline result is that the ranking of generators by difficulty is not stable under
that choice.

## Result

Seed-averaged over two seeds; each of the 49 cells per arm is balanced at n = 1,000.

| Strategy | Diagonal (in-domain) | Off-diagonal (transfer) | Gap |
|---|---|---|---|
| `center_crop` | 0.908 | 0.655 | 0.254 |
| `random_crop` | 0.892 | 0.645 | 0.247 |
| `rescale` | 0.902 | **0.563** | 0.338 |
| `pad` | 0.978 | 0.828 | 0.150 |
| square rule (0 parameters) | 0.977 | **0.977** | 0.000 |
| size-lookup rule (0 parameters) | 1.000 | 0.619 | 0.381 |

Rank agreement between strategies, Spearman rho with exact permutation p-values (n = 7, all
5,040 orderings enumerated):

| Pair | rho (p) | Pair | rho (p) |
|---|---|---|---|
| `center_crop` / `random_crop` | **+0.964 (0.003)** | `center_crop` / `pad` | -0.286 (0.556) |
| `center_crop` / `rescale` | +0.000 (1.000) | `random_crop` / `pad` | -0.321 (0.498) |
| `random_crop` / `rescale` | -0.179 (0.713) | `rescale` / `pad` | +0.643 (0.139) |

The two strategies that do not resample agree almost perfectly. Rescaling produces an
unrelated ordering, and ADM moves from first place to last.

`pad` posts the best transfer but should **not** be read as the best correction: every
generator emits square images and only real photographs have variable aspect ratio, so padding
turns the size cue into a border that marks the real class exactly. A border-only detector
ceilings at 0.977, and `pad`'s measured diagonal is 0.978.

## Install

```bash
pip install -r requirements.txt
```

`torch` is imported lazily inside the functions that need it, so the analysis and summary
paths run on a machine with no GPU and no torch installed.

## Reproducing the results

### 1. Build the cache

Decodes each row once and writes four uint8 arrays of shape `(N, 128, 128, 3)`, one per
strategy, plus per-row metadata. Run the validation split first: every reported cell comes
from it, and it carries the metadata the baselines need.

```bash
python -m src.data --split validation --out-dir /path/to/cache --keep-native
python -m src.data --split train --out-dir /path/to/cache
```

Do not pass `--keep-native` on train. The build is deterministic given `--seed`.

### 2. Train one cell

```bash
python -m src.train --cache-dir /path/to/cache --strategy center_crop --source ADM --seed 0
```

Writes `results/runs/<strategy>/<source>/seed<N>/metrics.json` and `model.pt`. Checkpoints are
gitignored; only `metrics.json` is committed.

### 3. Reproduce the full 56-run matrix

```bash
for strategy in center_crop random_crop rescale pad; do
  for source in ADM BigGAN GLIDE Midjourney SD15 VQDM Wukong; do
    for seed in 0 1; do
      python -m src.train --cache-dir /path/to/cache \
        --strategy "$strategy" --source "$source" --seed "$seed"
    done
  done
done
```

About 127 s per run, roughly 2 GPU-hours in total on a free-tier T4. `--force` re-runs a cell
whose `metrics.json` already exists.

### 4. Zero-parameter baselines

Needs metadata only, never pixels, so it runs on a laptop.

```bash
python -m src.baseline --cache /path/to/cache --out results/baseline
```

### 5. Figures and analysis

```bash
python -m src.analyze matrices --baseline-dir results/baseline --out-dir results/figures
python -m src.analyze ranking  --out-dir results/figures
```

The defaults already point at `results/runs`; passing `--results-dir results` breaks them.
`--baseline-dir` is what adds the two zero-parameter rules as extra panels.

Both read committed JSON only, so they need no GPU and no cache. The two cue-level analyses
need more:

```bash
# spectra: images only, no checkpoints, no GPU
python -m src.analyze spectra --cache-dir /path/to/cache \
  --strategies center_crop random_crop rescale pad --out-dir results/figures --n-images 500

# attribution: needs the model.pt checkpoints for the strategies listed
python -m src.analyze attribution --cache-dir /path/to/cache \
  --results-dir /path/to/checkpoints --strategies pad random_crop \
  --seed 0 --n-images 200 --device cuda --out-dir results/figures
```

### 6. Diagnostics

```bash
# train vs validation accuracy, re-scored from saved checkpoints. No retraining.
python -m src.trainval --cache-dir /path/to/cache --ckpt-root /path/to/checkpoints --owner <name>

# dump per-cell test logits (needs checkpoints; produces the .npz used below)
python -m src.logits --cache-dir /path/to/cache --ckpt-root /path/to/checkpoints --owner <name>

# combine the arms off-GPU from dumped logits. Runs on a laptop from committed files.
python -m src.ensemble --logits results/figures/logits_ido.npz

# epoch-budget control, written to results/experiments/ctrl80
python -m src.train --cache-dir /path/to/cache --strategy center_crop --source VQDM \
  --seed 0 --epochs 80 --results-dir results/experiments/ctrl80
```

## Layout

```
src/data.py        dataset download, decode, the four equalization transforms, cache build
src/model.py       CompactCNN, 1,173,473 parameters, trained from scratch
src/train.py       training loop, checkpoint selection, per-run metrics
src/baseline.py    square rule and size-lookup rule, zero parameters
src/analyze.py     transfer matrices, ranking, Spearman, attribution, spectra
src/trainval.py    train/val gap diagnostic, re-scored from checkpoints
src/logits.py      dumps per-cell test logits so arms can be combined off-GPU
src/ensemble.py    off-GPU combination of arms from dumped logits
src/rotation.py    rotation-sensitivity ablation (implemented, not run)
src/pilot.py       cache-build pilot, used to verify shards before the full build
notebooks/         Colab drivers; all logic lives in src/
results/runs/         56 metrics.json, committed
results/baseline/     14 metrics.json, committed
results/experiments/  epoch-budget and capacity controls (ctrl80, d4_fc256)
results/figures/      figures and analysis JSON
```

## Data

[Tiny-GenImage](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage), a 35,000-image
subset of GenImage (Zhu et al., 2023), CC BY-NC-SA 4.0. 17,500 real ImageNet photographs and
2,500 images from each of seven generators: ADM, BigGAN, GLIDE, Midjourney, SD 1.5, VQDM and
Wukong.

The shipped `validation` split is used as the **test** set; our validation set is carved from
10% of `train`. Per run: 3,600 train / 400 validation / 4,000 test. Model selection uses the
internal validation set only, so no reported number ever influenced training.

## Notes on reproducibility

- Seeds are fixed and the cache build is deterministic, so each reported cell reproduces from
  a clean clone.
- The analysis is reproducible without a GPU: re-running `analyze matrices` regenerates
  `transfer_matrices.json` byte-identical to the committed file, and `analyze ranking`
  reproduces `ranking.json` to floating-point precision (differences of order 1e-16 in the
  Spearman coefficients, from summation order).
- `*.pt` and `cache/` are gitignored. Every `metrics.json` is committed, which is what lets
  the analysis run without the weights.
- Reported cells come from the official test split and never influenced model selection.
- The train/val diagnostic checks itself: recomputed validation accuracy must reproduce the
  value recorded during training, and it does so exactly across all 28 runs.
