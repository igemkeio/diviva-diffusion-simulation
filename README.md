# model-2 — DivIVA / PopZ localization model

Code for a reaction–diffusion model that asks whether DivIVA, which localizes to both
poles of *E. coli*, can be made asymmetric by PopZ, which localizes to a single pole —
together with its wet-lab calibration.

This corresponds to the [Model page](https://2026.igem.wiki/keio-japan/model/) of our Wiki. Each of the four DBTL cycles is its own
directory, and **each cycle can be run independently**. Later cycles were made by copying
and modifying the code of earlier ones, so files such as `model.py` and `cluster.py` appear
under the same name in several cycles, but their contents differ (each is a snapshot
frozen for that cycle).

| Cycle | Question | Conclusion | Main script |
| --- | --- | --- | --- |
| **1** Monomer, 2 fields | Can bipolar localization be explained by putting curvature into the monomer membrane-binding rate? | No (energy short by 231×, FRAP off by 2–3 orders of magnitude, ceiling of 1/a) | `cycle1_monomer/code/model.py` |
| **2** Cluster field | Does moving curvature into the cluster lifetime resolve all three failures at once? | Yes (requires 92 molecules' worth, biphasic FRAP) | `cycle2_cluster/code/cluster.py` |
| **3** PopZ | Which mechanism to adopt: spatial occupancy (S1) or local degradation (S2)? | S1. S2 buys left/right ratio by eroding the scaffold | `cycle3_popz/code/popz.py` |
| **4** Wet calibration | Does the verdict hold after calibrating the model to measurements? | It holds. 72% agreement; $p \le 1$ confirmed | `cycle4_calibration/code/forward.py` |

## Directory layout

```
model-2/
├── cycle1_monomer/        code/  figure_code/  figures/  outputs/
├── cycle2_cluster/        code/  figure_code/  figures/  outputs/
├── cycle3_popz/           code/  figure_code/  figures/  outputs/
├── cycle4_calibration/    code/  figure_code/  figures/  outputs/  data/
└── wiki_figures/roadmap/  DBTL roadmap figure for the Wiki (HTML → PNG)
```

| | Contents |
| --- | --- |
| `code/` | The model itself, validation, and numerics. **Does not make figures** |
| `figure_code/` | Scripts that make the figures used on the Wiki. They import `code/` and write to `figures/` |
| `outputs/` | Run logs from `code/`. The numbers in the Wiki text are taken from here |
| `figures/` | Generated output. Not tracked (`.gitignore`) since it can be regenerated |

The scripts in `figure_code/` **resolve paths from their own file location**, so they
import `code/` and write to `figures/` no matter where they are run from.

## Why this layout

The model grows more complex step by step. Each cycle is built so that it **contains the
previous cycle as a special case**, and this is guaranteed by regression tests.

- In Cycle 2, setting the cluster formation rate to zero reproduces the Cycle 1 answer to 10 significant figures
- In Cycle 3, setting PopZ to zero returns the Cycle 2 answer with a relative difference of 0 (exact match)

Because of this property, **differences between cycles can be attributed solely to the
presence or absence of the added field**. This is why the code is frozen per cycle rather
than merged into a single codebase.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.10 or later. `tifffile` is only needed for reading raw images in Cycle 4.

## Running

Run from inside `code/`. Only the figure-making scripts live in `figure_code/`, and their
output goes to each cycle's `figures/`.

### Cycle 1 — Monomer two-field model

```bash
cd cycle1_monomer/code
python model.py                        # 6 validation checks (~20 s) ★ make sure this passes first
python cycle1_numbers.py               # recompute every number in the text (~1.5 min)
python ../figure_code/figures_cycle1.py   # figures A–K (~2 min). Append e.g. A C to select specific ones
```

If `model.py` fails, don't trust any of the numbers that follow. In particular, it assumes
that **V3 (mass conservation, $7.9\times10^{-14}$)** and **V4 (with $\rho=1$, polar
enrichment is exactly 1)** pass.
Example output: `outputs/cycle1_out.txt`.

### Cycle 2 — Cluster field

```bash
cd cycle2_cluster/code
python cluster.py                      # 13 validation checks (incl. V7 = regression test against Cycle 1)
python results2.py                     # numbers in the text
python ../figure_code/figures2.py      # figures 1–8. Append e.g. 3 5 to select specific ones
python ../figure_code/figures_notion.py   # ★ c2fig1–6 used on the Wiki (run after figures2.py)
python ../figure_code/fig_sideview.py  # side-view still image and GIF of the cell
```

⚠️ **`figures_notion.py` reuses the output of `figures2.py`** (5 of the 6 figures are
renamed copies; it only draws `c2fig5_bistability.png` itself). Running them in the reverse
order will fail.
Example output: `outputs/results2_out.txt`, `outputs/figures2_out.txt`.

### Cycle 3 — PopZ (S1 spatial occupancy / S2 local degradation)

```bash
cd cycle3_popz/code
python popz.py                         # 36 validation checks (~4 min) ★ assumes VR1 passes
python results_cycle3.py               # numbers used for the verdict
python ../figure_code/figures_cycle3.py     # figures 1–9 (those ending in _en are English versions)
python ../figure_code/fig_occupancy_gif.py  # time-evolution GIF at θ = 0.659 (5× left/right ratio)
```

Example output: `outputs/results_cycle3_out.txt`, `outputs/validate_cycle3_out.txt`.

### Cycle 4 — Calibration with wet data

```bash
cd cycle4_calibration/code
python forward.py                      # self-check of the forward model (projection → PSF → pixels)
python cycle5.py                       # recompute all numbers in the text (~1 min). Example output: outputs/results5_out.txt
python final_p.py                      # estimate the growth order p (α = 1.076 ± 0.061)
python s1s2_curves.py                  # writes figures/_s1s2_curves.json, read by fig_s1s2.py
python ../figure_code/fig_model_vs_image.py   # simulated vs. real images
python ../figure_code/fig_agreement.py        # agreement before/after calibration (48% → 72%)
```

The data are in `cycle4_calibration/data/`. Paths are resolved from file locations, so the
working directory doesn't matter.

## Wiki figures and the scripts that make them

★ marks figures used on the Wiki (Model page / DBTL page). The same scripts also produce
figures that were not used.

| Figure | Script |
| --- | --- |
| ★ `figE_energy.png`, ★ `figK_total_sideview.png/.gif`, `figA`–`figJ` | `cycle1_monomer/figure_code/figures_cycle1.py` |
| ★ `c2fig1_energy.png`, `c2fig2`–`c2fig6` | `cycle2_cluster/figure_code/figures_notion.py` (+ `figures2.py`) |
| ★ `c2figK_total_sideview.png/.gif` | `cycle2_cluster/figure_code/fig_sideview.py` |
| ★ `fig8_cleavable_en.png`, ★ `fig9_theta_en.png`, `fig1`–`fig7` | `cycle3_popz/figure_code/figures_cycle3.py` |
| ★ `c3figK_occupancy.png/.gif` | `cycle3_popz/figure_code/fig_occupancy_gif.py` |
| ★ `cal_fig6_model_vs_image.png` | `cycle4_calibration/figure_code/fig_model_vs_image.py` |
| ★ `cal_fig_agreement.png` | `cycle4_calibration/figure_code/fig_agreement.py` |
| ★ `cal_fig_alpha.png` | `cycle4_calibration/figure_code/fig_alpha_fit.py` |
| ★ `cal_fig_baseline.png` | `cycle4_calibration/figure_code/fig_baseline_membrane.py` |
| ★ `cal_fig5_iptg_expression.png` | `cycle4_calibration/figure_code/fig_iptg_expression.py` |
| ★ `cal_fig7_membrane_baseline.png` | `cycle4_calibration/figure_code/fig_membrane_baseline.py` |
| ★ `cal_fig2_validation.png` | `cycle4_calibration/figure_code/fig_validation.py` |
| `cal_fig_s1_theta.png`, `cal_fig_s1s2_scaffold.png` | `cycle4_calibration/figure_code/fig_s1s2.py` (run `code/s1s2_curves.py` first) |
| ★ `fig_roadmap.png` / `fig_roadmap_ja.png` | `wiki_figures/roadmap/` (`shot.py`) |

## Data

`cycle4_calibration/data/` contains the analyzed tables plus just two raw images used in
the figures. Wet-lab image analysis is not done in this repository; we use the results
from the following two repositories.

| Repository | What we used |
| --- | --- |
| [igemkeio/ecoli-imaging](https://github.com/igemkeio/ecoli-imaging) | Cell segmentation from phase-contrast and fluorescence images, and inter-channel misalignment correction |
| [igemkeio/DivIVA-anl](https://github.com/igemkeio/DivIVA-anl) | Line profiles along the long axis, per-cell metrics, and comparison with free FP (null) |

The CSVs in `data/` are copies of the output of the DivIVA-anl analysis (folder name
`anl/` at the time of the work). See `data/SOURCES.md` for provenance and md5 checksums.

| File | Contents |
| --- | --- |
| `cells_final_w5.csv` | Per-cell line profiles (line width 5 px) and metrics. 1643 cells |
| `cells_integrated.csv` | Information needed for cropping: ROI coordinates, cell length, SNR, etc. |
| `cells_profiles_wide_w3.csv` | The same, taken with a line width of 3 px (for checking line-width dependence) |
| `profiles_w3.csv` | Line-width 3 px profiles (long format, one pixel per row). Read by `cycle5.py` |
| `q2_cells.csv` | Per-cell values for Question 2 of DivIVA-anl. Read by `cycle5.py` |
| `pool_features.csv` | Features of the population before manual selection (for checking selection bias) |
| `1010_mc_Div/5/5aa.tif` | Raw image of DivIVA-mCherry (IPTG 10 µM) |
| `1010_FPnull/mche/mche4.tif` | Raw image of free mCherry (non-localizing control) |

⚠️ **The `end_mid` column in `cells_final_w5.csv` is max(poleA, poleB).**
If you need the mean of the two poles, recompute it from `poleA_mid` and `poleB_mid`.
All numbers in the text use the recomputed values.

## Adapting to your own system

| What to change | Where |
| --- | --- |
| Cell geometry ($R$, $L$) | `Par` in each cycle's `model.py` |
| Rate constants | Same as above (from Cycle 2 on, `Par` in `cluster.py`) |
| PopZ coverage ($\theta$, $r_{popz}$) | `Par.theta` / `Par.r_popz` in `popz.py` |
| Pixel size / PSF | `PX_UM` in `forward.py`, and `PSF` in each script |

$\rho$, $\rho_n$, $k_{on}$, and $k_{grow}$ are **not independent parameters**. They are
dependent quantities back-calculated each time from the measured partition ratio and polar
fraction, so do not edit them directly.

## Checking that results are valid

Every cycle is designed to pass its validation checks before producing results.

| | Cycle 1 | Cycle 2 | Cycle 3 |
| --- | --- | --- | --- |
| Number of checks | 6 | 13 | 36 |
| Mass conservation / balance | $7.9\times10^{-14}$ | $6.1\times10^{-13}$ | $2.0\times10^{-16}$ |
| Regression against previous cycle | — | $1.5\times10^{-11}$ | relative difference 0 |

## Known issues

- `cycle5.py` reads `data/` by default. If the environment variable `WET_DIR` points to a
  DivIVA-anl output directory (one containing `extracted/` and `results/`), it reads that instead
- The parentheses in the save step at the end of `fit_R_short_axis.py` were broken and have
  been fixed (2026-10-21). The printed short-axis fit values (R and PSF) are unchanged

## Known limitations

- **The geometry has no septum.** The calibration target "66% of cluster molecules are at
  the poles/septum" is attributed entirely to the poles, so the polar effect is overestimated
- **There is no cluster size distribution.** Clusters are treated as a surface density, so
  the granular structures seen in real images cannot be reproduced
- **All rate constants come from *B. subtilis*.** DivIVA does not exist in *E. coli*, so
  there are no native values to substitute
- **There are no independent replicates on the wet side.** All conditions differ in IPTG
  concentration; they are not replicates

## License

MIT.
