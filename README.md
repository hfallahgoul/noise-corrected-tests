# Noise-Corrected Tests for High-Dimensional Forecasting Models — replication package

Hasan Fallahgoul, Monash University.

This package reproduces every generated table and figure in the paper and contains a small
module, `ccpa.py`, that runs the noise-corrected test on any pair of nested forecasts.
It ships the code and the stored output the tables are built from. Rebuilding the tables
from that output takes about two minutes; rerunning the simulations and the applications
from scratch takes several hours and, for the realized-variance results, network access to
the Risk Lab data service.

## Quick start

```
pip install -r requirements.txt
python verify.py
```

`verify.py` runs three checks and prints one line for each:

```
stored output intact
all generated tables reproduce exactly from output/
ccpa agrees with the simulation engine
```

The first compares every stored output file with `MANIFEST.sha256` (plain `sha256sum`
format, so `sha256sum -c MANIFEST.sha256` works without this code). The second rebuilds all
generated tables from the stored output and compares them byte for byte with the copies in
the paper, which a clone finds in `reference_tables/`. The third runs `tests/test_ccpa.py`, which computes the oracle, difference and
window-residual versions of the corrected statistic with `ccpa.py` on a simulated path and
checks them against the vectorised engine used in the simulations.

`python build_tables.py` writes the tables and figures. Inside the author's tree, where the
paper sits at `../Latex`, it writes there; a clone on its own writes to `./tables` and
`./figures`. `MC_TEX`, `MC_FIG` and `MC_OUT` override the three directories.

## Using the test

The statistic needs, at each forecast origin, the two forecasts, the realised target, the
squared length $\hat\Psi_t=\|a_t\|^2$ of the weights the increment $\hat y_{2,t}-\hat y_{1,t}$
places on the training targets, and an estimate of the innovation variance.

```python
import ccpa

# a: (n, T) increment weights, one row per forecast origin. For a ridge on top of a
# benchmark that projects on regressors H, ccpa.ridge_increment_weights(X, s, z, H) gives a row.
psi = ccpa.psihat(a)

# window-residual variance, one value per origin (thm:feasible-w)
s2w = ccpa.sigma2_window(Ywin, Hwin)            # Hwin=None for a rolling-mean benchmark
cc_w = ccpa.cc_test(y, yhat1, yhat2, psi, s2w, bandwidth=b)

# difference estimator with its influence-function term (thm:feasible)
s2d, h = ccpa.sigma2_difference(y)
cc_d = ccpa.cc_test(y, yhat1, yhat2, psi, s2d, bandwidth=b, influence=h)

ccpa.intersection(cc_d, cc_w)                   # reject only if both reject
ccpa.sanity_ratio(yhat1, yhat2, psi, s2w)       # report beside every statistic
```

For applied work the same test is available as a pip-installable package,
[`nctest`](https://github.com/hfallahgoul/nctest) (`pip install nctest`), which also runs the
Diebold–Mariano and Clark–West tests on their own. `ccpa.py` stays in this package because
`verify.py` checks it against the simulation engine.

The long-run variance is the Bartlett estimator with sample autocovariances normalised by
$n$, as in the paper. The bandwidth should satisfy $T \ll b \ll \sqrt{n/T}$; the simulations
use $b=\max(18, \lceil 1.5T\rceil)$.

## What produces what

Labels are the paper's `\label`s, which do not move when sections are renumbered.
"Checked" means `build_tables.py --check` rebuilds the table byte for byte.

### Section 5 and the simulation appendix

| Paper object | Script that produces the output | Stored output | Checked |
|---|---|---|---|
| `tab:mc-conform`, `tab:mc-conform-n`, figure `fig-mc-conform` | `mc_conform.py --run` | `output/mc_conform/` | tables yes |
| `tab:mc-size`, `tab:mc-power`, `tab:mc-closedform` | `mc_study.py A` | `output/armA.json` | yes |
| `tab:mc-grid`, figures `pgrid_*` | `mc_study.py grid` simulates; `build_tables.py` draws the figures from the stored JSON | `output/pgrid_*.json` | table yes |
| `tab:mc-features` | `mc_study.py B` | `output/armB.json` | yes |
| `tab:mc-breakdown` | `mc_study.py C` | `output/armC.json` | yes |
| `tab:mc-signal` | `mc_study.py S` | `output/armS.json` | yes |
| `tab:mc-complexity` | `mc_study.py D` | `output/armD.json` | yes |
| `tab:mc-overparam` (panels A and B, the size block and the diagnostic rows) | `mc_overparam.py`, `mc_overparam.py diag` | `output/armO2.json`, `output/armO2_diag.json` | yes |
| `tab:mc-overparam` (panel C, $R^2=0.005$) | `mc_overparam.py lowsig` | `output/armO2_lowsig.json` | yes |
| `tab:window` | `mc_window.py` | `output/mc_window/` | yes |
| `tab:intersection` | `mc_intersection.py` | `output/mc_intersection/` | yes |
| `tab:kernel-check`, `tab:delta` | `mc_kernel_delta.py` | `output/mc_kernel/` | yes |
| `tab:kappa` | `mc_kappa_tables.py` (panels), `mc_har_diagnosis.py` (window sweep in the caption) | `output/mc_har/` | yes |
| `tab:cw-har` | `ldv_scope/cw_under_har.py`, `ldv_scope/verify.py` | `ldv_scope/output/` | yes |
| `tab:har-power` | `ldv_scope/har_size_power.py`, `ldv_scope/verify.py` | `ldv_scope/output/` | yes |
| `tab:har-omitted` | `ldv_scope/har_omitted.py` | `ldv_scope/output/` | yes |
| `tab:pathfunc` | `ldv_scope/kappa_pathfunc.py` | `ldv_scope/output/` | yes |
| `prop:dof` (numbers in the text) | `mc_dof.py` | `output/mc_dof/` | `mc_dof.py --check` |
| `app:signal-feasible` (numbers in the text) | `omitted_feasible.py` | `output/armS_feasible.json` | no |
| `eq:wedge-general` (numbers in Section 3.3) | `mc_fitted_benchmark.py` | `output/mc_fitted_benchmark/` | no |
| window dependence of $\mu_{\mathrm{CC}}$ (numbers in the text of `sec:mc-window`) | `omitted_window.py` | `output/omitted_window.json` | no |

### Section 6 and the empirical appendix

| Paper object | Script | Stored output | Checked |
|---|---|---|---|
| `tab:rv-main` | `rv_application.py --ticker 84398` | `output/rv_tables/rv_rows.json` | yes |
| SPY five-estimator numbers in the text of Section 6.3 | `rv_application.py --ticker 84398 --rv-col c5 --est-cols c2,c6,c7,c10,c11 --est-lags 5 --tag multiest` | `output/rv_tables/rv_rows_multiest.json` | no |
| figure `fig-rv-data` | `fig_rv_data.py` | `output/rv_series/84398.npz` | no |
| figure `fig-rv-stats` | `rv_application.py` | `output/rv_figures/` | no |
| `tab:rv-cross-psi`, `tab:rv-cross-stats`, figure `fig-rv-cross` | `rv_cross_section.py` | `output/rv_cross/` | tables yes |
| `tab:rv-multiest` | `rv_cross_multiest.py` | `output/rv_cross_multiest/panel.json` | yes |
| figure `fig-rv-disagree` | `fig_rv_disagree.py` | `output/rv_cross/`, `output/rv_cross_multiest/panel.json` | no |
| `tab:rv-predict`, `tab:rv-predict-robust` | `rv_window_store.py`, then `rv_window_bootstrap.py` | `output/rv_window_bootstrap_*.json` | yes |
| `tab:voc-psi`, `tab:voc-grid`, figure `fig-voc-psi` | `voc_application.py` | `output/voc_tables/`, `output/voc_figures/` | tables yes |
| `tab:voc-weighted` | `voc_weighted.py` | `output/voc_tables/voc_weighted.json` | yes |

`build_tables.py` also copies the figures the application scripts draw into the paper's figure
directory, so one run of it leaves no stale figure behind.

The layout of `tab:rv-main` is defined once, in `rv_application.write_output`; `build_tables.py`
calls it on the stored rows, so a fresh run and the rebuilt table cannot differ in format.
`fig_rv_data.py` and `fig_rv_disagree.py` read only stored files and run in seconds.

`rv_window.py` is the first, network-bound pass of the split-sample exercise; its per-asset
results are in `output/rv_window/`.

## Running from scratch

The Monte Carlo scripts are deterministic: seeds are derived from the design tuple, and each
arm checkpoints after every cell, so an interrupted run resumes and a completed run
reproduces the stored output exactly. Rerunning them is not needed to verify the paper.

```
python mc_conform.py --run    # the design of tab:mc-conform, about 30 minutes
python omitted_window.py      # the window-dependence numbers of sec:mc-window, about 5 minutes
python mc_study.py A          # 9-16 minutes each for A, B, C; arm D about 110 minutes
python mc_study.py grid       # the p-value grid of tab:mc-grid alone, about 15 minutes
python mc_window.py --run --signal spiked --R2 0.20 --T 12,18,24,36,48 --reps 1000
python mc_window.py --run --signal spiked --R2 0.10 --T 12,18,24,36,48,60,72,96
python mc_window.py --run --signal spiked --R2 0.05 --T 12,24,48,96,144
python mc_window.py --run --signal spiked --R2 0.02 --T 12,48,144,300,600
python mc_window.py --run --signal dense  --R2 0.20 --T 12,18,24,36,48
python mc_window.py --run --signal dense  --R2 0.10 --T 12,18,24,36,48,60,72,96
for s in 1 2 3 4 5; do for rho in 0.3 0.95; do for r2 in 0.02 0.1; do
  python mc_intersection.py $rho $r2 12 200 $s; done; done; done
python mc_overparam.py        # about 45 minutes on four cores
python mc_overparam.py diag   # about 15 minutes
python mc_overparam.py lowsig # about 30 minutes
```

`mc_overparam.py quick` runs a few replications of every cell to test an installation; its
output is not used by the paper. The run logs of the three full calls are stored next to their
output as `output/run_O2_*.log`.

The realized-variance scripts download series from the Risk Lab service of Dacheng Xiu
(`dachxiu.chicagobooth.edu`) at run time, one request per series, so their results depend on
the date of the download. The panels used in the paper are stored in `output/rv_cross/` and
`output/rv_cross_multiest/`; they hold computed statistics, not the underlying series. The one exception is
`output/rv_series/84398.npz`, the SPY series (Risk Lab column `c5`, five-minute subsampled
realized volatility from trades) behind `fig-rv-data`. The
split-sample exercise first stores per-date forecasts for the 983 series of
`panel_hac_check.json` (about 409 MB, not shipped) and then bootstraps from that store:

```
python rv_window_store.py --workers 8
python rv_window_bootstrap.py --variants base
python rv_window_bootstrap.py --variants third,ztilde
python rv_window_bootstrap.py --variants same
python rv_window_bootstrap.py --variants base,same --block 126
```

The bootstrap draws from one random stream per call, in the order the variants are listed,
so these four calls, in this form, are what reproduce the stored files.

`voc_application.py` needs only `data/PredictorData2021.csv`, the Goyal–Welch predictor
file, and runs in under a minute. `voc_weighted.py` adds the heteroskedasticity-weighted
correction on the same data and features; it reads nothing `voc_application.py` writes, and
also runs in under a minute.

## What the package does not reproduce

- `tab:ldv-scope` is assembled by hand. Panels (a), (b), (d) and (e) and the nesting figures in
  its caption can be read from `ldv_scope/output/kappa_map*.json`, `kappa_repair.json` and
  `nest_check.json`; the penalty sweep behind panel (c) is not in the package.
- The window, penalty and jump sweeps in `ldv_scope/output/kappa_pathfunc_{zt,robust,jumps}.json`
  were produced with the `sweep()` helper of `kappa_pathfunc.py` in an interactive session, and
  the exact calls were not recorded. `tab:pathfunc` is rebuilt from those files.
- The numerical checks reported inside Appendix A (`tab:psi-check`, `tab:kurt-check`,
  `tab:size-check`, `tab:stud-composite`, `tab:termA`, `tab:ned-fail`) are not produced by a
  script in this package.
- A few numbers quoted in the text come from checks run outside this package: the $28$
  percent by which the closed form misstates the realised forecast variance at $\rho_x=0.9$
  (Section 4.5 and Appendix A), the factor of $68$ at $T=8$ and $\rho_x=0.9$ in the footnote
  on the effective window (Section 5), the agreement to within $0.015$ in `rem:bias-size`, and
  the $98$ and $77$ percent variance recovery in `rem:hac-finite`.
- `tab:conditions`, `tab:gates` and `tab:mc-map` state conditions or summarise designs and
  contain no computed numbers.
- `rv_application.py` normalises the sample autocovariances in its long-run variance by
  $n-\ell$ rather than $n$. The relative difference is at most $b/n$, under $0.2$ percent at
  the bandwidth and sample size of Section 6.

## Changes

**v1.3.** Two scripts are new. `mc_conform.py` runs the one design that meets every condition of the
limit theory with $\sigma^2$ known, a zero benchmark with i.i.d. Gaussian feature rows and
innovations, and reaches each null without rescaling the forecast (`tab:mc-conform`,
`tab:mc-conform-n`, `fig-mc-conform`). `omitted_window.py` produces the window-dependence
numbers of `sec:mc-window`. `reference_tables/` now holds the two new tables.

The SPY sample is now stated as the one the stored results were computed on, $7{,}711$
trading days from 2 January 1996 to 9 October 2026 (the caption of `tab:rv-main` said
$7{,}678$ days), and `fig_rv_data.py` draws the series to the same date instead of 19 August
2026. `tab:rv-multiest` now leads with the unscreened panel and keeps the screened one as
panel B, and `fig_rv_disagree.py` uses the unscreened panel as well. The remaining table
changes are to captions, notation and column spacing.

**v1.2.** The simulations with a rolling-mean benchmark measured $\mu_{\mathrm{CC}}=2E[fg]-E[g^2]$
against the raw signal $f_{t-1}$. The benchmark's systematic error is $f_{t-1}$ less its window
mean, and that is what the estimand pairs with $g$. `mc_study.py` (the pilot that sets the
$H_0^{CC}$ and $H_0^{MSPE}$ rescalings, and the reported $\mu_{\mathrm{CC}}$ and $E[fg]$),
`mc_window.py`, `mc_intersection.py` and `pvalue_grid.py` now use it, and the pilot is four
times larger. Seeds are unchanged. `tab:mc-grid`, its figures and `tab:intersection` were
rerun, and the $\mu_{\mathrm{CC}}$ and $2E[fg]$ columns of `tab:mc-signal` and `tab:window`
recomputed; every rejection frequency outside the rescaled rows is unchanged. The p-value
grid now stores the MSPE gap in its JSON, so `build_tables.py` no longer reads a run log.
The two equity tables now state $1{,}141$ months and $1{,}128$ forecasts.
`output/armS_feasible.json`, behind numbers quoted in the text, still records the old
$\mu_{\mathrm{CC}}$ field, which the paper does not use; its rejection frequencies are unaffected.

## Software

Parts of the simulation and replication code were written with the assistance of Claude
(Anthropic). All code was reviewed and run by the author, and every generated table is
reproduced from stored output by `verify.py`.

Tested with Python 3.10 and 3.13 and the versions pinned in `requirements.txt`.

## Licence and citation

MIT licence (`LICENSE`). Cite the paper; `CITATION.cff` has the details.
