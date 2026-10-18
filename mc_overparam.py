"""Arm O: the overparametrised region, c > 1, with signal present.

    python mc_overparam.py          full run   (output/armO2.json, about 45 minutes on 4 cores)
    python mc_overparam.py quick    a few replications of every cell, to test the installation
    python mc_overparam.py diag     the 2x2 random-feature diagnostic (output/armO2_diag.json)
    python mc_overparam.py lowsig   panel B at R^2 = 0.005, the equity calibration
                                    (output/armO2_lowsig.json)

Why this arm exists.  Every size result in the paper at c > 1 (arm D, tab:mc-complexity) is
at R^2 = 0, the theta = 0 design, where H_cov and H_CC coincide and Clark-West is itself
correctly sized.  So no design shows the case the corrected test is for -- signal present,
mu_CC = 2E[fg] - E[g^2] <= 0, the bigger model overfitting -- once P exceeds T.  This arm
supplies it, in two parts.

  Part G  generic features.  Gaussian rows, Sigma = I, dense signal theta in the span of the
          features, R^2 = 0.10.  c in {0.5, 1.5, 2, 5, 10} at the feasible cell
          (T, n, b) = (12, 8000, 18), so P runs to 120.  c = 0.5 is the anchor back to the
          main grid.  Penalty held at ztilde = z/c in {0.1, 0.001}; the second is the
          near-ridgeless, minimum-norm-interpolator corner.

  Part F  random Fourier features.  The raw predictors are d = 5 AR(1) series with rho = 0.3,
          exactly arm B (tab:mc-features); the signal is LINEAR in the raw predictors, so the
          feature map is misspecified for it, as in the random-feature return-prediction
          literature.  gamma = 0.3 and z = 0.5, the arm B column with the largest Psihat, at
          P = 200 (c = 16.7) and P = 2000 (c = 166.7).

Rows.  For each cell the pilot (separate seed, no look-ahead into the replications) measures
E[fg] and E[g^2] for the fitted increment and sets a_CC = 2E[fg]/E[g^2], so that scaling the
increment by a gives mu_CC(a) = 2a E[fg] - a^2 E[g^2].  Three rows, all on the SAME simulated
paths (the seed depends on the cell, not the row), so they are paired:

    H0-CC   a = a_CC      mu_CC = 0.  H_CC holds; H_cov is false (E[fg] > 0), so CW
                          rejecting here is CW testing a different null, not an error of CW.
    alt     a = a_CC / 2  mu_CC at its maximum E[fg]^2/E[g^2].  Power.
    raw     a = 1         the forecast as fitted.  Classified null or alternative by the sign
                          of the realised mean of mu_CC, which is reported with its MC s.e.

The rescaling is the device of pvalue_grid (H0-CC row of tab:mc-grid); F4 there asks
0.3 <= a_CC <= 3 and is recorded here too.

Statistics are mc_study.run_path unchanged, so every number is computed by the same code as
tab:mc-size and tab:mc-complexity.  This file writes only output/armO2.json and its run log;
it touches no other output.  Cells run in parallel, one process per cell, each with a seed
derived from the cell tuple, so results do not depend on the number of workers.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")                    # one BLAS thread per worker process

import sys, math, time, datetime
import numpy as np
from multiprocessing import Pool

import mc_study as mc

T, N, B = 12, 8000, 18
R2 = 0.10
CRIT = 1.645

G_CS  = [0.5, 1.5, 2.0, 5.0, 10.0]
G_ZTS = [0.1, 0.001]
F_D, F_RHO, F_GAMMA, F_Z = 5, 0.3, 0.3, 0.5
F_PS  = [200, 2000]

ROWS = (("H0-CC", 1.0), ("alt", 0.5), ("raw", None))   # multiplier on a_CC; None means a = 1


# ------------------------------------------------------------------ path generators
def path_G(P, g, theta, r2=R2):
    return mc.gen_path(T, P, N, 0.0, r2, g, theta=theta)


def path_F(P, g, beta, rho=F_RHO, signal="raw", r2=R2):
    """Raw AR(1) predictors -> RFF features, with a signal at R^2 = r2 (0.10 by default).

    Same construction as arm B (mc.ar1_rows, mc.rff), with a signal added.  The timing
    convention matches mc.gen_path: y_t = f_{t-1} + eps_t, features dated t-1.

    signal="raw"   f linear in the raw predictors (beta in R^d), outside the feature span.
    signal="span"  f = feats @ beta (beta in R^P), inside the span.  The map W, b is redrawn
                   on every path, as in arm B, so f is rescaled on each path to variance R2;
                   the scale is one number per path of 8,014 rows.
    """
    m = N + T + 2
    Rraw = mc.ar1_rows(m, F_D, rho, g)
    feats = mc.rff(Rraw, P, F_GAMMA, g)
    if signal == "span":
        f = feats @ beta
        f *= math.sqrt(r2) / f.std()
    else:
        f = Rraw @ beta                           # Var(f) = R2 * sigma^2, sigma^2 = 1
    eps = g.standard_normal(m)
    y = np.roll(f, 1) + eps; y[0] = eps[0]
    return y, feats, f


def signal_G(P, g, r2=R2):
    th = g.standard_normal(P); th /= np.linalg.norm(th)
    return th * math.sqrt(r2)


def signal_F(g, r2=R2):
    b = g.standard_normal(F_D); b /= np.linalg.norm(b)
    return b * math.sqrt(r2)                      # raw rows have unit variance


# ------------------------------------------------------------------ pilot
def pilot(make_path, z, g, reps=50, stride=7):
    """E[fg], E[g^2] for the increment as fitted (a = 1).  Same algebra as run_path."""
    I = np.eye(T); one = np.ones(T); M = I - np.outer(one, one) / T
    Efg = Eg2 = 0.0; K = 0
    for _ in range(reps):
        y, S, f = make_path(g)
        for i in range(0, N, stride):
            X = S[i:i+T]; F = f[i:i+T]; s = S[i+T]
            Bm = np.linalg.inv(z*I + X @ X.T / T)
            a = (M @ (Bm @ (X @ s))) / T
            gi = a @ F
            # f relative to the benchmark: E[y_t|F_{t-1}] - b_t'F_{t-1} with b_t = iota/T
            # (eq:wedge-general).  run_path's mu_CC uses the raw f_t and omits -2E[fbar g];
            # that term is zero up to noise in the linear Gaussian designs but not under
            # random Fourier features, where it moved the "H0-CC" rows off the null.
            Efg += (f[i+T] - F.mean()) * gi; Eg2 += gi * gi; K += 1
    return Efg / K, Eg2 / K


# ------------------------------------------------------------------ one cell
def run_cell(spec):
    part, key, P, z, R = spec["part"], spec["key"], spec["P"], spec["z"], spec["R"]
    # R^2 travels in the spec rather than as a module global, so that it reaches worker
    # processes started by "spawn" (the macOS default), which re-import this module
    r2 = spec.get("R2", R2)
    t0 = time.time()
    gs = np.random.default_rng(mc.seed_of("O-signal", *key))
    if part == "G":
        sig = signal_G(P, gs, r2)
        make = lambda g: path_G(P, g, sig, r2)
    elif part in ("Fd", "FL"):
        rho_d, sg = spec["rho"], spec["signal"]
        if sg == "span":
            sig = gs.standard_normal(P)
        else:
            sig = signal_F(gs, r2)
        make = lambda g: path_F(P, g, sig, rho=rho_d, signal=sg, r2=r2)
    else:
        sig = signal_F(gs, r2)
        make = lambda g: path_F(P, g, sig, r2=r2)

    Efg, Eg2 = pilot(make, z, np.random.default_rng(mc.seed_of("O-pilot", *key)),
                     reps=spec.get("pilot_reps", 50))
    a_cc = 2 * Efg / Eg2 if Eg2 > 0 else float("nan")

    g = np.random.default_rng(mc.seed_of("O", *key))
    acc = {lab: [] for lab, _ in ROWS}
    for _ in range(R):
        y, S, f = make(g)
        for lab, mult in ROWS:
            a = 1.0 if mult is None else mult * a_cc
            acc[lab].append(mc.run_path(y, S, f, T, z, B, a, 1.0, both=True))

    se = lambda p: float(math.sqrt(max(p * (1 - p), 0.0) / R))
    out = []
    for lab, mult in ROWS:
        A = {k: np.array([r[k] for r in acc[lab]]) for k in acc[lab][0]}
        scale = 1.0 if mult is None else mult * a_cc
        mu = A["mu_CC"]
        # E[dCC] with the oracle sigma^2 = 1 equals the paper's mu_CC exactly
        # (eq:wedge-general, b'a = 0, iid innovations).  Unbiased, noisier than mu_CC.
        dcc = A["mspe_gap"] + A["Psihat"]
        row = dict(part=part, row=lab, key=list(key), T=T, n=N, b=B, P=P, c=P / T, z=z,
                   ztilde=z / (P / T), R2=r2, R=R, scale=float(scale), a_CC=float(a_cc),
                   F4=bool(0.3 <= a_cc <= 3.0), pilot_Efg=float(Efg), pilot_Eg2=float(Eg2),
                   Psi_closed=float(mc.psi_closed(P / T, z)),
                   Psihat=float(A["Psihat"].mean()),
                   theta_T=float(np.nanmean(A["theta_T"])),
                   mu_CC_raw=float(mu.mean()), mu_CC_raw_se=float(mu.std() / math.sqrt(R)),
                   mu_CC=float(dcc.mean()), mu_CC_se=float(dcc.std() / math.sqrt(R)),
                   Efg=float(A["Efg"].mean()), Eg2=float(A["Eg2"].mean()),
                   mspe_gap=float(A["mspe_gap"].mean()),
                   rej_DM=float(np.mean(A["t_DM"] > CRIT)),
                   rej_CW=float(np.mean(A["t_CW"] > CRIT)),
                   sd_t_CW=float(A["t_CW"].std()))
        for s2 in ("oracle", "diff"):
            sfx = "__" + s2
            t = A["t_CC_hac" + sfx]
            p = float(np.mean(t > CRIT))
            row[f"rej_CC_{s2}"] = p
            row[f"rej_CC_samp_{s2}"] = float(np.mean(A["t_CC_samp" + sfx] > CRIT))
            row[f"ratio_{s2}"] = float(np.nanmean(A["ratio" + sfx]))
            row[f"sd_t_CC_{s2}"] = float(t.std())
            row[f"mc_se_{s2}"] = se(p)
        row["rej_CC_w"] = float(np.mean(A["t_CC_w"] > CRIT))
        if part == "F":
            row.update(d=F_D, rho_x=F_RHO, gamma=F_GAMMA, signal="raw")
        elif part in ("Fd", "FL"):
            row.update(d=F_D, rho_x=spec["rho"], gamma=F_GAMMA, signal=spec["signal"])
        else:
            row.update(rho_x=0.0)
        out.append(row)
    return {"key": list(key), "rows": out, "minutes": (time.time() - t0) / 60}


def cells(quick):
    R = 40 if quick else 1000
    n_ = None
    specs = []
    for c in (G_CS if not quick else [1.5, 5.0]):
        P = int(round(c * T))
        for zt in (G_ZTS if not quick else [0.001]):
            z = zt * (P / T)
            specs.append(dict(part="G", key=("G", c, zt), P=P, z=z, R=R))
    for P in (F_PS if not quick else [200]):
        specs.append(dict(part="F", key=("F", F_D, P, F_GAMMA, F_Z), P=P, z=F_Z, R=R))
    return specs


def diag_cells(R=1000):
    """Why do the random-feature cells over-reject with the oracle sigma^2?

    The F cell differs from the generic G cells in two ways at once: persistent raw
    predictors (rho_x = 0.3) and a signal outside the feature span.  The 2x2 below varies
    them one at a time at P = 200; (0.3, raw) re-runs the F cell under a new seed as a
    check that the 0.143 is not a seed artefact.
    """
    P = 200
    return [dict(part="Fd", key=("Fd", rho, sg, P), P=P, z=F_Z, R=R, rho=rho, signal=sg)
            for rho in (0.0, 0.3) for sg in ("raw", "span")]


def lowsig_cells(R=1000, r2=0.005):
    """Panel B again at the monthly equity calibration R^2 = 0.005 of sec:fixedT.

    The four P = 200 designs of the diagnostic and the P = 2,000 cell.  With so little
    signal E[fg] is small and a_CC = 2E[fg]/E[g^2] noisy, so the pilot uses 200 paths
    instead of 50; whether each H0-CC row sits on the null is read off mu_CC and its s.e.
    """
    out = [dict(part="FL", key=("FL", r2, rho, sg, 200), P=200, z=F_Z, R=R, rho=rho,
                signal=sg, R2=r2, pilot_reps=200)
           for rho, sg in ((0.3, "raw"), (0.0, "raw"), (0.3, "span"), (0.0, "span"))]
    out.append(dict(part="FL", key=("FL", r2, 0.3, "raw", 2000), P=2000, z=F_Z, R=R,
                    rho=0.3, signal="raw", R2=r2, pilot_reps=200))
    return out


def show(res):
    for r in res["rows"]:
        print(f"  {r['part']} P={r['P']:5d} c={r['c']:7.2f} zt={r['ztilde']:7.4f} {r['row']:6s} "
              f"a={r['scale']:6.3f} (a_CC={r['a_CC']:.3f} F4={r['F4']}) "
              f"Psihat={r['Psihat']:6.3f} mu_CC={r['mu_CC']:+.4f}({r['mu_CC_se']:.4f}) "
              f"| DM {r['rej_DM']:.3f} CW {r['rej_CW']:.3f} "
              f"CC orc {r['rej_CC_oracle']:.3f} diff {r['rej_CC_diff']:.3f}")
    print(f"  [{res['minutes']:.1f} min]", flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "full"
    quick = what == "quick"
    diag = what == "diag"
    lowsig = what == "lowsig"
    # v2 file names: the first run (armO.json, armO_diag.json) used raw f in the pilot and
    # is kept for the audit trail
    name = ("armO2_quick.json" if quick else "armO2_diag.json" if diag
            else "armO2_lowsig.json" if lowsig else "armO2.json")
    sys.stdout = mc.Tee(mc.opath(f"run_O2_{what}_{datetime.datetime.now():%Y%m%d_%H%M}.log"))
    print(f"# mc_overparam {what}  started {datetime.datetime.now():%Y-%m-%d %H:%M}  -> {mc.OUTDIR}/")
    mc.acquire_lock("O2" + ("q" if quick else "d" if diag else "l" if lowsig else ""))
    done = mc.ckpt_load(name, []) or []
    have = {tuple(r["key"]) for r in done}
    pool_ = diag_cells() if diag else lowsig_cells() if lowsig else cells(quick)
    todo = [s for s in pool_ if tuple(s["key"]) not in have]
    print(f"  {len(have)} cells done, {len(todo)} to run; R = {todo[0]['R'] if todo else '-'}")
    # slowest first, so the P = 2000 cells do not finish last on an idle pool
    todo.sort(key=lambda s: -s["P"])
    workers = int(os.environ.get("MC_WORKERS", min(4, os.cpu_count() or 1)))
    with Pool(workers) as pool:
        for res in pool.imap_unordered(run_cell, todo):
            done.extend(res["rows"])
            mc.ckpt_save(name, done)
            show(res)
    print(f"\n# finished in {(time.time() - mc._T0) / 60:.1f} min.  {len(done)} rows -> {mc.opath(name)}")
