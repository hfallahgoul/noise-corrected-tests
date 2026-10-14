"""Main Monte Carlo study (Section 5 and Appendix B.7-B.8).

    python mc_study.py A      size, power and the 3x3 p-value grids   (tab:mc-size, tab:mc-power,
                              tab:mc-closedform, tab:mc-grid, fig pgrid_*)
    python mc_study.py B      feature maps at fixed c                  (tab:mc-features)
    python mc_study.py C      breakdown designs                         (tab:mc-breakdown)
    python mc_study.py S      signal structure: spiked, dense, omitted  (tab:mc-signal)
    python mc_study.py D      complexity sweep through c = 1            (tab:mc-complexity)
    python mc_study.py all
    python mc_study.py quick  a few replications of everything, to test the installation

Cells are indexed by the target value of Psi, and the ridge penalty z is solved for.  The
feasibility conditions F1-F4 of tab:gates are checked and printed for every cell; failing
cells are marked, not dropped.  The innovation variance is run both as the oracle and as the
difference estimator.  All forecasts are computed in the dual (T x T) form, so the cost does
not grow with P.  Seeds are derived from the design tuple (seed_of), so every arm is
reproducible cell by cell and resumes from its checkpoint.

Output: output/arm{A,B,C,D,S}.json, output/pgrid_*.json, and a run log.
Runtime on a laptop: arms A-C 9-16 minutes each, arm D about 110 minutes.
"""
import numpy as np, math, sys, json, os, time, datetime, zlib


# ------------------------------------------------------------ reproducible seeds
def seed_of(*parts):
    """Deterministic seed from arbitrary arguments.

    Python's built-in hash() is salted per process for strings, so
    default_rng(hash(("C1", 0.2)) % 2**32) draws a DIFFERENT stream in every run
    and, with the checkpoint-and-resume logic below, a run that is interrupted
    silently fills the remaining cells from a different stream than the completed
    ones.  crc32 of the repr is stable across processes and machines.
    """
    return zlib.crc32(repr(parts).encode()) & 0xFFFFFFFF

# ------------------------------------------------------------ output plumbing
OUTDIR = os.environ.get("MC_OUT", "output");
os.makedirs(OUTDIR, exist_ok=True)
_T0 = time.time()

def opath(name):
    return os.path.join(OUTDIR, name)

def save_json(name, obj):
    with open(opath(name), "w") as fh:
        json.dump(obj, fh, indent=1, default=float)
    print(f"  [saved] {opath(name)}")

def load_json(name, default=None):
    try:
        with open(opath(name)) as fh: return json.load(fh)
    except Exception: return default

# --- checkpoints -------------------------------------------------------------
# Default is OUTDIR, so everything a run produces lands in one folder.  Set
# MC_CKPT to keep progress somewhere durable if you intend to move MC_OUT around;
# writes are atomic either way, so a kill mid-write cannot truncate the file.
# MC_FRESH=1 ignores any existing checkpoint and recomputes every cell.
CKPT  = os.environ.get("MC_CKPT", OUTDIR)
FRESH = os.environ.get("MC_FRESH", "") not in ("", "0", "no", "false")
os.makedirs(CKPT, exist_ok=True)

def ckpt_save(name, obj):
    tmp = os.path.join(CKPT, name + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=float)
    os.replace(tmp, os.path.join(CKPT, name))     # atomic: no truncated file on kill

def ckpt_load(name, default=None):
    if FRESH:
        print(f"  [checkpoint] MC_FRESH set - ignoring any existing {name}")
        return default
    p = os.path.join(CKPT, name)
    try:
        with open(p) as fh: return json.load(fh)
    except Exception: return default

def acquire_lock(tag):
    """Refuse to start if another run is already going.  Stale locks are cleared."""
    lk = os.path.join(CKPT, f".lock_{tag}")
    if os.path.exists(lk):
        try:
            pid = int(open(lk).read().strip()); os.kill(pid, 0)
            print(f"\n*** REFUSING TO START: '{tag}' is already running as pid {pid}.")
            print(f"*** Kill it first, or delete {lk} if you know it is dead.\n")
            sys.exit(1)
        except (ValueError, ProcessLookupError, PermissionError):
            print(f"  [lock] clearing stale lock {lk}")
    with open(lk, "w") as fh: fh.write(str(os.getpid()))
    import atexit; atexit.register(lambda: os.path.exists(lk) and os.remove(lk))

class Tee:
    """Mirror stdout to a log file so a long run is never lost to a closed terminal."""
    def __init__(self, fn):
        self.f = open(fn, "a"); self.o = sys.__stdout__
    def write(self, x): self.o.write(x); self.f.write(x); self.f.flush()
    def flush(self): self.o.flush(); self.f.flush()

def progress(i, total, label="", every=1):
    if i % every and i != total: return
    el = time.time() - _T0
    rate = i/max(el, 1e-9); eta = (total-i)/max(rate, 1e-9)
    bar = "#"*int(30*i/total) + "."*(30-int(30*i/total))
    sys.stdout.write(f"\r  [{bar}] {i}/{total} {label}  elapsed {el/60:5.1f}m  eta {eta/60:5.1f}m   ")
    sys.stdout.flush()
    if i == total: sys.stdout.write("\n")

def savefig(fig, stem):
    """Publication-quality output: PDF and EPS for LaTeX, PNG for quick viewing."""
    for ext in ("pdf", "eps", "png"):
        fig.savefig(opath(f"{stem}.{ext}"), format=ext, dpi=300, bbox_inches="tight")
    print(f"  [saved] {opath(stem)}.pdf / .eps / .png")

def latex_table(rows, cols, headers, caption, label, fname, fmt=None):
    """Emit a booktabs table ready for \\input{} from the paper."""
    fmt = fmt or {}
    al = "l"*1 + "r"*(len(cols)-1)
    L = ["\\begin{table}[htbp]", "\\centering", f"\\caption{{{caption}}}",
         f"\\label{{{label}}}", "\\small", f"\\begin{{tabular}}{{{al}}}", "\\toprule",
         " & ".join(headers) + " \\\\", "\\midrule"]
    for r in rows:
        cells = []
        for k in cols:
            v = r.get(k, "")
            cells.append(fmt.get(k, "{}").format(v) if not isinstance(v, float)
                         else fmt.get(k, "{:.3f}").format(v))
        L.append(" & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(opath(fname), "w") as fh: fh.write("\n".join(L) + "\n")
    print(f"  [saved] {opath(fname)}")

try:
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "figure.dpi": 300, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False})
except Exception:
    pass

# ----------------------------------------------------------------- primitives
def psi_closed(c, z):
    u = 1.0 + c + z
    q = u*u - 4*c
    return np.inf if q <= 0 else 0.5*(u/math.sqrt(q) - 1.0)

def solve_z(c, psi_target, lo=1e-8, hi=1e4):
    """Psi is strictly decreasing in z, so bisect."""
    if psi_closed(c, lo) < psi_target:
        return None                                  # target unreachable at this c
    for _ in range(200):
        mid = math.sqrt(lo*hi)
        if psi_closed(c, mid) > psi_target: lo = mid
        else:                               hi = mid
    return math.sqrt(lo*hi)

def bartlett(x, bw):
    x = x - x.mean(); m = len(x); s = x @ x / m
    for l in range(1, bw+1):
        s += 2*(1 - l/(bw+1)) * (x[l:] @ x[:-l]) / m
    return s

Phi = lambda t: 0.5*(1 + math.erf(t/math.sqrt(2)))

def feasibility(T, rho, n, b):
    Teff = T*(1-rho)/(1+rho)
    return {"T_eff": Teff,
            "F1": Teff >= 5.0,                       # window not collinear
            "F2": n >= 4*T**3,                       # bandwidth window non-empty
            "F3": T < b < math.sqrt(n/T),            # bias and variance both controlled
            "b_hi": math.sqrt(n/T)}

# ------------------------------------------------------------------- designs
def ar1_rows(m, P, rho, g):
    S = np.empty((m, P)); S[0] = g.standard_normal(P)
    if rho == 0.0:
        S[1:] = g.standard_normal((m-1, P)); return S
    for t in range(1, m):
        S[t] = rho*S[t-1] + math.sqrt(1-rho**2)*g.standard_normal(P)
    return S

def rff(Rraw, P, gamma, g):
    d = Rraw.shape[1]
    W = g.standard_normal((P, d))*gamma
    b = g.uniform(0, 2*np.pi, P)
    return math.sqrt(2.0)*np.cos(Rraw @ W.T + b)

def zt_ratio(z, c, R2):
    """How far below the MSE-optimal penalty a cell sits.

    prop:optimal-z gives the optimal EFFECTIVE penalty zt* = 1/R^2, so
        zt/zt* = (z/c) * R^2.
    A value of 1e-3 means the cell runs three orders of magnitude under-shrunk,
    which is the regime the paper is about.  Returns nan under the sharp null.
    """
    return (z/c)*R2 if R2 > 0 else float("nan")


def gen_path(T, P, n, rho, R2, g, delta=0.0, innov="gauss", theta=None,
             signal="dense", xi=0.5, feat_vol=0.0):
    """Returns (y, S, f).  delta > 0 induces Stambaugh endogeneity.

    signal:
      "dense"/"spiked"  f = S theta, with theta supplied by the caller
      "omitted"         (S3) f is driven by a predictor NOT in S but correlated
                        with S[:,0] at xi, so the fitted model is misspecified and
                        g can approximate f only up to the unspanned part.

    feat_vol: scales each row of S by (sigma_t/mean sigma)**feat_vol, so the
      dispersion of the features moves with the conditional volatility of the target.
      Only meaningful with innov="garch".  Conditional heteroskedasticity on its own
      costs the correction nothing -- what matters is the within-window covariance
      between a_{t,s}^2 and sigma_s^2, and that requires the interaction.  Set
      feat_vol=+1 for features that disperse in turbulent periods, -1 for the reverse.
    """
    m = n + T + 2
    if rho == 0.0:
        eta = g.standard_normal((m, P)); S = eta.copy()
    else:
        eta = g.standard_normal((m, P))
        S = np.empty((m, P)); S[0] = eta[0]
        for t in range(1, m):
            S[t] = rho*S[t-1] + math.sqrt(1-rho**2)*eta[t]
    sig_path = None
    if innov == "t5":
        u = g.standard_t(5, m)/math.sqrt(5/3)
    elif innov == "garch":
        u = np.empty(m); hh = np.empty(m); h = 1.0; e = g.standard_normal(m)
        for t in range(m):
            hh[t] = h; u[t] = math.sqrt(h)*e[t]; h = 0.05 + 0.90*h + 0.05*u[t]**2
        sd = u.std(); u /= sd; sig_path = np.sqrt(hh)/sd
    else:
        u = g.standard_normal(m)
    if feat_vol and sig_path is not None:
        S = S * ((sig_path/sig_path.mean())**feat_vol)[:, None]
    if delta > 0.0:                                   # eps_t correlated with eta_t
        v = eta[:, 0]/1.0
        eps = delta*v + math.sqrt(max(1-delta**2, 0))*u
    else:
        eps = u
    if signal == "omitted" and R2 > 0:
        e_ = g.standard_normal(m)
        w  = xi*S[:, 0] + math.sqrt(max(1-xi**2, 0))*e_
        f  = math.sqrt(R2)*w/w.std()
    else:
        if theta is None:
            theta = np.zeros(P)
        f = S @ theta
    y = np.roll(f, 1) + eps; y[0] = eps[0]
    return y, S, f

# --------------------------------------------------------------------- engine
def run_path(y, feats, f, T, z, b, scale=1.0, sigma2=None, both=False):
    """Dual-form rolling ridge on demeaned targets.  Returns a dict of statistics.

    The benchmark is the window mean, so M = I - iota iota'/T.  The ridge is solved on the
    RAW design and the resulting weights are residualised, a = M w with
    w = X (Sighat + z I)^{-1} s / T, so the plug-in psi = ||a||^2 is eq:psihat for this
    pair and prop:dof applies with q = 1.  rv_application.py residualises at the other
    point, putting M inside the ridge; see its header.


    sigma2 : float   use this known value (oracle)
             None    estimate by the difference estimator of prop:sigma-diff
    both   : if True, return BOTH the oracle and the difference-estimator statistics
             from this single path, suffixed "__oracle" and "__diff", so the two are
             computed on identical data.  The oracle value is then taken from sigma2,
             which must be supplied.  Pairing removes almost all of the Monte Carlo
             noise from the oracle-versus-estimated comparison, which is the one
             sec:mc-size rests on.

    Also returned:
      t_CC_w    the heteroskedasticity-weighted correction.  lem:psi-t is the
                homoskedastic special case of E[(a'E)^2 | D] = sum_s a_s^2 sigma_s^2.
                Replacing sigma^2 ||a||^2 by sum_s a_s^2 * (1/2)(y_s - y_{s-1})^2 is
                exactly unbiased for sum_s a_s^2 * (sigma_s^2 + sigma_{s-1}^2)/2 --
                the LAG-SMOOTHED weighted variance, not sum_s a_s^2 sigma_s^2 itself.
                See prop:sigma-weighted and eq:psitil-residue: a scalar's error is the
                covariance of a_s^2 with the LEVEL sigma_s^2, this estimator's is the
                covariance with the INCREMENT, which vanishes when either sequence is
                flat.  In the GARCH designs below the two targets differ by about 0.1%
                against the 8-18% a scalar misses, so the smoothing costs little here.
                No extra cost, since a is already computed.
    """
    n = len(y) - T - 1
    I = np.eye(T); one = np.ones(T); M = I - np.outer(one, one)/T
    g_ = np.empty(n); v_ = np.empty(n); psi = np.empty(n); y1 = np.empty(n)
    wpsi = np.empty(n)                                   # weighted correction
    d2h = 0.5*np.r_[0.0, np.diff(y)**2]                  # (1/2)(y_s - y_{s-1})^2
    d2h[0] = d2h[1:].mean()
    for i in range(n):
        X = feats[i:i+T]; Y = y[i+1:i+T+1]; F = f[i:i+T]; s = feats[i+T]
        B = np.linalg.inv(z*I + X @ X.T / T)
        a = (M @ (B @ (X @ s))) / T
        y1[i] = Y.mean(); g_[i] = a @ F; v_[i] = a @ (Y - F); psi[i] = a @ a
        wpsi[i] = float((a**2) @ d2h[i+1:i+T+1])         # rows of the training window
    yt = y[T+1:T+n+1]; ft = f[T:T+n]
    gs, vs, ps_ = scale*g_, scale*v_, scale**2*psi
    wp_ = scale**2*wpsi
    yh = y1 + gs + vs
    D    = yh - y1
    dDM  = (yt-y1)**2 - (yt-yh)**2
    dCW  = dDM + D**2
    rn   = math.sqrt(n)
    h    = 0.5*np.r_[0., np.diff(yt)**2]; h[0] = h[1:].mean()   # h.mean() == s2_diff
    s2_diff = 0.5*np.mean(np.diff(yt)**2)

    def block(s2v, estimated):
        corr = s2v*ps_
        dCC  = dDM + corr
        # influence function: the third term carries sqrt(n)(s2hat - s2) and is
        # present only when s2 is estimated
        sc_ = dCC - dCC.mean() + (ps_.mean()*(h - h.mean()) if estimated else 0.0)
        om2 = max(bartlett(sc_, b), 1e-14)
        return {"t_CC_hac":  rn*dCC.mean()/math.sqrt(om2),
                "t_CC_samp": rn*dCC.mean()/dCC.std(),
                "t_CC_cw":   rn*dCC.mean()/math.sqrt(np.mean(dCW**2)),
                "ratio": np.mean(D**2)/np.mean(corr) if np.mean(corr) > 0 else np.nan}

    # weighted correction: no scalar sigma^2 to estimate, so the influence function
    # is that of the loss differential alone
    dCCw = dDM + wp_
    om2w = max(bartlett(dCCw - dCCw.mean(), b), 1e-14)

    # dDM inherits the same (T-1)-dependence as dCC from overlapping windows, so it
    # gets the same HAC treatment; dCW is a martingale difference by lem:centring and
    # is left studentised by its sample variance, as Clark-West prescribe.
    om2dm = max(bartlett(dDM - dDM.mean(), b), 1e-14)

    common = {"t_DM": rn*dDM.mean()/math.sqrt(om2dm),
              "t_DM_samp": rn*dDM.mean()/dDM.std(),
              "t_CW": rn*dCW.mean()/dCW.std(),
              "t_CC_w": rn*dCCw.mean()/math.sqrt(om2w),
              "ratio_w": np.mean(D**2)/np.mean(wp_) if np.mean(wp_) > 0 else np.nan,
              "Psihat": ps_.mean(), "Psihat_w": wp_.mean(),
              "s2_diff": s2_diff,
              "theta_T": ps_.var()/ps_.mean()**2 if ps_.mean() > 0 else np.nan,
              "mu_CC": np.mean(2*ft*gs - gs**2),
              # The two halves of mu_CC, reported separately because H_cov is a statement
              # about E[fg] ALONE while H_cc is about the difference.  Without the split one
              # cannot tell a design where Clark--West rejects correctly (E[fg]>0, so H_cov
              # is false) from one where it rejects a true H_cov.
              "Efg": float(np.mean(ft*gs)), "Eg2": float(np.mean(gs**2)),
              "mspe_gap": np.mean(dDM)}
    if both:
        if sigma2 is None:
            raise ValueError("both=True needs the oracle sigma2")
        out = dict(common)
        for tag, s2v, est in (("__oracle", sigma2, False), ("__diff", s2_diff, True)):
            for k, v in block(s2v, est).items():
                out[k + tag] = v
        return out
    s2 = sigma2 if sigma2 is not None else s2_diff
    return {**common, **block(s2, sigma2 is None)}

def pilot_scales(T, P, n, rho, R2, z, g, reps=25, stride=7):
    """Population moments -> the two FIXED rescalings.  No look-ahead."""
    th = g.standard_normal(P); th /= np.linalg.norm(th); th *= math.sqrt(R2)
    I = np.eye(T); one = np.ones(T); M = I - np.outer(one, one)/T
    Efg = Eg2 = Ev2 = 0.0; K = 0
    for _ in range(reps):
        y, S, f = gen_path(T, P, n, rho, R2, g, theta=th)
        for i in range(0, n, stride):
            X = S[i:i+T]; Y = y[i+1:i+T+1]; F = f[i:i+T]; s = S[i+T]
            B = np.linalg.inv(z*I + X @ X.T / T); a = (M @ (B @ (X @ s))) / T
            Efg += f[i+T]*(a @ F); Eg2 += (a @ F)**2; Ev2 += (a @ (Y-F))**2; K += 1
    Efg, Eg2, Ev2 = Efg/K, Eg2/K, Ev2/K
    return th, 2*Efg/Eg2, 2*Efg/(Eg2+Ev2), {"Efg": Efg, "Eg2": Eg2, "Ev2": Ev2}

# =========================================================== ARM A
def arm_A(R=1000, quick=False):
    print("="*100); print("ARM A - size, power, and the 3x3 p-value grid"); print("="*100)
    CELLS  = [(8, 2000, 12), (12, 8000, 18)] if not quick else [(8, 1200, 12)]
    RHOS   = [0.0, 0.3]
    PSIS   = [0.05, 0.2, 0.5, 1.0]
    CFORS  = {0.05: 0.2, 0.2: 0.3, 0.5: 0.5, 1.0: 0.7}     # c chosen so the target is reachable
    R2S    = [0.0, 0.02, 0.05, 0.10, 0.20]
    if quick: PSIS, R2S, R = [0.2, 0.5], [0.0, 0.10], 60
    rows = ckpt_load("armA.json", []) or []
    done = {(r["T"], r["n"], r["rho"], r["psi_t"], r["s2"], r["R2"]) for r in rows}
    if done: print(f"  resuming: {len(done)} cells already complete")
    total = len(CELLS)*len(RHOS)*len(PSIS)*len(R2S); k = 0
    for (T, n, b) in CELLS:
        for rho in RHOS:
            fe = feasibility(T, rho, n, b)
            tag = "OK" if all([fe["F1"], fe["F2"], fe["F3"]]) else "INFEASIBLE"
            print(f"\n--- T={T} n={n} b={b} rho={rho}  T_eff={fe['T_eff']:.1f} "
                  f"F1={fe['F1']} F2={fe['F2']} F3={fe['F3']} ({tag}) ---")
            for psit in PSIS:
                c = CFORS[psit]; z = solve_z(c, psit)
                if z is None: print(f"  Psi={psit}: unreachable at c={c}"); continue
                P = max(1, int(round(c*T)))
                # oracle and difference-estimator statistics are computed from the SAME
                # simulated paths, so the comparison between them is paired
                for R2 in R2S:
                    k += 1
                    if (T, n, rho, psit, "oracle", R2) in done and \
                       (T, n, rho, psit, "diff", R2) in done:
                        continue
                    g = np.random.default_rng(seed_of(T, n, rho, psit, R2))
                    th = np.zeros(P)
                    if R2 > 0:
                        th = g.standard_normal(P); th /= np.linalg.norm(th); th *= math.sqrt(R2)
                    acc = []
                    for _ in range(R):
                        y, S, f = gen_path(T, P, n, rho, R2, g, theta=th)
                        acc.append(run_path(y, S, f, T, z, b, 1.0, 1.0, both=True))
                    A = {kk: np.array([a[kk] for a in acc]) for kk in acc[0]}
                    for s2mode in ("oracle", "diff"):
                        sfx = "__" + s2mode
                        rows.append(dict(T=T, n=n, b=b, rho=rho, psi_t=psit, c=c, z=z, P=P,
                                         s2=s2mode, R2=R2, feasible=tag,
                                         Psihat=A["Psihat"].mean(),
                                         Psihat_w=A["Psihat_w"].mean(),
                                         theta_T=A["theta_T"].mean(),
                                         ratio=A["ratio"+sfx].mean(),
                                         ratio_w=A["ratio_w"].mean(),
                                         mu_CC=A["mu_CC"].mean(),
                                         rej_DM=np.mean(A["t_DM"] > 1.645),
                                         rej_DM_samp=np.mean(A["t_DM_samp"] > 1.645),
                                         rej_CW=np.mean(A["t_CW"] > 1.645),
                                         rej_CC=np.mean(A["t_CC_hac"+sfx] > 1.645),
                                         rej_CC_samp=np.mean(A["t_CC_samp"+sfx] > 1.645),
                                         rej_CC_cw=np.mean(A["t_CC_cw"+sfx] > 1.645),
                                         rej_CC_w=np.mean(A["t_CC_w"] > 1.645)))
                    ckpt_save("armA.json", rows)          # checkpoint after every cell
                    progress(k, total, f"Psi*={psit} R2={R2}")
                    zr = zt_ratio(z, c, R2)
                    print(f"  Psi*={psit:4.2f} z={z:8.3g} P={P:3d} R2={R2:4.2f} "
                          f"| Psihat={A['Psihat'].mean():7.4f}/{A['Psihat_w'].mean():7.4f} "
                          f"ratio={A['ratio__oracle'].mean():5.2f} zt/zt*={zr:8.2e} "
                          f"| rej DM/CW = {np.mean(A['t_DM']>1.645):.3f} "
                          f"{np.mean(A['t_CW']>1.645):.3f} "
                          f"| CC orc/diff/wtd = {np.mean(A['t_CC_hac__oracle']>1.645):.3f} "
                          f"{np.mean(A['t_CC_hac__diff']>1.645):.3f} "
                          f"{np.mean(A['t_CC_w']>1.645):.3f}")
    ckpt_save("armA.json", rows)
    for r in rows:                                    # derived, so old checkpoints resume fine
        r["zt_ratio"] = zt_ratio(r["z"], r["c"], r["R2"])
    latex_table(sorted(rows, key=lambda r: (r["T"], r["rho"], r["psi_t"], r["s2"], r["R2"])),
                ["T", "rho", "psi_t", "s2", "R2", "Psihat", "zt_ratio",
                 "ratio", "rej_DM", "rej_CW", "rej_CC"],
                ["$T$", "$\\rho_x$", "$\\Psi^*$", "$\\hat\\sigma^2$", "$R^2$",
                 "$\\hat\\Psi_n$", "$\\ztil/\\ztil^*$", "$\\mathcal R$", "DM", "CW", "CC"],
                "Size and power. Rejection frequencies of the one-sided 5\\% tests. "
                "The column $\\ztil/\\ztil^*$ reports each cell's penalty as a fraction of "
                "the MSE-optimal one of \\cref{prop:optimal-z}; values far below one locate "
                "the design in the under-shrinkage region where the correction operates.",
                "tab:mc-size-power", "tab_size_power.tex",
                {"T": "{:.0f}", "rho": "{:.1f}", "psi_t": "{:.2f}", "s2": "{}",
                 "R2": "{:.2f}", "zt_ratio": "{:.1e}"})
    return rows


# =========================================================== ARM S (signal structure)
def arm_S(R=1000, quick=False):
    """(S1) spiked, (S2) dense, (S3) omitted-predictor signal, at one representative cell.

    Kept separate from arm A rather than folded into its loop: tripling a four-hour
    sweep to answer a robustness question is a poor trade, and the question is whether
    the ranking of the three statistics survives misspecification, not how it varies
    over the whole grid.
    """
    print("="*100); print("ARM S - signal structure: spiked, dense, omitted predictor")
    print("="*100)
    T, n, b, rho, psit = 12, 8000, 18, 0.3, 0.5
    c = 0.5; z = solve_z(c, psit); P = max(1, int(round(c*T)))
    R2S = [0.0, 0.02, 0.05, 0.10, 0.20]
    if quick: R2S, R = [0.0, 0.10], 60
    fe = feasibility(T, rho, n, b)
    print(f"T={T} n={n} b={b} rho={rho} c={c} z={z:.4g} P={P}  T_eff={fe['T_eff']:.1f} "
          f"F1={fe['F1']} F2={fe['F2']} F3={fe['F3']}")
    rows = ckpt_load("armS.json", []) or []
    done = {(r["signal"], r["R2"]) for r in rows}
    for sig in ["spiked", "dense", "omitted"]:
        for R2 in R2S:
            if (sig, R2) in done: continue
            g = np.random.default_rng(seed_of(sig, R2))
            th = np.zeros(P)
            if R2 > 0 and sig != "omitted":
                if sig == "spiked":
                    th = np.zeros(P); th[0] = 1.0
                else:
                    th = g.standard_normal(P); th /= np.linalg.norm(th)
                th *= math.sqrt(R2)
            acc = []
            for _ in range(R):
                y, S, f = gen_path(T, P, n, rho, R2, g, theta=th, signal=sig)
                acc.append(run_path(y, S, f, T, z, b, 1.0, 1.0))
            A = {k: np.array([a[k] for a in acc]) for k in acc[0]}
            rows.append(dict(signal=sig, R2=R2, T=T, n=n, c=c, z=z, P=P,
                             Psihat=A["Psihat"].mean(), ratio=A["ratio"].mean(),
                             mu_CC=A["mu_CC"].mean(), mspe_gap=A["mspe_gap"].mean(),
                             # H_cov is a statement about E[fg] alone; mu_CC cannot
                             # distinguish a correct Clark--West rejection from a wrong one.
                             Efg2=2*A["Efg"].mean(), Eg2=A["Eg2"].mean(),
                             zt_ratio=zt_ratio(z, c, R2),
                             rej_DM=np.mean(A["t_DM"] > 1.645),
                             rej_CW=np.mean(A["t_CW"] > 1.645),
                             rej_CC=np.mean(A["t_CC_hac"] > 1.645)))
            ckpt_save("armS.json", rows)
            print(f"  {sig:8s} R2={R2:4.2f} | Psihat={A['Psihat'].mean():7.4f} "
                  f"mu_CC={A['mu_CC'].mean():+8.5f} gap={A['mspe_gap'].mean():+8.5f} "
                  f"| rej DM/CW/CC = {np.mean(A['t_DM']>1.645):.3f} "
                  f"{np.mean(A['t_CW']>1.645):.3f} {np.mean(A['t_CC_hac']>1.645):.3f}")
    latex_table(rows, ["signal", "R2", "Psihat", "mu_CC", "mspe_gap",
                       "rej_DM", "rej_CW", "rej_CC"],
                ["signal", "$R^2$", "$\\hat\\Psi_n$", "$\\mu_{CC}$", "MSPE gap",
                 "DM", "CW", "CC"],
                "Signal structure. (S1) spiked, (S2) dense, (S3) the signal is driven by "
                "a predictor absent from the feature set, so the fitted model is "
                "misspecified. Rejection frequencies of the one-sided 5\\% tests.",
                "tab:mc-signal", "tab_signal.tex",
                {"signal": "{}", "R2": "{:.2f}"})
    return rows

def pvalue_grid(T=8, n=2000, b=12, rho=0.3, psit=0.5, R2=0.10, R=1000, s2mode="oracle"):
    """The 3x3 grid.  Rows = the three nulls, columns = the three statistics."""
    print("="*100); print(f"P-VALUE GRID  T={T} n={n} b={b} rho={rho} Psi*={psit} "
                          f"R2={R2} sigma2={s2mode}"); print("="*100)
    c = {0.05: 0.2, 0.2: 0.3, 0.5: 0.5, 1.0: 0.7}[psit]; z = solve_z(c, psit)
    P = max(1, int(round(c*T))); s2 = 1.0 if s2mode == "oracle" else None
    g = np.random.default_rng(20260817)
    fe = feasibility(T, rho, n, b)
    print(f"T_eff={fe['T_eff']:.1f} F1={fe['F1']} F2={fe['F2']} F3={fe['F3']}  c={c} z={z:.4g} P={P}")
    th, aCC, aDM, mom = pilot_scales(T, P, n, rho, R2, z, g)
    print(f"pilot: E[fg]={mom['Efg']:.5f} E[g2]={mom['Eg2']:.5f} E[v2]={mom['Ev2']:.5f}")
    print(f"       a_CC={aCC:.4f}  a_DM={aDM:.4f}   F4 = {0.3 <= aCC <= 3}")
    ROWS = [("H0-cov ", np.zeros(P), 1.0), ("H0-CC  ", th, aCC), ("H0-MSPE", th, aDM)]
    # Column order must track the row order so that panel (i,i) is the null in row i
    # tested by the statistic that null belongs to.  Rows are Hcov, Hcc, Hmspe, so the
    # columns are CW, CC, DM.  Getting this wrong puts three degenerate panels on the
    # diagonal and contradicts the caption.
    COLS = [("CW", "t_CW"), ("CC", "t_CC_hac"), ("DM", "t_DM")]
    COLTITLE = {"CW": "Clark–West", "CC": "noise-corrected",
                "DM": "Diebold–Mariano"}
    ROWTITLE = {"H0-cov": r"$H_0^{\mathrm{cov}}$", "H0-CC": r"$H_0^{\mathrm{CC}}$",
                "H0-MSPE": r"$H_0^{\mathrm{MSPE}}$"}
    out = {}
    for lab, theta, sc in ROWS:
        g2 = np.random.default_rng(seed_of(lab))
        acc = []
        for _ in range(R):
            y, S, f = gen_path(T, P, n, rho, R2, g2, theta=theta)
            acc.append(run_path(y, S, f, T, z, b, sc, s2))
        A = {k: np.array([a[k] for a in acc]) for k in acc[0]}
        out[lab] = A
        print(f"\n{lab}  mu_CC={A['mu_CC'].mean():+.5f}  mspe_gap={A['mspe_gap'].mean():+.5f}"
              f"  ratio={A['ratio'].mean():.2f}")
        print(f"{'':9}{'rej 5%':>9}{'sqrt(m)D':>10}  uniform?")
        for nm, key in COLS + [("CCw", "t_CC_w")]:
            pv = np.array([1-Phi(t) for t in A[key]]); x = np.sort(pv); m = len(x)
            D = max(np.max(np.arange(1, m+1)/m - x), np.max(x - np.arange(0, m)/m))
            print(f"{nm:>9}{np.mean(pv < 0.05):9.3f}{math.sqrt(m)*D:10.2f}"
                  f"  {'UNIFORM' if math.sqrt(m)*D < 1.36 else 'not uniform'}")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 3, figsize=(11, 9), sharex=True)
        band = 1.96*math.sqrt(R*0.1*0.9)          # pointwise 95% band on a bin count
        colors     = {"CW": "#b22222", "CC": "#1f77b4", "DM": "#7f7f7f"}
        colors_off = {"CW": "#d99a9a", "CC": "#9dc3e6", "DM": "#c4c4c4"}
        for i, (lab, _, _) in enumerate(ROWS):
            for j, (nm, key) in enumerate(COLS):
                pv = np.array([1-Phi(t) for t in out[lab][key]]); A_ = ax[i, j]
                on_diag = (i == j)
                counts, _ = np.histogram(pv, bins=10, range=(0, 1))
                # the diagonal carries the row's own statistic and is shaded; it is
                # drawn on a fifth of the vertical range so that a uniform histogram
                # fills the panel and the degenerate off-diagonal panels stay legible
                A_.set_facecolor("#f5f5f3" if on_diag else "#ffffff")
                # The vertical range keys on whether the panel is DEGENERATE, not on
                # whether it is diagonal.  A panel whose mass is spread out gets R/5, so
                # the uniform reference and its band fill it; only a panel with a spike
                # expands to contain the spike.  Keying on the diagonal instead crushes
                # the near-uniform Hcov x CC panel, which the text discusses and a reader
                # has to be able to see.
                A_.set_ylim(0, max(R*0.2, 1.15*counts.max()))
                A_.hist(pv, bins=10, range=(0, 1),
                        color=(colors if on_diag else colors_off)[nm],
                        edgecolor="white")
                A_.axhline(R/10, ls="--", c="k", lw=1)
                A_.axhline(R/10+band, ls=":", c="k", lw=1)
                A_.axhline(max(R/10-band, 0), ls=":", c="k", lw=1)
                if i == 0: A_.set_title(COLTITLE[nm], fontsize=11)
                if j == 0: A_.set_ylabel(f"{ROWTITLE[lab.strip()]}\ncount",
                                         fontsize=10, labelpad=12)
                if i == 2: A_.set_xlabel("p-value")
        fig.tight_layout(); savefig(fig, f"pgrid_psi{psit}_{s2mode}")
    except Exception as e:
        print("plot skipped:", e)

    dump = {lab: {k: A[k].tolist() for k in ("t_DM","t_DM_samp","t_CW","t_CC_hac",
                                             "t_CC_samp","t_CC_cw","t_CC_w",
                                             "Psihat","Psihat_w","ratio","ratio_w",
                                             "mu_CC")}
            for lab, A in out.items()}
    fn = f"pgrid_psi{psit}_{s2mode}.json"
    save_json(fn, {"design": dict(T=T, n=n, b=b, rho=rho, psi_t=psit, c=c, z=z, P=P,
                              R2=R2, R=R, s2=s2mode, a_CC=aCC, a_DM=aDM, **mom),
               "stats": dump})
    return out

# =========================================================== ARM B
def arm_B(R=200, quick=False):
    print("="*100); print("ARM B - the feature map: c fixed, Psi varies"); print("="*100)
    T, n, b, z = 12, 8000, 18, 0.5
    if quick: T, n, b, R = 12, 1500, 18, 40
    DS, PS, GAMMAS = ([5, 15, 50], [200, 2000], [0.05, 0.1, 0.3, 1.0, 3.0])
    if quick: DS, PS, GAMMAS = [15], [200], [0.1, 1.0]
    print(f"T={T} n={n} z={z}   (closed form knows only c = P/T)")
    print(f"{'d':>4}{'P':>7}{'c':>8}{'gamma':>7}{'Psi(c,z)':>11}{'Psihat':>10}{'ratio':>9}")
    rows = []
    for d in DS:
        for P in PS:
            for gam in GAMMAS:
                g = np.random.default_rng(seed_of(d, P, gam))
                acc = []
                for _ in range(R):
                    Rraw = ar1_rows(n+T+2, d, 0.3, g)
                    feats = rff(Rraw, P, gam, g)
                    y = g.standard_normal(n+T+2)
                    acc.append(run_path(y, feats, np.zeros(n+T+2), T, z, b, 1.0, 1.0))
                ph = np.mean([a["Psihat"] for a in acc]); pc = psi_closed(P/T, z)
                rows.append(dict(d=d, P=P, c=P/T, gamma=gam, psi_closed=pc, psihat=ph))
                print(f"{d:4d}{P:7d}{P/T:8.1f}{gam:7.2f}{pc:11.4f}{ph:10.4f}{ph/pc:9.1f}")
    ckpt_save("armB.json", rows)
    latex_table(rows, ["d", "P", "c", "gamma", "psi_closed", "psihat"],
                ["$d$", "$P$", "$c$", "$\\gamma$", "$\\Psi(c,z)$", "$\\hat\\Psi_n$"],
                "Nominal against effective complexity: $c$ fixed down each block, "
                "$\\hat\\Psi_n$ varying with the feature map.",
                "tab:mc-features", "tab_features.tex",
                {"d": "{:.0f}", "P": "{:.0f}", "c": "{:.1f}", "gamma": "{:.2f}"})
    return rows

# =========================================================== ARM C
def arm_C(R=2000, quick=False):
    """R raised from 400.  At 400 the Monte Carlo standard error on a rejection rate
    near 0.05 is 0.011, which is coarser than the differences sec:mc-breakdown quotes
    to three decimals; two independent runs moved the endogeneity endpoint from 0.198
    to 0.235 on noise alone."""
    print("="*100); print("ARM C - where the approximations break"); print("="*100)
    if quick: R = 60
    OUT = {"C1": [], "C2": [], "C3": []}

    print("\n[C1] endogeneity sweep, with the shift diagnostic")
    T, n, b, rho, c = 8, 2000, 12, 0.3, 0.5
    z = solve_z(c, 0.5); P = max(1, int(round(c*T)))
    print(f"T={T} n={n} c={c} z={z:.4g} P={P}")
    print(f"{'delta':>7}{'ratio':>9}{'ratio shifted':>15}{'rej CC':>9}{'diagnosis':>34}")
    for delta in [0.0, 0.2, 0.5, 0.8]:
        g = np.random.default_rng(seed_of("C1", delta))
        ru, rs, rej, rejw, rw = [], [], [], [], []
        for _ in range(R):
            y, S, f = gen_path(T, P, n, rho, 0.0, g, delta=delta)
            a = run_path(y, S, f, T, z, b, 1.0, 1.0); ru.append(a["ratio"])
            rej.append(a["t_CC_hac"] > 1.645); rejw.append(a["t_CC_w"] > 1.645)
            rw.append(a["ratio_w"])
            Ssh = np.roll(S, n//3, axis=0)
            rs.append(run_path(y, Ssh, f, T, z, b, 1.0, 1.0)["ratio"])
        u, s = np.mean(ru), np.mean(rs)
        dg = ("A2 violated" if s > 0.9 and u < 0.9 else
              "correction suspect" if s <= 0.9 else "no problem")
        OUT["C1"].append(dict(delta=delta, ratio=u, ratio_shift=s,
                              ratio_w=float(np.mean(rw)),
                              rej_CC=float(np.mean(rej)),
                              rej_CC_w=float(np.mean(rejw)), diagnosis=dg))
        print(f"{delta:7.2f}{u:9.3f}{s:15.3f}{np.mean(rej):9.3f}{dg:>34}")

    print("\n[C2] innovation distribution, and the volatility-feature interaction")
    print("     feat_vol != 0 ties the dispersion of the features to the conditional")
    print("     volatility of the target.  Heteroskedasticity ALONE costs nothing; what")
    print("     the scalar correction cannot see is the within-window covariance between")
    print("     a_s^2 and sigma_s^2, and that needs the interaction.")
    print(f"{'innov':>8}{'featvol':>9}{'Psihat':>9}{'ratio':>8}{'ratio_w':>9}"
          f"{'rej CC':>9}{'rej CCw':>9}{'rej CW':>9}")
    for innov, fv in [("gauss", 0.0), ("t5", 0.0), ("garch", 0.0),
                      ("garch", 1.0), ("garch", -1.0)]:
        g = np.random.default_rng(seed_of("C2", innov, fv)); acc = []
        for _ in range(R):
            y, S, f = gen_path(T, P, n, rho, 0.0, g, innov=innov, feat_vol=fv)
            acc.append(run_path(y, S, f, T, z, b, 1.0, 1.0))
        A = {k: np.array([a[k] for a in acc]) for k in acc[0]}
        OUT["C2"].append(dict(innov=innov, feat_vol=fv,
                              Psihat=float(A['Psihat'].mean()),
                              Psihat_w=float(A['Psihat_w'].mean()),
                              ratio=float(A['ratio'].mean()),
                              ratio_w=float(A['ratio_w'].mean()),
                              rej_CC=float(np.mean(A['t_CC_hac']>1.645)),
                              rej_CC_w=float(np.mean(A['t_CC_w']>1.645)),
                              rej_CW=float(np.mean(A['t_CW']>1.645))))
        print(f"{innov:>8}{fv:>9.0f}{A['Psihat'].mean():9.4f}{A['ratio'].mean():8.3f}"
              f"{A['ratio_w'].mean():9.3f}{np.mean(A['t_CC_hac']>1.645):9.3f}"
              f"{np.mean(A['t_CC_w']>1.645):9.3f}{np.mean(A['t_CW']>1.645):9.3f}")

    print("\n[C3] outside F2: the empirical calibration, deliberately infeasible")
    for (T_, n_, b_) in [(8, 2000, 12), (60, 600, 60)]:
        fe = feasibility(T_, rho, n_, b_)
        z_ = solve_z(c, 0.5); P_ = max(1, int(round(c*T_)))
        g = np.random.default_rng(seed_of("C3", T_)); acc = []
        for _ in range(max(R//2, 40)):
            y, S, f = gen_path(T_, P_, n_, rho, 0.0, g)
            acc.append(run_path(y, S, f, T_, z_, b_, 1.0, 1.0))
        A = {k: np.array([a[k] for a in acc]) for k in acc[0]}
        tag = "OK" if all([fe["F1"], fe["F2"], fe["F3"]]) else "INFEASIBLE"
        OUT["C3"].append(dict(T=T_, n=n_, b=b_, T_eff=fe['T_eff'], F2=bool(fe['F2']),
                              F3=bool(fe['F3']), tag=tag,
                              size_CC=float(np.mean(A['t_CC_hac']>1.645)),
                              size_CC_w=float(np.mean(A['t_CC_w']>1.645))))
        print(f"  T={T_:3d} n={n_:5d} b={b_:3d}  T_eff={fe['T_eff']:.1f} F2={fe['F2']} "
              f"F3={fe['F3']} ({tag})  size CC={np.mean(A['t_CC_hac']>1.645):.3f}")
    ckpt_save("armC.json", OUT)

def arm_D(R=1000, quick=False):
    """The complexity sweep: does the corrected test survive at the interpolation threshold?

    Arm A indexes its cells by a target Psi*, and that is exactly what caps its complexity
    ratio at c = 0.7.  Psi is unbounded as c -> 1 at a fixed penalty, so a fixed Psi* target
    becomes unreachable further up and solve_z returns None.  The consequence is that the
    region Section 3's closed form is written about -- the pole at (c,z) = (1,0), the
    c/(1-c) and 1/(c-1) branches, and the degradation Theorem R2c describes -- has no
    simulation evidence anywhere in the paper.

    This arm indexes by c directly and holds the scale-free penalty ztilde = zT/P fixed,
    which is what a forecaster does, then reports the realised Psi-hat rather than
    targeting it.  Everything else matches arm A's feasible cell, so the c <= 0.7 rows are
    comparable with it.

    Writes output/armD.json.  It touches no existing output file: arms A, B, C, S and the
    p-value grids are left exactly as published.
    """
    print("="*100); print("ARM D - complexity sweep through the interpolation threshold"); print("="*100)
    T, n, b, rho = 12, 8000, 18, 0.0
    CS  = [0.5, 0.7, 0.9, 1.0, 1.1, 1.5, 2.0, 3.0]
    ZTS = [1.0, 0.1]                       # moderate and light effective penalty
    R2S = [0.0, 0.05]                      # size, and one alternative for power
    if quick:
        CS, ZTS, R2S, R = [0.7, 1.0, 2.0], [1.0], [0.0], 60

    rows = ckpt_load("armD.json", []) or []
    done = {(r["c_target"], r["ztilde"], r["R2"], r["s2"]) for r in rows}
    if done: print(f"  resuming: {len(done)//2} cells already complete")
    fe = feasibility(T, rho, n, b)
    print(f"  design T={T} n={n} b={b} rho={rho}  T_eff={fe['T_eff']:.1f} "
          f"F1={fe['F1']} F2={fe['F2']} F3={fe['F3']}")
    total = len(CS)*len(ZTS)*len(R2S); k = 0
    for ct in CS:
        P = max(1, int(round(ct*T)))
        c = P/T                                        # realised ratio after rounding
        for zt in ZTS:
            z = zt*c                                   # ztilde = zT/P = z/c
            psi_th = psi_closed(c, z)
            for R2 in R2S:
                k += 1
                if all((ct, zt, R2, m) in done for m in ("oracle", "diff")):
                    continue
                g = np.random.default_rng(seed_of("D", ct, zt, R2))
                th = np.zeros(P)
                if R2 > 0:
                    th = g.standard_normal(P); th /= np.linalg.norm(th); th *= math.sqrt(R2)
                acc = []
                for _ in range(R):
                    y, S, f = gen_path(T, P, n, rho, R2, g, theta=th)
                    acc.append(run_path(y, S, f, T, z, b, 1.0, 1.0, both=True))
                A = {kk: np.array([a[kk] for a in acc]) for kk in acc[0]}

                def moments(v):
                    v = np.asarray(v, float); sd = v.std()
                    return float(sd), float(np.mean((v - v.mean())**4)/sd**4) if sd > 0 else float("nan")
                se = lambda p: float(math.sqrt(max(p*(1-p), 0.0)/R))

                for s2mode in ("oracle", "diff"):
                    sfx = "__" + s2mode
                    rj_cc = float(np.mean(A["t_CC_hac"+sfx] > 1.645))
                    sd_cc, ku_cc = moments(A["t_CC_hac"+sfx])
                    sd_cw, ku_cw = moments(A["t_CW"])
                    rows.append(dict(
                        T=T, n=n, b=b, rho=rho, c_target=ct, c=c, P=P, ztilde=zt, z=z,
                        s2=s2mode, R2=R2, R=R,
                        Psi_closed=float(psi_th), Psihat=float(A["Psihat"].mean()),
                        psi_ratio=float(A["Psihat"].mean()/psi_th) if np.isfinite(psi_th) and psi_th > 0 else float("nan"),
                        theta_T=float(A["theta_T"].mean()),
                        ratio=float(A["ratio"+sfx].mean()),
                        sd_t_CC=sd_cc, kurt_t_CC=ku_cc, sd_t_CW=sd_cw, kurt_t_CW=ku_cw,
                        rej_DM=float(np.mean(A["t_DM"] > 1.645)),
                        rej_CW=float(np.mean(A["t_CW"] > 1.645)),
                        rej_CC=rj_cc,
                        rej_CC_samp=float(np.mean(A["t_CC_samp"+sfx] > 1.645)),
                        rej_CC_cw=float(np.mean(A["t_CC_cw"+sfx] > 1.645)),
                        rej_CC_w=float(np.mean(A["t_CC_w"] > 1.645)),
                        mc_se=se(rj_cc)))
                ckpt_save("armD.json", rows)
                progress(k, total, f"c={ct} zt={zt} R2={R2}")
                o = [r for r in rows if r["c_target"]==ct and r["ztilde"]==zt
                     and r["R2"]==R2 and r["s2"]=="oracle"][-1]
                print(f"  c={ct:4.1f} (P={P:2d}) zt={zt:4.1f} R2={R2:4.2f} "
                      f"| Psi={psi_th:8.3f} Psihat={o['Psihat']:8.3f} ratio={o['psi_ratio']:5.2f} "
                      f"theta_T={o['theta_T']:5.2f} "
                      f"| rej DM/CW/CC = {o['rej_DM']:.3f} {o['rej_CW']:.3f} {o['rej_CC']:.3f} "
                      f"(mc se {o['mc_se']:.3f}) sd(t_CC)={o['sd_t_CC']:.2f}")
    ckpt_save("armD.json", rows)
    print(f"\n  arm D complete: {len(rows)} rows -> {OUTDIR}/armD.json")
    return rows


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "quick"
    sys.stdout = Tee(opath(f"run_{what}_{datetime.datetime.now():%Y%m%d_%H%M}.log"))
    print(f"# mc_study {what}  started {datetime.datetime.now():%Y-%m-%d %H:%M}  -> {OUTDIR}/")
    acquire_lock(what)
    q = (what == "quick")
    if what in ("A", "all", "quick"):
        arm_A(quick=q)
        for m in ["oracle", "diff"]:                 # the practical estimator matters
            for pt in ([0.5] if q else [0.2, 0.5, 1.0]):
                pvalue_grid(psit=pt, R=(80 if q else 1000), s2mode=m)
    if what in ("B", "all", "quick"): arm_B(quick=q)
    if what in ("C", "all", "quick"): arm_C(quick=q)
    if what in ("S", "all", "quick"): arm_S(quick=q)
    if what in ("D", "all", "quick"): arm_D(quick=q)
    print(f"\n# finished in {(time.time()-_T0)/60:.1f} min.  All output in {OUTDIR}/")
