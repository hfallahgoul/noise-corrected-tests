"""The corrected test on daily realized variance: HAR against a high-dimensional nested
alternative (Section 6, tab:rv-main, figure fig-rv-stats).

The benchmark is the HAR model of Corsi (2009); the alternative adds lags 2, ..., P of log
RV, standardised on the window and fitted by ridge on the HAR residuals, so the comparison is
nested.  For every (P, z) cell the script reports Psihat_n, the sanity ratio, the incremental
fit, the realised MSPE gap, and t_DM, t_CW, t_CC under several innovation-variance
estimators (window residual, difference, and the weighted forms).

    python rv_application.py --probe 84398                     # inspect the Risk Lab payload
    python rv_application.py --ticker 84398                    # the series of tab:rv-main
    python rv_application.py --offline                          # synthetic data, no network

Needs network access to Risk Lab (dachxiu.chicagobooth.edu).  The rows behind tab:rv-main
are stored in output/rv_tables/rv_rows.json; build_tables.py writes the table from them.
"""
import re

_SIGNED = re.compile(r"(?<![\w\\$])([+-]\d*\.?\d+)")


def mathsign(body):
    """Wrap signed numbers in table rows in math mode, so a minus renders as a minus
    and not as a hyphen. Cells already inside $...$ are left alone."""
    out = []
    for line in body.split("\n"):
        if "&" not in line or "\\\\" not in line:
            out.append(line); continue
        spans = [(m.start(), m.end()) for m in re.finditer(r"\$[^$]*\$", line)]
        parts, last = [], 0
        for m in _SIGNED.finditer(line):
            if any(a <= m.start() < b for a, b in spans):
                continue
            parts.append(line[last:m.start()]); parts.append("$%s$" % m.group(1)); last = m.end()
        parts.append(line[last:])
        out.append("".join(parts))
    return "\n".join(out)


import argparse, datetime, json, math, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TABS = os.path.join(HERE, "output", "rv_tables")
FIGS = os.path.join(HERE, "output", "rv_figures")
LOGS = os.path.join(HERE, "output", "rv_logs")

# matplotlib writes a font cache to $HOME by default, which is not writable under some
# sandboxes. Point it at the system temp dir before matplotlib is imported anywhere; a
# font cache is not an output and does not belong in output/.
import tempfile
os.environ.setdefault("MPLCONFIGDIR",
                      os.path.join(tempfile.gettempdir(), "mplcache-rvapp"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)


class Tee:
    """Mirror stdout to a log file, so a run is never lost to a closed terminal.
    Same device as mc_study.py, kept deliberately identical."""
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a"); self.o = sys.__stdout__
    def write(self, x):
        self.o.write(x); self.f.write(x); self.f.flush()
    def flush(self):
        self.o.flush(); self.f.flush()


def save_rows(rows, meta):
    # --tag namespaces the OUTPUT, not just the log: a variant run must never overwrite
    # the stored own-lags run that tab-rv-main is built from.

    """Persist the numbers, not just the rendered tables. Anything in the paper must be
    rebuildable from this file without re-running the forecasts."""
    os.makedirs(TABS, exist_ok=True)
    path = os.path.join(TABS, f"rv_rows{_TAG}.json")
    with open(path, "w") as fh:
        json.dump({"meta": meta, "rows": rows}, fh, indent=1, default=float)
    return path


# ============================================================ data
# Risk Lab column layout (from the site's volatility.js), after the ID and Date columns.
# Volatilities are annualised; to_daily_variance() squares and divides by 252.
RISKLAB_COLUMNS = {
    "c2": "QMLE (Xiu 2010; Da-Xiu 2021), trades, all ticks",
    "c3": "MA order q chosen by AIC for c2",
    "c4": "standard error of c2",
    "c5": "5-min subsampled realized volatility, trades",
    "c6": "15-min subsampled realized volatility, trades",
    "c7": "QMLE (Xiu 2010; Da-Xiu 2021), mid-quotes, all ticks",
    "c8": "MA order q chosen by AIC for c7",
    "c9": "standard error of c7",
    "c10": "5-min subsampled realized volatility, mid-quotes",
    "c11": "15-min subsampled realized volatility, mid-quotes",
}


def fetch_risklab(ticker_id, verbose=True):
    """Fetch and parse one Risk Lab series.

    The payload is six metadata lines followed by a whitespace-delimited matrix. The
    metadata names the series, and -- unlike the version in HAR_estimate.ipynb -- we
    print all of it, because the column layout is not documented anywhere else and
    guessing which column is the 5-minute RV is how you end up modelling a column of
    zeros.
    """
    import requests
    from io import StringIO
    url = f"https://dachxiu.chicagobooth.edu/data.php?ticker={ticker_id}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    lines = resp.text.strip().split("\n")
    if verbose:
        print("--- Risk Lab metadata (first 6 lines) ---")
        for i, ln in enumerate(lines[:6]):
            print(f"  [{i}] {ln.strip()}")
        print("-----------------------------------------")
    df = pd.read_csv(StringIO("\n".join(lines[6:])), sep=r"\s+", header=None)
    ncol = df.shape[1]
    df.columns = ["ID", "Date"] + [f"c{i}" for i in range(2, ncol)]
    df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d")
    return df.set_index("Date").sort_index().drop(columns=["ID"])


def probe(df):
    """Report enough about each column to identify it. Risk Lab pads unavailable series
    with zeros, so a column can look fine in .head() and be empty over the sample."""
    print(f"\n{len(df):,} rows, {df.index.min().date()} to {df.index.max().date()}\n")
    hdr = f"{'column':>8}{'nonzero':>10}{'mean':>10}{'median':>10}{'min':>9}{'max':>9}"
    print(hdr); print("-" * len(hdr))
    for c in df.columns:
        v = pd.to_numeric(df[c], errors="coerce")
        nz = v[(v > 0) & v.notna()]
        frac = len(nz) / len(v)
        if len(nz) == 0:
            print(f"{c:>8}{frac:>10.1%}{'--':>10}{'--':>10}{'--':>9}{'--':>9}")
        else:
            print(f"{c:>8}{frac:>10.1%}{nz.mean():>10.4f}{nz.median():>10.4f}"
                  f"{nz.min():>9.4f}{nz.max():>9.4f}")
    print("""
Columns with 0% nonzero are unavailable for this asset -- do not model them. Risk Lab
serves QMLE plus 5-minute and 15-minute subsampled RV, from trades and from mid-quotes,
each with a standard error. Levels near 0.15-0.35 are annualized volatilities; a column
an order of magnitude smaller is a standard error, not an estimate.

Pick the 5-minute subsampled series if you want the result to be comparable to the HAR
literature, which models 5-minute RV. Using QMLE invites the question of whether the
finding is about the estimator. Pass your choice with --rv-col.""")


def synthetic_rv(n=4000, seed=0):
    """A genuine HAR process, for testing the pipeline without network access.

    Synthetic log RV from an AR(1) would be a poor test of the pipeline, since AR(1)
    then ties HAR exactly and beta_w, beta_m are zero. This
    generates log RV from the Corsi cascade itself, with coefficients in the range the
    empirical literature reports, so HAR is the correctly specified benchmark and should
    beat the naive baselines by the usual margin.
    """
    g = np.random.default_rng(seed)
    bd, bw, bm, sig = 0.36, 0.34, 0.25, 0.42
    mu = math.log(0.02 ** 2 / 252)                    # ~20% annualized vol
    burn = 300
    lv = np.full(n + burn, mu)
    for t in range(22, n + burn):
        d = lv[t - 1]
        w = lv[t - 5:t].mean()
        m = lv[t - 22:t].mean()
        lv[t] = mu * (1 - bd - bw - bm) + bd * d + bw * w + bm * m + sig * g.standard_normal()
    lv = lv[burn:]
    idx = pd.bdate_range("2005-01-03", periods=n)
    return pd.Series(np.exp(lv), index=idx, name="RV")


def to_daily_variance(vol_series, annualized=True):
    """Risk Lab reports annualized volatility. HAR is written in daily variance."""
    v = pd.to_numeric(vol_series, errors="coerce").replace(0.0, np.nan).dropna()
    return (v ** 2) / 252.0 if annualized else v


# ============================================================ design matrices
def har_design(rv, log=True):
    """Corsi (2009) HAR. Target and regressors are log RV by default; squared-error loss
    on log RV is what the paper's Lemma A.6 is stated for."""
    s = np.log(rv) if log else rv
    d = pd.DataFrame(index=s.index)
    d["y"]  = s.shift(-1)                    # one-step-ahead target
    d["rv_d"] = s
    d["rv_w"] = s.rolling(5).mean()
    d["rv_m"] = s.rolling(22).mean()
    return d.dropna()


def extra_lags(rv, n_lags, log=True):
    """The nested alternative: HAR plus lags 2..n_lags of the same series.

    This is a real question rather than a device. HAR compresses the lag structure into
    three averages; whether the compression loses anything is exactly the sort of claim
    the applied literature supports with Clark-West tests.
    """
    s = np.log(rv) if log else rv
    # Row d carries the target s(d+1) (har_design shifts it by -1), so lag k of the target
    # is s(d-k+1), that is s.shift(k-1). L1 = s(d) duplicates rv_d and callers drop it,
    # leaving lags 2..n_lags of the target.
    X = pd.concat({f"L{k}": s.shift(k - 1) for k in range(1, n_lags + 1)}, axis=1)
    return X.dropna()


# ============================================================ validation
def extra_estimators(frames, n_lags, log=True):
    """The nested alternative: HAR on one realized measure, enlarged by lags of OTHER
    estimators of the SAME latent variance.

    Risk Lab serves several estimators per asset -- QMLE and subsampled RV at two
    frequencies, from trades and from mid-quotes.  They estimate one quantity, so any
    incremental predictive content is measurement-error averaging: real, documented, and
    small.  That is the configuration in which the two nulls of Table 3.1 can disagree.
    A rejection needs 2E[fg] > 0, which weak genuine signal supplies; a reversal needs
    2E[fg] <= E[g^2], which many collinear regressors carrying little each supplies.

    Each column is a different functional of the same intraday path, so admissibility is
    the third case of the scope subsection, measured in tab:pathfunc (kappa_a ~ 1) and
    NOT covered by either panel of tab:ldv-scope.  Trades and mid-quotes are two noisy
    observations of one efficient price rather than two functionals of one recorded path,
    which is a step further out than the simulation covers; --est-cols lets the trades
    block be run alone as the conservative variant.
    """
    out = {}
    for col, rv in frames.items():
        rv = rv[~rv.index.duplicated(keep="first")]
        s_ = np.log(rv) if log else rv
        for k in range(1, n_lags + 1):
            # lag k of the target s(d+1) is the estimator's value on day d-k+1; L1 is
            # the same-day estimate, known when the forecast of s(d+1) is made
            out[f"{col}_L{k}"] = s_.shift(k - 1)
    return pd.concat(out, axis=1).dropna()


def synthetic_estimators(rv, k=5, seed=0):
    """--offline companion to extra_estimators: k noisy observations of the same latent
    variance, at the correlation the probe of ticker 84398 reports (0.88-0.98 in logs).
    For wiring tests only."""
    g = np.random.default_rng(seed + 991)
    lv = np.log(rv)
    sd = float(lv.std())
    return {f"synth{j}": pd.Series(
        np.exp(lv + 0.30 * sd * g.standard_normal(len(lv))), index=lv.index)
        for j in range(k)}


def qlike(y_true_var, y_pred_var):
    """Patton (2011) QLIKE, in VARIANCE units. Reported for comparison with the HAR
    literature; the paper's test is derived for squared error and not for this loss."""
    p = np.clip(y_pred_var, 1e-14, None)
    return float(np.mean(np.log(p) + y_true_var / p))


def validate(rv, T, log=True):
    """Check the benchmark before comparing anything to it.

    The comparison in this script reports the INCREMENTAL R^2 of the augmented model
    over HAR. That number is near zero or negative whenever HAR is hard to beat, which
    is the usual finding. It is not HAR's own fit, and confusing the two is easy. This
    stage reports HAR against three naive baselines so the benchmark is known to work
    before anything is tested against it.
    """
    s = np.log(rv) if log else rv
    print("\n" + "=" * 78)
    print("VALIDATION -- properties of the series and of the benchmark")
    print("=" * 78)

    ann = float(np.sqrt(rv.mean() * 252) * 100)
    print(f"\nRealized variance series: {len(rv):,} days, "
          f"{rv.index.min().date()} to {rv.index.max().date()}")
    print(f"  mean annualized volatility {ann:.1f}%   "
          f"log-RV mean {s.mean():.3f}, sd {s.std():.3f}")
    ac = [float(s.autocorr(k)) for k in (1, 5, 22, 66, 252)]
    print("  log-RV autocorrelation at lags 1, 5, 22, 66, 252: "
          + ", ".join(f"{a:.3f}" for a in ac))
    print("  (slow hyperbolic decay is the long-memory signature HAR is built for;"
          "\n   a fast geometric drop would say the series is not what we think it is)")

    d = har_design(rv, log=log)
    y = d["y"].to_numpy()
    X = np.column_stack([np.ones(len(d)), d[["rv_d", "rv_w", "rv_m"]].to_numpy()])

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    r2_in = 1 - r.var() / y.var()
    print(f"\nHAR in-sample (full sample, {len(d):,} obs):")
    print(f"  const {beta[0]:+.4f}   beta_d {beta[1]:+.4f}   "
          f"beta_w {beta[2]:+.4f}   beta_m {beta[3]:+.4f}   sum {beta[1:].sum():.4f}")
    print(f"  R^2 = {r2_in:.4f}")
    print("  (Corsi: all three positive, summing just under one. A negative or wildly"
          "\n   large coefficient means the design matrix is broken, not that HAR failed.)")

    # --- rolling out-of-sample, HAR against three naive benchmarks
    n = len(d) - T - 1
    yt = y[T:T + n]
    fc = {k: np.empty(n) for k in ("HAR", "mean", "RW", "AR1")}
    for i in range(n):
        Xw, Yw = X[i:i + T], y[i:i + T]
        b, *_ = np.linalg.lstsq(Xw, Yw, rcond=None)
        fc["HAR"][i]  = X[i + T] @ b
        fc["mean"][i] = Yw.mean()
        fc["RW"][i]   = X[i + T, 1]                       # yesterday's log RV
        Xa = np.column_stack([np.ones(T), Xw[:, 1]])
        ba, *_ = np.linalg.lstsq(Xa, Yw, rcond=None)
        fc["AR1"][i]  = ba[0] + ba[1] * X[i + T, 1]

    print(f"\nRolling out-of-sample, T = {T}, n = {n:,}:")
    hdr = f"  {'model':>6}{'MSE':>12}{'R2 vs mean':>13}{'QLIKE':>11}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    mse = {k: float(np.mean((yt - v) ** 2)) for k, v in fc.items()}
    for k in ("mean", "RW", "AR1", "HAR"):
        r2 = 1 - mse[k] / mse["mean"]
        ql = qlike(np.exp(yt), np.exp(fc[k])) if log else qlike(yt, fc[k])
        print(f"  {k:>6}{mse[k]:>12.5f}{r2:>13.4f}{ql:>11.4f}")
    print("""
  R^2 vs mean is HAR's own out-of-sample fit -- the 0.5-0.7 figure the HAR literature
  quotes. It is NOT the column reported in the comparison table below, which is the
  INCREMENTAL R^2 of the augmented model over HAR and is negative whenever the extra
  regressors do not help. Both are correct; they answer different questions.

  HAR should beat all three baselines. If it does not, stop: the problem is the data or
  the design matrix, and no amount of testing downstream will fix it.""")
    return dict(r2_in=r2_in, beta=beta.tolist(),
                mse={k: mse[k] for k in mse},
                r2_har_vs_mean=1 - mse["HAR"] / mse["mean"])


# ============================================================ the rolling loop
def rolling(y, Xhar, Xext, T, zgrid):
    """Rolling HAR benchmark plus ridge increment on the HAR residuals.

    The projection M is the HAR residual maker on the window and enters the ridge itself,
    so a_t = M Xe (G + z I)^{-1} s_new / T is the weight vector the increment places on the
    T training targets and Psihat_t = ||a_t||^2 is exact by Lemma A.6.  See the header for
    why this is not the same object as the Monte Carlo's a_t = M w_{t-1}.

    Computed in PRIMAL form. Here P < T (unlike the random-feature application, where
    P >> T and the dual form is the cheap one), so we eigendecompose the P x P matrix
    G = Xe' M Xe / T once per window and every penalty on the grid is then free:

        A = (G + zI)^{-1} = V diag(1/(g+z)) V'
        increment = s' A (Xe' M Y / T)
        Psihat    = a'a = s' V diag(g/(g+z)^2) V' s / T

    That is O(T P^2 + P^3) per window against O(T^3) for the dual, a factor of 25 at
    T = 250, P = 50.
    """
    n = len(y) - T - 1
    nz = len(zgrid)
    yb  = np.empty(n)
    yh  = np.empty((nz, n)); psi = np.empty((nz, n))
    psi_wd = np.empty((nz, n)); psi_wl = np.empty((nz, n))
    s2w = np.empty(n)
    k = Xhar.shape[1]
    for i in range(n):
        H  = Xhar[i:i + T]
        Xe = Xext[i:i + T]
        Y  = y[i:i + T]
        h_new, s_new = Xhar[i + T], Xext[i + T]

        # Standardise the extra features on the window's own moments. Lags of log RV are
        # in levels and near-collinear for a persistent series, which sends the smallest
        # eigenvalues of G toward zero and Psihat with them; this is the F1 failure mode
        # of Section 5.2. Ridge is not scale-invariant, so the scaling is a modelling
        # choice and this is the standard one. Window moments only, so no look-ahead.
        mu, sd = Xe.mean(0), Xe.std(0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        Xe = (Xe - mu) / sd
        s_new = (s_new - mu) / sd

        # --- benchmark: HAR fitted on the window, applied out of sample
        Qh, Rh = np.linalg.qr(H)
        beta = np.linalg.solve(Rh, Qh.T @ Y)
        yb[i] = h_new @ beta
        resid = Y - H @ beta
        s2w[i] = resid @ resid / max(T - k, 1)

        # --- residualise against HAR without forming the T x T projection
        QtXe = Qh.T @ Xe                    # k x P
        G  = (Xe.T @ Xe - QtXe.T @ QtXe) / T          # = Xe' M Xe / T
        r  = (Xe.T @ Y  - QtXe.T @ (Qh.T @ Y)) / T    # = Xe' M Y  / T

        g, V = np.linalg.eigh(G)
        g = np.clip(g, 0.0, None)
        Vs, Vr = V.T @ s_new, V.T @ r

        # The weighted corrections need a_t itself, not just its norm. One extra T x P
        # residualised design per window; the per-penalty cost is a single matvec.
        MXe = Xe - Qh @ QtXe                              # T x P
        dr = np.empty(T)
        dr[1:] = 0.5 * np.diff(resid) ** 2; dr[0] = dr[1]  # residual differences
        lv = local_var(resid, k)                           # smoothed local residual var

        for j, z in enumerate(zgrid):
            yh[j, i]  = yb[i] + Vs @ (Vr / (g + z))
            psi[j, i] = float(Vs @ (g / (g + z) ** 2 * Vs)) / T
            a2 = ((MXe @ (V @ (Vs / (g + z)))) / T) ** 2
            psi_wd[j, i] = float(a2 @ dr)
            psi_wl[j, i] = float(a2 @ lv)
    return y[T:T + n], yb, yh, psi, s2w, psi_wd, psi_wl


def local_var(r, k, half=10):
    """Smoothed local variance of the window residuals, rescaled so its mean matches the
    window estimate. Same definition as Code/mc_har.py and Code/ldv_scope/har_size_power.py."""
    r2 = r ** 2
    T = len(r)
    out = np.empty(T)
    cs = np.concatenate([[0.0], np.cumsum(r2)])
    for s in range(T):
        lo, hi = max(0, s - half), min(T, s + half + 1)
        out[s] = (cs[hi] - cs[lo]) / (hi - lo)
    return out * (r2.sum() / max(T - k, 1)) / max(out.mean(), 1e-300)


def statistics(yt, yb, yh, psi, s2, bandwidth=None, corrections=None, headline="win"):
    """corrections: {name: per-period correction array in loss units}. The headline
    t_CC uses corrections[headline]; every variant is reported as t_CC_<name>."""
    n = len(yt)
    e1, e2 = yt - yb, yt - yh
    D   = yh - yb
    dDM = e1 ** 2 - e2 ** 2
    dCW = dDM + D ** 2
    if corrections is None:
        corrections = {"win": s2 * psi}
        headline = "win"
    b   = bandwidth or max(1, int(np.ceil(4 * (n / 100) ** (2 / 9))))
    def hac(x):
        x = x - x.mean(); g = np.mean(x * x)
        for h in range(1, b + 1):
            g += 2 * (1 - h / (b + 1)) * np.mean(x[h:] * x[:-h])
        return max(g, 1e-14)
    t = lambda d, v: math.sqrt(n) * d.mean() / math.sqrt(v)
    C = corrections[headline]
    out = dict(
        t_DM=t(dDM, dDM.var()), t_CW=t(dCW, dCW.var()),
        t_CW_hac=t(dCW, hac(dCW)),   # robustness: CW with the same Bartlett LRV as t_CC
        t_CC=t(dDM + C, hac(dDM + C)),
        Psihat=float(np.mean(psi)),
        theta_T=float(np.var(psi) / np.mean(psi) ** 2) if np.mean(psi) > 0 else np.nan,
        ratio=float(np.mean(D ** 2) / max(np.mean(C), 1e-300)),
        R2_incr=float(1 - np.mean(e2 ** 2) / np.mean(e1 ** 2)),  # vs HAR, not vs mean
        mspe_gap=float(np.mean(dDM)),
        mse_bench=float(np.mean(e1 ** 2)), mse_alt=float(np.mean(e2 ** 2)),
        headline=headline,
    )
    for nm, c in corrections.items():
        out[f"t_CC_{nm}"]  = t(dDM + c, hac(dDM + c))
        out[f"corr_{nm}"]  = float(np.mean(c))
        out[f"ratio_{nm}"] = float(np.mean(D ** 2) / max(np.mean(c), 1e-300))
    return out


# ============================================================ output
def _g2(x):
    """Two significant figures, plain notation (0.0028, 28, 2800)."""
    return f"{float(f'{x:.2g}'):g}"


def write_output(rows, meta):
    os.makedirs(TABS, exist_ok=True); os.makedirs(FIGS, exist_ok=True)
    # n varies with P, because a longer lag block costs evaluation points at the start
    # of the sample.  Reporting meta["n"] alone (the first row) understates the range.
    _ns = sorted({r["n"] for r in rows})
    _nstr = (f"$n={_ns[0]:,}$ evaluation points" if len(_ns) == 1 else
             f"$n$ runs from ${_ns[0]:,}$ to ${_ns[-1]:,}$ evaluation points as $P$ varies")
    cap = (r"\caption{HAR against a high-dimensional nested alternative. Daily log "
           r"realized variance for " + meta["asset"] + r", " + _nstr +
           r", rolling window $T=" + str(meta["T"]) + r"$. The "
           r"benchmark is the HAR \eqref{eq:rv-har}; the alternative \eqref{eq:rv-ridge} adds "
           r"lags $2,\dots,P+1$ of the same series, fitted by ridge to the HAR residuals, so "
           r"the comparison is nested. $c=P/T$ and $\ztil=zT/P$. $\Psihatn$ is exact by "
           r"\cref{lem:psi-t}. $R^2_{\text{incr}}$ is the mean-square gap of the alternative "
           r"over HAR divided by HAR's mean squared error, negative when the larger model "
           r"forecasts worse. $\mathcal R$ is the sanity ratio of \eqref{eq:ratio}, which "
           r"cannot fall below one in population and equals one exactly when the systematic "
           r"component vanishes. $t_{CC}$ uses the scalar plug-in $\hat\sigma^2_w\Psihat_t$; "
           r"$t_{CC}^{w}$ uses the weighted correction of \cref{prop:sigma-weighted}, with "
           r"the inner variance taken from first differences of the benchmark residuals, as "
           r"a lagged-dependent-variable benchmark requires.}")
    L = [r"\begin{table}[htbp]", r"\centering", cap,
         r"\label{tab:rv-main}", r"\small",
         r"\begin{tabular}{rrrrrrrrrr}", r"\toprule",
         r"$P$ & $c$ & $\ztil$ & $\Psihatn$ & $\mathcal R$ & $R^2_{\text{incr}}$"
         r" & $t_{DM}$ & $t_{CW}$ & $t_{CC}$ &"
         r" $t_{CC}^{w}$ \\", r"\midrule"]
    prev = None
    for r in rows:
        if prev is not None and r["P"] != prev:
            L.append(r"\addlinespace")
        pp = str(r["P"]) if r["P"] != prev else ""
        cc = f"{r['c']:.2f}" if r["P"] != prev else ""
        dag = r"$^{\dagger}$" if r["ratio"] < 1 else ""
        L.append(f"{pp} & {cc} & {_g2(r['zt'])} & {r['Psihat']:.4f} & "
                 f"{r['ratio']:.2f}{dag} & {r['R2_incr']:.4f} & "
                 f"{r['t_DM']:.2f} & {r['t_CW']:.2f} & {r['t_CC']:.2f} & "
                 f"{r.get('t_CC_wdiff', float('nan')):.2f} \\\\")
        prev = r["P"]
    L += [r"\bottomrule",
          r"\multicolumn{10}{@{}l}{\footnotesize $^{\dagger}$ sanity ratio below one; the"
          r" corrected statistic should not be read in that row.}",
          r"\end{tabular}", r"\end{table}"]
    open(os.path.join(TABS, f"tab-rv-main{_TAG}.tex"), "w").write(mathsign("\n".join(L) + "\n"))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    Ps = sorted({r["P"] for r in rows})
    cm = plt.get_cmap("viridis")
    col = {P: cm(i / max(len(Ps) - 1, 1)) for i, P in enumerate(Ps)}

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for P in Ps:
        s = sorted([r for r in rows if r["P"] == P], key=lambda r: r["zt"])
        ax.loglog([r["zt"] for r in s], [max(r["Psihat"], 1e-10) for r in s],
                  "o-", ms=3.5, lw=1.2, color=col[P], label=rf"$P={P}$")
    ax.axhspan(1e-10, 0.05, color="0.88", alpha=0.6, zorder=0)
    ax.set_xlabel(r"effective penalty $\tilde z = zT/P$")
    ax.set_ylabel(r"$\hat\Psi_n$   (units of $\sigma^2$)")
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, f"fig-rv-psi{_TAG}.pdf"),
                                    bbox_inches="tight")
    fig.savefig(os.path.join(FIGS, f"fig-rv-psi{_TAG}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(Ps), figsize=(2.5 * len(Ps), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for k, P in enumerate(Ps):
        ax = axes[k]; s = sorted([r for r in rows if r["P"] == P], key=lambda r: r["zt"])
        zt = [r["zt"] for r in s]
        ax.axhspan(-1.645, 1.645, color="0.9", zorder=0)
        ax.axhline(1.645, ls=":", lw=0.8, color="k")
        for key, lab, sty in [("t_DM", "DM", "s--"), ("t_CW", "CW", "o-"),
                              ("t_CC", "CC", "^-")]:
            ax.semilogx(zt, [r[key] for r in s], sty, ms=3.5, lw=1.2, label=lab)
        bad = [r["zt"] for r in s if r["ratio"] < 1]
        if bad:
            ax.axvspan(min(bad), max(bad), color="crimson", alpha=0.10, zorder=0)
        ax.set_title(rf"$P={P}$", fontsize=9); ax.set_xlabel(r"$\tilde z$")
        if k == 0:
            ax.set_ylabel("one-sided $t$"); ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, f"fig-rv-stats{_TAG}.pdf"),
                                    bbox_inches="tight")
    fig.savefig(os.path.join(FIGS, f"fig-rv-stats{_TAG}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================ driver
_RAW_PAYLOAD = []
_TAG = ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", type=int, default=None,
                help="Risk Lab id; 84398 = SPY, 39693 = Barnes Group")
    ap.add_argument("--rv-col", default=None, help="column name from --probe")
    ap.add_argument("--probe", type=int, default=None)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--T", type=int, default=250, help="rolling estimation window")
    ap.add_argument("--sigma2", choices=["win", "diff", "wdiff", "wloc"], default="win",
                    help="which correction drives the headline t_CC; all four are "
                         "reported in every row regardless")
    ap.add_argument("--est-cols", default=None,
                    help="comma-separated Risk Lab columns to enlarge with, e.g. "
                         "c2,c6,c7,c10,c11. Switches the alternative from own-lags to "
                         "other estimators of the same latent variance.")
    ap.add_argument("--est-lags", type=int, default=5,
                    help="lags per estimator; P = len(--est-cols) * --est-lags")
    ap.add_argument("--tag", default="", help="suffix for the log file name")
    ap.add_argument("--label", default=None,
                    help="human-readable asset name for the table caption, e.g. \"SPDR S&P 500 ETF\"; the Risk Lab id is appended")
    args = ap.parse_args()

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    logpath = os.path.join(LOGS, f"rv_{stamp}{tag}.log")
    sys.stdout = Tee(logpath)
    print(f"# rv_application.py  {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"# command: {' '.join(sys.argv)}")
    print(f"# log: {logpath}\n")

    if args.probe is not None:
        probe(fetch_risklab(args.probe)); return

    if args.offline:
        rv, asset = synthetic_rv(), "SYNTHETIC (--offline)"
        print("!" * 72)
        print("!!  --offline: this is SIMULATED volatility, not Risk Lab data.")
        print("!!  Nothing below is an empirical result. Drop --offline and pass")
        print("!!  --ticker <id> --rv-col <col> for the real series.")
        print("!" * 72 + "\n")
    else:
        if args.ticker is None:
            sys.exit("give --ticker (after --probe) or use --offline")
        raw = fetch_risklab(args.ticker)
        if args.rv_col is None:
            probe(raw); sys.exit("\nre-run with --rv-col <column>")
        name = args.label or "Risk Lab"
        _RAW_PAYLOAD.append(raw)
        rv, asset = (to_daily_variance(raw[args.rv_col]),
                     f"{name} (Risk Lab identifier {args.ticker})" if args.label
                     else f"Risk Lab {args.ticker}")

    global _TAG
    _TAG = ("_" + args.tag) if args.tag else ""
    T = args.T
    Ps = [10, 25, 50, 100]
    zgrid = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])

    val = validate(rv, T)

    har = har_design(rv)
    print(f"\n{asset}: {len(rv):,} daily RV observations, "
          f"{rv.index.min().date()} to {rv.index.max().date()}")
    print(f"HAR sample {len(har):,}, window T={T}")

    rows = []
    hdr = (f"{'P':>5}{'c=P/T':>8}{'z~':>9}{'Psihat_n':>10}{'ratio':>7}{'R2incr':>9}"
           f"{'MSPEgap':>11}{'t_DM':>7}{'t_CW':>7}{'t_CC':>7}"
           f"{'CCwdiff':>8}{'CCwloc':>8}{'':>7}")
    print("\n" + hdr); print("-" * len(hdr))
    if args.est_cols:
        cols = [c.strip() for c in args.est_cols.split(",") if c.strip()]
        if args.offline:
            frames = synthetic_estimators(rv, k=len(cols))
        else:
            raw = _RAW_PAYLOAD[0]
            missing = [c for c in cols if c not in raw.columns]
            if missing:
                sys.exit(f"--est-cols: {missing} not in the payload; run --probe first")
            frames = {c: to_daily_variance(raw[c]) for c in cols}
        Ps = [len(frames) * args.est_lags]
        print(f"\nalternative: {len(frames)} other estimators x {args.est_lags} lags "
              f"-> P = {Ps[0]}  ({', '.join(frames)})")

    for P in Ps:
        ext = (extra_estimators(frames, args.est_lags) if args.est_cols
               else extra_lags(rv, P))
        # duplicated trading dates appear in a few Risk Lab payloads; without this the
        # design matrices come out with mismatched heights
        har_ = har[~har.index.duplicated(keep="first")]
        ext = ext[~ext.index.duplicated(keep="first")]
        idx = har_.index.intersection(ext.index)
        y    = har_.loc[idx, "y"].to_numpy()
        Xhar = np.column_stack([np.ones(len(idx)),
                                har_.loc[idx, ["rv_d", "rv_w", "rv_m"]].to_numpy()])
        Xext = ext.loc[idx].to_numpy()
        if not args.est_cols:
            # own-lag design only: lag 1 is already in HAR as rv_d, so the increment
            # adds lags 2..P.  With --est-cols every column is a lag of a DIFFERENT
            # estimator, none of which the HAR contains, so nothing is dropped.
            Xext = Xext[:, 1:]
        Peff = Xext.shape[1]
        yt, yb, yh, psi, s2w, psi_wd, psi_wl = rolling(y, Xhar, Xext, T, zgrid)
        s2d = 0.5 * np.mean(np.diff(yt) ** 2)
        n_ok = len(yt)
        for j, z in enumerate(zgrid):
            corrections = {
                "win":   s2w * psi[j],                          # scalar plug-in, default
                "diff":  np.full(n_ok, s2d) * psi[j],           # scalar, differenced
                "wdiff": psi_wd[j],                             # weighted, residual diffs
                "wloc":  psi_wl[j],                             # weighted, local residual var
            }
            st = statistics(yt, yb, yh[j], psi[j], s2w,
                            corrections=corrections, headline=args.sigma2)
            st.update(P=Peff, c=Peff / T, z=float(z), zt=float(z) * T / Peff, n=n_ok)
            rows.append(st)
            note = "R<1" if st["ratio"] < 1 else ("CW ok" if st["Psihat"] < .05 else "")
            print(f"{Peff:>5}{Peff/T:>8.3f}{st['zt']:>9.1e}{st['Psihat']:>10.4f}"
                  f"{st['ratio']:>7.2f}{st['R2_incr']:>+9.4f}{st['mspe_gap']:>+11.2e}"
                  f"{st['t_DM']:>+7.2f}{st['t_CW']:>+7.2f}{st['t_CC']:>+7.2f}"
                  f"{st['t_CC_wdiff']:>+8.2f}{st['t_CC_wloc']:>+8.2f}{note:>7}")

    meta = dict(asset=asset, n=rows[-1]["n"], T=T, sigma2=args.sigma2,
                validation=val,
                offline=bool(args.offline), ticker=args.ticker, rv_col=args.rv_col,
                rv_measure=RISKLAB_COLUMNS.get(args.rv_col or "", "unknown"),
                run=stamp)
    write_output(rows, meta)
    rowpath = save_rows(rows, meta)
    print(f"\nfeasibility: T = {T}, n^(1/3) = {rows[-1]['n'] ** (1/3):.1f}  "
          f"-> {'INSIDE' if T <= rows[-1]['n'] ** (1/3) else 'OUTSIDE'} the bandwidth "
          f"condition (Table 5.1 states the gate, Section 5.6 measures the cost)")
    print(f"tables  -> {TABS}")
    print(f"figures -> {FIGS}")
    print(f"rows    -> {rowpath}")
    print(f"log     -> {logpath}")


if __name__ == "__main__":
    main()
