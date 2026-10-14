"""Monte Carlo matched to the realized-volatility application (Appendix B).

HAR benchmark with q = 4 estimated regressors, ridge alternative on persistent features,
conditionally heteroskedastic innovations, c between 0.04 and 0.4, and three innovation-
variance estimators (oracle, window-local HAR residual variance, and the weighted
sum_s a_s^2 sigma_s^2).

    python mc_har.py H1       size under the three nulls in the HAR design
    python mc_har.py H2       what drives the sanity ratio below one
    python mc_har.py H3       the empirical calibration and the bandwidth condition
    python mc_har.py all      [--reps 300] [--quick]

Output: output/mc_har/mc_har_H{1,2,3}.json and logs.
"""
import argparse, datetime, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "output", "mc_har")
LOGS = os.path.join(OUT, "logs")
os.makedirs(OUT, exist_ok=True); os.makedirs(LOGS, exist_ok=True)

import tempfile
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplcache-mchar"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)


class Tee:
    def __init__(self, path):
        self.f = open(path, "a"); self.o = sys.__stdout__
    def write(self, x): self.o.write(x); self.f.write(x); self.f.flush()
    def flush(self): self.o.flush(); self.f.flush()


def save(name, obj):
    p = os.path.join(OUT, name)
    with open(p, "w") as fh:
        json.dump(obj, fh, indent=1, default=float)
    print(f"  [saved] {p}")


# ============================================================ DGP
def gen_path(n, T, P, lam, rho_x=0.95, het=False, seed=0, burn=300):
    """log RV from a Corsi cascade, plus an exogenous persistent signal block.

    The target follows

        l_t = mu(1 - b_d - b_w - b_m) + b_d l_{t-1} + b_w l_w + b_m l_m
              + lam * x_{t-1}'theta + eps_t

    so HAR on own lags is the correctly specified benchmark when lam = 0, and the
    exogenous block x carries whatever incremental signal lam buys. Making the extra
    block exogenous rather than further own-lags is deliberate: own lags of a persistent
    series are so collinear that Psihat measures the conditioning of the design rather
    than its complexity, which is a different phenomenon and not the one under test.

    het=True makes sigma_t follow its own persistent process, so the innovations cluster
    the way log-RV innovations do. sigma_t is returned so the oracle correction can use
    the true within-window variances.
    """
    g = np.random.default_rng(seed)
    m = n + T + burn + 30
    bd, bw, bm, sig = 0.36, 0.34, 0.25, 0.42
    mu = math.log(0.02 ** 2 / 252)

    # exogenous persistent panel, standardised marginally
    x = np.empty((m, P)); x[0] = g.standard_normal(P)
    for t in range(1, m):
        x[t] = rho_x * x[t-1] + math.sqrt(1 - rho_x ** 2) * g.standard_normal(P)
    theta = g.standard_normal(P); theta /= np.linalg.norm(theta)

    if het:                       # persistent vol-of-vol, mean one
        lv = np.zeros(m)
        for t in range(1, m):
            lv[t] = 0.97 * lv[t-1] + 0.30 * g.standard_normal()
        s_t = np.exp(lv - lv.var() / 2)
    else:
        s_t = np.ones(m)
    eps = sig * s_t * g.standard_normal(m)

    f = lam * (x @ theta)                       # incremental signal, dated t
    l = np.full(m, mu)
    for t in range(22, m):
        d = l[t-1]; w = l[t-5:t].mean(); mm = l[t-22:t].mean()
        l[t] = mu * (1 - bd - bw - bm) + bd * d + bw * w + bm * mm + f[t-1] + eps[t]
    sl = slice(burn, burn + n + T + 22)
    return l[sl], x[sl], f[sl], (sig * s_t)[sl]


def har_matrix(l):
    """Target and HAR regressors from a log-RV path. Row t predicts l_{t+1}."""
    T_ = len(l)
    d = l[21:T_-1]
    w = np.array([l[t-4:t+1].mean() for t in range(21, T_-1)])
    m = np.array([l[t-21:t+1].mean() for t in range(21, T_-1)])
    y = l[22:T_]
    X = np.column_stack([np.ones(len(d)), d, w, m])
    return y, X, slice(21, T_-1)


# ============================================================ engine
def run_path(y, Xh, Xe, sig_true, T, z, scale=1.0):
    """Rolling HAR benchmark plus ridge increment on the HAR residuals, primal form.

    Returns the pieces the three statistics need, plus the three sigma^2 variants and
    the systematic/noise split of the increment used to impose the nulls.
    """
    n = len(y) - T - 1
    k = Xh.shape[1]
    yb = np.empty(n); gs = np.empty(n); vs = np.empty(n); psi = np.empty(n)
    s2w = np.empty(n); s2h = np.empty(n); s2o = np.empty(n)
    for i in range(n):
        H, Xw, Y = Xh[i:i+T], Xe[i:i+T], y[i:i+T]
        h_new, s_new = Xh[i+T], Xe[i+T]

        Qh, Rh = np.linalg.qr(H)
        beta = np.linalg.solve(Rh, Qh.T @ Y)
        yb[i] = h_new @ beta
        r = Y - H @ beta
        s2w[i] = r @ r / max(T - k, 1)

        QtXe = Qh.T @ Xw
        G = (Xw.T @ Xw - QtXe.T @ QtXe) / T
        rr = (Xw.T @ Y - QtXe.T @ (Qh.T @ Y)) / T
        gg, V = np.linalg.eigh(G); gg = np.clip(gg, 0.0, None)
        Vs = V.T @ s_new
        Ainv = 1.0 / (gg + z)
        # a = M Xe V diag(Ainv) Vs / T ; we need a itself for the robust sigma^2
        w_ = V @ (Ainv * Vs)                       # P-vector
        MXw = Xw - Qh @ QtXe
        a = (MXw @ w_) / T
        psi[i] = a @ a
        inc = a @ Y
        # split the increment into systematic and noise parts (needs the true mean)
        gs[i] = scale * (a @ (Y - r))              # a'(fitted part) = systematic
        vs[i] = scale * (a @ r)                    # a'(residual)    = estimation noise
        # sigma^2 variants
        s2o[i] = float(np.mean(sig_true[i:i+T] ** 2))                 # oracle, window
        loc = local_var(r, k)
        s2h[i] = float(np.sum(a ** 2 * loc) / max(psi[i], 1e-300))    # robust, in sigma^2 units
    return dict(yb=yb, psi=psi, gs=gs, vs=vs, s2w=s2w, s2h=s2h, s2o=s2o,
                yt=y[T:T+n], n=n)


def local_var(r, k, half=10):
    """Smoothed local variance of the window residuals, for the robust correction.
    A centred moving average of r^2, rescaled so its mean matches the window estimate."""
    r2 = r ** 2
    T = len(r)
    out = np.empty(T)
    cs = np.concatenate([[0.0], np.cumsum(r2)])
    for s in range(T):
        lo, hi = max(0, s - half), min(T, s + half + 1)
        out[s] = (cs[hi] - cs[lo]) / (hi - lo)
    return out * (r2.sum() / max(T - k, 1)) / max(out.mean(), 1e-300)


def statistics(R, s2, b):
    """DM, CW and CC from one path. s2 is a per-period array in variance units."""
    yt, yb, psi = R["yt"], R["yb"], R["psi"]
    D = R["gs"] + R["vs"]
    yh = yb + D
    e1, e2 = yt - yb, yt - yh
    dDM = e1 ** 2 - e2 ** 2
    dCW = dDM + D ** 2
    dCC = dDM + s2 * psi
    n = len(yt)
    def hac(x):
        x = x - x.mean(); gsum = np.mean(x * x)
        for h in range(1, b + 1):
            gsum += 2 * (1 - h / (b + 1)) * np.mean(x[h:] * x[:-h])
        return max(gsum, 1e-14)
    t = lambda d, v: math.sqrt(n) * d.mean() / math.sqrt(v)
    return dict(t_DM=t(dDM, dDM.var()), t_CW=t(dCW, dCW.var()), t_CC=t(dCC, hac(dCC)),
                Psihat=float(psi.mean()),
                ratio=float(np.mean(D ** 2) / np.mean(s2 * psi)),
                mspe_gap=float(np.mean(dDM)))


def one_rep(n, T, P, z, lam, het, seed, scale=1.0, s2mode="win"):
    l, x, f, sig = gen_path(n, T, P, lam, het=het, seed=seed)
    y, Xh, sl = har_matrix(l)
    Xe = x[sl]
    Xe = (Xe - Xe.mean(0)) / np.where(Xe.std(0) < 1e-12, 1.0, Xe.std(0))
    m = min(len(y), len(Xe))
    R = run_path(y[:m], Xh[:m], Xe[:m], sig[sl][:m], T, z, scale=scale)
    s2 = {"win": R["s2w"], "oracle": R["s2o"], "robust": R["s2h"] * R["s2w"] / 1.0}[s2mode]
    if s2mode == "robust":
        s2 = R["s2h"]
    b = max(1, int(np.ceil(4 * (R["n"] / 100) ** (2 / 9))))
    return statistics(R, s2, b), R


# ============================================================ H1
def arm_H1(reps=300, n=1200, T=250, quick=False):
    print("=" * 96); print("H1 - size and the three nulls in the HAR design"); print("=" * 96)
    Ps  = [10, 25, 50] if not quick else [25]
    zts = [0.1, 1.0, 10.0] if not quick else [1.0]
    lam = 0.06
    if quick: reps = 40
    rows = []
    for P in Ps:
        for zt in zts:
            z = zt * P / T
            # pilot: population moments -> the rescalings that impose each null
            Efg = Eg2 = Ev2 = 0.0
            for r in range(12):
                _, R = one_rep(400, T, P, z, lam, False, 90000 + r)
                Efg += float(np.mean(R["gs"] * (R["yt"] - R["yb"] - R["vs"])))
                Eg2 += float(np.mean(R["gs"] ** 2)); Ev2 += float(np.mean(R["vs"] ** 2))
            Efg /= 12; Eg2 /= 12; Ev2 /= 12
            a_CC = 2 * Efg / max(Eg2, 1e-12)
            a_DM = 2 * Efg / max(Eg2 + Ev2, 1e-12)
            print(f"\n--- P={P} c={P/T:.3f} zt={zt} | E[fg]={Efg:.5f} E[g2]={Eg2:.5f} "
                  f"E[v2]={Ev2:.5f} | a_CC={a_CC:.3f} a_DM={a_DM:.3f}")
            for null, sc in [("H0-cov", 0.0), ("H0-CC", a_CC), ("H0-MSPE", a_DM)]:
                acc = {k: [] for k in ("t_DM", "t_CW", "t_CC", "ratio", "mspe_gap", "Psihat")}
                for r in range(reps):
                    st, _ = one_rep(n, T, P, z, lam, False, 1000 + r, scale=sc)
                    for k in acc: acc[k].append(st[k])
                rec = dict(P=P, c=P/T, zt=zt, null=null, scale=sc,
                           rej_DM=float(np.mean(np.array(acc["t_DM"]) > 1.645)),
                           rej_CW=float(np.mean(np.array(acc["t_CW"]) > 1.645)),
                           rej_CC=float(np.mean(np.array(acc["t_CC"]) > 1.645)),
                           ratio=float(np.mean(acc["ratio"])),
                           mspe_gap=float(np.mean(acc["mspe_gap"])),
                           Psihat=float(np.mean(acc["Psihat"])), reps=reps)
                rows.append(rec)
                own = {"H0-cov": "CW", "H0-CC": "CC", "H0-MSPE": "DM"}[null]
                print(f"   {null:<8} Psihat={rec['Psihat']:.4f} R={rec['ratio']:.2f} "
                      f"gap={rec['mspe_gap']:+.4f} | rej DM/CW/CC = "
                      f"{rec['rej_DM']:.3f} {rec['rej_CW']:.3f} {rec['rej_CC']:.3f}"
                      f"   (own: {own})")
    save("mc_har_H1.json", rows)
    return rows


# ============================================================ H2
def arm_H2(reps=300, n=1200, T=250, quick=False):
    print("=" * 96)
    print("H2 - what drives the sanity ratio below one? homoskedastic vs clustered")
    print("=" * 96)
    print("""
Correct specification throughout: HAR is the true benchmark, the extra block is exogenous,
there is no endogeneity of any kind. The only thing that changes across rows is whether
the innovations are conditionally heteroskedastic, and which sigma^2 estimator is used.
R must exceed one in population.
""")
    Ps = [25, 50] if not quick else [25]
    zts = [0.1, 1.0] if not quick else [1.0]
    if quick: reps = 40
    rows = []
    hdr = (f"{'P':>4}{'zt':>7}{'het':>6}{'sigma2':>9}{'Psihat':>9}{'R':>7}"
           f"{'P(R<1)':>9}{'rej_CC':>9}")
    print(hdr); print("-" * len(hdr))
    for P in Ps:
        for zt in zts:
            z = zt * P / T
            for het in (False, True):
                for s2mode in ("oracle", "win", "robust"):
                    rr, cc, ps = [], [], []
                    for r in range(reps):
                        st, _ = one_rep(n, T, P, z, 0.0, het, 2000 + r, s2mode=s2mode)
                        rr.append(st["ratio"]); cc.append(st["t_CC"]); ps.append(st["Psihat"])
                    rec = dict(P=P, zt=zt, het=bool(het), s2=s2mode,
                               Psihat=float(np.mean(ps)), ratio=float(np.mean(rr)),
                               frac_bad=float(np.mean(np.array(rr) < 1)),
                               rej_CC=float(np.mean(np.array(cc) > 1.645)), reps=reps)
                    rows.append(rec)
                    print(f"{P:>4}{zt:>7}{str(het):>6}{s2mode:>9}{rec['Psihat']:>9.4f}"
                          f"{rec['ratio']:>7.2f}{rec['frac_bad']:>9.2f}{rec['rej_CC']:>9.3f}")
    save("mc_har_H2.json", rows)
    return rows


# ============================================================ H3
def arm_H3(reps=200, T=250, quick=False):
    print("=" * 96)
    print("H3 - the empirical calibration and the bandwidth condition")
    print("=" * 96)
    if quick: reps = 30
    rows = []
    print(f"{'T':>5}{'n':>7}{'n^(1/3)':>9}{'F2':>5}{'size CC':>9}{'size CW':>9}")
    print("-" * 44)
    for (T_, n_) in [(20, 1200), (60, 1200), (250, 1200), (250, 4000)]:
        cc, cw = [], []
        for r in range(reps):
            st, _ = one_rep(n_, T_, 25, 1.0 * 25 / T_, 0.0, False, 3000 + r)
            cc.append(st["t_CC"]); cw.append(st["t_CW"])
        f2 = n_ >= 4 * T_ ** 3
        rec = dict(T=T_, n=n_, F2=bool(f2),
                   size_CC=float(np.mean(np.array(cc) > 1.645)),
                   size_CW=float(np.mean(np.array(cw) > 1.645)), reps=reps)
        rows.append(rec)
        print(f"{T_:>5}{n_:>7}{n_**(1/3):>9.1f}{'yes' if f2 else 'no':>5}"
              f"{rec['size_CC']:>9.3f}{rec['size_CW']:>9.3f}")
    save("mc_har_H3.json", rows)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["H1", "H2", "H3", "all"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reps", type=int, default=300)
    a = ap.parse_args()
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sys.stdout = Tee(os.path.join(LOGS, f"mc_har_{a.arm}_{stamp}.log"))
    print(f"# mc_har.py {a.arm}  {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"# command: {' '.join(sys.argv)}\n")
    t0 = time.time()
    if a.arm in ("H1", "all"): arm_H1(reps=a.reps, quick=a.quick)
    if a.arm in ("H2", "all"): arm_H2(reps=a.reps, quick=a.quick)
    if a.arm in ("H3", "all"): arm_H3(reps=a.reps, quick=a.quick)
    print(f"\n# finished in {(time.time()-t0)/60:.1f} min. Output in {OUT}")
