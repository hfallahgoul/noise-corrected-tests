"""The theorems' own design (tab:mc-conform, fig:mc-grid; Section 5).

Every other simulation in the paper relaxes at least one condition of the limit theory: a
rolling-mean or HAR benchmark, persistent feature rows, an estimated sigma^2, or a window too
long for T^3 << n.  This script runs the one design that meets all of them at once:

    zero benchmark (yhat1 = 0); feature rows S_t i.i.d. N(0, I_P); innovations eps_t i.i.d.
    N(0, 1); sigma^2 known (oracle); T = 8, P = 4 (P = 6 at Psi* = 1); Bartlett HAC with
    T < b < sqrt(n/T).

and it reaches each null WITHOUT rescaling the forecast.  The signal is

    f_{t-1} = theta' S_{t-1} + u_{t-1},    u_t i.i.d. N(0, tau^2),

where u is a predictor the forecaster observes but leaves out of the feature set, independent
of the features and of the innovations.  It is part of the predictor path, so f is measurable
with respect to it and sigma^2 = 1 is the innovation variance.  theta carries the part the
features span and u the part they cannot.  u_{t-1} is outside
the training window, so it adds nothing to E[fg]; inside the window it enters the training
targets and so the fitted increment g, adding tau^2 E[Psihat] to E[g^2].  Hence

    mu_CC = m(theta) - tau^2 E[Psihat],     m(theta) = 2E[f_s g_s] - E[g_s^2] (spanned part),
    MSPE gap = mu_CC - sigma^2 E[Psihat],

and m(theta) is proportional to ||theta||^2.  A pilot measures m per unit R^2 and E[Psihat];
the rows below then follow exactly, up to the pilot's sampling error:

    sharp     theta = 0, tau = 0                        f = 0: every null holds
    cov       theta = 0, tau^2 = 0.10                    E[fg] = 0 with f != 0: H_cov boundary
    cc        R^2 = 0.10 spanned, tau^2 = m / E[Psihat]  mu_CC = 0 with E[fg] > 0: H_CC boundary
    mspe      tau^2 = 0.5, R^2 such that gap = 0          H_MSPE boundary
    alt       R^2 = 0.10 spanned, tau = 0                 mu_CC > 0: power

    python mc_conform.py --validate           # engine against ccpa.py on one path
    python mc_conform.py --run                # every cell (about 30 minutes)
    python mc_conform.py --run --psi 0.5 --n 8000 --rows cc,mspe
    python mc_conform.py --table              # print every stored cell

Output: output/mc_conform/<row>_psi<Psi*>_n<n>.json, one file per cell, holding the
replication-level t-statistics.  Seeds derive from the cell, so a rerun reproduces the file.
"""
import argparse, hashlib, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output", "mc_conform")
T = 8
PMAP = {0.2: 4, 0.5: 4, 1.0: 6}       # P per target: Psi(c, z) = 1 needs c > 0.5 at z > 0
REPS = 2000
BW = {2000: 12, 8000: 18, 32000: 26}          # T < b < sqrt(n/T) in every cell
GRID = [(0.5, 8000, ("sharp", "cov", "cc", "mspe", "alt")),     # panel A and the figure
        (0.2, 8000, ("sharp", "cov", "cc", "mspe", "alt")),
        (1.0, 8000, ("sharp", "cov", "cc", "mspe", "alt")),
        (0.5, 2000, ("sharp", "cc")),                            # panel B: size against n
        (0.5, 32000, ("sharp", "cc"))]
CRIT = 1.6448536269514722
R2_SPAN, TAU2_COV, TAU2_MSPE = 0.10, 0.10, 0.5


def seed(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def psi_closed(c, z):
    u = 1.0 + c + z
    return 0.5 * (u / math.sqrt(u * u - 4 * c) - 1.0)


def solve_z(c, target, lo=1e-8, hi=1e4):
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if psi_closed(c, mid) > target else (lo, mid)
    return math.sqrt(lo * hi)


def bartlett(x, bw):
    x = x - x.mean(); m = len(x); s = x @ x / m
    for l in range(1, bw + 1):
        s += 2 * (1 - l / (bw + 1)) * (x[l:] @ x[:-l]) / m
    return s


def gen(n, theta, tau, g):
    """y[s] = f[s-1] + eps[s], f[s] = theta'S[s] + u[s]; rows of S i.i.d. N(0, I_P)."""
    m = n + T + 1
    P = len(theta)
    S = g.standard_normal((m, P))
    u = tau * g.standard_normal(m)
    eps = g.standard_normal(m)
    f = S @ theta + u
    y = np.r_[eps[0], f[:-1] + eps[1:]]
    return y, S, f


def engine(y, S, f, z, b, sigma2=1.0):
    """Zero benchmark, ridge on the window rows i..i+T-1 with targets y[i+1..i+T], forecast
    of y[i+T+1] from S[i+T].  Returns the three statistics and the realised moments."""
    n = len(y) - T - 1
    P = S.shape[1]
    idx = np.arange(n)[:, None] + np.arange(T)[None, :]
    X, Yw, Fw = S[idx], y[idx + 1], f[idx]
    A = np.einsum("ntp,ntq->npq", X, X) / T + z * np.eye(P)
    w = np.linalg.solve(A, S[np.arange(n) + T][..., None])[..., 0]
    a = np.einsum("ntp,np->nt", X, w) / T                 # weights on the training targets
    g = np.einsum("nt,nt->n", a, Fw)
    yh = np.einsum("nt,nt->n", a, Yw)
    psi = np.einsum("nt,nt->n", a, a)
    yt, ft = y[np.arange(n) + T + 1], f[np.arange(n) + T]
    dDM = yt ** 2 - (yt - yh) ** 2
    dCW = dDM + yh ** 2
    dCC = dDM + sigma2 * psi
    rn = math.sqrt(n)
    return dict(t_DM=rn * dDM.mean() / math.sqrt(max(bartlett(dDM, b), 1e-14)),
                t_CW=rn * dCW.mean() / dCW.std(),
                t_CC=rn * dCC.mean() / math.sqrt(max(bartlett(dCC, b), 1e-14)),
                mu_CC=float(np.mean(2 * ft * g - g * g)), Efg=float(np.mean(ft * g)),
                mspe_gap=float(dDM.mean()), Psihat=float(psi.mean()))


def pilot(psit, n):
    """m per unit spanned R^2 and E[Psihat], from 100 independent paths."""
    P = PMAP[psit]
    z = solve_z(P / T, psit)
    g = np.random.default_rng(seed("pilot", psit, n))
    th1 = g.standard_normal(P); th1 /= np.linalg.norm(th1)      # unit spanned variance
    m, ps = [], []
    for _ in range(100):
        y, S, f = gen(n, th1, 0.0, g)
        r = engine(y, S, f, z, BW[n])
        m.append(r["mu_CC"]); ps.append(r["Psihat"])
    return z, th1, float(np.mean(m)), float(np.mean(ps))


def design(row, th1, m1, psi):
    """(theta, tau) for each row; m1 = m(theta) at unit spanned R^2."""
    if row == "sharp":
        return 0 * th1, 0.0
    if row == "cov":
        return 0 * th1, math.sqrt(TAU2_COV)
    if row == "cc":
        return math.sqrt(R2_SPAN) * th1, math.sqrt(R2_SPAN * m1 / psi)
    if row == "mspe":                     # m1 R2 - tau^2 psi - psi = 0
        return math.sqrt(psi * (1 + TAU2_MSPE) / m1) * th1, math.sqrt(TAU2_MSPE)
    if row == "alt":
        return math.sqrt(R2_SPAN) * th1, 0.0
    raise ValueError(row)


def run(psit, n, rows, reps=REPS):
    os.makedirs(OUT, exist_ok=True)
    todo = [r for r in rows if not os.path.exists(os.path.join(OUT, f"{r}_psi{psit}_n{n}.json"))]
    if not todo:
        return
    z, th1, m1, psi = pilot(psit, n)
    print(f"Psi*={psit} n={n} z={z:.4g} b={BW[n]} | pilot m per unit R2={m1:.4f} "
          f"E[Psihat]={psi:.4f}", flush=True)
    for row in todo:
        theta, tau = design(row, th1, m1, psi)
        g = np.random.default_rng(seed("cell", row, psit, n))
        t0 = time.time()
        acc = [engine(*gen(n, theta, tau, g), z, BW[n]) for _ in range(reps)]
        A = {k: [a[k] for a in acc] for k in acc[0]}
        rec = dict(row=row, psi_t=psit, n=n, T=T, P=len(theta), c=len(theta) / T, z=z, b=BW[n], reps=reps,
                   R2_span=float(theta @ theta), tau2=tau * tau,
                   pilot_m_unit=m1, pilot_psi=psi, **A)
        json.dump(rec, open(os.path.join(OUT, f"{row}_psi{psit}_n{n}.json"), "w"))
        r = lambda k: np.mean(np.array(A[k]) > CRIT)
        print(f"  {row:5s} R2={rec['R2_span']:.3f} tau2={rec['tau2']:.3f} | "
              f"mu_CC={np.mean(A['mu_CC']):+.4f} E[fg]={np.mean(A['Efg']):+.4f} "
              f"gap={np.mean(A['mspe_gap']):+.4f} Psihat={np.mean(A['Psihat']):.3f} | "
              f"DM {r('t_DM'):.3f} CW {r('t_CW'):.3f} CC {r('t_CC'):.3f} "
              f"[{time.time() - t0:.0f}s]", flush=True)


def table():
    import glob
    for fn in sorted(glob.glob(os.path.join(OUT, "*.json"))):
        d = json.load(open(fn))
        r = lambda k: np.mean(np.array(d[k]) > CRIT)
        print(f"{d['row']:5s} Psi*={d['psi_t']} n={d['n']:5d} | mu_CC={np.mean(d['mu_CC']):+.4f} "
              f"gap={np.mean(d['mspe_gap']):+.4f} | DM {r('t_DM'):.3f} CW {r('t_CW'):.3f} "
              f"CC {r('t_CC'):.3f}")


def validate():
    """The engine against ccpa.py, path by path, on one simulated path."""
    sys.path.insert(0, HERE)
    import ccpa
    psit, n = 0.5, 400
    P = PMAP[psit]; z = solve_z(P / T, psit); b = BW[2000]
    g = np.random.default_rng(1)
    th = g.standard_normal(P); th *= math.sqrt(0.1) / np.linalg.norm(th)
    y, S, f = gen(n, th, 0.3, g)
    e = engine(y, S, f, z, b)
    yh, ps = np.empty(n), np.empty(n)
    for i in range(n):
        a = ccpa.ridge_increment_weights(S[i:i + T], S[i + T], z)
        yh[i] = a @ y[i + 1:i + T + 1]; ps[i] = a @ a
    yt = y[T + 1:T + n + 1]
    cc = ccpa.cc_test(yt, np.zeros(n), yh, ps, 1.0, b)
    dmcw = ccpa.dm_cw(yt, np.zeros(n), yh)
    print(f"t_CC  engine={e['t_CC']:+.12f}  ccpa={cc['t']:+.12f}")
    print(f"t_CW  engine={e['t_CW']:+.12f}  ccpa={dmcw['t_CW']:+.12f}")
    print(f"t_DM  engine={e['t_DM']:+.12f}  ccpa(HAC)={math.sqrt(n) * (yt**2 - (yt - yh)**2).mean() / math.sqrt(ccpa.bartlett_lrv(yt**2 - (yt - yh)**2, b)):+.12f}")
    ok = abs(e["t_CC"] - cc["t"]) < 1e-9 and abs(e["t_CW"] - dmcw["t_CW"]) < 1e-9
    print("engine agrees with ccpa" if ok else "MISMATCH")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--psi", type=float, default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--rows", default=None)
    a = ap.parse_args()
    if a.validate:
        validate()
    if a.run:
        for psit, n, rows in GRID:
            if (a.psi is None or a.psi == psit) and (a.n is None or a.n == n):
                run(psit, n, a.rows.split(",") if a.rows else rows)
    if a.table:
        table()
