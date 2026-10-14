"""Window sweep for tab:window (Section 5.4, sec:mc-window).

The design of tab:mc-signal (mc_study.arm_S): P = 6 AR(1) predictors with rho_x = 0.3, a
rolling-mean benchmark, a ridge alternative at the penalty z solving Psi(0.5, z) = 0.5,
oracle sigma^2 = 1 and n = 8000.  The penalty is held at its T = 12 value and only the window
T changes, so c = P/T and the estimation-noise term sigma^2 Psi(P/T, z) fall with T.  The
table reports, for each T, the mean-square gap and the DM and CC rejection frequencies, and
the predicted crossover T* solving sigma^2 Psi(P/T*, z) = mu_CC(T = 12).

The engine is the primal form of mc_study.run_path, vectorised with rolling sums; by the
push-through identity it equals run_path's dual form, and --validate prints both.

    python mc_window.py --validate
    python mc_window.py --run --signal spiked --R2 0.20 --T 12,24,48,96,192 --reps 1000
    python mc_window.py --table                 # print every stored cell
    python mc_window.py --latex tab-window.tex  # write the table

Output: output/mc_window/<signal>_R2<R2>_T<T>.json, one file per cell.
Runtime: about 20 seconds per cell at 1,000 replications.
"""
import argparse, json, math, os, sys, hashlib, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output", "mc_window")
os.makedirs(OUT, exist_ok=True)
RHO, P, N, C0, PSI0 = 0.3, 6, 8000, 0.5, 0.5

def psi_closed(c, z):
    u = 1.0 + c + z; q = u*u - 4*c
    return np.inf if q <= 0 else 0.5*(u/math.sqrt(q) - 1.0)

def solve_z(c, target, lo=1e-8, hi=1e4):
    for _ in range(200):
        mid = math.sqrt(lo*hi)
        if psi_closed(c, mid) > target: lo = mid
        else: hi = mid
    return math.sqrt(lo*hi)

Z = solve_z(C0, PSI0)          # the penalty of the T=12 cell, held fixed throughout

def seed(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)

def bartlett(x, bw):
    x = x - x.mean(); m = len(x); s = x @ x / m
    for l in range(1, bw+1):
        s += 2*(1 - l/(bw+1)) * (x[l:] @ x[:-l]) / m
    return s

def theta_for(signal, R2, g):
    if signal == "spiked":
        th = np.zeros(P); th[0] = 1.0
    else:
        th = g.standard_normal(P); th /= np.linalg.norm(th)
    return th*math.sqrt(R2)

def gen(T, n, R2, th, g):
    """Same DGP as mc_study.gen_path (dense/spiked, gauss): y_{t+1} = S_t'theta + eps."""
    m = n + T + 2
    eta = g.standard_normal((m, P)); S = np.empty((m, P)); S[0] = eta[0]
    for t in range(1, m):
        S[t] = RHO*S[t-1] + math.sqrt(1-RHO**2)*eta[t]
    eps = g.standard_normal(m)
    f = S @ th
    y = np.roll(f, 1) + eps; y[0] = eps[0]
    return y, S, f

def roll(A, T):
    """Sums over windows [i, i+T) for i = 0..len(A)-T, along axis 0."""
    cs = np.concatenate([np.zeros((1,)+A.shape[1:]), np.cumsum(A, axis=0)])
    return cs[T:] - cs[:-T]

def engine(y, S, f, T, z, b, sigma2=1.0):
    n = len(y) - T - 1
    X = S[:n+T]                                   # rows used in windows i..i+T-1
    Yn = y[1:n+T+1]                               # y_{s+1} paired with S_s
    Fn = f[:n+T]
    sX = roll(X, T)[:n]; sF = roll(Fn, T)[:n]; sY = roll(Yn, T)[:n]
    XX = roll(np.einsum("ti,tj->tij", X, X), T)[:n]
    XF = roll(X*Fn[:, None], T)[:n]; XY = roll(X*Yn[:, None], T)[:n]
    s = S[T:T+n]
    A = z*np.eye(P)[None] + XX/T
    w = np.linalg.solve(A, s[..., None])[..., 0]          # (n, P)
    xbar = sX/T
    # a = M X w / T  ->  a'v = (w'(X'v) - T (xbar'w) vbar)/T
    g_ = (np.einsum("ij,ij->i", w, XF) - np.einsum("ij,ij->i", xbar, w)*sF)/T
    aY = (np.einsum("ij,ij->i", w, XY) - np.einsum("ij,ij->i", xbar, w)*sY)/T
    v_ = aY - g_
    XcX = XX - T*np.einsum("ti,tj->tij", xbar, xbar)
    psi = np.einsum("ti,tij,tj->t", w, XcX, w)/T**2
    y1 = sY/T
    yt = y[T+1:T+n+1]; ft = f[T:T+n]
    D = g_ + v_; yh = y1 + D
    dDM = (yt-y1)**2 - (yt-yh)**2
    dCW = dDM + D**2
    dCC = dDM + sigma2*psi
    rn = math.sqrt(n)
    return dict(
        t_DM=rn*dDM.mean()/math.sqrt(max(bartlett(dDM, b), 1e-14)),
        t_CW=rn*dCW.mean()/dCW.std(),
        t_CC=rn*dCC.mean()/math.sqrt(max(bartlett(dCC, b), 1e-14)),
        mspe_gap=float(dDM.mean()), mu_CC=float(np.mean(2*ft*g_ - g_**2)),
        Psihat=float(psi.mean()), Efg=float(np.mean(ft*g_)), Eg2=float(np.mean(g_**2)))

def bw(T):
    return max(18, int(math.ceil(1.5*T)))      # b = 18 at T = 12, as in arm_S; b > T after

def validate():
    sys.path.insert(0, HERE)
    import mc_study as M
    g = np.random.default_rng(1)
    th = theta_for("dense", 0.2, g)
    y, S, f = gen(12, 600, 0.2, th, g)
    a = engine(y, S, f, 12, Z, 18)
    r = M.run_path(y, S, f, 12, Z, 18, 1.0, 1.0)
    for k in ("t_DM", "t_CW", "mspe_gap", "mu_CC", "Psihat"):
        print(f"{k:9s} primal={a[k]:+.10f}  run_path={r[k]:+.10f}")
    print(f"t_CC      primal={a['t_CC']:+.10f}  run_path={r['t_CC_hac']:+.10f}")

def run(signal, R2, Ts, reps):
    for T in Ts:
        fn = os.path.join(OUT, f"{signal}_R2{R2}_T{T}.json")
        if os.path.exists(fn):
            print(f"[skip] {fn}"); continue
        g = np.random.default_rng(seed("w1a", signal, R2, T))
        th = theta_for(signal, R2, np.random.default_rng(seed("theta", signal, R2)))
        b = bw(T); t0 = time.time(); acc = []
        for _ in range(reps):
            y, S, f = gen(T, N, R2, th, g)
            acc.append(engine(y, S, f, T, Z, b))
        A = {k: np.array([a[k] for a in acc]) for k in acc[0]}
        rec = dict(signal=signal, R2=R2, T=T, c=P/T, z=Z, b=b, n=N, reps=reps,
                   F3=bool(T < b < math.sqrt(N/T)),
                   Psi_closed=psi_closed(P/T, Z), Psihat=float(A["Psihat"].mean()),
                   mu_CC=float(A["mu_CC"].mean()), mspe_gap=float(A["mspe_gap"].mean()),
                   mspe_gap_se=float(A["mspe_gap"].std()/math.sqrt(reps)),
                   Efg2=2*float(A["Efg"].mean()), Eg2=float(A["Eg2"].mean()),
                   rej_DM=float(np.mean(A["t_DM"] > 1.645)),
                   rej_CW=float(np.mean(A["t_CW"] > 1.645)),
                   rej_CC=float(np.mean(A["t_CC"] > 1.645)),
                   secs=round(time.time()-t0, 1))
        json.dump(rec, open(fn, "w"), indent=1)
        print(f"{signal} R2={R2} T={T:4d} c={P/T:.3f} b={b} F3={rec['F3']} | "
              f"Psi={rec['Psihat']:.4f} mu_CC={rec['mu_CC']:+.4f} gap={rec['mspe_gap']:+.4f} "
              f"| DM/CW/CC = {rec['rej_DM']:.3f} {rec['rej_CW']:.3f} {rec['rej_CC']:.3f} "
              f"[{rec['secs']}s]", flush=True)

def table():
    import glob
    R = [json.load(open(f)) for f in glob.glob(os.path.join(OUT, "*.json"))]
    R.sort(key=lambda r: (r["signal"], r["R2"], r["T"]))
    for r in R:
        print(f"{r['signal']:7s} {r['R2']:.2f} T={r['T']:4d} c={r['c']:.3f} F3={str(r['F3']):5s} "
              f"Psi={r['Psihat']:.4f} mu_CC={r['mu_CC']:+.4f} gap={r['mspe_gap']:+.4f}"
              f"(±{r['mspe_gap_se']:.4f}) DM={r['rej_DM']:.3f} CW={r['rej_CW']:.3f} "
              f"CC={r['rej_CC']:.3f}")


def latex(path):
    """tab:window - spiked signal; one block per R^2, predicted T* from mu_CC at T = 12."""
    import glob
    R = [json.load(open(f)) for f in glob.glob(os.path.join(OUT, "spiked_*.json"))]
    blocks = {}
    for r in R:
        blocks.setdefault(r["R2"], []).append(r)
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{The window at which the enlarged model starts to pay. The design of "
         r"\cref{tab:mc-signal} (spiked signal, $P=6$ AR(1) predictors with $\rho_x=0.3$, rolling-mean "
         r"benchmark, oracle $\sigma^2=1$, $n=8{,}000$, $1{,}000$ replications per cell, drawn independently of "
         r"\cref{tab:mc-signal}) with the ridge "
         r"penalty held at its $T=12$ value, $z=0.133$ ($\Psi=0.5$ at $c=0.5$), and only the window $T$ "
         r"varied. gap is the mean-square gap (positive when the enlarged model is more accurate). "
         r"Bartlett bandwidth $b=\max(18,\lceil1.5T\rceil)$. $T^*$ is the smallest window with "
         r"$\sigma^2\Psi(P/T^*,z)\le\mu_{\mathrm{CC}}$, with $\mu_{\mathrm{CC}}$ taken at $T=12$, "
         r"the value a user has. One-sided $5\%$ rejection frequencies. Dense signal, run at $R^2=0.20$ "
         r"and $0.10$, gives the same crossovers to the second decimal of the gap.}",
         r"\label{tab:window}", r"\small", r"\setlength{\tabcolsep}{5pt}",
         r"\begin{tabular}{rrrrrrrrr}", r"\toprule",
         r"$R^2$ & $T^*$ & $T$ & $c$ & $\Psihatn$ & $\mu_{\mathrm{CC}}$ & gap & DM & CC \\", r"\midrule"]
    first = True
    for R2 in sorted(blocks, reverse=True):
        rows = sorted(blocks[R2], key=lambda r: r["T"])
        mu12 = [r for r in rows if r["T"] == 12][0]["mu_CC"]
        T = 6
        while psi_closed(P/T, Z) > mu12: T += 1
        if not first: L.append(r"\addlinespace")
        first = False
        for k, r in enumerate(rows):
            g = r["mspe_gap"]
            gs = f"${g:.3f}$" if g > 0 else f"$-{abs(g):.3f}$"
            L.append(f"{(f'{R2:.2f}' if k == 0 else '')} & {(str(T) if k == 0 else '')} & {r['T']} & "
                     f"{r['c']:.3f} & {r['Psihat']:.3f} & {r['mu_CC']:.3f} & {gs} & "
                     f"{r['rej_DM']:.3f} & {r['rej_CC']:.3f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(path, "w").write("\n".join(L) + "\n")
    print("[wrote]", path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--latex", default=None, help="write tab:window to this path")
    ap.add_argument("--signal", default="spiked")
    ap.add_argument("--R2", type=float, default=0.20)
    ap.add_argument("--T", default="12,24,48,96")
    ap.add_argument("--reps", type=int, default=1000)
    a = ap.parse_args()
    if a.validate: validate()
    if a.run: run(a.signal, a.R2, [int(t) for t in a.T.split(",")], a.reps)
    if a.table: table()
    if a.latex: latex(a.latex)
