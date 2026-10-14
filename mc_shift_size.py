"""Size and power of the corrected test when the correction is estimated by circular
shifting rather than by the plug-in sigma^2 Psihat_t (rem:sigma-outside).

Reports, under the sharp null and a sequence of alternatives, the corrected statistic with
the plug-in correction, the corrected statistic with the shifted squared forecast averaged
over several shifts, and Clark-West for reference.

    python mc_shift_size.py            # a few minutes
    python mc_shift_size.py --quick

Output: output/mc_har/shift_size.json and shift_size.log.
"""
import argparse, datetime, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output", "mc_har")
os.makedirs(OUT, exist_ok=True)


class Tee:
    def __init__(self, p):
        self.f = open(p, "a"); self.o = sys.__stdout__
    def write(self, x): self.o.write(x); self.f.write(x); self.f.flush()
    def flush(self): self.o.flush(); self.f.flush()


def one_path(bench, rho, lam, T, P, zt, n, rng, taus):
    """One replication. Returns the three statistics.

    bench : 'exog' benchmark regresses on exogenous variables (Lemma A.6 holds)
            'ar'   benchmark regresses on lags of the target (Lemma A.6 fails when the
                   extra features are exogenous -- the case this script is about)
    lam   : incremental signal loading on the exogenous feature block
    """
    z = zt * P / T
    m = n + T + 80
    sig = 0.42
    eps = sig * rng.standard_normal(m)
    Z = np.empty((m, 3)); Z[0] = rng.standard_normal(3)
    for t in range(1, m):
        Z[t] = 0.9 * Z[t-1] + math.sqrt(1 - 0.81) * rng.standard_normal(3)
    Xe = np.empty((m, P)); Xe[0] = rng.standard_normal(P)
    for t in range(1, m):
        Xe[t] = 0.95 * Xe[t-1] + math.sqrt(1 - 0.95**2) * rng.standard_normal(P)
    th = rng.standard_normal(P); th /= np.linalg.norm(th)
    f = lam * (Xe @ th)                       # incremental signal, dated t, enters y_{t+1}

    y = np.zeros(m)
    if bench == "ar":
        for t in range(1, m):
            y[t] = rho * y[t-1] + f[t-1] + eps[t]
    else:
        y = Z @ np.array([0.5, 0.3, 0.2]) + np.r_[0.0, f[:-1]] + eps

    L = 8
    Y = y[L:]; M_ = len(Y)
    if bench == "ar":
        H = np.column_stack([np.ones(M_), y[L-1:L-1+M_], y[L-2:L-2+M_], y[L-3:L-3+M_]])
    else:
        H = np.column_stack([np.ones(M_), Z[L:L+M_]])
    XE = Xe[L:L+M_]
    XS = [np.roll(XE, t_, axis=0) for t_ in taus]

    N = M_ - T - 1
    yb = np.empty(N); yh = np.empty(N); psi = np.empty(N)
    s2w = np.empty(N); shf = np.empty(N)
    k = H.shape[1]
    for i in range(N):
        Hw, Yw = H[i:i+T], Y[i:i+T]
        Qh, Rh = np.linalg.qr(Hw)
        beta = np.linalg.solve(Rh, Qh.T @ Yw)
        yb[i] = H[i+T] @ beta
        r = Yw - Hw @ beta
        s2w[i] = r @ r / max(T - k, 1)

        def increment(X_):
            Xw = X_[i:i+T]; s_ = X_[i+T]
            QtX = Qh.T @ Xw
            G = (Xw.T @ Xw - QtX.T @ QtX) / T
            gg, V = np.linalg.eigh(G); gg = np.clip(gg, 0.0, None)
            w_ = V @ ((V.T @ s_) / (gg + z))
            a = ((Xw - Qh @ QtX) @ w_) / T
            return a @ Yw, a @ a

        d, p_ = increment(XE)
        yh[i] = yb[i] + d; psi[i] = p_
        shf[i] = float(np.mean([increment(XSj)[0] ** 2 for XSj in XS]))

    yt = Y[T:T+N]
    e1, e2 = yt - yb, yt - yh
    D = yh - yb
    dDM = e1**2 - e2**2
    dCW = dDM + D**2
    dCCp = dDM + s2w * psi        # plug-in correction
    dCCs = dDM + shf              # shift correction
    b = max(1, int(np.ceil(4 * (N / 100) ** (2/9))))

    def hac(x):
        x = x - x.mean(); g = np.mean(x*x)
        for h in range(1, b+1):
            g += 2 * (1 - h/(b+1)) * np.mean(x[h:] * x[:-h])
        return max(g, 1e-14)
    t = lambda d_, v: math.sqrt(N) * d_.mean() / math.sqrt(v)
    return dict(t_CW=t(dCW, dCW.var()),
                t_CCp=t(dCCp, hac(dCCp)),
                t_CCs=t(dCCs, hac(dCCs)),
                R_plug=float(np.mean(D**2) / np.mean(s2w*psi)),
                R_shift=float(np.mean(D**2) / np.mean(shf)),
                Psihat=float(psi.mean()))


def run(bench, rho, lams, T, P, zt, n, reps, taus, seed=11):
    out = []
    for lam in lams:
        rng = np.random.default_rng(seed)
        acc = {k: [] for k in ("t_CW", "t_CCp", "t_CCs", "R_plug", "R_shift", "Psihat")}
        for _ in range(reps):
            s = one_path(bench, rho, lam, T, P, zt, n, rng, taus)
            for k in acc: acc[k].append(s[k])
        rec = dict(bench=bench, rho=rho, lam=lam, T=T, P=P, zt=zt, n=n, reps=reps,
                   rej_CW=float(np.mean(np.array(acc["t_CW"]) > 1.645)),
                   rej_CCp=float(np.mean(np.array(acc["t_CCp"]) > 1.645)),
                   rej_CCs=float(np.mean(np.array(acc["t_CCs"]) > 1.645)),
                   R_plug=float(np.mean(acc["R_plug"])),
                   R_shift=float(np.mean(acc["R_shift"])),
                   Psihat=float(np.mean(acc["Psihat"])),
                   sd_CCp=float(np.std(acc["t_CCp"])),
                   sd_CCs=float(np.std(acc["t_CCs"])))
        out.append(rec)
        print(f"  lam={lam:<5} Psihat={rec['Psihat']:.4f} | R plug/shift = "
              f"{rec['R_plug']:.2f}/{rec['R_shift']:.2f} | rej CW={rec['rej_CW']:.3f} "
              f"CC-plug={rec['rej_CCp']:.3f} CC-shift={rec['rej_CCs']:.3f} | "
              f"sd(t) {rec['sd_CCp']:.2f}/{rec['sd_CCs']:.2f}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args()
    sys.stdout = Tee(os.path.join(OUT, "shift_size.log"))
    print(f"# mc_shift_size.py  {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
    T, P, zt = 250, 25, 1.0
    n = a.n or (400 if a.quick else 700)
    reps = a.reps or (60 if a.quick else 250)
    taus = [151, 251, 351, 451]
    lams = [0.0, 0.02, 0.05] if a.quick else [0.0, 0.01, 0.02, 0.04, 0.08]
    t0 = time.time(); rows = []

    print("=" * 92)
    print("A. Benchmark regressors EXOGENOUS -- Lemma A.6 holds, both corrections valid")
    print("=" * 92)
    print("   lam = 0 is the sharp null; nominal size is 0.05.")
    rows += run("exog", 0.0, lams, T, P, zt, n, reps, taus)

    print("\n" + "=" * 92)
    print("B. Benchmark = lags of the target, rho = 0.85 -- the plug-in fails here")
    print("=" * 92)
    rows += run("ar", 0.85, lams, T, P, zt, n, reps, taus)

    with open(os.path.join(OUT, "shift_size.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"\n[saved] {os.path.join(OUT, 'shift_size.json')}")
    print(f"# finished in {(time.time()-t0)/60:.1f} min")
