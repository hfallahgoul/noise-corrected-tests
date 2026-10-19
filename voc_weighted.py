"""The heteroskedasticity-weighted correction on the equity random-feature design.

Same data, features, window and penalty grid as voc_application.py (tab:voc-psi,
tab:voc-grid). Adds the weighted correction of prop:sigma-weighted,

    Psitil_t = sum_s a_{t,s}^2 * (1/2)(y_s - y_{s-1})^2 ,

which gives each training date its own noise estimate instead of one scalar sigma^2, and
its sanity ratio R_w = mean(Delta_t^2) / mean(Psitil_t). The computation of a_t, of the
squared changes and of the statistic is the one in mc_study.run_path (t_CC_w, ratio_w), so
the simulated and the empirical numbers come from the same algebra.

    python voc_weighted.py            # under a minute

Output: output/voc_tables/voc_weighted.json and output/voc_tables/tab-voc-weighted.tex.
It reads nothing that voc_application.py writes and writes nothing it reads, so the
published equity tables are untouched.
"""
import json, math, os, sys
import numpy as np

from voc_application import goyal_welch, rff, mathsign, HERE, TABS

GAMMAS = [0.1, 0.3, 1.0, 3.0]
ZGRID = np.array([1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5])
CRIT = 1.645


def sweep_w(y, feats, T, zgrid):
    """voc_application.sweep, plus the weighted correction for every (window, z)."""
    n = len(y) - T - 1
    nz = len(zgrid)
    I, one = np.eye(T), np.ones(T)
    M = I - np.outer(one, one) / T
    d2h = 0.5 * np.r_[0.0, np.diff(y) ** 2]          # (1/2)(y_s - y_{s-1})^2, as run_path
    d2h[0] = d2h[1:].mean()
    yb = np.empty(n)
    yh = np.empty((nz, n)); psi = np.empty((nz, n)); wpsi = np.empty((nz, n))
    for i in range(n):
        X = feats[i:i + T]
        Y = y[i + 1:i + T + 1]
        s = feats[i + T]
        lam, Q = np.linalg.eigh(X @ X.T / T)
        QtXs = Q.T @ (X @ s)
        yb[i] = Y.mean()
        h = d2h[i + 1:i + T + 1]                       # rows of the training window
        for j, z in enumerate(zgrid):
            a = (M @ (Q @ (QtXs / (z + lam)))) / T
            yh[j, i] = yb[i] + a @ Y
            psi[j, i] = a @ a
            wpsi[j, i] = (a ** 2) @ h
    return y[T + 1:T + n + 1], yb, yh, psi, wpsi


def stats(yt, yb, yh, psi, wpsi):
    n = len(yt)
    e1, e2 = yt - yb, yt - yh
    D = yh - yb
    dDM = e1 ** 2 - e2 ** 2
    dCW = dDM + D ** 2
    s2 = 0.5 * np.mean(np.diff(yt) ** 2)
    dCC = dDM + s2 * psi
    dCCw = dDM + wpsi
    b = max(1, int(np.ceil(4 * (n / 100) ** (2 / 9))))

    def hac(x):
        x = x - x.mean(); g = np.mean(x * x)
        for h in range(1, b + 1):
            g += 2 * (1 - h / (b + 1)) * np.mean(x[h:] * x[:-h])
        return max(g, 1e-14)
    t = lambda d, v: math.sqrt(n) * d.mean() / math.sqrt(v)
    return dict(t_DM=t(dDM, dDM.var()), t_CW=t(dCW, dCW.var()),
                t_CC=t(dCC, hac(dCC)), t_CCw=t(dCCw, hac(dCCw)),
                Psihat=float(psi.mean()), Psitil=float(wpsi.mean()), sigma2=float(s2),
                ratio=float(np.mean(D ** 2) / np.mean(s2 * psi)),
                ratio_w=float(np.mean(D ** 2) / np.mean(wpsi)),
                mspe_gap=float(np.mean(dDM)))


def main():
    T, P = 12, 2000
    y, R, names = goyal_welch(os.path.join(HERE, "data", "PredictorData2021.csv"))
    R = (R - R.mean(0)) / R.std(0)
    rows = []
    print(f"{'gamma':>6}{'z':>8}{'Psihat':>9}{'Psitil/s2':>10}{'R':>7}{'R_w':>7}"
          f"{'t_CW':>7}{'t_CC':>7}{'t_CCw':>7}")
    for gam in GAMMAS:
        F = rff(R, P, gam)
        yt, yb, yh, psi, wpsi = sweep_w(y, F, T, ZGRID)
        for j, z in enumerate(ZGRID):
            s = stats(yt, yb, yh[j], psi[j], wpsi[j])
            s.update(gamma=gam, z=float(z), zt=float(z) * T / P)
            rows.append(s)
            print(f"{gam:>6.2f}{z:>8.0e}{s['Psihat']:>9.4f}{s['Psitil']/s['sigma2']:>10.4f}"
                  f"{s['ratio']:>7.2f}{s['ratio_w']:>7.2f}{s['t_CW']:>+7.2f}"
                  f"{s['t_CC']:>+7.2f}{s['t_CCw']:>+7.2f}")
    os.makedirs(TABS, exist_ok=True)
    json.dump(dict(n=len(yt), T=T, P=P, d=R.shape[1], rows=rows),
              open(os.path.join(TABS, "voc_weighted.json"), "w"), indent=1)

    ok_s = [r for r in rows if r["ratio"] >= 1]
    ok_w = [r for r in rows if r["ratio_w"] >= 1]
    rej = lambda rs, k: sum(r[k] > CRIT for r in rs)
    print(f"\ncells passing R >= 1: scalar {len(ok_s)}/{len(rows)}, weighted {len(ok_w)}/{len(rows)}")
    print(f"rejections, all cells: CW {rej(rows,'t_CW')}, CC {rej(rows,'t_CC')}, CC_w {rej(rows,'t_CCw')}")
    print(f"rejections where R_w >= 1: CW {rej(ok_w,'t_CW')}, CC_w {rej(ok_w,'t_CCw')} of {len(ok_w)}")

    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{The weighted correction on the equity design. Same data, features and"
         r" grid as \cref{tab:voc-grid}: monthly excess returns, $n=" + f"{len(yt):,}".replace(",", "{,}") +
         r"$ forecasts, $P=2{,}000$ random Fourier features of the $14$ predictors on a"
         r" rolling window of $T=12$. $\Psihatn$ is the plug-in of the scalar correction"
         r" $\hat\sigma^2_\Delta\Psihat_t$; $\Psitil_n/\hat\sigma^2_\Delta$ is the average"
         r" weighted correction of \cref{prop:sigma-weighted} in the same units. $\mathcal R$"
         r" and $\mathcal R_w$ are the sanity ratios of the scalar and weighted corrections,"
         r" the average squared forecast differential divided by the average correction; both"
         r" must exceed one in population. $t$-statistics are one-sided, CW studentised by its"
         r" sample variance and both corrected statistics by the HAC estimator with the"
         r" bandwidth of \cref{tab:voc-grid}.}",
         r"\label{tab:voc-weighted}", r"\small",
         r"\begin{tabular}{rrrrrrrrr}", r"\toprule",
         r"$\gamma$ & $z$ & $\Psihatn$ & $\Psitil_n/\hat\sigma^2_\Delta$ & $\mathcal R$"
         r" & $\mathcal R_w$ & $\tCW$ & $\tCC$ & $\tCC$, weighted \\", r"\midrule"]
    for gam in GAMMAS:
        first = True
        for r in [r for r in rows if r["gamma"] == gam]:
            L.append(f"{('%.1f' % gam) if first else ''} & {r['z']:.0e} & {r['Psihat']:.4f} & "
                     f"{r['Psitil']/r['sigma2']:.4f} & {r['ratio']:.2f} & {r['ratio_w']:.2f} & "
                     f"{r['t_CW']:.2f} & {r['t_CC']:.2f} & {r['t_CCw']:.2f} \\\\")
            first = False
        L.append(r"\addlinespace")
    L.pop()
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(TABS, "tab-voc-weighted.tex"), "w").write(mathsign("\n".join(L) + "\n"))
    print(f"-> {TABS}/voc_weighted.json, tab-voc-weighted.tex")


if __name__ == "__main__":
    main()
