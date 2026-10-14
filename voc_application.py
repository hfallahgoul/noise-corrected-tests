"""Psihat_n and the three statistics in the random-feature design of Kelly, Malamud and
Zhou (2024) (tab:voc-psi, tab:voc-grid, figure fig-voc-psi).

P random Fourier features of the Goyal-Welch predictors, ridge on a rolling T = 12 month
window, rolling-mean benchmark.  The script reports Psihat_n against the closed form
(1 - q/T) Psi(c, z) with q = 1 (prop:dof), the scale-free penalty ztil = zT/P, and t_DM,
t_CW, t_CC.  It does not adjudicate whether complexity helps in return prediction.

    python voc_application.py                  # default grid, under a minute
    python voc_application.py --P 12000

Input: data/PredictorData2021.csv.  Output: output/voc_tables/, output/voc_figures/.
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


import argparse, math, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TABS = os.path.join(HERE, "output", "voc_tables")
FIGS = os.path.join(HERE, "output", "voc_figures")


# ----------------------------------------------------------------- data
def goyal_welch(path):
    raw = pd.read_csv(path)
    raw.columns = [c.strip() for c in raw.columns]
    for c in raw.columns:
        if raw[c].dtype == object:
            raw[c] = pd.to_numeric(raw[c].astype(str).str.replace(",", "").str.strip(),
                                   errors="coerce")
    d, idx = pd.DataFrame(index=raw.index), raw["Index"]
    d["dp"]   = np.log(raw["D12"]) - np.log(idx)
    d["dy"]   = np.log(raw["D12"]) - np.log(idx.shift(1))
    d["ep"]   = np.log(raw["E12"]) - np.log(idx)
    d["de"]   = np.log(raw["D12"]) - np.log(raw["E12"])
    d["svar"] = raw["svar"]
    d["bm"]   = raw["b/m"]
    d["ntis"] = raw["ntis"]
    d["tbl"]  = raw["tbl"]
    d["lty"]  = raw["lty"]
    d["ltr"]  = raw["ltr"]
    d["tms"]  = raw["lty"] - raw["tbl"]
    d["dfy"]  = raw["BAA"] - raw["AAA"]
    d["dfr"]  = raw["corpr"] - raw["ltr"]
    d["infl"] = raw["infl"].shift(1)                   # CPI released with a lag
    y = raw["CRSP_SPvw"] - raw["Rfree"]                # excess return, the target
    ok = d.notna().all(axis=1) & y.notna()
    return y[ok].to_numpy(), d[ok].to_numpy(), list(d.columns)


def rff(R, P, gamma, seed=0):
    g = np.random.default_rng(seed)
    W = g.standard_normal((P, R.shape[1])) * gamma
    b = g.uniform(0, 2 * np.pi, P)
    return math.sqrt(2.0) * np.cos(R @ W.T + b)


def Psi_closed(c, z):
    s = 1 + c + z
    return 0.5 * (s / math.sqrt(s * s - 4 * c) - 1)


# ------------------------------------------------- rolling loop, all z at once
def sweep(y, feats, T, zgrid):
    """One T x T eigendecomposition per window; the whole z-grid is then free, since
    (zI + K)^{-1} = Q diag(1/(z+lam)) Q'. Cost is O(n T^3 + n T^2 P), not O(n|z|T^3)."""
    n = len(y) - T - 1
    nz = len(zgrid)
    I, one = np.eye(T), np.ones(T)
    M = I - np.outer(one, one) / T
    yb = np.empty(n)
    yh = np.empty((nz, n)); psi = np.empty((nz, n))
    for i in range(n):
        X = feats[i:i + T]                             # T x P
        Y = y[i + 1:i + T + 1]
        s = feats[i + T]
        lam, Q = np.linalg.eigh(X @ X.T / T)           # one eig, reused for every z
        Xs = X @ s                                     # T-vector
        QtXs = Q.T @ Xs
        yb[i] = Y.mean()
        for j, z in enumerate(zgrid):
            a = (M @ (Q @ (QtXs / (z + lam)))) / T
            yh[j, i]  = yb[i] + a @ Y
            psi[j, i] = a @ a
    return y[T + 1:T + n + 1], yb, yh, psi


def statistics(yt, yb, yh, psi, bandwidth=None):
    n = len(yt)
    e1, e2 = yt - yb, yt - yh
    D   = yh - yb
    dDM = e1 ** 2 - e2 ** 2
    dCW = dDM + D ** 2
    s2  = 0.5 * np.mean(np.diff(yt) ** 2)              # Prop 4.15 difference estimator
    dCC = dDM + s2 * psi
    b   = bandwidth or max(1, int(np.ceil(4 * (n / 100) ** (2 / 9))))
    def hac(x):
        x = x - x.mean(); g = np.mean(x * x)
        for h in range(1, b + 1):
            g += 2 * (1 - h / (b + 1)) * np.mean(x[h:] * x[:-h])
        return max(g, 1e-14)
    t = lambda d, v: math.sqrt(n) * d.mean() / math.sqrt(v)
    return dict(t_DM=t(dDM, dDM.var()), t_CW=t(dCW, dCW.var()), t_CC=t(dCC, hac(dCC)),
                Psihat=psi.mean(), sigma2=s2,
                ratio=np.mean(D ** 2) / np.mean(s2 * psi),
                R2oos=1 - np.mean(e2 ** 2) / np.mean(e1 ** 2),
                mspe_gap=np.mean(dDM))


# ----------------------------------------------------------------- output
def write_tables(rows, meta):
    os.makedirs(TABS, exist_ok=True)
    c = meta["c"]
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{The noise-corrected test on the virtue-of-complexity design."
         rf" Monthly US excess returns, $n={meta['n']:,}$, $d={meta['d']}$ Goyal--Welch"
         rf" predictors, $P={meta['P']:,}$ random Fourier features, $T={meta['T']}$, so"
         rf" $c=P/T={c:,.0f}$. $\ztil=zT/P$ is the scale-free penalty. $\Psihatn$ is"
         r" exact by \cref{lem:psi-t}; the closed form predicts"
         rf" $(1-1/T)/(c-1)={(1-1/meta['T'])/(c-1):.4f}$ for every cell in the table,"
         r" the factor $1-q/T$ with $q=1$ being the rolling mean's own degree of freedom"
         r" (\cref{prop:dof}). $\mathcal R$ is the"
         r" sanity ratio, which must exceed one in population; rows with"
         r" $\mathcal R<1$ have a mis-scaled $\hat\sigma^2$ and the corrected statistic"
         r" should not be read there. $\overline{\Delta\MSPE}$ is benchmark minus ridge,"
         r" negative when the ridge forecast is worse.}",
         r"\label{tab:voc-grid}", r"\small",
         r"\begin{tabular}{rrrrrrrrrr}", r"\toprule",
         r"$\gamma$ & $z$ & $\ztil$ & $\Psihatn$ & $\mathcal R$ & $R^2_{\text{oos}}$"
         r" & $\overline{\Delta\MSPE}$ & $\tDM$ & $\tCW$ & $\tCC$ \\"]
    L[-1] = (r"$\gamma$ & $z$ & $\ztil$ & $\Psihatn$ & $\mathcal R$ & $R^2_{\text{oos}}$"
             r" & $\overline{\Delta\MSPE}$ & $t_{DM}$ & $t_{CW}$ & $t_{CC}$ \\")
    L.append(r"\midrule")
    prev = None
    for r in rows:
        if prev is not None and r["gamma"] != prev:
            L.append(r"\addlinespace")
        g = f"{r['gamma']:.2f}" if r["gamma"] != prev else ""
        cc = "" if r["ratio"] >= 1 else r"$^{\dagger}$"
        L.append(f"{g} & {r['z']:.0e} & {r['zt']:.1e} & {r['Psihat']:.4f} & "
                 f"{r['ratio']:.2f}{cc} & {r['R2oos']:.3f} & {r['mspe_gap']:.1e} & "
                 f"{r['t_DM']:.2f} & {r['t_CW']:.2f} & {r['t_CC']:.2f} \\\\")
        prev = r["gamma"]
    L += [r"\bottomrule",
          r"\multicolumn{10}{@{}l}{\footnotesize $^{\dagger}$ sanity ratio below one;"
          r" see \cref{sec:mc-breakdown}.}",
          r"\end{tabular}", r"\end{table}"]
    open(os.path.join(TABS, "tab-voc-grid.tex"), "w").write(mathsign("\n".join(L) + "\n"))

    # diagnostic-only table: Psihat and the scale-free penalty,
    # no test statistics, since the sanity ratio blocks the corrected test on this data.
    D = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Measuring the wedge in a random-feature design."
         rf" Monthly US excess returns, $n={meta['n']:,}$, $d={meta['d']}$ Goyal--Welch"
         rf" predictors, $P={meta['P']:,}$ random Fourier features on a rolling window of"
         rf" $T={meta['T']}$, so $c=P/T={c:,.0f}$ in every cell. $\ztil=zT/P$ is the"
         r" scale-free penalty of \cref{sec:optimal-penalty}. $\Psihatn$ is exact by"
         r" \cref{lem:psi-t} and requires no assumption on the design. The closed form"
         r" \eqref{eq:closed-main} sees only $c$ and $z$; at this $c$ it is within a percent"
         rf" of its ridgeless limit $1/(c-1)={1/(c-1):.4f}$ for the two lightest penalties,"
         rf" falling to ${_psi_cf(c, 1e2):.4f}$ at $z=10^2$ and"
         rf" ${_psi_cf(c, 1e4)*1e6:.0f}\times10^{{-6}}$ at $z=10^4$. Against a rolling-mean"
         rf" benchmark \cref{{prop:dof}} scales all of these by $1-1/T={1-1/meta['T']:.3f}$,"
         r" which is immaterial at the discrepancies this table reports."
         r" $\mathcal R$ is the sanity ratio of \cref{eq:ratio}.}",
         r"\label{tab:voc-psi}", r"\small",
         r"\begin{tabular}{rrrrr}", r"\toprule",
         r"$\gamma$ & $z$ & $\ztil=zT/P$ & $\Psihatn$ & $\mathcal R$ \\", r"\midrule"]
    prev = None
    for r in rows:
        if r["z"] not in (1e-2, 1e0, 1e2, 1e4):
            continue
        if prev is not None and r["gamma"] != prev:
            D.append(r"\addlinespace")
        g = f"{r['gamma']:.1f}" if r["gamma"] != prev else ""
        D.append(f"{g} & {r['z']:.0e} & {r['zt']:.1e} & {r['Psihat']:.4f} & "
                 f"{r['ratio']:.2f} \\\\")
        prev = r["gamma"]
    D += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(TABS, "tab-voc-psi.tex"), "w").write(mathsign("\n".join(D) + "\n"))

    # headline: the cells where CW and CC disagree and the ratio passes
    dis = [r for r in rows if r["ratio"] >= 1 and (r["t_CW"] > 1.645) != (r["t_CC"] > 1.645)]
    H = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Cells in \cref{tab:voc-grid} where the two statistics disagree at the"
         r" one-sided $5\%$ level and the sanity ratio passes. These are the designs in"
         r" which the distinction between the two nulls changes the reported conclusion.}",
         r"\label{tab:voc-headline}", r"\small",
         r"\begin{tabular}{rrrrrrr}", r"\toprule",
         r"$\gamma$ & $\ztil$ & $\Psihatn$ & $\mathcal R$ & $\overline{\Delta\MSPE}$"
         r" & $t_{CW}$ & $t_{CC}$ \\", r"\midrule"]
    if dis:
        for r in dis:
            H.append(f"{r['gamma']:.2f} & {r['zt']:.1e} & {r['Psihat']:.4f} & "
                     f"{r['ratio']:.2f} & {r['mspe_gap']:.1e} & {r['t_CW']:.2f} & "
                     f"{r['t_CC']:.2f} \\\\")
    else:
        H.append(r"\multicolumn{7}{c}{\emph{no cell on this grid reverses}} \\")
    H += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(TABS, "tab-voc-headline.tex"), "w").write(mathsign("\n".join(H) + "\n"))


def write_figures(rows, meta):
    os.makedirs(FIGS, exist_ok=True)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    gammas = sorted({r["gamma"] for r in rows})
    cmap = plt.get_cmap("viridis")
    col = {g: cmap(i / max(len(gammas) - 1, 1)) for i, g in enumerate(gammas)}
    c = meta["c"]; T_ = meta["T"]

    # --- Figure 1: Psihat_n against the scale-free penalty, versus the closed form
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for g in gammas:
        s = [r for r in rows if r["gamma"] == g]
        ax.loglog([r["zt"] for r in s], [max(r["Psihat"], 1e-8) for r in s],
                  "o-", ms=3.5, lw=1.2, color=col[g], label=rf"$\gamma={g:g}$")
    ax.axhline(1 / (c - 1), ls="--", lw=1.3, color="crimson", zorder=1)
    ax.axhspan(1e-8, 0.05, color="0.88", alpha=0.6, zorder=0)
    x0, x1 = ax.get_xlim()
    ax.text(x1 * 0.75, 1 / (c - 1) * 1.6,
            rf"closed form, every cell: $(1-1/T)/(c-1)={(1-1/T_)/(c-1):.1e}$",
            color="crimson", fontsize=7.5, ha="right")
    ax.text(x0 * 1.5, 2e-7, "correction negligible below here:\nClark–West and CC agree",
            fontsize=7.5, color="0.3", va="bottom")
    ax.set_ylim(1e-8, None)
    ax.set_xlabel(r"effective penalty $\tilde z = zT/P$")
    ax.set_ylabel(r"$\hat\Psi_n$   (units of $\sigma^2$)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig-voc-psi.pdf"),
                                    bbox_inches="tight")
    fig.savefig(os.path.join(FIGS, "fig-voc-psi.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: the three statistics across the penalty grid
    fig, axes = plt.subplots(1, len(gammas), figsize=(2.5 * len(gammas), 3.0),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for k, g in enumerate(gammas):
        ax = axes[k]; s = sorted([r for r in rows if r["gamma"] == g], key=lambda r: r["zt"])
        zt = [r["zt"] for r in s]
        ax.axhspan(-1.645, 1.645, color="0.9", zorder=0)
        ax.axhline(1.645, ls=":", lw=0.8, color="k")
        for key, lab, sty in [("t_DM", "DM", "s--"), ("t_CW", "CW", "o-"),
                              ("t_CC", "CC", "^-")]:
            ax.semilogx(zt, [r[key] for r in s], sty, ms=3.5, lw=1.2, label=lab)
        bad = [r["zt"] for r in s if r["ratio"] < 1]
        if bad:
            ax.axvspan(min(bad), max(bad), color="crimson", alpha=0.10, zorder=0)
        ax.set_title(rf"$\gamma={g:g}$", fontsize=9)
        ax.set_xlabel(r"$\tilde z$")
        if k == 0:
            ax.set_ylabel("one-sided $t$")
            ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig-voc-stats.pdf"),
                                    bbox_inches="tight")
    fig.savefig(os.path.join(FIGS, "fig-voc-stats.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- driver
def _psi_cf(c, z):
    """Deterministic equivalent of Theorem 4.3, for the caption of tab:voc-psi.  The
    table's own numbers are the realised plug-in; this is only the comparison the
    caption draws, and it is NOT 1/(c-1) except in the ridgeless limit."""
    s = 1.0 + c + z
    return 0.5 * (s / math.sqrt(s * s - 4.0 * c) - 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(HERE, "data", "PredictorData2021.csv"))
    ap.add_argument("--T", type=int, default=12)
    ap.add_argument("--P", type=int, default=2000)
    args = ap.parse_args()

    y, R, names = goyal_welch(args.csv)
    R = (R - R.mean(0)) / R.std(0)
    c = args.P / args.T
    meta = dict(n=len(y), d=R.shape[1], P=args.P, T=args.T, c=c)
    print(f"n = {len(y)} months, d = {R.shape[1]}: {', '.join(names)}")
    print(f"T = {args.T}, P = {args.P:,}, c = P/T = {c:,.0f}, "
          f"closed form (1-1/T)/(c-1) = {(1-1/args.T)/(c-1):.5f}  "
          f"[prop:dof; q=1 for the rolling mean]\n")

    gammas = [0.1, 0.3, 1.0, 3.0]
    zgrid  = np.array([1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5])

    rows = []
    hdr = (f"{'gamma':>6}{'z':>8}{'z~':>9}{'Psihat_n':>10}{'ratio':>7}{'R2oos':>8}"
           f"{'MSPEgap':>10}{'t_DM':>7}{'t_CW':>7}{'t_CC':>7}{'':>7}")
    print(hdr); print("-" * len(hdr))
    for gam in gammas:
        F = rff(R, args.P, gam)
        yt, yb, yh, psi = sweep(y, F, args.T, zgrid)
        for j, z in enumerate(zgrid):
            s = statistics(yt, yb, yh[j], psi[j])
            s.update(gamma=gam, z=float(z), zt=float(z) * args.T / args.P)
            rows.append(s)
            note = "R<1" if s["ratio"] < 1 else ("CW ok" if s["Psihat"] < .05 else "")
            print(f"{gam:>6.2f}{z:>8.0e}{s['zt']:>9.1e}{s['Psihat']:>10.4f}"
                  f"{s['ratio']:>7.2f}{s['R2oos']:>+8.3f}{s['mspe_gap']:>+10.1e}"
                  f"{s['t_DM']:>+7.2f}{s['t_CW']:>+7.2f}{s['t_CC']:>+7.2f}{note:>7}")

    write_tables(rows, meta); write_figures(rows, meta)
    lo = min(r["Psihat"] for r in rows); hi = max(r["Psihat"] for r in rows)
    print(f"\nPsihat_n spans {lo:.2e} to {hi:.2f} at identical P, T, c "
          f"(closed form: {(1-1/args.T)/(c-1):.5f} everywhere)")
    print(f"tables  -> {TABS}")
    print(f"figures -> {FIGS}")


if __name__ == "__main__":
    main()
