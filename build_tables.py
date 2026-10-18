"""Write every generated table and figure of the paper from the stored output.

    python build_tables.py            # write into ../Latex (or ./tables, ./figures)
    python build_tables.py --check    # rebuild into a temporary directory and compare
                                      # byte for byte with the copies in ../Latex

Inside the author's tree the paper sits at ../Latex and the tables are written there; a
clone on its own writes to ./tables and ./figures.  MC_TEX and MC_FIG override both, and
MC_OUT points at a different output directory.  README.md maps every table to its source.

  tab-mc-size, tab-mc-power, tab-mc-closedform   output/armA.json
  tab-mc-grid, pgrid_*.pdf                       output/pgrid_*.json and the run log
  tab-mc-features                                output/armB.json
  tab-mc-breakdown                               output/armC.json
  tab-mc-complexity                              output/armD.json
  tab-mc-overparam                               output/armO2.json, output/armO2_diag.json
  tab-mc-signal                                  output/armS.json
  tab-window                                     output/mc_window/spiked_*.json
  tab-intersection                               output/mc_intersection/*.json
  tab-kernel-check, tab-delta                    output/mc_kernel/kernel_delta.json
  tab-kappa                                      output/mc_har/
  tab-cw-har, tab-har-power, tab-har-omitted,
  tab-pathfunc                                   ldv_scope/output/
  tab-rv-main                                    output/rv_tables/rv_rows.json
  tab-rv-cross-psi, tab-rv-cross-stats           output/rv_cross/panel.json
  tab-rv-multiest                                output/rv_cross/ and rv_cross_multiest/
  tab-rv-predict, tab-rv-predict-robust          output/rv_window_bootstrap_*.json
  tab-voc-psi, tab-voc-grid                      output/voc_tables/ (voc_application.py)
  tab-voc-weighted                               output/voc_tables/ (voc_weighted.py)

The p-value figures are drawn here rather than by mc_study.py, so that each row's own
statistic sits on the diagonal; the t-statistics are the ones the run stored.
"""
import json, os, re, sys, glob, math, shutil, filecmp, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
# Source directory for the Monte Carlo output.  Overridable so that a run written to
# another directory can be turned into tables without editing this file:
#     MC_OUT=my_run python build_tables.py
OUT  = os.path.join(HERE, os.environ.get("MC_OUT", "output"))
# Where the generated artefacts go.  Inside the author's tree the paper sits
# alongside this package and the tables are written straight into it; a clone of
# this repository on its own has no ../Latex, so fall back to directories inside
# the package.  Both are overridable.  Neither is committed: the repository ships
# code and the Monte Carlo output that code consumes, and nothing rendered.
_PAPER_TEX = os.path.abspath(os.path.join(HERE, "..", "Latex", "sections", "tables"))
_PAPER_FIG = os.path.abspath(os.path.join(HERE, "..", "Latex", "figures"))
STANDALONE = not os.path.isdir(_PAPER_TEX)
TEX  = os.environ.get("MC_TEX") or (os.path.join(HERE, "tables") if STANDALONE else _PAPER_TEX)
FIG  = os.environ.get("MC_FIG") or (os.path.join(HERE, "figures") if STANDALONE else _PAPER_FIG)
# --check compares against the paper; a clone without the paper compares against the
# copies of the paper's tables shipped in reference_tables/.
REF  = os.environ.get("MC_TEX") or (os.path.join(HERE, "reference_tables") if STANDALONE else _PAPER_TEX)
os.makedirs(TEX, exist_ok=True); os.makedirs(FIG, exist_ok=True)
LOG  = sorted(glob.glob(os.path.join(OUT, "run_all_*.log")))

CHECK = "--check" in sys.argv


def load(name):
    with open(os.path.join(OUT, name)) as fh:
        return json.load(fh)


def Psi(c, z):
    """Closed form of Theorem 4.3. Kept here so the table cannot drift from the text."""
    s = 1 + c + z
    return 0.5 * (s / math.sqrt(s * s - 4 * c) - 1)


def mspe_gaps_from_log():
    """mspe_gap is printed by the p-value grid but not stored in its JSON."""
    if not LOG:
        raise SystemExit("no run log found; cannot recover mspe_gap")
    txt = open(LOG[-1]).read()
    out, psi = {}, None
    for line in txt.splitlines():
        m = re.search(r"P-VALUE GRID .*Psi\*=([\d.]+) .*sigma2=(\w+)", line)
        if m:
            psi, s2 = m.group(1), m.group(2)
            continue
        m = re.match(r"\s*(H0-\S+)\s+mu_CC=([+-][\d.]+)\s+mspe_gap=([+-][\d.]+)", line)
        if m and psi:
            out[(psi, m.group(1).strip())] = (float(m.group(2)), float(m.group(3)))
    return out


# ---------------------------------------------------------------- table writers
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


def w(path, lines):
    body = mathsign("\n".join(lines) + "\n")
    dest = os.path.join(TEXTMP if CHECK else TEX, path)
    with open(dest, "w") as fh:
        fh.write(body)


def tab_size(A):
    # Bold a rejection frequency (as printed, three decimals) that lies outside the 95%
    # Monte Carlo band of an exact 5% test with 1,000 replications: 0.05 +/- 0.0135.
    def bf(x):
        s3 = f"{x:.3f}"
        return r"\textbf{%s}" % s3 if (float(s3) < 0.0365 or float(s3) > 0.0635) else s3
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Size under the sharp null, and the choice among studentisations."
         r" Rejection frequencies of the one-sided $5\%$ tests at $R^2=0$, $1{,}000$"
         r" replications, $(n,b)=(2{,}000,12)$ at $T=8$ and $(8{,}000,18)$ at $T=12$. $\hat\omega^2_n$ is the influence-function-corrected HAC of"
         r" \cref{prop:hac-main}; ``samp'' is the sample variance of $\dCC$;"
         r" ``s.e.$_{\mathrm{CW}}$'' is the CW standard error applied to the"
         r" corrected numerator, which is a studentisation of the corrected statistic and"
         r" not the CW test; the CW test is the $\tCW$ column. The four $\tCC$ columns differ only in the"
         r" standard error, except $\Psitil_t$, which changes the correction term."
         r" The $\Psitil_t$ column replaces the scalar correction by the"
         r" heteroskedasticity-weighted one of \cref{prop:sigma-weighted}, studentised by"
         r" $\hat\omega^2_n$; it is reported here because the design is homoskedastic, so any"
         r" difference from the $\hat\omega^2_n$ column is an efficiency cost of the weighting"
         r" rather than a bias it removes."
         r" $\vartheta_T=\Var(\Psihat_t)/(\E\Psihat_t)^2$ is the dispersion diagnostic of"
         r" \cref{prop:kurtosis}. A row is fixed by $(T,\rho_x,\Psi^*)$ alone: the target"
         r" selects the complexity ratio at which it is reachable and the penalty that"
         r" attains it, giving $(\Psi^*,c,z)=(0.05,0.2,0.947)$, $(0.20,0.3,0.265)$,"
         r" $(0.50,0.5,0.133)$ and $(1.00,0.7,0.0748)$ in the four rows of every block,"
         r" with $P=\mathrm{round}(cT)$, so $P$ runs $2,2,4,6$ at $T=8$ and $2,4,6,8$ at"
         r" $T=12$. The Monte Carlo standard error of any entry is at most $0.016$."
         r" Bold marks a rejection frequency outside the $95\%$ Monte Carlo band for an exact"
         r" test, $0.05\pm1.96\sqrt{0.05\cdot0.95/1{,}000}$, that is, below $0.0365$ or above"
         r" $0.0635$. With $80$ such entries, about four would fall outside the band by chance"
         r" even if every test were exact.}",
         r"\label{tab:mc-size}", r"\small",
         r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
         r"& & & & \multicolumn{4}{c}{$\tCC$} & $\tCW$ & \\",
         r"\cmidrule(lr){5-8}\cmidrule(lr){9-9}",
         r"$T$ & $\rho_x$ & $\Psi^*$ & $\Psihatn$ & $\hat\omega^2_n$ & samp"
         r" & s.e.$_{\mathrm{CW}}$ & $\Psitil_t$ & & $\vartheta_T$ \\", r"\midrule"]
    for k, (T, rho) in enumerate([(8, 0.0), (8, 0.3), (12, 0.0), (12, 0.3)]):
        for i, p in enumerate([0.05, 0.2, 0.5, 1.0]):
            r = one(A, T=T, rho=rho, s2="oracle", psi_t=p, R2=0.0)
            rr = format(rho,'.1f') if i==0 else ''
            L.append(f"{T if i==0 else ''} & {rr} & {p:.2f} & "
                     f"{r['Psihat']:.3f} & {bf(r['rej_CC'])} & {bf(r['rej_CC_samp'])} & "
                     f"{bf(r['rej_CC_cw'])} & {bf(r['rej_CC_w'])} & "
                     f"{bf(r['rej_CW'])} & {r['theta_T']:.2f} \\\\")
        L.append(r"\midrule" if k == 1 else (r"\addlinespace" if k < 3 else ""))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab-mc-size.tex", [x for x in L if x])


def tab_power(A):
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Power, and what the difference estimator costs."
         r" $(T,n,b)=(12,8000,18)$, $\rho_x=0$, $1{,}000$ replications. Each $\Psi^*$"
         r" block runs at the complexity ratio and penalty of \cref{tab:mc-size}, so"
         r" $(c,z,P)$ is $(0.2,0.947,2)$, $(0.3,0.265,4)$, $(0.5,0.133,6)$ and"
         r" $(0.7,0.0748,8)$ down the four blocks. $\ztil/\ztil^*$: the penalty as a"
         r" fraction of the MSE-optimal one of \cref{prop:optimal-z}. $\mathcal R$: the"
         r" sanity ratio of \cref{eq:ratio} under $\hat\sigma^2_\Delta$. $\Delta$CC: power"
         r" under $\hat\sigma^2_\Delta$ less power under the oracle $\sigma^2$. The Monte"
         r" Carlo standard error of any entry is at most $0.016$. Bold marks a CW or"
         r" corrected-test rejection frequency in an $R^2=0$ row outside the $95\%$ Monte"
         r" Carlo band of an exact test, $0.0365$ to $0.0635$.}"
         r"\label{tab:mc-power}", r"\small",
         r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
         r"& & & \multicolumn{3}{c}{oracle $\sigma^2$} & \multicolumn{2}{c}{$\hat\sigma^2_\Delta$} & \\",
         r"\cmidrule(lr){4-6}\cmidrule(lr){7-8}",
         r"$\Psi^*$ & $R^2$ & $\ztil/\ztil^*$ & DM & CW & CC & CC & $\mathcal R$ & $\Delta$CC \\",
         r"\midrule"]
    for p in [0.05, 0.2, 0.5, 1.0]:
        for j, R2 in enumerate([0.0, 0.02, 0.05, 0.10, 0.20]):
            o = one(A, T=12, rho=0.0, s2="oracle", psi_t=p, R2=R2)
            d = one(A, T=12, rho=0.0, s2="diff",   psi_t=p, R2=R2)
            zr = f"{(o['z']/o['c'])*R2:.2g}" if R2 > 0 else "---"
            dv = round(d['rej_CC'] - o['rej_CC'], 3)
            dcc = "$0.000$" if dv == 0 else f"${dv:.3f}$"
            flag = r"\,$^\dagger$" if d["ratio"] < 0.995 else ""
            pp = format(p,'.2f') if j==0 else ''
            bb = (lambda x: (r"\textbf{%.3f}" % x) if (R2 == 0 and abs(x - 0.05) > 0.0135)
                  else "%.3f" % x)
            L.append(f"{pp} & {R2:.2f} & {zr} & {o['rej_DM']:.3f} & "
                     f"{bb(o['rej_CW'])} & {bb(o['rej_CC'])} & {bb(d['rej_CC'])} & "
                     f"{d['ratio']:.3f}{flag} & {dcc} \\\\")
        if p != 1.0:
            L.append(r"\addlinespace")
    L += [r"\bottomrule",
          r"\multicolumn{9}{l}{\footnotesize $^\dagger$ sanity ratio below $0.995$: the"
          r" correction exceeds the realised squared forecast differential.}",
          r"\end{tabular}", r"\end{table}"]
    w("tab-mc-power.tex", L)


def tab_closedform(A):
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Finite-window accuracy of the closed form. $\Psi(c,z)$ is evaluated at"
         r" the \emph{realised} $c=P/T$ after integer rounding of $P$. The last column is"
         r" the relative error scaled by the window length. It sits near $-1$ over most of"
         r" the table, which is \cref{prop:dof}: every design here uses a rolling-mean"
         r" benchmark, so $q=1$ and the closed form must be multiplied by $1-1/T$. After"
         r" that correction the cells with $c\ge0.5$ are within $2$ percent at both window"
         r" lengths. At smaller $c$ the error grows, to about $-3$ percent without persistence"
         r" and $-10$ percent with it, and the two cells at the smallest complexity ratio under"
         r" persistence sit near $-1.8$; that excess is not explained by degrees of"
         r" freedom. $\Psihatn$ is averaged over the $1{,}000$ replications of"
         r" \cref{tab:mc-size}.}",
         r"\label{tab:mc-closedform}", r"\small",
         r"\begin{tabular}{lrrrrrrr}", r"\toprule",
         r"$T$ & $\rho_x$ & $P$ & $c=P/T$ & $z$ & $\Psi(c,z)$ & $\Psihatn$"
         r" & $T\cdot(\Psihatn/\Psi-1)$ \\", r"\midrule"]
    for T in [8, 12]:
        for rho in [0.0, 0.3]:
            for i, p in enumerate([0.05, 0.2, 0.5, 1.0]):
                r = one(A, T=T, rho=rho, s2="oracle", psi_t=p, R2=0.0)
                cr = r["P"] / T
                pr = Psi(cr, r["z"])
                rr = format(rho,'.1f') if i==0 else ''
                L.append(f"{T if i==0 else ''} & {rr} & {r['P']} & "
                         f"{cr:.3f} & {r['z']:.4f} & {pr:.4f} & {r['Psihat']:.4f} & "
                         f"${T*(r['Psihat']/pr-1):.2f}$ \\\\")
            if not (T == 12 and rho == 0.3):
                L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab-mc-closedform.tex", L)


def tab_grid():
    import numpy as np
    gaps = mspe_gaps_from_log()
    NM = {"H0-cov": r"$\Hcov$", "H0-CC": r"$\Hcc$", "H0-MSPE": r"$\Hmspe$"}
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Each statistic against each null. Rejection frequencies of the"
         r" one-sided $5\%$ tests at $(T,n,b)=(8,2000,12)$, $\rho_x=0.3$,"
         r" $1{,}000$ replications. The corrected statistic is studentised by"
         r" $\hat\omega^2_n$, as in \cref{tab:mc-size}, and the replications are drawn"
         r" independently of that table. The $\Hcc$ and $\Hmspe$ rows carry $R^2=0.10$ and"
         r" impose their null exactly by rescaling the forecast differential by a factor"
         r" computed from an independent pilot sample. $\Hcov$ is not reachable that way,"
         r" since a positive rescaling multiplies $\E[fg]$ by the same factor; that row is"
         r" instead the $\theta=0$ design at scale one, which sets $\E[fg]$ and $\E[g^2]$"
         r" to zero together. It is therefore the degenerate point at which $\Hcov$ and"
         r" $\Hcc$ share a boundary, and the only row in which the equality of $\Psihatn$"
         r" with $-\overline{\Delta\MSPE}$ is an exact prediction of \cref{lem:psi-t}."
         r" $\overline{\Delta\MSPE}$ is the"
         r" realised mean squared prediction error gap, negative when the candidate"
         r" forecasts worse, and $\Psihatn$ the realised plug-in correction. $D_{\mathrm{KS}}=\sup_u|F_n(u)-u|$ is the Kolmogorov--Smirnov"
         r" statistic against uniformity of the $1{,}000$ $p$-values of that row's own"
         r" statistic under the oracle $\sigma^2$: the largest error in the $p$-value"
         r" distribution, measured in probability units rather than as a significance"
         r" level, so that a reader can judge the size of a departure and not only its"
         r" detectability. With $1{,}000$ draws, $D_{\mathrm{KS}}>0.043$ rejects uniformity"
         r" at the $5\%$ level, which two of the nine cells do; two more sit just under"
         r" the threshold, at $0.040$ and $0.037$."
         r" Shading marks each null's own statistic, and bold an own-statistic rate outside the"
         r" $95\%$ Monte Carlo band of an exact test, $0.0365$ to $0.0635$. The Monte Carlo standard"
         r" error of a rejection frequency is $0.007$ at a nominal $0.05$.}",
         r"\label{tab:mc-grid}", r"\small",
         r"\begin{tabular}{lrrrrrrrrrr}", r"\toprule",
         r"& & & & & \multicolumn{3}{c}{oracle $\sigma^2$} & \multicolumn{2}{c}{$\hat\sigma^2_\Delta$} & \\",
         r"\cmidrule(lr){6-8}\cmidrule(lr){9-10}",
         r"$\Psi^*$ & null imposed & $\mu_{CC}$ & $\overline{\Delta\MSPE}$ & $\Psihatn$ & DM & CW & CC"
         r" & CW & CC & $D_{\mathrm{KS}}$ \\", r"\midrule"]
    OWN = {"H0-cov": "CW", "H0-CC": "CC", "H0-MSPE": "DM"}
    for p in ["0.2", "0.5", "1.0"]:
        do = load(f"pgrid_psi{p}_oracle.json")
        dd = load(f"pgrid_psi{p}_diff.json")
        for i, key in enumerate(["H0-cov ", "H0-CC  ", "H0-MSPE"]):
            k = key.strip()
            So, Sd = do["stats"][key], dd["stats"][key]
            rj = lambda S, t: float(np.mean(np.array(S[t]) > 1.645))
            from scipy.stats import norm as _nrm, kstest as _ks
            _own = {"H0-cov": "t_CW", "H0-CC": "t_CC_hac", "H0-MSPE": "t_DM"}[k]
            _p = 1 - _nrm.cdf(np.asarray(So[_own], float))
            ksd = float(_ks(_p, "uniform").statistic)
            psihat = float(np.mean(np.asarray(So["Psihat"], float)))
            cells = {"DM": rj(So, "t_DM"), "CW": rj(So, "t_CW"), "CC": rj(So, "t_CC_hac")}
            def _f(c, v):
                if OWN[k] != c:
                    return "%.3f" % v
                s3 = (r"\textbf{%.3f}" % v) if abs(round(v, 3) - 0.05) > 0.0135 else "%.3f" % v
                return r"\cellcolor{black!10}" + s3
            fmt = {c: _f(c, v) for c, v in cells.items()}
            mu, gap = gaps[(p, k)]
            kss = "%.3f" % ksd
            L.append(f"{p if i==0 else ''} & {NM[k]} & {mu:.4f} & {gap:.3f} & "
                     f"{psihat:.3f} & "
                     f"{fmt['DM']} & {fmt['CW']} & {fmt['CC']} & "
                     f"{rj(Sd,'t_CW'):.3f} & {rj(Sd,'t_CC_hac'):.3f} & {kss} \\\\")
        if p != "1.0":
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab-mc-grid.tex", L)


def tab_features(B):
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Nominal complexity does not determine the wedge. $T=12$, $n=8{,}000$,"
         r" $b=18$, $z=0.5$, $200$ replications per cell, targets pure noise, so the measured object is exactly the null forecast"
         r" variance. Within each block $c=P/T$ is constant and the closed form predicts the"
         r" single value in the $\Psi(c,z)$ column; only the bandwidth of the feature map"
         r" varies. The last column is $\Psihatn/\Psi(c,z)$.}",
         r"\label{tab:mc-features}", r"\small",
         r"\begin{tabular}{rrrrrrr}", r"\toprule",
         r"$d$ & $P$ & $c$ & $\gamma$ & $\Psi(c,z)$ & $\Psihatn$ & ratio \\", r"\midrule"]
    prev = None
    for r in B:
        key = (r["d"], r["P"])
        if prev is not None and key != prev:
            L.append(r"\addlinespace")
        n = key != prev
        cc = format(r["c"], ".1f") if n else ""
        pc = format(r["psi_closed"], ".4f") if n else ""
        L.append(f"{r['d'] if n else ''} & {format(r['P'],',') if n else ''} & "
                 f"{cc} & {r['gamma']:.2f} & {pc} & {r['psihat']:.4f} & "
                 f"{r['psihat']/r['psi_closed']:.1f} \\\\")
        prev = key
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab-mc-features.tex", L)


def tab_breakdown(C):
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Where the approximations break. All three panels run at $T=8$,"
         r" $n=2{,}000$, $b=12$, $\rho_x=0.3$ and $c=0.5$ unless a row says otherwise,"
         r" which is the $\Psi^*=0.5$ cell of \cref{tab:mc-size}. Panel (a) correlates"
         r" innovations with $\eps_t$ at strength $\delta$, violating \cref{as:errors};"
         r" $\mathcal R$ is the sanity ratio and $\mathcal R^{\text{shift}}$ its value after"
         r" a large circular shift. Panel (b) varies the innovation distribution and then,"
         r" holding that distribution fixed at GARCH$(1,1)$ with $h_t=0.05+0.90h_{t-1}+0.05u_{t-1}^2$,"
         r" standardised to unit variance, the sign of the coupling"
         r" between the dispersion of the feature rows and the conditional volatility"
         r" $\sigma_t$: independent, rising with $\sigma_t$, and falling with it. The three"
         r" GARCH rows therefore share an innovation process and differ only in that"
         r" coupling, which is the quantity \eqref{eq:psi-t-het} says a scalar correction"
         r" cannot see. Panel (c)"
         r" runs designs failing the bandwidth gates of \cref{tab:gates}, the second of which is a"
         r" monthly-frequency design of the kind \cref{sec:empirics} works with."
         r" Entries headed ``size'' are one-sided $5\%$ rejection frequencies under"
         r" the sharp null, so $0.05$ is correct. Panels (a) and (b) use $2{,}000$"
         r" replications and carry a Monte Carlo standard error of $0.005$ at that level;"
         r" panel (c) uses $1{,}000$ and carries $0.007$. The $\Psitil_t$ column is the corrected test with the"
         r" heteroskedasticity-weighted correction of \cref{prop:sigma-weighted} in place of"
         r" the scalar one, studentised identically. The panels draw independent"
         r" replications, so the $\delta=0$ row of panel (a) and the Gaussian row of panel"
         r" (b) repeat one design on different draws. Bold marks a size entry outside the"
         r" $95\%$ Monte Carlo band of an exact test, $0.040$ to $0.060$ in panels (a) and"
         r" (b) and $0.0365$ to $0.0635$ in panel (c).}",
         r"\label{tab:mc-breakdown}", r"\small",
         r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"\multicolumn{6}{@{}l}{\emph{(a) Endogeneity}, violating \cref{as:errors}}\\",
         r"\cmidrule(r){1-6}",
         r"$\delta$ & $\mathcal R$ & $\mathcal R^{\text{shift}}$ & size CC"
         r" & size $\Psitil_t$ & \\"]
    def bz(x, h):
        return (r"\textbf{%.3f}" % x) if abs(x - 0.05) > h else "%.3f" % x
    h2, h1 = 1.96 * math.sqrt(0.0475 / 2000), 1.96 * math.sqrt(0.0475 / 1000)
    for r in C["C1"]:
        L.append(f"{r['delta']:.2f} & {r['ratio']:.3f} & {r['ratio_shift']:.3f} & "
                 f"{bz(r['rej_CC'], h2)} & {bz(r['rej_CC_w'], h2)} & \\\\")
    L += [r"\addlinespace",
          r"\multicolumn{6}{@{}l}{\emph{(b) Innovation distribution, and the"
          r" feature--volatility coupling}}\\",
          r"\cmidrule(r){1-6}",
          r"innovations & $\Psihatn$ & $\mathcal R$ & size CC & size $\Psitil_t$"
          r" & size CW \\"]
    NM = {("gauss", 0.0): "Gaussian",
          ("t5", 0.0): "$t_5$",
          ("garch", 0.0): r"GARCH$(1,1)$, features independent",
          ("garch", 1.0): r"GARCH$(1,1)$, dispersion $\uparrow$ with $\sigma_t$",
          ("garch", -1.0): r"GARCH$(1,1)$, dispersion $\downarrow$ with $\sigma_t$"}
    for r in C["C2"]:
        L.append(f"{NM[(r['innov'], r['feat_vol'])]} & {r['Psihat']:.4f} & "
                 f"{r['ratio']:.3f} & {bz(r['rej_CC'], h2)} & {bz(r['rej_CC_w'], h2)} & "
                 f"{bz(r['rej_CW'], h2)} \\\\")
    L += [r"\addlinespace",
          r"\multicolumn{6}{@{}l}{\emph{(c) Outside the bandwidth gates}}\\",
          r"\cmidrule(r){1-6}",
          r"design & $T_{\mathrm{eff}}$ & F2 & F3 & size CC & size $\Psitil_t$ \\"]
    for r in C["C3"]:
        L.append(f"$T={r['T']}$, $n={r['n']}$, $b={r['b']}$ & {r['T_eff']:.1f} & "
                 f"{'yes' if r['F2'] else 'no'} & {'yes' if r['F3'] else 'no'} & "
                 f"{bz(r['size_CC'], h1)} & {bz(r['size_CC_w'], h1)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab-mc-breakdown.tex", L)


def tab_complexity():
    """Arm D. Skipped silently when armD.json is absent, so the package still verifies
    before the sweep has been run."""
    import numpy as np
    src = os.path.join(OUT, "armD.json")
    if not os.path.exists(src):
        print("  (armD.json not found; complexity sweep table skipped)")
        return
    D = json.load(open(src))
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Through the interpolation threshold. $(T,n,b)=(12,8000,18)$,"
         r" $\rho_x=0$, $1{,}000$ replications, one-sided at $5\%$. Complexity is swept"
         r" directly and the scale-free penalty $\ztil=zT/P$ held fixed, rather than"
         r" targeting $\Psi^*$ as in \cref{tab:mc-size}: a fixed $\Psi^*$ is unreachable"
         r" as $c\to1$, which is why the other tables stop at $c=0.7$. $\Psi(c,z)$ is the"
         r" closed form \eqref{eq:closed-main} and $\Psihatn$ the realised plug-in."
         r" The targets swept are $0.5$, $0.7$, $0.9$, $1$, $1.1$, $1.5$, $2$ and $3$;"
         r" the column reports the \emph{realised} $c=P/T$ after integer rounding of $P$,"
         r" at which both $\Psi(c,z)$ and $z=\ztil c$ are evaluated, so the two cells"
         r" bracketing the threshold sit at $0.917$ and $1.083$ while the threshold cell"
         r" itself is exact. $\vartheta_T$ is the dispersion diagnostic of"
         r" \cref{prop:kurtosis}, whose boundedness is what \cref{thm:R2c} asserts, while"
         r" ``sd'' and ``kurt'' are the standard deviation and standardised fourth moment"
         r" of the studentised statistic itself, which is the quantity a user actually"
         r" relies on. Size columns are at $R^2=0$, with CC$_\Delta$ replacing the oracle"
         r" $\sigma^2$ by $\hat\sigma^2_\Delta$; power is at $R^2=0.05$. Monte Carlo"
         r" standard error at a nominal $0.05$ is $0.007$.}",
         r"\label{tab:mc-complexity}", r"\small",
         r"\begin{tabular}{rrrrrrrrrrrrr}", r"\toprule",
         r"& & & & & & & \multicolumn{2}{c}{studentised $t$}"
         r" & \multicolumn{3}{c}{size} & power \\",
         r"\cmidrule(lr){8-9}\cmidrule(lr){10-12}",
         r"$c$ & $P$ & $\ztil$ & $\Psi(c,z)$ & $\Psihatn$ & ratio & $\vartheta_T$"
         r" & sd & kurt & CW & CC & CC$_\Delta$ & CC \\", r"\midrule"]
    for zt in sorted({r["ztilde"] for r in D}, reverse=True):
        first = True
        for ct in sorted({r["c_target"] for r in D}):
            sz = [r for r in D if r["ztilde"]==zt and r["c_target"]==ct
                  and r["R2"]==0.0 and r["s2"]=="oracle"]
            sd_ = [r for r in D if r["ztilde"]==zt and r["c_target"]==ct
                   and r["R2"]==0.0 and r["s2"]=="diff"]
            pw = [r for r in D if r["ztilde"]==zt and r["c_target"]==ct
                  and r["R2"]>0 and r["s2"]=="oracle"]
            if not sz: continue
            s0 = sz[0]
            psi = "$\\infty$" if not np.isfinite(s0["Psi_closed"]) else "%.3f" % s0["Psi_closed"]
            rat = "---" if not np.isfinite(s0.get("psi_ratio", float("nan"))) else "%.2f" % s0["psi_ratio"]
            L.append(f"{s0['c']:.3f} & {s0['P']} & "
                     f"{('%.1f' % zt) if first else ''} & {psi} & "
                     f"{s0['Psihat']:.3f} & {rat} & {s0['theta_T']:.2f} & "
                     f"{s0['sd_t_CC']:.2f} & {s0['kurt_t_CC']:.1f} & "
                     f"{s0['rej_CW']:.3f} & {s0['rej_CC']:.3f} & "
                     f"{(('%.3f' % sd_[0]['rej_CC']) if sd_ else '---')} & "
                     f"{(('%.3f' % pw[0]['rej_CC']) if pw else '---')} \\\\")
            first = False
        L.append(r"\addlinespace")
    while L[-1] == r"\addlinespace": L.pop()
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab-mc-complexity.tex", L)


def tab_overparam():
    """Arm O (mc_overparam.py): more predictors than observations, with signal present.
    Reads output/armO2.json, output/armO2_diag.json and output/armO2_lowsig.json; skipped
    when any is absent.
    The first run (armO.json, armO_diag.json) measured mu_CC against the raw signal
    rather than the benchmark-relative one and is not used."""
    srcs = [os.path.join(OUT, n) for n in ("armO2.json", "armO2_diag.json", "armO2_lowsig.json")]
    if not all(os.path.exists(s) for s in srcs):
        print("  (armO2*.json not found; overparametrised table skipped)")
        return
    rows = sum((json.load(open(s)) for s in srcs), [])

    def get(part, P, zt=None, rho=None, signal=None, row="raw"):
        hit = [r for r in rows if r["part"] == part and r["P"] == P and r["row"] == row
               and (zt is None or abs(r["ztilde"] - zt) < 1e-9)
               and (rho is None or r.get("rho_x") == rho)
               and (signal is None or r.get("signal") == signal)]
        assert len(hit) == 1, (part, P, zt, rho, signal, row, len(hit))
        return hit[0]

    def b(p):          # bold outside the 95% Monte Carlo band of an exact 5% test
        s = "%.3f" % p
        return r"\textbf{%s}" % s if (p < 0.0365 or p > 0.0635) else s

    def line(c, zt, rho, span, h0, fit):
        cs = ("%.1f" % c) if c < 100 else "%.0f" % c
        cs = cs[:-2] if cs.endswith(".0") else cs
        return (f"{cs} & {zt:g} & {rho:.1f} & {span} & "
                f"{h0['rej_CW']:.3f} & {b(h0['rej_CC_oracle'])} & "
                f"{b(h0['rej_CC_samp_oracle'])} & {b(h0['rej_CC_diff'])} & "
                f"{fit['Psihat']:.3f} & {fit['mu_CC']:.3f} & {fit['mspe_gap']:.3f} & "
                f"{fit['rej_DM']:.3f} & {fit['rej_CC_oracle']:.3f} \\\\")

    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{More predictors than observations. Each row is one design: $T=12$,"
         r" $n=8{,}000$ forecast origins, bandwidth $b=18$, $1{,}000$ replications,"
         r" one-sided tests at $5\%$. Panel A uses the Gaussian predictors of"
         r" \eqref{eq:mc-predictors} with $\rho_x=0$ and $P=cT$, and a signal that is a"
         r" linear combination of them. Panels B and C use $P$ random Fourier features"
         r" $\sqrt2\cos(\gamma WR_t+b)$ of $d=5$ raw AR($1$) predictors $R_t$ with"
         r" persistence $\rho_x$, bandwidth $\gamma=0.3$ and raw penalty $z=0.5$; the signal"
         r" is a linear combination either of the features (in span) or of the raw"
         r" predictors (outside). The signal has $R^2=0.10$ in panels A and B and $0.005$,"
         r" the monthly equity calibration of \cref{sec:fixedT}, in panel C. Each design is"
         r" simulated once and evaluated twice. The left block multiplies the forecast"
         r" increment $\yhat2-\yhat1$ by $a_{\mathrm{CC}}=2\E[fg]/\E[g^2]$, estimated on an"
         r" independent pilot sample of $50$ paths ($200$ in panel C), which puts the design"
         r" on the boundary of $\Hcc$, $\mu_{\mathrm{CC}}=0$, while $\E[fg]>0$; its entries"
         r" are rejection frequencies under a true null, that is, size. CW is the"
         r" Clark--West test; CC is the corrected test with the true $\sigma^2$,"
         r" studentised by $\hat\omega^2_n$, and samp the same statistic studentised by"
         r" the sample variance of $\dCC$; the $\hat\sigma^2_\Delta$ column replaces"
         r" $\sigma^2$ by the difference estimator and studentises by $\hat\omega^2_n$. The"
         r" right block uses the forecast as estimated, with no rescaling. There"
         r" $\mu_{\mathrm{CC}}>0$, so its rejection frequencies are power. $\Psihatn$ is"
         r" the average plug-in correction, $\mu_{\mathrm{CC}}$ the average of $\dCC$ with"
         r" the true $\sigma^2$, which estimates the population value without bias by"
         r" \cref{lem:psi-t}, and the MSPE gap the average of $\ehat1^2-\ehat2^2$, negative"
         r" when the enlarged model forecasts worse; DM and CC use the true $\sigma^2$. The"
         r" Monte Carlo standard error of a rejection frequency is $0.007$ at a nominal"
         r" $0.05$; bold marks a size entry of the corrected test outside $0.0365$ to"
         r" $0.0635$, the $95\%$ band of an exact test. The two $c=0.5$ rows connect panel A"
         r" to \cref{tab:mc-size}. Generated by \texttt{mc\_overparam.py}.}",
         r"\label{tab:mc-overparam}", r"\small", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{rrrlrrrrrrrrr}", r"\toprule",
         r"& & & & \multicolumn{4}{c}{boundary of $\Hcc$ (size)}"
         r" & \multicolumn{5}{c}{forecast as estimated} \\",
         r"\cmidrule(lr){5-8}\cmidrule(lr){9-13}",
         r"& & & & \multicolumn{3}{c}{true $\sigma^2$} & & & & & & \\",
         r"\cmidrule(lr){5-7}",
         r"$c$ & $\ztil$ & $\rho_x$ & signal & CW & CC & samp & $\hat\sigma^2_\Delta$"
         r" & $\Psihatn$ & $\mu_{\mathrm{CC}}$ & MSPE gap & DM & CC \\",
         r"\midrule",
         r"\multicolumn{13}{l}{\emph{A. Generic features, $R^2=0.10$}} \\"]
    for c in (0.5, 1.5, 2.0, 5.0, 10.0):
        P = int(round(c * 12))
        for zt in (0.1, 0.001):
            L.append(line(c, zt, 0.0, "in span", get("G", P, zt, row="H0-CC"), get("G", P, zt)))
    L += [r"\addlinespace", r"\multicolumn{13}{l}{\emph{B. Random Fourier features, $R^2=0.10$}} \\"]
    for rho, sg in ((0.3, "raw"), (0.0, "raw"), (0.3, "span"), (0.0, "span")):
        if (rho, sg) == (0.3, "raw"):
            h0, fit = get("F", 200, row="H0-CC"), get("F", 200)
        else:
            h0 = get("Fd", 200, rho=rho, signal=sg, row="H0-CC")
            fit = get("Fd", 200, rho=rho, signal=sg)
        L.append(line(200 / 12, 0.5 * 12 / 200, rho, "in span" if sg == "span" else "outside", h0, fit))
    L.append(line(2000 / 12, 0.5 * 12 / 2000, 0.3, "outside",
                  get("F", 2000, row="H0-CC"), get("F", 2000)))
    L += [r"\addlinespace",
          r"\multicolumn{13}{l}{\emph{C. Random Fourier features, $R^2=0.005$}} \\"]
    for P, rho, sg in ((200, 0.3, "raw"), (200, 0.0, "raw"), (200, 0.3, "span"),
                       (200, 0.0, "span"), (2000, 0.3, "raw")):
        L.append(line(P / 12, 0.5 * 12 / P, rho, "in span" if sg == "span" else "outside",
                      get("FL", P, rho=rho, signal=sg, row="H0-CC"),
                      get("FL", P, rho=rho, signal=sg)))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab-mc-overparam.tex", L)


# ------------------------------------------------- cross-section (rv_cross/panel.json)
def _rv_cross_rows():
    with open(os.path.join(OUT, "rv_cross", "panel.json")) as fh:
        return json.load(fh)["rows"]


def tab_rv_cross_psi():
    import numpy as np
    rows = _rv_cross_rows()
    N = f"{len({r['id'] for r in rows}):,}".replace(",", "{,}")
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{The cross-section: the correction. The comparison of \cref{tab:rv-main}"
         r" --- \citet{corsi2009simple} HAR against ridge on lags $2,\dots,25$ of the same"
         rf" series --- run on {N} realized-variance series drawn at random from the Risk Lab"
         r" universe, at three penalties fixed before the panel was run. $T=250$ throughout"
         r" and no penalty is chosen per asset. $\Psi(c,z)$ is the deterministic equivalent"
         r" \eqref{eq:closed-main}, evaluated at the raw penalty $z=\ztil c$; it sees only"
         rf" $c$ and $z$ and therefore returns one number for all {N} series. $\Psihatn$ is"
         r" the realised correction, reported as the cross-sectional median and interquartile"
         r" range. The last column expresses $\hat\sigma^2\Psihatn$ as a share of the"
         r" benchmark's own out-of-sample mean squared error.}",
         r"\label{tab:rv-cross-psi}", r"\small",
         r"\begin{tabular}{rrrrlr}", r"\toprule",
         r"$\ztil$ & $c$ & $\Psi(c,z)$ & median $\Psihatn$ & IQR"
         r" & $\hat\sigma^2\Psihatn/\MSPE_1$ \\", r"\midrule"]
    for zt in (0.1, 1.0, 10.0):
        R = [r for r in rows if r["zt"] == zt]
        c = float(np.median([r["c"] for r in R]))
        ps = np.array([r["Psihat"] for r in R])
        sh = np.array([r["corr_win"]/r["mse_bench"] for r in R])
        L.append(f"{zt} & {c:.3f} & {Psi(c, zt*c):.4f} & {np.median(ps):.4f} & "
                 f"[{np.percentile(ps,25):.4f}, {np.percentile(ps,75):.4f}] & "
                 f"{np.median(sh):.3f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab-rv-cross-psi.tex", L)


def tab_rv_cross_stats():
    import numpy as np
    rows = _rv_cross_rows()
    N = f"{len({r['id'] for r in rows}):,}".replace(",", "{,}")
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{The cross-section: the three statistics. Same panel and same penalties as"
         r" \cref{tab:rv-cross-psi}. $\overline{\Delta\MSPE}$ is the median out-of-sample"
         r" deterioration of the enlarged model as a share of the benchmark's own mean squared"
         rf" error, and the next column counts how many of the {N} series have a negative gap."
         r" Means and standard deviations are taken across series; they are not sizes, because"
         r" each series is its own data-generating process. $\#$CW is the number of"
         r" CW rejections, one-sided at $5\%$, and \emph{only CW} counts those the"
         r" corrected statistic declines to make. $\mathcal R\ge1$ counts the series passing"
         r" the diagnostic \eqref{eq:ratio}, and the last column counts those whose realised"
         r" ordering of $\tCW$ and $\tCC$ agrees with what \cref{tab:three-nulls} predicts from"
         r" that diagnostic. The series share a common volatility factor and are not"
         r" independent draws.}",
         r"\label{tab:rv-cross-stats}", r"\small",
         r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{rrrrrrrrrr}", r"\toprule",
         r"$\ztil$ & $\overline{\Delta\MSPE}$ & gap$<0$ & $t_{DM}$ & $\tCW$ & $\tCC$ & \#CW"
         r" & only CW & $\mathcal R\ge1$ & ordering \\",
         r" & median & count & mean & mean (sd) & mean (sd) & & & & agrees \\", r"\midrule"]
    for zt in (0.1, 1.0, 10.0):
        R = [r for r in rows if r["zt"] == zt]
        gap = np.array([r["mspe_gap"]/r["mse_bench"] for r in R])
        tdm = np.array([r["t_DM"] for r in R])
        tcw = np.array([r["t_CW"] for r in R])
        tcc = np.array([r["t_CC"] for r in R])
        rat = np.array([r["ratio"] for r in R])
        cw, cc = tcw > 1.645, tcc > 1.645
        L.append(f"{zt} & ${np.median(gap):.3f}$ & {int((gap<0).sum())} & "
                 f"${tdm.mean():.2f}$ & ${tcw.mean():.2f}$ ({tcw.std():.2f}) & "
                 f"${tcc.mean():.2f}$ ({tcc.std():.2f}) & {int(cw.sum())} & "
                 f"\\textbf{{{int((cw & ~cc).sum())}}} & {int((rat>=1).sum())} & "
                 f"{int(((rat>=1) == (tcw>tcc)).sum())} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab-rv-cross-stats.tex", L)


def tab_rv_multiest():
    """The paired cross-section: same assets, same benchmark, two alternatives.

    The own-lags panel and the multi-estimator panel run on slightly different assets, so reporting
    the two at their own sample sizes would confound the alternative with the sample.  Every
    cell below is computed on the assets present in both panels AND passing the diagnostic in
    both at that penalty.
    """
    import numpy as np
    with open(os.path.join(OUT, "rv_cross_multiest", "panel.json")) as fh:
        mrows = json.load(fh)["rows"]
    orows = _rv_cross_rows()

    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{The cross-section, paired: what changes when the enlarging columns"
         r" change. Both rows of each block use the HAR benchmark \eqref{eq:rv-har} on the"
         r" same series, the same rolling window $T=250$ and the same"
         r" penalties, all fixed before either panel was run. They differ in what the"
         r" extra columns are: lags $2,\dots,25$ of the target itself ($P=24$), or five lags each"
         r" of five other Risk Lab estimators of the same latent variance \eqref{eq:rv-multiest}"
         r" ($P=25$)."
         r" Each cell uses the assets present in both panels that pass \eqref{eq:ratio} in"
         r" both at that penalty, so the two rows of a block are the same series and differ"
         r" in the alternative alone. Every entry after the median is a share of those"
         r" assets, computed from each asset's own out-of-sample run: gap$<0$, the share whose"
         r" mean squared error gap is negative; CW and CC, the shares on which each test rejects"
         r" one-sided at $5\%$; only CW, the share on which CW rejects and the corrected"
         r" statistic does not. The series share a common volatility factor and are not"
         r" independent draws, so these are frequencies across series and not sizes.}",
         r"\label{tab:rv-multiest}", r"\small",
         r"\setlength{\tabcolsep}{5pt}",
         r"\begin{tabular}{llrrrrrr}", r"\toprule",
         r"$\ztil$ & enlarging columns & assets & median $\Psihatn$ & gap$<0$ & CW"
         r" & CC & only CW \\", r"\midrule"]

    for k, zt in enumerate((0.1, 1.0, 10.0)):
        o = {r["id"]: r for r in orows if r["zt"] == zt}
        m = {r["id"]: r for r in mrows if abs(r["zt"] - zt) < 1e-12}
        ids = sorted(i for i in set(o) & set(m)
                     if o[i]["ratio"] >= 1.0 and m[i]["ratio"] >= 1.0)
        if not ids:
            continue
        for lab, d in (("own lags", o), ("five estimators", m)):
            f = lambda key: np.array([d[i][key] for i in ids])
            cw, cc = f("t_CW") > 1.645, f("t_CC") > 1.645
            L.append(f"{zt if lab == 'own lags' else ''} & {lab} & {len(ids)} & "
                     f"{np.median(f('Psihat')):.4f} & {np.mean(f('mspe_gap') < 0):.3f} & "
                     f"{cw.mean():.3f} & {cc.mean():.3f} & "
                     f"{(cw & ~cc).mean():.3f} \\\\")
        if k < 2:
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab-rv-multiest.tex", L)


# ------------------------------------------------------- ldv_scope (HAR benchmark)
LDV = os.path.join(HERE, "ldv_scope", "output")


def _ldv(name):
    with open(os.path.join(LDV, name)) as fh:
        return json.load(fh)


def tab_cw_har():
    """Replicated cells are the mean over the main run and the two verify seeds."""
    import numpy as np
    main, ver = _ldv("cw_under_har.json"), _ldv("verify.json")
    def cells(scale, T):
        v = [main[f"scale{scale}_T{T}"]]
        v += [ver[k] for k in ver
              if k.startswith("V1|") and k.endswith(f"|{scale}|{T}")]
        size = [x.get("cw_size", x.get("size")) for x in v]
        return size, [x["mean_t"] for x in v]
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{CW under the sharp null with a rolling-window HAR benchmark and an exogenous",
         r"extra block. The correct rejection frequency is $0.05$ throughout. $P=25$, $\ztil=1$,",
         r"$n=700$, $300$ replications per seed, one-sided at $5\%$. $k$ is the number of",
         r"independent seeds averaged in the entry beside it. Bold marks a rate outside the",
         r"$95\%$ Monte Carlo band of an exact test, $0.05\pm1.96\sqrt{0.05\cdot0.95/(300k)}$.}",
         r"\label{tab:cw-har}", r"\small",
         r"\begin{tabular}{lcrcrc}", r"\toprule",
         r"& & \multicolumn{2}{c}{$T=250$} & \multicolumn{2}{c}{$T=1000$} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r"HAR coefficients scaled by & Spectral radius & size & $k$ & size & $k$ \\",
         r"\midrule"]
    for sc in ("0.4", "0.7", "0.9", "1.0"):
        s250, _ = cells(sc, 250); s1000, _ = cells(sc, 1000)
        srad = main[f"scale{sc}_T250"]["srad"]
        def bfk(v):
            # bold outside the 95% MC band of an exact 5% test, m = 300k draws
            x = float(f"{np.mean(v):.3f}"); h = 1.96*math.sqrt(0.05*0.95/(300*len(v)))
            return (r"\textbf{%.3f}" if abs(x-0.05) > h else "%.3f") % x
        L.append(f"{float(sc):.2f} & {srad:.3f} & {bfk(s250)} & {len(s250)} & "
                 f"{bfk(s1000)} & {len(s1000)} \\\\")
    _, t250 = cells("1.0", 250); _, t1000 = cells("1.0", 1000)
    L += [r"\midrule",
          r"\multicolumn{2}{@{}l}{mean $t_{\mathrm{CW}}$ at scale $1.00$} &",
          f"  \\multicolumn{{2}}{{c}}{{${np.mean(t250):.2f}$}} & "
          f"\\multicolumn{{2}}{{c}}{{${np.mean(t1000):.2f}$}} \\\\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab-cw-har.tex", L)


def tab_har_omitted():
    """Size under H_cc with an ESTIMATED innovation variance, in the application's class.

    Every other H_cc cell in the paper is oracle: tab:mc-grid imposes the null by rescaling
    and the difference estimator fails there, and the misspecification block of
    tab:mc-signal runs at sigma^2 known.  Both use a rolling-mean benchmark, so nothing
    strips the signal before sigma^2 is estimated.  This arm uses the fitted HAR benchmark
    and sigmahat^2_w, the estimator the application uses, and reaches mu_CC < 0 by putting
    the extra signal in lags the P-column feature set cannot span.

    The T=40 and T=60 blocks are the mean of two independent seeds; T=250 and T=100 are one.
    """
    import numpy as np, glob
    rows = []
    for fn in sorted(glob.glob(os.path.join(LDV, "har_omitted_*.json"))):
        with open(fn) as fh:
            rows.extend(json.load(fh))
    if not rows:
        return
    key = lambda r: (r["T"], r["lam"])
    seen = {}
    for r in rows:
        seen.setdefault(key(r), []).append(r)

    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Size under $\Hcc$ with an estimated innovation variance, HAR class."
         r" Benchmark: rolling HAR. Alternative: ridge on lags $1,\dots,25$ of the target."
         r" Extra signal: a zero-sum combination of lags $6,\dots,60$, scaled by $\lambda$."
         r" $\mu_{CC}$ is recovered from $\overline{\Delta\MSPE}=\mu_{CC}-\sigma^2\Psihatn$"
         r" at the true $\sigma^2$, and $\Hcc$ holds in rows with $\mu_{CC}<0$. The corrected"
         r" test uses $\hat\sigma^2_w$, the in-window residual variance of the fitted HAR."
         r" $P=25$, $\ztil=1$, $n=700$, one-sided at $5\%$, $300$ replications per seed; the"
         r" $T=40$ and $T=60$ blocks average two seeds. Bold marks a rejection rate, in a row"
         r" where $\Hcc$ holds, outside the $95\%$ Monte Carlo band of an exact test.}",
         r"\label{tab:har-omitted}", r"\small",
         r"\setlength{\tabcolsep}{5pt}",
         r"\begin{tabular}{rrrrrrrrr}", r"\toprule",
         r"$T$ & $c$ & $\lambda$ & $\mu_{CC}$ & $\hat\sigma^2_w/\sigma^2$ & $\mathcal R$"
         r" & DM & $\tCW$ & $\tCC$ \\", r"\midrule"]
    for i, T in enumerate((250, 100, 60, 40)):
        lams = sorted({k[1] for k in seen if k[0] == T})
        for j, lam in enumerate(lams):
            v = seen[(T, lam)]
            m = lambda f: float(np.mean([r[f] for r in v]))
            mu = m("mu_CC")
            # bold the cells where H_cc holds: a rejection there is an error, and those are
            # the only rows that carry a size claim
            cw, cc = m("rej_CW"), m("rej_CC_w")
            h = 1.96*math.sqrt(0.05*0.95/(300*len(v)))
            fmt = lambda x: (r"\textbf{%.3f}" % x) if (mu < 0 and abs(round(x, 3)-0.05) > h) else "%.3f" % x
            L.append(f"{T if j == 0 else ''} & {25/T:.2f} & {lam:.2f} & "
                     f"${mu:.5f}$ & {m('s2w')/v[0]['s2_true']:.3f} & {m('R_w'):.2f} & "
                     f"{m('rej_DM'):.3f} & {fmt(cw)} & {fmt(cc)} \\\\")
        if i < 3:
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab-har-omitted.tex", L)


def tab_har_power():
    import numpy as np
    from decimal import Decimal, ROUND_HALF_UP
    rows = _ldv("har_size_power.json")
    seed2 = _ldv("har_size_power_seed4711.json")
    ver = _ldv("verify.json")["V2"]
    get = lambda feat: [r for r in rows if r["feat"] == feat]
    # half-up, so a value landing exactly on a half (dMSPE = -0.006750) prints as the
    # table does rather than as Python's round-half-to-even would
    def q(x, n):
        d = Decimal(repr(x)).quantize(Decimal(1).scaleb(-n), rounding=ROUND_HALF_UP)
        return f"{d:+.{n}f}" if x < 0 else f"{d:.{n}f}"
    lag0 = get("lags")[0]; exo0 = [r for r in rows if r["feat"] == "exog"][0]
    seeds_a = [f"${x['rej_CC']:.3f}$" for x in ver] + \
              [f"${x['rej_CC']:.3f}$" for x in seed2 if x["feat"] == "lags"
               and x["lam"] in (0.0, 0.1, 0.2)]
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Size and power with a HAR benchmark. The benchmark regresses on",
         r"$\{1,\,y_{t-1},\,\overline y_{t-5:t-1},\,\overline y_{t-22:t-1}\}$ and the data are generated",
         r"by that HAR, so at $\lambda=0$ the extra features carry no incremental signal and the",
         r"nominal size is $0.05$. Under $\lambda>0$ a zero-sum combination of lags $7$--$25$ enters the",
         r"conditional mean, which HAR cannot span. $T=250$, $P=25$, $\ztil=1$, $n=700$, $300$",
         r"replications, one-sided at $5\%$. The corrected test uses the window-residual $\hat\sigma^2_w$ and a",
         r"Bartlett long-run variance; CW uses the sample variance. Bold marks a CW or corrected-test",
         r"rejection frequency at $\lambda=0$ outside the $95\%$ Monte Carlo band of an exact test at",
         r"$300$ replications, $0.025$ to $0.075$. $\mathcal R$ is the sanity ratio \eqref{eq:ratio}, which",
         r"equals $\kldv$ at $\lambda=0$ and exceeds it thereafter because the realised squared",
         r"differential contains the systematic component. The mean-squared-error gap is reported only in panel (a), where the corrected test",
         r"rejects, and the mean corrected statistic only in panel (b), where it never does.",
         r"Two independent seeds give " + ", ".join(seeds_a[:2]) + r",",
         seeds_a[2] + r" and " + ", ".join(seeds_a[3:5]) + r", " + seeds_a[5] +
         r" for the corrected test in panel (a) at $\lambda=0$,",
         r"$0.10$ and $0.20$, and $" +
         f"{[x for x in seed2 if x['feat']=='exog' and x['lam']==0.0][0]['rej_CW']:.3f}" +
         r"$ for CW in panel (b) at $\lambda=0$. Under the sharp",
         r"null the mean and standard deviation of $t_{\mathrm{CW}}$ are $" +
         f"{lag0['mean_CW']:.2f}$ and ${lag0['sd_CW']:.2f}$ in panel",
         r"(a) against $" + f"{exo0['mean_CW']:.2f}$ and ${exo0['sd_CW']:.2f}$ in panel (b)" + r".}",
         r"\label{tab:har-power}", r"\small",
         r"\begin{tabular}{lccrcccr}", r"\toprule",
         r"\multicolumn{8}{@{}l}{\emph{(a) alternative is ridge on the $P=25$ individual lags of $y$}}\\",
         r"\cmidrule(r){1-8}",
         r"$\lambda$ & $\Psihatn$ & $\mathcal R$ & $\MSPE_1-\MSPE_2$ & DM & CW & CC"
         r" & mean $t_{\mathrm{CC}}$ \\", r"\midrule"]
    for r in get("lags"):
        hb = 1.96 * math.sqrt(0.0475 / 300)
        bl = lambda x: ("\\textbf{%.3f}" if (r["lam"] == 0.0 and abs(x - 0.05) > hb) else "%.3f") % x
        cc = bl(r["rej_CC"])
        L.append(f"{r['lam']:.2f} & {r['Psihat']:.3f} & {r['R_plug']:.2f} & "
                 f"${q(r['dMSPE'],4)}$ & {r['rej_DM']:.3f} & {bl(r['rej_CW'])} & {cc} & \\\\")
    L += [r"\addlinespace",
          r"\multicolumn{8}{@{}l}{\emph{(b) alternative is ridge on $P=25$ exogenous features}}\\",
          r"\cmidrule(r){1-8}"]
    for r in [x for x in rows if x["feat"] == "exog"]:
        hb = 1.96 * math.sqrt(0.0475 / 300)
        bl = lambda x: ("\\textbf{%.3f}" if (r["lam"] == 0.0 and abs(x - 0.05) > hb) else "%.3f") % x
        cw = bl(r["rej_CW"])
        L.append(f"{r['lam']:.2f} & {r['Psihat']:.3f} & {r['R_plug']:.2f} & & "
                 f"{r['rej_DM']:.3f} & {cw} & {bl(r['rej_CC'])} & "
                 f"${q(r['mean_CC'],2)}$ \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab-har-power.tex", L)


def tab_voc_psi():
    """Emitted by voc_application.py, which needs only the bundled Goyal-Welch file.
    Its output is not shipped, so run that script first; this copies its result so
    the caption fix cannot drift again."""
    src = os.path.join(OUT, "voc_tables", "tab-voc-psi.tex")
    if not os.path.exists(src):
        print("  (voc_tables/tab-voc-psi.tex not found; run "
              "`python voc_application.py` first. Skipping tab-voc-psi.)")
        return
    shutil.copyfile(src, os.path.join(TEXTMP if CHECK else TEX, "tab-voc-psi.tex"))


def _mchar(name):
    with open(os.path.join(OUT, "mc_har", name)) as fh:
        return json.load(fh)


def _kdel():
    with open(os.path.join(OUT, "mc_kernel", "kernel_delta.json")) as fh:
        return json.load(fh)


def tab_kernel():
    """Arm 'mc_kernel': the design-conditional kernel, measured in the right moment.

    The entries are root-mean-square across t, the moment E[2 rho_h^2] that enters
    omega^2 and the HAC bias bound; the average of rho_h across t is close to zero and
    says nothing about the size of the kernel.  See mc_kernel_delta.py.
    """
    K = _kdel()["kernel"]
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Design-conditional kernel at $c=0.5$, $z=0.1$, computed exactly from",
         r"\eqref{eq:w} rather than estimated from a time average, $n=400$ and $24$",
         r"replications. The entries are root-mean-square across $t$, which is the moment",
         r"$\gamma_\ell=\E[2\varrho_\ell^2]$ depends on; the average across $t$ of",
         r"$\varrho_h$ itself is a mean-zero quantity and reports only its own sampling",
         r"error. The last two columns are what \eqref{eq:kernel-decay} asks for:",
         r"$\sqrt T$ times the $h=1$ ratio is flat, so $\varrho_h=O(\varrho_0T^{-1/2})$,",
         r"and the summed absolute autocovariance of the innovation-driven score is",
         r"bounded in $T$. $\Psi(0.5,0.1)=0.5690$.}",
         r"\label{tab:kernel-check}", r"\small",
         r"\begin{tabular}{lrrrrrr@{\qquad}rr}", r"\toprule",
         r"& $\varrho_0$ & \multicolumn{5}{c}{$\mathrm{rms}_t(\varrho_h)/\varrho_0$}"
         r" & $\sqrt T\,\frac{\mathrm{rms}_t(\varrho_1)}{\varrho_0}$"
         r" & $\sum_{\ell<T}|\gamma_\ell|$\\",
         r"\cmidrule(lr){3-7}",
         r"& & $h=1$ & $h=2$ & $h=T/4$ & $h=T/2$ & $h=T-1$ & & \\",
         r"\midrule"]
    for r in K:
        T = r["T"]; m = r["rms_rel"]
        keys = ["1", "2", str(max(1, T // 4)), str(max(1, T // 2)), str(T - 1)]
        L.append(f"$T={T}$ & {r['rho0']:.4f} & "
                 + " & ".join(f"{m[k]:.4f}" for k in keys)
                 + f" & {math.sqrt(T) * m['1']:.3f} & {r['sumabs_gamma']:.4f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab-kernel-check.tex", L)


def tab_delta():
    """Arm 'mc_kernel': the remainder Delta = V_off - 2 Cov(I,II) of prop:omega.

    Both constituents are design-conditional Gaussian objects, so Isserlis gives them
    in closed form in terms of the same weights w_{t-1} the kernel table already builds;
    no fourth-order deterministic equivalent is needed.  See mc_kernel_delta.py.
    """
    D = _kdel()["delta"]
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{The remainder $\Delta$ of \eqref{eq:omega}, evaluated. Both",
         r"constituents follow from \cref{lem:kernel} and Isserlis' theorem and are",
         r"computed exactly from the weights \eqref{eq:w}; nothing here is estimated from",
         r"a simulated loss differential. Gaussian i.i.d.\ design and innovations,",
         r"$\sigma^2=1$, $n=400$, $24$ replications; Monte Carlo standard errors are",
         r"below $0.005$ throughout. The last column is the standard deviation the",
         r"CW-studentised corrected statistic would have if $\Delta$ were the",
         r"only departure from $4\sigma^4\Psi$, that is",
         r"$\sqrt{(4\sigma^4\Psi+\Delta)/(4\sigma^4\Psi)}$; the $(c,z)$ pairs in rows two,",
         r"five and six are those of \cref{tab:size-check}, which reports $0.961$, $1.194$",
         r"and $1.204$ against the $0.917$, $1.139$ and $1.159$ here. The residue is the",
         r"finite-$T$ term of \cref{rem:omega-finite-T}, which enters with the opposite",
         r"sign.}",
         r"\label{tab:delta}", r"\small",
         r"\begin{tabular}{rrr rrr rr rr}", r"\toprule",
         r"$c$ & $z$ & $T$ & $\Psi$ & $4\sigma^4\Psi$ & $V_{\mathrm{off}}$"
         r" & $\Cov(\mathrm I,\mathrm{II})$ & $\Delta$"
         r" & $\Delta/4\sigma^4\Psi$ & implied s.d. \\", r"\midrule"]
    for r in D:
        L.append(f"{r['c']:g} & {r['z']:g} & {r['T']} & {r['Psi']:.4f} & {r['lead']:.4f} & "
                 f"{r['Voff']:.4f} & {r['covI2']:.4f} & {r['delta']:.4f} & "
                 f"{r['delta_rel']:.3f} & {r['sd_cw']:.3f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab-delta.tex", L)


def tab_kappa():
    """Arm 'mc_har': the contamination factor when the benchmark regresses on lags of y.

    Read from output/mc_har/, written by mc_har_diagnosis.py.
    """
    ke = _mchar("kappa_exploration.json")
    pg = _mchar("diagnosis.json")["persistence_grid"]
    A = ke["by_rho_and_k_T250_P25_zt1"]
    B = ke["by_penalty_and_P_T250_rho085_k3"]
    lo, hi = pg["rho0.85_T125"]["R"], pg["rho0.85_T500"]["R"]
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{What \cref{lem:psi-t} costs in the one configuration that fails: a benchmark",
         r"regressing on lags of the target against features drawn from outside the target's own past.",
         r"The entries are $\kldv=\E[(a_t'\eps_t)^2]/(\sigma^2\E\|a_t\|^2)$, the ratio of the true null",
         r"forecast variance to the plug-in correction; \cref{lem:psi-t} forces $\kldv=1$. Panel (a)",
         r"varies the persistence of the target and the number of benchmark lags at $T=250$, $P=25$,",
         r"$\ztil=1$; panel (b) varies the penalty and the feature count at $\rho=0.85$, $k=3$. The extra",
         r"features are exogenous throughout and the benchmark is correctly specified, so nothing here is",
         r"a misspecification effect. Read against panel (a) of \cref{tab:ldv-scope}, which shows that",
         r"replacing those exogenous features by lags of the target returns $\kldv$ to one. The failure",
         r"does not shrink in $T$: at $\rho=0.85$ a fourfold increase in the window moves $\kldv$ only",
         f"from ${lo:.2f}$ at $T=125$ to ${hi:.2f}$ at $T=500$, in a run of the same script not"
         f" shown in the table.}}",
         r"\label{tab:kappa}", r"\small",
         r"\begin{tabular}{lrrr}", r"\toprule",
         r"\multicolumn{4}{@{}l}{\emph{(a) target persistence $\rho$ and benchmark lags $k$}}\\",
         r"\cmidrule(r){1-4}",
         r"$\rho$ & $k=1$ & $k=3$ & $k=6$ \\", r"\midrule"]
    for lab, key in (("0", "rho0"), ("0.5", "rho0.5"), ("0.85", "rho0.85"), ("0.95", "rho0.95")):
        L.append(f"{lab} & " + " & ".join(f"{A[key][k]:.2f}" for k in ("k1", "k3", "k6")) + r" \\")
    L += [r"\addlinespace",
          r"\multicolumn{4}{@{}l}{\emph{(b) effective penalty $\ztil$ and feature count $P$}}\\",
          r"\cmidrule(r){1-4}",
          r"$\ztil$ & $P=10$ & $P=25$ & $P=60$ \\", r"\midrule"]
    for lab, key in (("0.1", "zt0.1"), ("1", "zt1.0"), ("10", "zt10")):
        L.append(f"{lab} & " + " & ".join(f"{B[key][p]:.2f}" for p in ("P10", "P25", "P60")) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab-kappa.tex", L)


def tab_pathfunc():
    """Does the scope condition reach functionals of the target's own path?

    kappa_pathfunc.py.  tab:ldv-scope covers lag features and exogenous features; every
    common realized-volatility extension is neither.  The 'lags' and 'exog'
    rows are controls and must track kappa_map.json's HAR rows -- that is what validates
    the harness, so they are printed beside the new arms rather than assumed.
    """
    base = _ldv("kappa_pathfunc.json")
    zt   = _ldv("kappa_pathfunc_zt.json")
    rob  = _ldv("kappa_pathfunc_robust.json")
    jmp  = _ldv("kappa_pathfunc_jumps.json")
    km   = _ldv("kappa_map.json")
    ctrl_lags = km["A|har|har|lags"]["kappa"]
    ctrl_exog = km["A|har|har|exog"]["kappa"]

    NM = {"subsamp": r"second estimator, same path",
          "quartic": r"realized quarticity",
          "bipower": r"bipower variation",
          "semivar": r"upside semivariance"}
    ZG = ["0.001", "0.01", "0.1", "1.0", "10.0"]
    TG = ["100", "250", "500"]
    MG = ["26", "78", "390"]
    JK = sorted(jmp, key=float)

    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{The scope condition reaches functionals of the target's own path.",
         r"Entries are $\kldv=\E[(a_t'E_{t-1})^2]/(\sigma^2\E\|a_t\|^2)$, which",
         r"\cref{lem:psi-t} forces to one. The benchmark is a correctly specified HAR and the",
         r"target is exactly HAR with innovations that are i.i.d.\ across days but built from a",
         r"within-day path, so the extra block carries no incremental signal by construction and",
         r"$\kldv$ is identified. The extra block is a functional of that same path, at the same",
         r"lag structure and the same $P$ as the lag-feature case, so the only thing varying",
         r"across rows is which functional the features are. \emph{Level} functionals share the",
         r"innovation's sign, \emph{magnitude} functionals only its scale. The first two rows are",
         f"controls for \\cref{{tab:ldv-scope}}'s HAR rows (${ctrl_lags:.2f}$ with lag features,",
         f"${ctrl_exog:.2f}$ with exogenous ones); they are what validates the harness. Panel (a)",
         r"uses $40$ replications and panels (b) and (c) $20$ per cell, drawn independently, so the",
         r"control rows differ across panels by Monte Carlo error.",
         r"Panel (b) varies the effective penalty, the window and the intraday grid; panel (c)",
         r"adds a compound-Poisson jump component, so that $RV-BPV$ is genuinely non-zero, which",
         r"is the configuration a jump decomposition exists for.}",
         r"\label{tab:pathfunc}", r"\small",
         r"\begin{tabular}{llr}", r"\toprule",
         r"\multicolumn{3}{@{}l}{\emph{(a) base cell: $T=250$, $P=25$, $\ztil=1$, $M=78$}}\\",
         r"\cmidrule(r){1-3}",
         r"extra features & channel & $\kldv$ \\", r"\midrule"]
    L.append(f"lags (control) & --- & {base['lags']['kappa_a']:.3f} \\\\")
    L.append(f"exogenous (control) & --- & {base['exog']['kappa_a']:.3f} \\\\")
    L.append(r"\addlinespace")
    for k in ("subsamp", "quartic", "bipower", "semivar"):
        L.append(f"{NM[k]} & {base[k]['channel']} & {base[k]['kappa_a']:.3f} \\\\")

    L += [r"\bottomrule", r"\end{tabular}", r"", r"\vspace{6pt}",
          r"\begin{tabular}{l" + "r" * (len(ZG) + len(TG) + len(MG)) + r"}", r"\toprule",
          r"\multicolumn{%d}{@{}l}{\emph{(b) penalty, window length, intraday grid}}\\"
          % (1 + len(ZG) + len(TG) + len(MG)),
          r"\cmidrule(r){1-%d}" % (1 + len(ZG) + len(TG) + len(MG)),
          r" & \multicolumn{%d}{c}{$\ztil$} & \multicolumn{%d}{c}{$T$} & \multicolumn{%d}{c}{$M$} \\"
          % (len(ZG), len(TG), len(MG)),
          r"\cmidrule(lr){2-%d}\cmidrule(lr){%d-%d}\cmidrule(lr){%d-%d}"
          % (1 + len(ZG), 2 + len(ZG), 1 + len(ZG) + len(TG),
             2 + len(ZG) + len(TG), 1 + len(ZG) + len(TG) + len(MG)),
          " & " + " & ".join(ZG) + " & " + " & ".join(TG) + " & " + " & ".join(MG) + r" \\",
          r"\midrule"]
    for k in ("lags", "subsamp", "bipower", "quartic"):
        cells = []
        cells += [f"{zt[k][g]['kappa_a']:.3f}" if k in zt and g in zt[k] else "---" for g in ZG]
        cells += [f"{rob['T'][k][g]['kappa_a']:.3f}" if k in rob["T"] else "---" for g in TG]
        cells += [f"{rob['M'][k][g]['kappa_a']:.3f}" if k in rob["M"] else "---" for g in MG]
        lab = "lags (control)" if k == "lags" else NM[k]
        L.append(f"{lab} & " + " & ".join(cells) + r" \\")

    L += [r"\bottomrule", r"\end{tabular}", r"", r"\vspace{6pt}",
          r"\begin{tabular}{lrrrrr}", r"\toprule",
          r"\multicolumn{6}{@{}l}{\emph{(c) jump intensity}}\\", r"\cmidrule(r){1-6}",
          r"$\lambda$ & $(RV-BPV)/RV$ & lags & second est. & bipower & quarticity \\",
          r"\midrule"]
    for k in JK:
        r_ = jmp[k]
        L.append(f"{float(k):g} & {r_['jump_share']*100:.1f}\\% & "
                 + " & ".join(f"{r_[f]['kappa_a']:.3f}"
                              for f in ("lags", "subsamp", "bipower", "quartic")) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab-pathfunc.tex", L)


def tab_signal(S):
    """Signal structure.  Built from armS.json rather than copied from the .tex
    mc_study.py emits, so the table has the same provenance as every other one
    and the package needs to ship no rendered output."""
    NM = {"spiked": "spiked", "dense": "dense", "omitted": "omitted"}
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Signal structure, and whether CW is rejecting a true null."
         r" $T=12$, $n=8{,}000$, $b=18$, $\rho_x=0.3$, $\Psi^*=0.5$ ($c=0.5$, $z=0.133$,"
         r" $P=6$), oracle $\sigma^2$, $1{,}000$ replications. Spiked: signal in one"
         r" predictor. Dense: signal across all six. Omitted: signal from a variable outside"
         r" the model, correlated $0.5$ with the first predictor. $\Hcov$ holds when"
         r" $2\E[fg]\le0$, $\Hcc$ when $\mu_{CC}\le0$. MSPE gap: mean of"
         r" $\ehat1^2-\ehat2^2$, negative when the enlarged model forecasts worse."
         r" One-sided $5\%$ rejection frequencies. Bold marks a CW or corrected-test rate in an"
         r" $R^2=0$ row outside the $95\%$ Monte Carlo band of an exact test, $0.0365$ to"
         r" $0.0635$.}",
         r"\label{tab:mc-signal}", r"\small",
         r"\setlength{\tabcolsep}{4.5pt}",
         r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
         r"signal & $R^2$ & $\hat\Psi_n$ & $2\E[fg]$ & $\E[g^2]$ & $\mu_{CC}$"
         r" & MSPE gap & DM & CW & CC \\",
         r"\midrule"]
    sb = lambda r, x: (r"\textbf{%.3f}" % x) if (r["R2"] == 0 and abs(x - 0.05) > 0.0135) else "%.3f" % x
    for sig in ("spiked", "dense", "omitted"):
        for r in sorted([x for x in S if x["signal"] == sig], key=lambda x: x["R2"]):
            L.append(f"{NM[sig]} & {r['R2']:.2f} & {r['Psihat']:.3f} & "
                     f"{r['Efg2']:.4f} & {r['Eg2']:.4f} & "
                     f"{r['mu_CC']:.3f} & ${r['mspe_gap']:.3f}$ & {r['rej_DM']:.3f} & "
                     f"{sb(r, r['rej_CW'])} & {sb(r, r['rej_CC'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab-mc-signal.tex", L)

def figures():
    import numpy as np
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import norm, binom
    plt.rcParams.update({"font.family": "serif", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    COLS = [("Clark-West", "t_CW"), ("noise-corrected", "t_CC_hac"),
            ("Diebold-Mariano", "t_DM")]
    ROWS = [("H0-cov ", r"$H_0^{\mathrm{cov}}$"), ("H0-CC  ", r"$H_0^{\mathrm{CC}}$"),
            ("H0-MSPE", r"$H_0^{\mathrm{MSPE}}$")]
    COLOR = {0: "#B02418", 1: "#1F5FA8", 2: "#6E6E6E"}
    for f in sorted(glob.glob(os.path.join(OUT, "pgrid_*.json"))):
        d = json.load(open(f)); R = d["design"]["R"]; nb = 10
        lo, hi = binom.ppf(0.025, R, 1/nb), binom.ppf(0.975, R, 1/nb)
        fig, axes = plt.subplots(3, 3, figsize=(9.0, 6.4))
        for i, (rk, rlab) in enumerate(ROWS):
            S = d["stats"][rk]
            for j, (cl, ck) in enumerate(COLS):
                ax = axes[i, j]; p = 1 - norm.cdf(np.array(S[ck])); own = (i == j)
                ax.hist(p, bins=nb, range=(0, 1), color=COLOR[j],
                        alpha=1.0 if own else 0.5, edgecolor="white", linewidth=0.6)
                ax.axhline(R/nb, ls="--", lw=0.8, color="k")
                ax.axhline(lo, ls=":", lw=0.7, color="k")
                ax.axhline(hi, ls=":", lw=0.7, color="k")
                if own:
                    ax.set_ylim(0, max(R/nb*2.0, hi*1.6)); ax.set_facecolor("#F6F6F2")
                    for s in ("left", "bottom"): ax.spines[s].set_linewidth(1.5)
                else:
                    ax.set_ylim(0, R*1.02)
                if i == 0: ax.set_title(cl, fontsize=10)
                if j == 0: ax.set_ylabel(rlab + "\ncount", fontsize=9)
                if i == 2: ax.set_xlabel("p-value", fontsize=9)
                ax.tick_params(labelsize=7)
        fig.tight_layout()
        stem = os.path.basename(f).replace(".json", "")
        fig.savefig(os.path.join(FIGTMP if CHECK else FIG, stem + ".pdf"),
                    format="pdf", bbox_inches="tight")
        plt.close(fig)



# ------------------------------------------------------- tables added in the revision
def w_raw(path, text):
    """Write a table whose cells are already in math mode."""
    with open(os.path.join(TEXTMP if CHECK else TEX, path), "w") as fh:
        fh.write(text)


def tab_rv_main():
    """tab:rv-main from output/rv_tables/rv_rows.json. The layout lives in ONE place,
    rv_application.write_output, so the paper and a fresh run cannot drift apart."""
    import rv_application as A
    R = load(os.path.join("rv_tables", "rv_rows.json"))
    meta = dict(R["meta"])
    meta["asset"] = (r"the SPDR S\&P 500 ETF (Risk Lab identifier 84398), 7,678 trading days "
                     r"from 1996 to 2026")
    old, oldfig = A.TABS, A.FIGS
    A.TABS = TEXTMP if CHECK else TEX
    # write_output also redraws fig-rv-stats and fig-rv-psi. Those are stored output of
    # rv_application.py; redrawing them here would change their PDF timestamps and break
    # the manifest, so the redraw goes to a throwaway directory.
    A.FIGS = tempfile.mkdtemp()
    try:
        A.write_output(R["rows"], meta)
    finally:
        A.TABS, A.FIGS = old, oldfig


def tab_voc_grid():
    """Emitted by voc_application.py; copied here so --check covers it."""
    src = os.path.join(OUT, "voc_tables", "tab-voc-grid.tex")
    if not os.path.exists(src):
        print("  (voc_tables/tab-voc-grid.tex not found; run `python voc_application.py`.)")
        return
    shutil.copyfile(src, os.path.join(TEXTMP if CHECK else TEX, "tab-voc-grid.tex"))


def tab_voc_weighted():
    """Emitted by voc_weighted.py; copied here so --check covers it."""
    src = os.path.join(OUT, "voc_tables", "tab-voc-weighted.tex")
    if not os.path.exists(src):
        print("  (voc_tables/tab-voc-weighted.tex not found; run `python voc_weighted.py`.)")
        return
    shutil.copyfile(src, os.path.join(TEXTMP if CHECK else TEX, "tab-voc-weighted.tex"))


def tab_window():
    """tab:window from output/mc_window/spiked_*.json (mc_window.py)."""
    import mc_window
    mc_window.latex(os.path.join(TEXTMP if CHECK else TEX, "tab-window.tex"))


def _pm(x):
    return f"${x:.3f}$" if x >= 0 else f"$-{abs(x):.3f}$"


def _ci(a):
    return f"[{_pm(a[0])}, {_pm(a[1])}]"


def tab_rv_predict():
    """tab:rv-predict and tab:rv-predict-robust from output/rv_window_bootstrap_*.json."""
    base = load("rv_window_bootstrap_base_b63.json")
    same = load("rv_window_bootstrap_same_b63.json")
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Which statistic, computed at a short window, predicts the gap a long window "
         r"delivers later. The five-estimator design of \cref{tab:rv-multiest}; series passing "
         r"\eqref{eq:ratio} in the decision period ("
         + ", ".join(f"${int(b['N'])}$" for b in base[:-1]) + f" and ${int(base[-1]['N'])}$"
         + r" at the three penalties). "
         r"Decision statistics use a $T=250$ window on dates before the outcome period; the outcome is "
         r"the relative mean-square gap (benchmark minus enlarged, over the benchmark's MSE; positive when the enlarged model is more accurate) of the same "
         r"procedure with the penalty held fixed, at $T=250$ or $T=1000$, over the second half of the "
         r"dates a $T=1000$ window can forecast (here 2024--2026). $\rho$ is the Spearman rank correlation, across "
         r"series, between each statistic computed in the decision period and the gap realised later. Intervals are $90\%$ moving-block bootstrap intervals over calendar dates ($63$-day "
         r"blocks, the same blocks for every series, $300$ draws), so common volatility shocks are "
         r"resampled jointly. The last two columns compare the median outcome gap with its prediction "
         r"from the $T=250$ decision period and the closed form, "
         r"$\overline{\mathrm{gap}}_{250}+\Psihatn\,[1-\Psi(P/T,z)/\Psi(P/250,z)]$. Robustness to the "
         r"outcome split, the penalty rule and the block length is in \cref{app:rv-predict}.}",
         r"\label{tab:rv-predict}", r"\small", r"\setlength{\tabcolsep}{3.5pt}",
         r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{rrrrrccrr}", r"\toprule",
         r"& & \multicolumn{3}{c}{$\rho$ with the later gap} & \multicolumn{2}{c}{difference in $\rho$} & \multicolumn{2}{c}{median gap} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
         r"$\ztil$ & outcome $T$ & DM & CW & CC & CC $-$ DM & CC $-$ CW & predicted & actual \\", r"\midrule"]
    for k, (s, b) in enumerate(zip(same, base)):
        if k:
            L.append(r"\addlinespace")
        L.append(f"{b['zt']:g} & 250 & {_pm(s['rho']['DM'])} & {_pm(s['rho']['CW'])} & "
                 f"{_pm(s['rho']['CC'])} & {_ci(s['CCmDM_date'])} & {_ci(s['CCmCW_date'])} & --- & --- \\\\")
        L.append(f" & 1000 & {_pm(b['rho']['DM'])} & {_pm(b['rho']['CW'])} & {_pm(b['rho']['CC'])} & "
                 f"{_ci(b['CCmDM_date'])} & {_ci(b['CCmCW_date'])} & {_pm(b['pred_med'])} & "
                 f"{_pm(b['act_med'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    w_raw("tab-rv-predict.tex", "\n".join(L) + "\n")

    V = [("half, penalty held (63-day blocks)", "base_b63"),
         ("half, penalty held (126-day blocks)", "base_b126"),
         ("last third, penalty held", "third_b63"),
         ("half, penalty rescaled with $T$", "ztilde_b63")]
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Robustness of \cref{tab:rv-predict} at the long window ($T=1000$). Each block varies "
         r"one choice: the block length of the date bootstrap, the share of the sample used as the "
         r"outcome period, and whether the ridge penalty is held at its $T=250$ value or rescaled with "
         r"the window. Notation as in \cref{tab:rv-predict}.}",
         r"\label{tab:rv-predict-robust}", r"\small", r"\setlength{\tabcolsep}{3.5pt}",
         r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lrrrrccrr}", r"\toprule",
         r"variant & $\ztil$ & DM & CW & CC & CC $-$ DM & CC $-$ CW & pred. & actual \\", r"\midrule"]
    for k, (lab, fn) in enumerate(V):
        R = load(f"rv_window_bootstrap_{fn}.json")
        if k:
            L.append(r"\addlinespace")
        for j, b in enumerate(R):
            L.append(f"{lab if j == 0 else ''} & {b['zt']:g} & {_pm(b['rho']['DM'])} & "
                     f"{_pm(b['rho']['CW'])} & {_pm(b['rho']['CC'])} & {_ci(b['CCmDM_date'])} & "
                     f"{_ci(b['CCmCW_date'])} & {_pm(b['pred_med'])} & {_pm(b['act_med'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    w_raw("tab-rv-predict-robust.tex", "\n".join(L) + "\n")


def tab_intersection():
    """tab:intersection from output/mc_intersection/inter_*_s<seed>.json (mc_intersection.py)."""
    import numpy as np
    cells = {}
    for fn in sorted(glob.glob(os.path.join(OUT, "mc_intersection", "inter_*_s[1-9].json"))):
        d = load(fn)
        cells.setdefault((d["rho"], d["R2"], d["T"]), []).append(d)
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Size and power of the corrected test under three innovation-variance estimators and "
         r"the intersection rule of \cref{cor:intersection}. Rolling-mean benchmark, $P=6$ AR(1) features "
         r"with persistence $\rho_x$, dense signal, $T=12$, $n=4{,}000$, $b=18$, ridge penalty as in "
         r"\cref{tab:window}. Boundary: the candidate forecast rescaled so that "
         r"$\mu_{\mathrm{CC}}=0$, a point of $\Hcc$ carrying signal; alternative: the unscaled "
         r"forecast, $\mu_{\mathrm{CC}}>0$. Oracle uses the true $\sigma^2$; difference uses "
         r"$\hat\sigma^2_\Delta$ with the influence-function long-run variance of \cref{thm:feasible}; "
         r"window uses $\hat\sigma^2_{w,t}$ with the long-run variance of $d^{\mathrm{CC},w}_t$ "
         r"(\cref{thm:feasible-w}); both rejects only when the last two do. The two bias columns are "
         r"$\E[(\hat\sigma^2-\sigma^2)\Psihat_t]$ for each estimator. One-sided $5\%$ rejection "
         r"frequencies over $1{,}000$ replications (Monte Carlo s.e.\ $0.007$ at $0.05$). Bold marks a "
         r"boundary-row rate outside the $95\%$ Monte Carlo band of an exact test, $0.0365$ to $0.0635$.}",
         r"\label{tab:intersection}", r"\small", r"\setlength{\tabcolsep}{4.5pt}",
         r"\begin{tabular}{rrlrrrrrrr}", r"\toprule",
         r"$\rho_x$ & $R^2$ & point & $\mu_{\mathrm{CC}}$ & bias, diff. & bias, window & oracle & difference & window & both \\",
         r"\midrule"]
    first = True
    for key in sorted(cells, key=lambda k: (k[0], k[1])):
        ch = cells[key]
        if not first:
            L.append(r"\addlinespace")
        first = False
        for j, lab in enumerate(("boundary", "unscaled")):
            raw = np.vstack([np.array(c["raw"][lab]) for c in ch])
            rej = (raw > 1.645).mean(0)
            both = ((raw[:, 1] > 1.645) & (raw[:, 2] > 1.645)).mean()
            mu = np.mean([c[lab]["mu"] for c in ch])
            bd = np.mean([c[lab]["bias_d"] for c in ch])
            bw = np.mean([c[lab]["bias_w"] for c in ch])
            h = 1.96*math.sqrt(0.05*0.95/raw.shape[0])
            fb = lambda x: (r"\textbf{%.3f}" % x) if (lab == "boundary" and abs(round(x, 3)-0.05) > h) else "%.3f" % x
            sg = lambda x: "$%.4f$" % (0.0 if abs(x) < 5e-5 else x)
            L.append(f"{(key[0] if j == 0 else '')} & {(f'{key[1]:.2f}' if j == 0 else '')} & "
                     f"{'boundary' if lab == 'boundary' else 'alternative'} & "
                     f"{sg(mu)} & {sg(bd)} & {sg(bw)} & {fb(rej[0])} & {fb(rej[1])} & "
                     f"{fb(rej[2])} & {fb(both)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w_raw("tab-intersection.tex", "\n".join(L) + "\n")


def one(A, **kw):
    hits = [r for r in A if all(r[k] == v for k, v in kw.items())]
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one cell for {kw}, found {len(hits)}")
    return hits[0]


# Figures drawn by the application scripts land in output/; the paper reads them from FIG.
# Copying them here means one build_tables.py run leaves no stale figure in the paper.
SCRIPT_FIGURES = {
    "fig-rv-stats.pdf":    os.path.join("rv_figures", "fig-rv-stats.pdf"),     # rv_application.py
    "fig-rv-data.pdf":     os.path.join("rv_figures", "fig-rv-data.pdf"),      # fig_rv_data.py
    "fig-rv-disagree.pdf": os.path.join("rv_figures", "fig-rv-disagree.pdf"),  # fig_rv_disagree.py
    "fig-rv-cross.pdf":    os.path.join("rv_cross", "figures", "fig-rv-cross.pdf"),  # rv_cross_section.py
    "fig-voc-psi.pdf":     os.path.join("voc_figures", "fig-voc-psi.pdf"),     # voc_application.py
}


def copy_script_figures():
    if CHECK:
        return
    for dst, src in SCRIPT_FIGURES.items():
        shutil.copyfile(os.path.join(OUT, src), os.path.join(FIG, dst))


if __name__ == "__main__":
    if CHECK:
        TEXTMP = tempfile.mkdtemp(); FIGTMP = tempfile.mkdtemp()
    A, B, C = load("armA.json"), load("armB.json"), load("armC.json")
    S = load("armS.json")
    print(f"armA {len(A)} cells | armB {len(B)} rows | armC "
          f"{sum(len(v) for v in C.values())} rows")
    tab_size(A); tab_power(A); tab_closedform(A); tab_grid()
    tab_features(B); tab_breakdown(C); tab_signal(S)
    tab_rv_cross_psi(); tab_rv_cross_stats()
    tab_cw_har(); tab_har_power(); tab_har_omitted(); tab_voc_psi(); tab_kappa(); tab_complexity(); tab_overparam()
    tab_kernel(); tab_delta(); tab_pathfunc(); tab_rv_multiest(); 
    tab_rv_main(); tab_voc_grid(); tab_voc_weighted(); tab_window(); tab_rv_predict(); tab_intersection()
    figures()
    copy_script_figures()
    if CHECK:
        bad = []
        for n in os.listdir(TEXTMP):
            ref = os.path.join(REF, n)
            if not os.path.exists(ref) or not filecmp.cmp(os.path.join(TEXTMP, n), ref, shallow=False):
                bad.append(n)
        print("MISMATCH: " + ", ".join(bad) if bad
              else "all generated tables reproduce exactly from output/")
    else:
        print("rebuilt all generated tables and figures from output/")
