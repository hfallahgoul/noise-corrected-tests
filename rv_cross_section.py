"""The own-lag cross-section of realized-variance series (app:rv-cross; tab:rv-cross-psi,
tab:rv-cross-stats, figure fig-rv-cross).

Runs the comparison of rv_application.py across a pseudo-random sample of the Risk Lab
universe and reports the distribution of Psihat_n, the rejection rates of the three
statistics, how often Clark-West and the corrected test disagree, and how often Clark-West
rejects in favour of a model whose realised out-of-sample MSPE is worse.  Risk Lab publishes
no directory, so --discover probes a seeded permutation of the identifier space and keeps the
first series meeting a fixed history requirement; the seed is recorded in the manifest.

    python rv_cross_section.py --discover --target 600
    python rv_cross_section.py --run --n-assets 400
    python rv_cross_section.py --tables

Needs network access to Risk Lab.  The panel used in the paper is stored in
output/rv_cross/panel.json.
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


import argparse, datetime, json, math, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output", "rv_cross")
TABS = os.path.join(OUT, "tables")
FIGS = os.path.join(OUT, "figures")
for d in (OUT, TABS, FIGS):
    os.makedirs(d, exist_ok=True)

import tempfile
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplcache-rvx"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

# The estimator itself is imported, never copied. If Section 6.2 changes, this changes.
sys.path.insert(0, HERE)
from rv_application import (                       # noqa: E402
    fetch_risklab, to_daily_variance, har_design, extra_lags, rolling, statistics)

BASE_URL = "https://dachxiu.chicagobooth.edu/data.php?ticker={}"
MANIFEST = os.path.join(OUT, "manifest.json")
PANEL = os.path.join(OUT, "panel.json")

# Pre-specified design. Do not tune these after looking at the panel.
P_EXTRA = 25
ZTILDES = (0.1, 1.0, 10.0)
T_WINDOW = 250
CRIT = 1.645                      # one-sided 5%


# ================================================================= discovery
def probe_header(tid, timeout=15):
    """Read only the metadata block of one Risk Lab payload.

    Returns None when the identifier is absent or the response is not a data payload.
    Streams and aborts after a couple of kilobytes: the six metadata lines are all we
    need to decide whether the series is worth downloading in full.
    """
    import requests
    try:
        with requests.get(BASE_URL.format(tid), timeout=timeout, stream=True) as r:
            if r.status_code != 200:
                return None
            buf = b""
            for chunk in r.iter_content(chunk_size=1024):
                buf += chunk
                if len(buf) > 3072 or buf.count(b"\n") > 8:
                    break
    except Exception:
        return None
    lines = buf.decode("utf-8", "replace").split("\n")
    if len(lines) < 6:
        return None
    try:
        sym = lines[0].strip()
        pid = int(lines[1].strip())
        name = lines[2].strip()
        ndays = int(lines[3].strip())
        d0 = int(lines[4].strip())
        d1 = int(lines[5].strip())
    except (ValueError, IndexError):
        return None
    if pid != tid or ndays <= 0:
        return None
    return dict(id=tid, symbol=sym, name=name, ndays=ndays, start=d0, end=d1)


def discover(target, id_min, id_max, min_days, end_after, workers, seed, timeout):
    """Probe a pseudo-random permutation of the identifier space until `target` series
    meet the history requirement. Appends to the manifest so a run can be resumed."""
    prior = {}
    if os.path.exists(MANIFEST):
        M = json.load(open(MANIFEST))
        prior = {int(k): v for k, v in M.get("hits", {}).items()}
        tried_n = M.get("tried", 0)
        print(f"resuming: {len(prior)} qualifying series already found "
              f"({tried_n} identifiers probed)")
    else:
        tried_n = 0

    ids = list(range(id_min, id_max + 1))
    random.Random(seed).shuffle(ids)
    ids = [i for i in ids if i not in prior]

    hits = dict(prior)
    tried = tried_n
    t0 = time.time()
    print(f"probing for {target} series with >= {min_days} days ending after {end_after}")
    print(f"{'probed':>8}{'found':>8}{'rate':>8}{'elapsed':>10}   last hit")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {}
        it = iter(ids)
        # keep the pool topped up rather than submitting 80,000 futures at once
        for _ in range(workers * 4):
            try:
                t = next(it)
            except StopIteration:
                break
            futs[ex.submit(probe_header, t, timeout)] = t
        while futs and len(hits) < target:
            done = next(as_completed(futs))
            futs.pop(done)
            tried += 1
            rec = done.result()
            if rec and rec["ndays"] >= min_days and rec["end"] >= end_after:
                hits[rec["id"]] = rec
                if len(hits) % 10 == 0 or len(hits) < 5:
                    print(f"{tried:>8}{len(hits):>8}{len(hits)/max(tried,1):>8.3f}"
                          f"{time.time()-t0:>9.0f}s   {rec['symbol']:<8} {rec['name'][:38]}")
            try:
                t = next(it)
                futs[ex.submit(probe_header, t, timeout)] = t
            except StopIteration:
                pass
        for f in futs:
            f.cancel()

    man = dict(seed=seed, id_min=id_min, id_max=id_max, min_days=min_days,
               end_after=end_after, tried=tried, target=target,
               generated=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               hits={str(k): v for k, v in hits.items()})
    json.dump(man, open(MANIFEST, "w"), indent=1)
    print(f"\n[manifest] {len(hits)} series from {tried} probes "
          f"({(time.time()-t0)/60:.1f} min) -> {MANIFEST}")
    return man


# ================================================================= one asset
def run_one(rec, P_extra=P_EXTRA, T=T_WINDOW, ztildes=ZTILDES, min_eval=750):
    """Download one series and run the comparison at every pre-specified penalty.

    Returns a list of row dicts, or a dict with 'error' set. Kept import-clean so it can
    run in a worker process.
    """
    tid = rec["id"]
    try:
        raw = fetch_risklab(tid, verbose=False)
    except Exception as e:
        return dict(id=tid, error=f"fetch: {type(e).__name__}")

    # Pick the column with the most usable observations. Risk Lab pads absent series with
    # zeros and the column layout is not documented, so choosing by data content rather
    # than by position is the only safe rule. The plausibility filter keeps the choice
    # from landing on a column of counts or codes: an annualised volatility should sit
    # between one and five hundred percent.
    best, best_n = None, 0
    for c in raw.columns:
        v = pd.to_numeric(raw[c], errors="coerce").replace(0.0, np.nan).dropna()
        if len(v) < 100:
            continue
        med = float(v.median())
        if not (0.01 <= med <= 5.0):
            continue
        if len(v) > best_n:
            best, best_n = c, len(v)
    if best is None:
        return dict(id=tid, error="no column looks like an annualised volatility")
    if best_n < min_eval + T + P_extra + 40:
        return dict(id=tid, error=f"too few usable observations ({best_n})")

    rv = to_daily_variance(raw[best])
    # A series with long flat runs is a padded or stale quote, not a volatility path.
    if float((rv.diff() == 0).mean()) > 0.05:
        return dict(id=tid, error="more than 5% of days repeat the previous value")

    har = har_design(rv)
    ext = extra_lags(rv, P_extra)
    # A handful of Risk Lab payloads repeat a trading date. A duplicated index makes
    # .loc[idx] return more rows than len(idx) and the design matrices come out with
    # mismatched heights, which is what dropped five series from the first panel run.
    har = har[~har.index.duplicated(keep="first")]
    ext = ext[~ext.index.duplicated(keep="first")]
    idx = har.index.intersection(ext.index)
    if len(idx) < min_eval + T + 2:
        return dict(id=tid, error=f"short overlap ({len(idx)})")

    y = har.loc[idx, "y"].to_numpy()
    Xhar = np.column_stack([np.ones(len(idx)),
                            har.loc[idx, ["rv_d", "rv_w", "rv_m"]].to_numpy()])
    Xext = ext.loc[idx].to_numpy()[:, 1:]        # lag 1 is already in HAR as rv_d
    Peff = Xext.shape[1]

    zgrid = np.array([zt * Peff / T for zt in ztildes])
    yt, yb, yh, psi, s2w, psi_wd, psi_wl = rolling(y, Xhar, Xext, T, zgrid)
    n = len(yt)
    if n < min_eval:
        return dict(id=tid, error=f"n={n} below floor")

    e1 = yt - yb
    mse_bench = float(np.mean(e1 ** 2))
    rows = []
    for j, zt in enumerate(ztildes):
        corr = {"win": s2w * psi[j], "wdiff": psi_wd[j], "wloc": psi_wl[j]}
        st = statistics(yt, yb, yh[j], psi[j], s2w, corrections=corr, headline="win")
        st.update(id=tid, symbol=rec.get("symbol"), name=rec.get("name"),
                  P=Peff, c=Peff / T, zt=zt, n=n, T=T,
                  # recorded so this panel and the multi-estimator panel can be checked to
                  # sit on the SAME benchmark series asset by asset, which is what makes
                  # the two comparable in anything but name.
                  bench_col=best,
                  start=rec.get("start"), end=rec.get("end"),
                  mse_bench=mse_bench,
                  corr_share=float(np.mean(corr["win"])) / max(mse_bench, 1e-300))
        rows.append(st)
    return rows


# ================================================================= the panel
def build_panel(n_assets, workers, min_days, timeout_note=True):
    if not os.path.exists(MANIFEST):
        sys.exit("no manifest; run --discover first")
    man = json.load(open(MANIFEST))
    recs = [v for v in man["hits"].values() if v["ndays"] >= min_days]
    # Deterministic order: the manifest's own discovery order is already a random
    # permutation of permno space, so taking a prefix keeps the sample random.
    recs = sorted(recs, key=lambda r: r["id"])
    random.Random(man["seed"]).shuffle(recs)
    recs = recs[:n_assets]
    print(f"running {len(recs)} assets, P={P_EXTRA}, ztilde={ZTILDES}, T={T_WINDOW}")

    rows, errors = [], []
    t0 = time.time()
    done_n = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_one, r): r for r in recs}
        for f in as_completed(futs):
            r = futs[f]
            done_n += 1
            try:
                res = f.result()
            except Exception as e:
                errors.append(dict(id=r["id"], error=f"worker: {type(e).__name__}: {e}"))
                continue
            if isinstance(res, dict) and "error" in res:
                errors.append(res)
            else:
                rows.extend(res)
            if done_n % 20 == 0:
                el = time.time() - t0
                print(f"  {done_n}/{len(recs)} done, {len(errors)} dropped, "
                      f"{el/60:.1f} min elapsed, {el/done_n*(len(recs)-done_n)/60:.1f} to go")

    meta = dict(P=P_EXTRA, ztildes=list(ZTILDES), T=T_WINDOW, crit=CRIT,
                n_assets_requested=n_assets, n_assets_ok=len({r["id"] for r in rows}),
                n_dropped=len(errors), manifest_seed=man["seed"],
                manifest_tried=man["tried"],
                run=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    json.dump({"meta": meta, "rows": rows, "errors": errors},
              open(PANEL, "w"), indent=1, default=float)
    print(f"\n[panel] {meta['n_assets_ok']} assets, {len(errors)} dropped "
          f"({(time.time()-t0)/60:.1f} min) -> {PANEL}")
    return meta, rows, errors


# ================================================================= reporting
def closed_form(c, z):
    """Psi(c,z) of the deterministic equivalent, for comparison with the panel median.
    It sees only c and z: no feature covariance, no persistence, no long memory."""
    s = 1.0 + c + z
    return 0.5 * (s / math.sqrt(max(s * s - 4.0 * c, 1e-300)) - 1.0)


def rates(rows, zt):
    """Rejection and disagreement frequencies at one penalty."""
    R = [r for r in rows if abs(r["zt"] - zt) < 1e-12]
    if not R:
        return None
    n = len(R)
    cw = np.array([r["t_CW"] for r in R])
    cc = np.array([r["t_CC"] for r in R])
    ccw = np.array([r["t_CC_wdiff"] for r in R])
    dm = np.array([r["t_DM"] for r in R])
    gap = np.array([r["mspe_gap"] for r in R])
    ratio = np.array([r["ratio_win"] for r in R])
    share = np.array([r["corr_share"] for r in R])
    psi = np.array([r["Psihat"] for r in R])
    rej = lambda t: t > CRIT
    ok = ratio >= 1.0                      # cells where the corrected statistic is readable
    nCW = int(np.sum(rej(cw)))
    c = float(np.median([r["c"] for r in R]))
    # Table 3.1 makes t_CW > t_CC exactly when the diagnostic passes. The two use
    # different studentisations, so the realised ordering can differ near the boundary;
    # recording how often they agree is the check that identity survives contact with data.
    agree = float(np.mean((ok & (cc <= cw)) | (~ok & (cc > cw))))
    return dict(
        c=c, Psi_closed=closed_form(c, zt * c),
        order_agree=agree, n_order_agree=int(round(agree * n)),
        mean_DM=float(dm.mean()), mean_CW=float(cw.mean()), sd_CW=float(cw.std()),
        mean_CC=float(cc.mean()), sd_CC=float(cc.std()),
        gap_med=float(np.median(gap / np.array([r["mse_bench"] for r in R]))),
        n_both=int(np.sum(rej(cw) & rej(cc))),
        n_disagree=int(np.sum(rej(cw) & ~rej(cc))),
        zt=zt, n_assets=n,
        rej_DM=float(np.mean(rej(dm))), rej_CW=float(np.mean(rej(cw))),
        rej_CC=float(np.mean(rej(cc))), rej_CCw=float(np.mean(rej(ccw))),
        # the two headline numbers, unconditional
        disagree=float(np.mean(rej(cw) & ~rej(cc))),
        pathology=float(np.mean(rej(cw) & (gap < 0))),
        # and conditional on Clark-West having rejected, which is the form a practitioner
        # reads: of the rejections you would report, what share does not survive
        n_CW_rej=nCW,
        disagree_cond=float(np.sum(rej(cw) & ~rej(cc)) / nCW) if nCW else float("nan"),
        pathology_cond=float(np.sum(rej(cw) & (gap < 0)) / nCW) if nCW else float("nan"),
        # the same two restricted to assets where the diagnostic passes
        disagree_ok=float(np.mean((rej(cw) & ~rej(cc))[ok])) if ok.any() else float("nan"),
        pathology_ok=float(np.mean((rej(cw) & (gap < 0))[ok])) if ok.any() else float("nan"),
        both=float(np.mean(rej(cw) & rej(cc))),
        neither=float(np.mean(~rej(cw) & ~rej(cc))),
        cc_only=float(np.mean(~rej(cw) & rej(cc))),
        ratio_pass=float(np.mean(ok)),
        psi_med=float(np.median(psi)), psi_q1=float(np.percentile(psi, 25)),
        psi_q3=float(np.percentile(psi, 75)),
        share_med=float(np.median(share)), share_q1=float(np.percentile(share, 25)),
        share_q3=float(np.percentile(share, 75)),
        gap_neg=float(np.mean(gap < 0)))


def write_tables(meta, rows):
    S = [rates(rows, zt) for zt in meta["ztildes"]]
    S = [x for x in S if x]
    nA = meta["n_assets_ok"]

    capA = (r"\caption{The cross-section: the correction. The comparison of "
            r"\cref{tab:rv-main} --- \citet{corsi2009simple} HAR against ridge on lags "
            r"$2,\dots," + str(meta["P"]) + r"$ of the same series --- run on " + f"{nA}" +
            r" realized-variance series drawn at random from the Risk Lab universe, at "
            r"three penalties fixed before the panel was run. $T=" + str(meta["T"]) +
            r"$ throughout and no penalty is chosen per asset. $\Psi(c,\ztil)$ is the "
            r"deterministic equivalent \eqref{eq:closed-main}, which sees only $c$ and the "
            r"penalty and therefore returns one number for all " + f"{nA}" + r" series. "
            r"$\Psihatn$ is the realised correction, reported as the cross-sectional median "
            r"and interquartile range. The last column expresses $\hat\sigma^2\Psihatn$ as "
            r"a share of the benchmark's own out-of-sample mean squared error.}")
    L = [r"\begin{table}[htbp]", r"\centering", capA, r"\label{tab:rv-cross-psi}",
         r"\small", r"\begin{tabular}{rrrrlr}", r"\toprule",
         r"$\ztil$ & $c$ & $\Psi(c,\ztil)$ & median $\Psihatn$ & IQR &"
         r" $\hat\sigma^2\Psihatn/\MSPE_1$ \\", r"\midrule"]
    for x in S:
        L.append(f"{x['zt']:.1f} & {x['c']:.3f} & {x['Psi_closed']:.4f} & "
                 f"{x['psi_med']:.4f} & [{x['psi_q1']:.4f}, {x['psi_q3']:.4f}] & "
                 f"{x['share_med']:.3f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(TABS, "tab-rv-cross-psi.tex"), "w").write(mathsign("\n".join(L) + "\n"))

    capB = (r"\caption{The cross-section: the three statistics. Same panel and same "
            r"penalties as \cref{tab:rv-cross-psi}. $\overline{\Delta\MSPE}$ is the median "
            r"out-of-sample deterioration of the enlarged model as a share of the "
            r"benchmark's own mean squared error, and the next column counts how many of "
            r"the " + f"{nA}" + r" series have a negative gap. "
            r"Means and standard deviations are taken across series; they are not sizes, "
            r"because each series is its own data-generating process. $\#$CW is the number "
            r"of Clark--West rejections, one-sided at $5\%$, and \emph{only CW} counts those "
            r"the corrected statistic declines to make. $\mathcal R\ge1$ counts the "
            r"series passing the diagnostic \eqref{eq:ratio}, and the last column counts "
            r"those whose realised ordering of $\tCW$ and $\tCC$ agrees with what "
            r"\cref{tab:three-nulls} predicts from that diagnostic. The series share a "
            r"common volatility factor and are not independent draws.}")
    M = [r"\begin{table}[htbp]", r"\centering", capB, r"\label{tab:rv-cross-stats}",
         r"\small", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{rrrrrrrrrr}", r"\toprule",
         r"$\ztil$ & $\overline{\Delta\MSPE}$ & gap$<0$ & $t_{DM}$ &"
         r" $\tCW$ & $\tCC$ & \#CW & only CW & $\mathcal R\ge1$ & ordering \\",
         r" & median & count & mean & mean (sd) & mean (sd) & & & & agrees \\",
         r"\midrule"]
    for x in S:
        ngap = int(round(x["gap_neg"] * x["n_assets"]))
        M.append(f"{x['zt']:.1f} & {x['gap_med']:+.3f} & {ngap} & "
                 f"{x['mean_DM']:+.2f} & {x['mean_CW']:+.2f} ({x['sd_CW']:.2f}) & "
                 f"{x['mean_CC']:+.2f} ({x['sd_CC']:.2f}) & {x['n_CW_rej']} & "
                 f"\\textbf{{{x['n_disagree']}}} & "
                 f"{int(round(x['ratio_pass']*x['n_assets']))} & "
                 f"{x['n_order_agree']} \\\\")
    M += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(TABS, "tab-rv-cross-stats.tex"), "w").write(mathsign("\n".join(M) + "\n"))

    print("\n" + "=" * 112)
    print(f"CROSS-SECTION: {nA} assets, {meta['n_dropped']} dropped, "
          f"P={meta['P']}, T={meta['T']}")
    print("=" * 112)
    h = (f"{'ztilde':>7}{'Psi(c,z)':>10}{'med Psi':>9}{'ratio':>7}{'corr/MSE':>10}"
         f"{'gap/MSE':>9}{'t_DM':>8}{'t_CW':>14}{'t_CC':>14}"
         f"{'#CW':>5}{'both':>6}{'onlyCW':>8}{'R>=1':>6}{'order':>7}")
    print(h); print("-" * len(h))
    for x in S:
        print(f"{x['zt']:>7.1f}{x['Psi_closed']:>10.4f}{x['psi_med']:>9.4f}"
              f"{x['psi_med']/x['Psi_closed']:>7.2f}{x['share_med']:>10.3f}"
              f"{x['gap_med']:>+9.3f}{x['mean_DM']:>8.2f}"
              f"{x['mean_CW']:>+9.2f} ({x['sd_CW']:.2f}){x['mean_CC']:>+9.2f} ({x['sd_CC']:.2f})"
              f"{x['n_CW_rej']:>5}{x['n_both']:>6}{x['n_disagree']:>8}"
              f"{int(round(x['ratio_pass']*x['n_assets'])):>6}{x['n_order_agree']:>7}")
    print(f"\nAll counts are out of {S[0]['n_assets']} series.")
    print("onlyCW  = Clark-West rejects and the corrected statistic does not.")
    print("order   = realised ordering of t_CW and t_CC matches what the diagnostic predicts.")
    json.dump(S, open(os.path.join(TABS, "rv_cross_rates.json"), "w"), indent=1)
    return S


def write_figure(meta, rows, zt=1.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    R = [r for r in rows if abs(r["zt"] - zt) < 1e-12]
    cw = np.array([r["t_CW"] for r in R])
    cc = np.array([r["t_CC"] for r in R])
    gap = np.array([r["mspe_gap"] for r in R])
    ok = np.array([r["ratio_win"] >= 1 for r in R])

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    lo = float(min(cw.min(), cc.min())) - 0.6
    hi = float(max(cw.max(), cc.max())) + 0.6
    # the disagreement quadrant: CW rejects, CC does not
    ax.axhspan(lo, CRIT, xmin=(CRIT - lo) / (hi - lo), xmax=1.0,
               color="#f2c9c9", alpha=0.55, zorder=0, lw=0)
    ax.plot([lo, hi], [lo, hi], ls="--", c="0.6", lw=1, zorder=1)
    ax.axhline(CRIT, c="k", lw=0.9, zorder=2)
    ax.axvline(CRIT, c="k", lw=0.9, zorder=2)
    worse = gap < 0
    ax.scatter(cw[ok & worse], cc[ok & worse], s=16, c="#b22222", alpha=0.75,
               lw=0, zorder=3, label="alternative worse in MSPE")
    ax.scatter(cw[ok & ~worse], cc[ok & ~worse], s=16, c="#1f77b4", alpha=0.75,
               lw=0, zorder=3, label="alternative better in MSPE")
    ax.scatter(cw[~ok], cc[~ok], s=13, facecolors="none", edgecolors="0.65",
               lw=0.7, zorder=2, label=r"$\mathcal{R}<1$, not readable")
    ax.set_xlabel(r"Clark–West $t$")
    ax.set_ylabel(r"noise-corrected $t$")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    n_dis = int(np.sum((cw > CRIT) & (cc <= CRIT)))
    ax.set_title(rf"$\tilde z={zt}$, {len(R)} series; "
                 rf"{n_dis} in the shaded quadrant", fontsize=10)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"fig-rv-cross.{ext}"), dpi=250,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {FIGS}/fig-rv-cross.pdf")


# ================================================================= cli
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--discover", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--target", type=int, default=600,
                    help="how many qualifying series to find during discovery")
    ap.add_argument("--id-min", type=int, default=10000)
    ap.add_argument("--id-max", type=int, default=93500)
    ap.add_argument("--min-days", type=int, default=2500,
                    help="history floor; 2500 days is about ten years")
    ap.add_argument("--end-after", type=int, default=20200101,
                    help="drop series that stop before this date (yyyymmdd)")
    ap.add_argument("--n-assets", type=int, default=400)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260823)
    ap.add_argument("--timeout", type=int, default=15)
    args = ap.parse_args()

    if not (args.discover or args.run or args.tables):
        ap.error("pick at least one of --discover, --run, --tables")

    if args.discover:
        discover(args.target, args.id_min, args.id_max, args.min_days,
                 args.end_after, args.workers, args.seed, args.timeout)

    if args.run:
        meta, rows, errors = build_panel(args.n_assets, args.workers, args.min_days)
        if rows:
            write_tables(meta, rows)
            write_figure(meta, rows)
        if errors:
            print(f"\n{len(errors)} series dropped. First ten reasons:")
            for e in errors[:10]:
                print(f"   {e['id']}: {e['error']}")

    if args.tables and not args.run:
        D = json.load(open(PANEL))
        write_tables(D["meta"], D["rows"])
        write_figure(D["meta"], D["rows"])


if __name__ == "__main__":
    main()
