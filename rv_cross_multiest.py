"""The multi-estimator cross-section (Section 6.4, tab:rv-multiest).

The benchmark is HAR on one realized-variance estimator; the enlarged model adds lags of
several other Risk Lab estimators of the same latent variance (QMLE and subsampled RV at two
frequencies, from trades and from mid-quotes), so any incremental content is
measurement-error averaging.  Each column is a different functional of the same intraday
path, the admissible case measured in tab:pathfunc.  Discovery, fetching, the rolling
estimator and the statistics are imported from rv_cross_section.py and rv_application.py,
so the two panels differ in the alternative only.

    python rv_cross_multiest.py --run --n-assets 400
    python rv_cross_multiest.py --tables
    python rv_cross_multiest.py --run --tag hac_check     # the panel used by rv_window*.py

Needs network access to Risk Lab.  The panels used in the paper are stored in
output/rv_cross_multiest/panel.json and panel_hac_check.json.
"""
import argparse, json, math, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rv_application import (fetch_risklab, to_daily_variance, har_design,
                            extra_estimators, rolling, statistics)
import rv_cross_section as X

OUT = os.path.join(HERE, "output", "rv_cross_multiest")
os.makedirs(OUT, exist_ok=True)

T_WINDOW = X.T_WINDOW
ZTILDES  = X.ZTILDES
CRIT     = X.CRIT


def volatility_columns(raw, min_obs=100):
    """Every column that is an annualised-volatility ESTIMATE, in descending coverage.

    Two filters.  The absolute one is rv_cross_section's: a median between one and five
    hundred percent, which removes counts and codes.  It is not enough on its own -- a
    standard-error column passes it too (on ticker 84398, c9 has median 0.0155).  The
    relative filter removes those: an estimate and its own standard error differ by an
    order of magnitude, so a column whose median is below a third of the largest is a
    standard error and not a second estimate.  On 84398 this keeps exactly
    {c2,c5,c6,c7,c10,c11} and drops {c3,c4,c8,c9}.
    """
    cand = {}
    for c in raw.columns:
        v = pd.to_numeric(raw[c], errors="coerce").replace(0.0, np.nan).dropna()
        if len(v) < min_obs:
            continue
        # Risk Lab serves integer code/count columns alongside the estimates (on ticker
        # 84398, c3 and c8 take values 1..12).  They must go FIRST.  The absolute filter
        # below is inclusive at 5.0, so a code column whose median happens to be 5 passes
        # it, becomes the maximum, and the relative filter then discards every genuine
        # estimate as though it were a standard error.  That is what rejected 553 assets
        # as "stale": the stale series was the code column, not the volatility.
        if float(np.isclose(v, np.round(v)).mean()) > 0.99:
            continue
        med = float(v.median())
        if 0.01 <= med <= 5.0:
            cand[c] = (med, len(v))
    if not cand:
        return []
    # Relative filter: an estimate and its own standard error differ by an order of
    # magnitude, so a column whose median is below a third of the largest is the latter.
    top = max(m for m, _ in cand.values())
    keep = {c: nv for c, (m, nv) in cand.items() if m >= top / 3.0}
    return sorted(keep, key=lambda c: -keep[c])


def run_one(rec, n_est=5, min_est=3, est_lags=5, T=T_WINDOW, ztildes=ZTILDES,
            min_eval=750, trades_only=False):
    """HAR on the best-covered estimator, enlarged by lags of the others."""
    tid = rec["id"]
    try:
        raw = fetch_risklab(tid, verbose=False)
    except Exception as e:
        return dict(id=tid, error=f"fetch: {type(e).__name__}")

    try:
        raw = raw[~raw.index.duplicated(keep="first")]
        cols = volatility_columns(raw)
        if len(cols) < 1 + min_est:
            return dict(id=tid, error=f"only {len(cols)} estimator columns (need {1+min_est})")
        bench, others = cols[0], cols[1:]
        if trades_only:
            # keep the estimators adjacent to the benchmark in the payload: Risk Lab groups a
            # source's columns together, so contiguity is the available proxy for "same source".
            bi = list(raw.columns).index(bench)
            others = sorted(others, key=lambda c: abs(list(raw.columns).index(c) - bi))[:2]
        others = others[:n_est]

        rv = to_daily_variance(raw[bench])
        if float((rv.diff() == 0).mean()) > 0.05:
            return dict(id=tid, error="more than 5% of days repeat the previous value")

        # A Risk Lab payload carries each estimate's own standard error beside it, and the
        # relative filter in volatility_columns only removes those an order of magnitude
        # below the LARGEST estimate -- on assets where the estimates sit close together an
        # SE can survive.  Record each enlarging column's median relative to the benchmark
        # so the composition of the design can be audited from the panel afterwards.
        _mb = float(pd.to_numeric(raw[bench], errors="coerce").replace(0.0, np.nan).median())
        med_ratio = [float(pd.to_numeric(raw[c], errors="coerce").replace(0.0, np.nan).median())
                     / max(_mb, 1e-300) for c in others]

        frames = {c: to_daily_variance(raw[c]) for c in others}
        har = har_design(rv)
        ext = extra_estimators(frames, est_lags)
        har = har[~har.index.duplicated(keep="first")]
        ext = ext[~ext.index.duplicated(keep="first")]
        idx = har.index.intersection(ext.index)
        if len(idx) < min_eval + T + 2:
            return dict(id=tid, error=f"short overlap ({len(idx)})")

        y = har.loc[idx, "y"].to_numpy()
        Xhar = np.column_stack([np.ones(len(idx)),
                                har.loc[idx, ["rv_d", "rv_w", "rv_m"]].to_numpy()])
        # every column is a lag of a DIFFERENT estimator, none of which the HAR contains,
        # so nothing is dropped here -- unlike the own-lags panel, where lag 1 is rv_d.
        Xext = ext.loc[idx].to_numpy()
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
                      bench_col=bench, est_cols=list(others), n_est=len(others),
                      med_ratio=med_ratio,
                      est_lags=est_lags, start=rec.get("start"), end=rec.get("end"),
                      mse_bench=mse_bench,
                      corr_share=float(np.mean(corr["win"])) / max(mse_bench, 1e-300))
            rows.append(st)
        return rows
    except Exception as e:
        return dict(id=tid, error=f"{type(e).__name__}: {e}")


def report(rows, meta):
    """The two rates, computed only where the diagnostic passes."""
    print("\n" + "=" * 92)
    print("MULTI-ESTIMATOR CROSS-SECTION")
    print("=" * 92)
    print(f"assets with usable panels: {meta['n_assets']}   "
          f"P: median {meta['P_median']}, range {meta['P_min']}-{meta['P_max']}")
    # gap<0 is printed because the "pathology" conjunction is only informative when it
    # differs from the CW rejection rate.  Where the enlarged model loses in every asset
    # the two coincide and the conjunction does no work, which the reader must be able
    # to see rather than infer.
    hdr = (f"{'ztil':>7}{'assets':>8}{'R>=1':>7}{'Psihat':>9}{'gap<0':>8}"
           f"{'CW rej':>9}{'CC rej':>9}{'disagree':>10}{'pathology':>11}")
    print("\n" + hdr); print("-" * len(hdr))
    out = []
    for zt in ZTILDES:
        R = [r for r in rows if abs(r["zt"] - zt) < 1e-12]
        if not R:
            continue
        ok = np.array([r["ratio_win"] >= 1.0 for r in R])
        cw = np.array([r["t_CW"] for r in R]); cc = np.array([r["t_CC"] for r in R])
        gap = np.array([r["mspe_gap"] for r in R])
        psi = np.array([r["Psihat"] for r in R])
        m = ok
        if m.sum() == 0:
            continue
        cwr = cw[m] > CRIT
        ccr = cc[m] > CRIT
        disagree = float(np.mean(cwr & ~ccr))
        pathology = float(np.mean(cwr & (gap[m] < 0)))
        negshare = float(np.mean(gap[m] < 0))
        row = dict(zt=zt, n=len(R), n_ok=int(m.sum()), psi_med=float(np.median(psi[m])),
                   gap_neg=negshare, cw_rej=float(np.mean(cwr)), cc_rej=float(np.mean(ccr)),
                   disagree=disagree, pathology=pathology,
                   mu_cc_neg=float(np.mean((gap[m] + np.array([r["corr_win"] for r in R])[m]) < 0)))
        out.append(row)
        print(f"{zt:>7}{len(R):>8}{int(m.sum()):>7}{np.median(psi[m]):>9.3f}{negshare:>8.3f}"
              f"{np.mean(cwr):>9.3f}{np.mean(ccr):>9.3f}{disagree:>10.3f}{pathology:>11.3f}")
    print("\n  disagree  = CW rejects at 5% and the corrected statistic does not")
    print("  pathology = CW rejects at 5% and the realised MSPE gap is negative")
    print("  both computed only on assets where the sanity ratio passes")
    print("  gap<0     = share where the enlarged model forecasts worse; when this is 1.000")
    print("              the pathology rate EQUALS the CW rejection rate and the")
    print("              conjunction is not a separate finding")

    # Composition audit: an enlarging column whose median is far below the benchmark's is
    # that estimator's standard error rather than a second estimate.  Both are functions of
    # the same intraday path, so neither breaks the theory, but the paper claims the design
    # is built from ALTERNATIVE ESTIMATORS and that claim has to be checkable.
    ratios = [x for r in rows if abs(r["zt"] - ZTILDES[0]) < 1e-12
              for x in r.get("med_ratio", [])]
    if ratios:
        ratios = np.array(ratios)
        print(f"\n  enlarging columns: {len(ratios)}   median/benchmark ratio: "
              f"median {np.median(ratios):.3f}, "
              f"share below 0.3 (standard-error-like) {np.mean(ratios < 0.3):.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--n-assets", type=int, default=400)
    ap.add_argument("--n-est", type=int, default=5)
    ap.add_argument("--min-est", type=int, default=3)
    ap.add_argument("--est-lags", type=int, default=5)
    ap.add_argument("--trades-only", action="store_true")
    ap.add_argument("--tag", default=None,
                    help="write to panel_<tag>.json instead of panel.json, so runs with "
                         "different --n-est/--est-lags do not overwrite one another")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--manifest", default=None,
                    help="reuse rv_cross_section's discovery manifest (default) so the "
                         "two panels run on the SAME assets and are comparable")
    a = ap.parse_args()

    man = a.manifest or os.path.join(X.OUT, "manifest.json")
    if not os.path.exists(man):
        sys.exit(f"no manifest at {man}\n"
                 f"run:  python rv_cross_section.py --discover --target 600")
    M = json.load(open(man))
    # rv_cross_section writes the discovered assets under "hits", keyed by permno as a
    # string.  Sorting by key makes the subset a deterministic function of the manifest,
    # so --n-assets 400 selects the same 400 assets on every run and the two panels are
    # comparable asset by asset.
    hits = M.get("hits")
    if not isinstance(hits, dict) or not hits:
        sys.exit(f"{man} has no 'hits' block; re-run rv_cross_section.py --discover")
    # Selection copied from rv_cross_section.build_panel EXACTLY -- history filter, sort
    # by id, then shuffle on the manifest seed -- so that at any --n-assets the two panels
    # run on the SAME assets and differ in the alternative alone.  Sorting by permno
    # instead would silently compare different samples.
    import random as _random
    recs = [v for v in hits.values() if v["ndays"] >= M.get("min_days", 2500)]
    recs = sorted(recs, key=lambda r: r["id"])
    _random.Random(M["seed"]).shuffle(recs)
    recs = recs[:a.n_assets]
    print(f"{len(recs)} of {len(hits)} discovered assets from {man} "
          f"(same selection rule as rv_cross_section)")

    if a.run:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        rows, errs = [], []
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(run_one, r, a.n_est, a.min_est, a.est_lags,
                              trades_only=a.trades_only): r for r in recs}
            for i, f in enumerate(as_completed(futs), 1):
                out = f.result()
                if isinstance(out, dict) and "error" in out:
                    errs.append(out)
                else:
                    rows.extend(out)
                if i % 25 == 0:
                    print(f"  {i}/{len(recs)}  ok={len(rows)//max(len(ZTILDES),1)} "
                          f"err={len(errs)}", flush=True)
        Ps = sorted({r["P"] for r in rows})
        meta = dict(n_assets=len({r["id"] for r in rows}), n_err=len(errs),
                    P_median=int(np.median([r["P"] for r in rows])) if rows else 0,
                    P_min=Ps[0] if Ps else 0, P_max=Ps[-1] if Ps else 0,
                    n_est=a.n_est, est_lags=a.est_lags, trades_only=a.trades_only,
                    T=T_WINDOW, ztildes=list(ZTILDES),
                    # stored package-relative: an absolute path here names the author's home
                    # directory, which deanonymises a paper under review and points a
                    # replicator at a directory that does not exist on their machine.
                    manifest=os.path.relpath(man, HERE))
        fn = f"panel_{a.tag}.json" if a.tag else "panel.json"
        json.dump(dict(meta=meta, rows=rows, errors=errs),
                  open(os.path.join(OUT, fn), "w"), indent=1)
        print(f"\n[saved] {OUT}/{fn}   ({meta['n_assets']} assets, {len(errs)} errors)")

        # Error census.  The own-lags panel runs on 995 of these 1000 assets because it
        # needs ONE usable column; this design needs two columns that overlap on enough
        # common dates, so a smaller panel is expected rather than a symptom.  Printing
        # the causes distinguishes that structural restriction from a filter that is still
        # rejecting assets it should not.
        import collections as _c
        def _cause(e):
            t = e["error"]
            if t.startswith("fetch"):                 return "fetch failed"
            if "estimator columns" in t:              return "fewer than 1+min_est columns survive the filter"
            if "repeat" in t:                         return "benchmark column stale (>5% repeated days)"
            if "short overlap" in t:                  return "columns do not overlap on enough common dates"
            if "below floor" in t:                    return "fewer than min_eval evaluation points"
            return t.split(":")[0]
        print("\n  why the other assets dropped out")
        for k, v in _c.Counter(_cause(e) for e in errs).most_common():
            print(f"    {v:>5}  {k}")
        print(f"    {meta['n_assets']:>5}  ran")

        report(rows, meta)

    if a.tables and not a.run:
        d = json.load(open(os.path.join(OUT, f"panel_{a.tag}.json" if a.tag else "panel.json")))
        report(d["rows"], d["meta"])


if __name__ == "__main__":
    main()
