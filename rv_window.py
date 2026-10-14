"""Split-sample predictive validity on the realized-variance panel, first pass
(Section 6.4 and app:rv-predict; the stored results are summarised by rv_window_bootstrap.py).

At window T the out-of-sample mean-square gap estimates mu_CC(T) - sigma^2 Psi(P/T, z), and
the estimation-noise term falls as T grows.  A test of H0-CC therefore speaks to whether the
enlarged model pays once the window is long; Diebold-Mariano, which tests the gap at T,
speaks to whether it pays at T; Clark-West, centred on 2E[fg], speaks to neither.

Per asset and per penalty ztil in {0.1, 1, 10}:
  outcome period   the second half of the dates on which a T = 1000 window can forecast
  decision period  every earlier date on which a T = 250 window can forecast
  decision stats   t_DM, t_CW, t_CW_hac, t_CC, sanity ratio R and the gap, at T = 250
  outcomes         the mean-square gap relative to the benchmark's MSE at T = 1000, 500, 250
The two periods do not overlap, so a decision statistic cannot predict an outcome through
shared forecast errors.  The absolute penalty is held at its T = 250 value (same estimator,
more data); --hold ztilde rescales it with the window instead.

Needs network access to Risk Lab (dachxiu.chicagobooth.edu).  Resumable, one JSON per asset:
    python rv_window.py --run --workers 8
    python rv_window.py --report
Output: output/rv_window/<asset id>.json.
"""
import argparse, json, math, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PANEL = os.path.join(HERE, "output", "rv_cross_multiest", "panel_hac_check.json")
OUT = os.path.join(HERE, "output", "rv_window")
T_DEC, T_OUT = 250, (250, 500, 1000)
ZT = (0.1, 1.0, 10.0)
CRIT = 1.645
MIN_DEC, MIN_OUT = 500, 500


def run_asset(tid, hold):
    import rv_cross_multiest as MX
    from rv_application import (fetch_risklab, to_daily_variance, har_design,
                                extra_estimators, rolling, statistics)
    fn = os.path.join(OUT, f"{tid}.json")
    if os.path.exists(fn):
        return tid, "skip"
    try:
        raw = fetch_risklab(tid, verbose=False)
        raw = raw[~raw.index.duplicated(keep="first")]
        cols = MX.volatility_columns(raw)
        if len(cols) < 4:
            raise ValueError(f"only {len(cols)} estimator columns")
        bench, others = cols[0], cols[1:6]
        rv = to_daily_variance(raw[bench])
        frames = {c: to_daily_variance(raw[c]) for c in others}
        har = har_design(rv); ext = extra_estimators(frames, 5)
        har = har[~har.index.duplicated(keep="first")]
        ext = ext[~ext.index.duplicated(keep="first")]
        idx = har.index.intersection(ext.index)
        y = har.loc[idx, "y"].to_numpy()
        Xh = np.column_stack([np.ones(len(idx)),
                              har.loc[idx, ["rv_d", "rv_w", "rv_m"]].to_numpy()])
        Xe = ext.loc[idx].to_numpy(); P = Xe.shape[1]
        L = len(y)
        # rolling(..., T) forecasts targets y[T], ..., y[L-2]; index forecasts by target.
        first_out_window = max(T_OUT)            # earliest target a T=1000 window reaches
        n_long = (L - 1) - first_out_window
        n_out = n_long // 2
        out_start = (L - 1) - n_out              # outcome targets: out_start .. L-2
        dec_start, dec_end = T_DEC, out_start    # decision targets: T_DEC .. out_start-1
        if n_out < MIN_OUT or dec_end - dec_start < MIN_DEC:
            raise ValueError(f"too short: decision {dec_end-dec_start}, outcome {n_out}")

        def zgrid(T):
            return np.array([zt*P/(T_DEC if hold == "z" else T) for zt in ZT])

        def window(T, lo, hi):
            yt, yb, yh, psi, s2w, _, _ = rolling(y, Xh, Xe, T, zgrid(T))
            a, b = lo - T, hi - T                # positions in rolling's output
            return yt[a:b], yb[a:b], yh[:, a:b], psi[:, a:b], s2w[a:b]

        rec = dict(id=int(tid), P=P, hold=hold, bench=bench, est_cols=list(others),
                   decision=dict(first=str(idx[dec_start].date()), last=str(idx[dec_end-1].date()),
                                 n=dec_end-dec_start),
                   outcome=dict(first=str(idx[out_start].date()), last=str(idx[L-2].date()),
                                n=n_out),
                   cells=[])
        yt, yb, yh, psi, s2w = window(T_DEC, dec_start, dec_end)
        dec = {}
        for j, zt in enumerate(ZT):
            st = statistics(yt, yb, yh[j], psi[j], s2w)
            dec[zt] = dict(t_DM=st["t_DM"], t_CW=st["t_CW"], t_CW_hac=st["t_CW_hac"],
                           t_CC=st["t_CC"], ratio=st["ratio"], Psihat=st["Psihat"],
                           gap_rel=st["mspe_gap"]/st["mse_bench"])
        outc = {zt: {} for zt in ZT}
        for T in T_OUT:
            yt, yb, yh, psi, s2w = window(T, out_start, L - 1)
            for j, zt in enumerate(ZT):
                st = statistics(yt, yb, yh[j], psi[j], s2w)
                outc[zt][f"gap_rel_T{T}"] = st["mspe_gap"]/st["mse_bench"]
                outc[zt][f"t_DM_T{T}"] = st["t_DM"]
        for zt in ZT:
            rec["cells"].append(dict(zt=zt, z=float(zt*P/T_DEC), decision=dec[zt],
                                     outcome=outc[zt]))
        json.dump(rec, open(fn, "w"), indent=1)
        return tid, "ok"
    except Exception as e:
        json.dump(dict(id=int(tid), error=f"{type(e).__name__}: {e}"),
                  open(os.path.join(OUT, f"{tid}.err.json"), "w"))
        return tid, f"error {type(e).__name__}: {e}"


def run(workers, hold):
    os.makedirs(OUT, exist_ok=True)
    ids = sorted({int(r["id"]) for r in json.load(open(PANEL))["rows"]})
    json.dump(dict(panel=os.path.relpath(PANEL, HERE), hold=hold, T_dec=T_DEC, T_out=T_OUT, zt=ZT, n_ids=len(ids)),
              open(os.path.join(OUT, "_meta.json"), "w"), indent=1)
    print(f"{len(ids)} assets; decision T={T_DEC}; outcome T in {T_OUT}; hold={hold}")
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_asset, i, hold) for i in ids]
        for f in as_completed(futs):
            tid, msg = f.result(); done += 1
            if msg not in ("skip", "ok") or done % 50 == 0:
                print(f"[{done}/{len(ids)} {time.time()-t0:6.0f}s] {tid}: {msg}", flush=True)
    print(f"done in {(time.time()-t0)/60:.1f} min")


def _rank(x):
    o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); r[o] = np.arange(len(x))
    return r

def spearman(x, y):
    return float(np.corrcoef(_rank(x), _rank(y))[0, 1])

def auc(score, label):
    """P(score of a series that pays > score of one that does not); 0.5 = no information."""
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = _rank(np.r_[pos, neg]) + 1
    return float((r[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg)))


def report(pass_only=True):
    res = []
    for f in os.listdir(OUT):
        if f.endswith(".json") and not f.startswith("_") and not f.endswith(".err.json"):
            res.append(json.load(open(os.path.join(OUT, f))))
    nerr = sum(f.endswith(".err.json") for f in os.listdir(OUT))
    print(f"assets with results: {len(res)}   errors: {nerr}")
    if res:
        print(f"example periods: decision {res[0]['decision']}, outcome {res[0]['outcome']}")
    stats = ("t_DM", "t_CW", "t_CW_hac", "t_CC")
    for zt in ZT:
        C = [c for r in res for c in r["cells"] if abs(c["zt"] - zt) < 1e-12]
        if pass_only:
            C = [c for c in C if c["decision"]["ratio"] >= 1]
        if not C:
            continue
        print(f"\n=== ztilde = {zt}   (N = {len(C)}{', decision-period R >= 1' if pass_only else ''}) ===")
        for T in T_OUT:
            y = np.array([c["outcome"][f"gap_rel_T{T}"] for c in C]); pays = y > 0
            print(f"\n outcome: relative MSE gap at T={T} in the outcome period; "
                  f"pays (gap > 0) on {pays.mean():.3f}; median gap {np.median(y):+.4f}")
            print(f"   {'decision stat':14s} {'Spearman':>9s} {'AUC':>7s} "
                  f"{'rejects':>8s} {'pays|rej':>9s} {'pays|not':>9s}")
            for s in stats:
                x = np.array([c["decision"][s] for c in C]); rej = x > CRIT
                pr = pays[rej].mean() if rej.any() else float("nan")
                pn = pays[~rej].mean() if (~rej).any() else float("nan")
                print(f"   {s:14s} {spearman(x, y):+9.3f} {auc(x, pays):7.3f} "
                      f"{rej.mean():8.3f} {pr:9.3f} {pn:9.3f}")
        # Where do the decision statistics disagree, and who is right at T=1000?
        y = np.array([c["outcome"]["gap_rel_T1000"] for c in C]); pays = y > 0
        cw = np.array([c["decision"]["t_CW"] > CRIT for c in C])
        cc = np.array([c["decision"]["t_CC"] > CRIT for c in C])
        dm = np.array([c["decision"]["t_DM"] > CRIT for c in C])
        for lab, m in (("CW rejects, CC does not", cw & ~cc), ("CC rejects, DM does not", cc & ~dm),
                       ("neither CW nor CC rejects", ~cw & ~cc)):
            if m.any():
                print(f"   {lab:27s} N={m.sum():4d}  pays at T=1000 on {pays[m].mean():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--all", action="store_true", help="report without the R >= 1 filter")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--hold", choices=["z", "ztilde"], default="z")
    a = ap.parse_args()
    if a.run: run(a.workers, a.hold)
    if a.report: report(pass_only=not a.all)
