"""Store per-date forecasts for the realized-variance panel, so that every split-sample and
bootstrap calculation of rv_window_bootstrap.py runs without network access.

For each asset in output/rv_cross_multiest/panel_hac_check.json, writes
output/rv_window_store/<asset id>.npz holding, on the asset's own date index (NaN where a
window cannot forecast):
    date (yyyymmdd), y
    for T in (250, 1000):  yb_T, s2w_T       benchmark forecast and window sigma^2
                           yh_T, psi_T       ridge forecast and Psihat_t, shape (6, L)
The six rows are ztil in (0.1, 1, 10) with the absolute penalty held at its T = 250 value,
then the same three with the penalty rescaled with T.  About 0.25 MB per asset, 409 MB in
all, which is why the store is not shipped.

Needs network access to Risk Lab (dachxiu.chicagobooth.edu):
    python rv_window_store.py --workers 8
"""
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PANEL = os.path.join(HERE, "output", "rv_cross_multiest", "panel_hac_check.json")
OUT = os.path.join(HERE, "output", "rv_window_store")
TS, ZT, T0 = (250, 1000), (0.1, 1.0, 10.0), 250


def store_asset(tid):
    import rv_cross_multiest as MX
    from rv_application import (fetch_risklab, to_daily_variance, har_design,
                                extra_estimators, rolling)
    fn = os.path.join(OUT, f"{tid}.npz")
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
        Xe = ext.loc[idx].to_numpy(); P = Xe.shape[1]; L = len(y)
        if L < max(TS) + 1000:
            raise ValueError(f"L={L} too short")
        rec = dict(date=np.array([int(d.strftime("%Y%m%d")) for d in idx], dtype=np.int32),
                   y=y.astype(np.float32), P=np.int32(P))
        for T in TS:
            zg = np.array([zt*P/T0 for zt in ZT] + [zt*P/T for zt in ZT])
            yt, yb, yh, psi, s2w, _, _ = rolling(y, Xh, Xe, T, zg)
            n = len(yb)                                   # targets T .. T+n-1
            def full(v):
                a = np.full(v.shape[:-1] + (L,), np.nan, dtype=np.float32)
                a[..., T:T+n] = v; return a
            rec[f"yb_{T}"], rec[f"s2w_{T}"] = full(yb), full(s2w)
            rec[f"yh_{T}"], rec[f"psi_{T}"] = full(yh), full(psi)
        np.savez_compressed(fn, **rec)
        return tid, "ok"
    except Exception as e:
        json.dump(dict(id=int(tid), error=f"{type(e).__name__}: {e}"),
                  open(os.path.join(OUT, f"{tid}.err.json"), "w"))
        return tid, f"error {type(e).__name__}: {e}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ids = sorted({int(r["id"]) for r in json.load(open(PANEL))["rows"]})
    print(f"{len(ids)} assets -> {OUT}")
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(store_asset, i) for i in ids]
        for f in as_completed(futs):
            tid, msg = f.result(); done += 1
            if msg not in ("skip", "ok") or done % 50 == 0:
                print(f"[{done}/{len(ids)} {time.time()-t0:6.0f}s] {tid}: {msg}", flush=True)
    print(f"done in {(time.time()-t0)/60:.1f} min")
