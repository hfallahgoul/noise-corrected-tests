"""Which statistic, computed at a short window, predicts the gap a long window delivers
later: tab:rv-predict (Section 6.4) and tab:rv-predict-robust (app:rv-predict).

Reads output/rv_window_store/*.npz (written by rv_window_store.py).  Decision statistics are
computed at T = 250 on series whose decision-period sanity ratio is at least one; the
outcome is the relative mean-square gap later in the sample.  Variants:
  base        outcome = last half of the dates a T = 1000 window forecasts; penalty held
  third       outcome = last third
  ztilde      penalty rescaled with the window
  same        outcome window T = 250 (the same window as the decision)
Inference is a moving-block bootstrap of calendar dates, the same blocks for every series,
so common volatility shocks are resampled jointly; studentisations are held at their
full-sample values inside the bootstrap.  An asset bootstrap is reported beside it.

    python rv_window_bootstrap.py --variants base,same,third,ztilde --block 63 --reps 300
    python rv_window_bootstrap.py --variants base,same --block 126 --reps 300
Output: output/rv_window_bootstrap_<variant>_b<block>.json.
"""
import argparse, glob, json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, "output", "rv_window_store")
ZT = (0.1, 1.0, 10.0)
T0, T1 = 250, 1000


def bartlett_lrv(x):
    n = len(x); b = max(1, int(math.ceil(4*(n/100)**(2/9))))
    x = x - x.mean(); g = np.mean(x*x)
    for h in range(1, b+1):
        g += 2*(1 - h/(b+1))*np.mean(x[h:]*x[:-h])
    return max(g, 1e-14)


def psi_closed(c, z):
    u = 1 + c + z; q = u*u - 4*c
    return 0.5*(u/math.sqrt(q) - 1)


def rk(x):
    o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); r[o] = np.arange(len(x)); return r


def spear(x, y):
    return float(np.corrcoef(rk(x), rk(y))[0, 1])


def load():
    """File list and the common calendar grid; arrays are read lazily in build()."""
    files = sorted(glob.glob(os.path.join(STORE, "*.npz")))
    dates = [np.load(f)["date"] for f in files]
    grid = np.unique(np.concatenate(dates))
    return files, grid


def _assets(files, grid):
    for f in files:
        d = np.load(f)
        a = {k: d[k] for k in d.files}
        a["id"] = int(os.path.basename(f)[:-4]); a["pos"] = np.searchsorted(grid, a["date"])
        yield a


def build(A, frac, hold, grid=None, Tout=T1):
    """Per asset and penalty: decision-period series (T0) and outcome-period series (T1)."""
    h = 0 if hold == "z" else 3
    S = []
    for a in (_assets(A, grid) if grid is not None else A):
        L = len(a["y"]); P = int(a["P"]); y = a["y"].astype(float)
        n_long = (L - 1) - T1
        out_lo = (L - 1) - int(n_long*frac); out_hi = L - 1
        dec_lo, dec_hi = T0, out_lo
        if out_hi - out_lo < 250 or dec_hi - dec_lo < 500:
            continue
        for j, zt in enumerate(ZT):
            r = dict(id=a["id"], zt=zt, P=P)
            # decision, T0 (hold irrelevant at T0: rows j and j+3 coincide)
            sl = slice(dec_lo, dec_hi)
            yb = a[f"yb_{T0}"][sl].astype(float); yh = a[f"yh_{T0}"][j, sl].astype(float)
            ps = a[f"psi_{T0}"][j, sl].astype(float); s2 = a[f"s2w_{T0}"][sl].astype(float)
            yt = y[sl]
            e1, e2, D = (yt-yb)**2, (yt-yh)**2, (yh-yb)**2
            dDM = e1 - e2; dCW = dDM + D; dCC = dDM + s2*ps
            n = len(dDM)
            r["dec"] = dict(pos=a["pos"][sl], dDM=dDM, dCW=dCW, dCC=dCC,
                            se=dict(DM=dDM.std(), CW=dCW.std(), CC=math.sqrt(bartlett_lrv(dCC))),
                            n=n, ratio=float(D.mean()/max((s2*ps).mean(), 1e-300)),
                            Psihat=float(ps.mean()), e1=e1)
            # outcome, T1
            sl = slice(out_lo, out_hi)
            yb = a[f"yb_{Tout}"][sl].astype(float)
            yh = a[f"yh_{Tout}"][j + (h if Tout != T0 else 0), sl].astype(float)
            yt = y[sl]
            e1o = (yt-yb)**2
            r["out"] = dict(pos=a["pos"][sl], dDM=e1o - (yt-yh)**2, e1=e1o)
            z0 = zt*P/T0; z1 = z0 if hold == "z" else zt*P/T1
            r["k"] = psi_closed(P/Tout, z1 if Tout != T0 else z0)/psi_closed(P/T0, z0)
            S.append(r)
    return S


def point(S, zt):
    C = [s for s in S if s["zt"] == zt and s["dec"]["ratio"] >= 1]
    t = {k: np.array([math.sqrt(c["dec"]["n"])*c["dec"][f"d{k}"].mean()/c["dec"]["se"][k]
                      for c in C]) for k in ("DM", "CW", "CC")}
    y = np.array([c["out"]["dDM"].mean()/c["out"]["e1"].mean() for c in C])
    g0 = np.array([c["dec"]["dDM"].mean()/c["dec"]["e1"].mean() for c in C])
    pred = g0 + np.array([c["dec"]["Psihat"]*(1 - c["k"]) for c in C])
    return C, t, y, pred


def block_index(G, block, rng, reps):
    nb = int(math.ceil(G/block))
    return rng.integers(0, G - block + 1, size=(reps, nb))


def boot_sums(pos, x, starts, block, G):
    """Sum of x and count over the union (with repetition) of blocks [s, s+block)."""
    cs = np.concatenate([[0.0], np.cumsum(x)])
    lo = np.searchsorted(pos, starts); hi = np.searchsorted(pos, starts + block)
    return (cs[hi] - cs[lo]).sum(1), (hi - lo).sum(1)


def bootstrap(C, starts, block, G):
    R = starts.shape[0]; N = len(C)
    t = {k: np.empty((R, N)) for k in ("DM", "CW", "CC")}; y = np.empty((R, N))
    for i, c in enumerate(C):
        d, o = c["dec"], c["out"]
        for k in ("DM", "CW", "CC"):
            s, m = boot_sums(d["pos"], d[f"d{k}"], starts, block, G)
            t[k][:, i] = np.sqrt(np.maximum(m, 1))*(s/np.maximum(m, 1))/d["se"][k]
        s, m = boot_sums(o["pos"], o["dDM"], starts, block, G)
        s1, _ = boot_sums(o["pos"], o["e1"], starts, block, G)
        y[:, i] = np.where(m > 0, s/np.maximum(s1, 1e-300), np.nan)
    return t, y


def run(variant, frac, hold, block, reps, A, grid, rng, Tout=T1):
    S = build(A, frac, hold, grid, Tout)
    print(f"\n##### {variant}: outcome = last {frac:.2f} of T={T1} dates, forecasts at T={Tout}, hold={hold}, "
          f"block={block} days, {reps} draws")
    rows = []
    for zt in ZT:
        C, t, y, pred = point(S, zt)
        N = len(C)
        rho = {k: spear(t[k], y) for k in t}
        starts = block_index(len(grid), block, rng, reps)
        tb, yb = bootstrap(C, starts, block, len(grid))
        dd, dw = [], []
        for r in range(reps):
            ok = ~np.isnan(yb[r])
            a = spear(tb["CC"][r, ok], yb[r, ok])
            dd.append(a - spear(tb["DM"][r, ok], yb[r, ok]))
            dw.append(a - spear(tb["CW"][r, ok], yb[r, ok]))
        ia = [rng.integers(0, N, N) for _ in range(reps)]
        ad = [spear(t["CC"][i], y[i]) - spear(t["DM"][i], y[i]) for i in ia]
        q = lambda v: (np.nanpercentile(v, 5), np.nanpercentile(v, 95))
        rec = dict(variant=variant, zt=zt, N=N, rho=rho, pays=float(np.mean(y > 0)),
                   CCmDM_date=q(dd), CCmCW_date=q(dw), CCmDM_asset=q(ad),
                   pred_med=float(np.median(pred)), act_med=float(np.median(y)),
                   pred_pos=float(np.mean(pred > 0)), act_pos=float(np.mean(y > 0)))
        rows.append(rec)
        print(f" ztil={zt:<4} N={N:4d} pays={rec['pays']:.3f} | rho DM {rho['DM']:+.3f} "
              f"CW {rho['CW']:+.3f} CC {rho['CC']:+.3f} | CC-DM date-block 90% "
              f"[{rec['CCmDM_date'][0]:+.3f},{rec['CCmDM_date'][1]:+.3f}] (asset "
              f"[{rec['CCmDM_asset'][0]:+.3f},{rec['CCmDM_asset'][1]:+.3f}]) | CC-CW "
              f"[{rec['CCmCW_date'][0]:+.3f},{rec['CCmCW_date'][1]:+.3f}] | gap med pred "
              f"{rec['pred_med']:+.4f} act {rec['act_med']:+.4f}; >0 pred "
              f"{rec['pred_pos']:.3f} act {rec['act_pos']:.3f}", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=63)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--variants", default="base,third,ztilde")
    a = ap.parse_args()
    rng = np.random.default_rng(20261002)
    A, grid = load()
    print(f"{len(A)} asset files, {len(grid)} calendar dates {grid[0]}-{grid[-1]}")
    spec = dict(base=(0.5, "z", T1), third=(1/3, "z", T1), ztilde=(0.5, "ztilde", T1),
                same=(0.5, "z", T0), same_third=(1/3, "z", T0))
    for v in a.variants.split(","):
        fr, ho, To = spec[v]
        rows = run(v, fr, ho, a.block, a.reps, A, grid, rng, To)
        json.dump(rows, open(os.path.join(HERE, "output", f"rv_window_bootstrap_{v}_b{a.block}.json"), "w"),
                  indent=1, default=float)
