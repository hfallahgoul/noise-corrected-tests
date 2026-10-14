"""tab:ldv-scope panels (d)-(e): kappa_a as the exogenous share of the features grows, and
residualisation on benchmark coefficients estimated on a disjoint earlier window.

    python ldv_scope/kappa_repair.py
Output: ldv_scope/output/kappa_repair.json.
"""
import json, math, os
import numpy as np

OUT = os.environ.get("LDV_OUT",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"))
os.makedirs(OUT, exist_ok=True)


def draw(g, m, rho, sig, P, target):
    eps = sig * g.standard_normal(m)
    Z = np.empty((m, 3)); Z[0] = g.standard_normal(3)
    for t in range(1, m):
        Z[t] = 0.9 * Z[t - 1] + math.sqrt(0.19) * g.standard_normal(3)
    Xex = np.empty((m, P)); Xex[0] = g.standard_normal(P)
    for t in range(1, m):
        Xex[t] = 0.95 * Xex[t - 1] + math.sqrt(1 - 0.95 ** 2) * g.standard_normal(P)
    y = np.zeros(m)
    if target == "ar":
        for t in range(1, m):
            y[t] = rho * y[t - 1] + eps[t]
    else:  # har
        for t in range(22, m):
            y[t] = 0.35 * y[t - 1] + 0.35 * y[t - 5:t].mean() + 0.25 * y[t - 22:t].mean() + eps[t]
    return y, Z, Xex, eps


def build(y, Z, Xex, L, M_, bench, feat, P, n_ex):
    if bench == "ar":
        H = np.column_stack([np.ones(M_)] + [y[L - k:L - k + M_] for k in (1, 2, 3)])
    else:  # har
        wk = np.array([y[L + i - 5:L + i].mean() for i in range(M_)])
        mo = np.array([y[L + i - 22:L + i].mean() for i in range(M_)])
        H = np.column_stack([np.ones(M_), y[L - 1:L - 1 + M_], wk, mo])
    n_lag = P - n_ex
    cols = []
    if n_lag > 0:
        cols.append(np.column_stack([y[L - 1 - k:L - 1 - k + M_] for k in range(n_lag)]))
    if n_ex > 0:
        cols.append(Xex[L:L + M_, :n_ex])
    return H, np.column_stack(cols)


def run(bench="ar", target="ar", n_ex=0, T=250, rho=0.85, P=25, zt=1.0,
        reps=40, n=1500, seed=1, gap=None):
    """gap=None -> within-window projection (status quo).
       gap=int  -> benchmark coefficients from a window ending `gap` obs earlier."""
    g = np.random.default_rng(seed); z = zt * P / T; per = []
    for _ in range(reps):
        m = n + T + 120; sig = 0.42
        y, Z, Xex, eps = draw(g, m, rho, sig, P, target)
        L = P + 30
        M_ = len(y) - L - 1
        H, XE = build(y, Z, Xex, L, M_, bench, "mix", P, n_ex)
        Y = y[L:L + M_]
        lo = T + (gap or 0)
        D2, SP = [], []
        for i in range(lo, M_ - T - 1, 5):
            Hw, Xw, Yw = H[i:i + T], XE[i:i + T], Y[i:i + T]
            s = XE[i + T]
            if gap is None:
                Qh, _ = np.linalg.qr(Hw)
                Xr = Xw - Qh @ (Qh.T @ Xw)
                Yr = Yw - Qh @ (Qh.T @ Yw)
            else:
                j = i - T - gap
                Hf, Xf, Yf = H[j:j + T], XE[j:j + T], Y[j:j + T]
                bY = np.linalg.lstsq(Hf, Yf, rcond=None)[0]
                bX = np.linalg.lstsq(Hf, Xf, rcond=None)[0]
                Xr = Xw - Hw @ bX
                Yr = Yw - Hw @ bY
            G = (Xr.T @ Xr) / T
            gg, V = np.linalg.eigh(G); gg = np.clip(gg, 0, None)
            w_ = V @ ((V.T @ s) / (gg + z))
            a = (Xr @ w_) / T
            D2.append((a @ Yr) ** 2); SP.append(sig ** 2 * (a @ a))
        per.append(np.mean(D2) / np.mean(SP))
    per = np.array(per)
    return float(per.mean()), float(per.std(ddof=1) / math.sqrt(len(per)))


if __name__ == "__main__":
    res = {}
    print("=" * 92)
    print("PANEL D. kappa as the extra features move from all-lags to all-exogenous")
    print("         (P = 25 throughout; benchmark correctly specified; sharp null)")
    print("=" * 92)
    print(f"{'exog features':>15}{'AR(3) bench, rho=.85':>28}{'HAR bench':>26}")
    print("-" * 92)
    for n_ex in (0, 5, 10, 15, 20, 25):
        a_, sa = run(bench="ar", target="ar", n_ex=n_ex, seed=4242)
        h_, sh = run(bench="har", target="har", n_ex=n_ex, seed=4242)
        print(f"{n_ex:>10} / 25{a_:>20.3f} +/-{sa:<5.3f}{h_:>18.3f} +/-{sh:<5.3f}")
        res[f"D|nex{n_ex}"] = dict(ar=a_, ar_se=sa, har=h_, har_se=sh)

    print()
    print("=" * 92)
    print("PANEL E. Out-of-window benchmark repair: residualise with coefficients fitted")
    print("         on a disjoint earlier window instead of projecting within-window.")
    print("=" * 92)
    print(f"{'design':>34}{'within-window':>20}{'out-of-window':>20}")
    print("-" * 92)
    for lab, kw in [
        ("AR(3) bench, exog feats, rho=.85", dict(bench="ar", target="ar", n_ex=25)),
        ("AR(3) bench, exog feats, rho=.95", dict(bench="ar", target="ar", n_ex=25, rho=0.95)),
        ("HAR bench, exog feats", dict(bench="har", target="har", n_ex=25)),
        ("HAR bench, exog feats, T=125 P=60", dict(bench="har", target="har", n_ex=60,
                                                   T=125, P=60, zt=0.1)),
    ]:
        w_, sw = run(seed=515, **kw)
        o_, so = run(seed=515, gap=5, **kw)
        print(f"{lab:>34}{w_:>13.3f} +/-{sw:<5.3f}{o_:>13.3f} +/-{so:<5.3f}")
        res[f"E|{lab}"] = dict(within=w_, within_se=sw, out=o_, out_se=so)

    with open(os.path.join(OUT, "kappa_repair.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"\n[saved] {OUT}/kappa_repair.json")
