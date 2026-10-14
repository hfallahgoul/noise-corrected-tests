"""Clark-West size under a rolling HAR benchmark on a persistent target, sharp null
(tab:cw-har, single-seed cells).

A rolling-window HAR benchmark carries the Kendall bias of order (1 + 3 rho)/T, so the
benchmark error has a predictable component and E[d^CW] need not vanish when the enlarged
block is exogenous.

    python ldv_scope/cw_under_har.py
Output: ldv_scope/output/cw_under_har.json and .log.
"""
import json, math, os
import numpy as np

OUT = os.environ.get("LDV_OUT",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"))


def one(scale, T, P, zt, n, rng, bench="har"):
    z = zt * P / T
    m = n + T + 140
    sig = 0.42
    eps = sig * rng.standard_normal(m)
    Xe = np.empty((m, P)); Xe[0] = rng.standard_normal(P)
    for t in range(1, m):
        Xe[t] = 0.95 * Xe[t - 1] + math.sqrt(1 - 0.95 ** 2) * rng.standard_normal(P)
    b = tuple(scale * v for v in (0.35, 0.35, 0.25))
    y = np.zeros(m)
    for t in range(30, m):
        y[t] = (b[0] * y[t - 1] + b[1] * y[t - 5:t].mean()
                + b[2] * y[t - 22:t].mean() + eps[t])
    L = P + 30
    M_ = len(y) - L
    wk = np.array([y[L + i - 5:L + i].mean() for i in range(M_)])
    mo = np.array([y[L + i - 22:L + i].mean() for i in range(M_)])
    H = np.column_stack([np.ones(M_), y[L - 1:L - 1 + M_], wk, mo])
    XE = Xe[L:L + M_]
    Y = y[L:L + M_]
    N = M_ - T - 1
    yb = np.empty(N); yh = np.empty(N)
    for i in range(N):
        Hw, Yw = H[i:i + T], Y[i:i + T]
        Qh, Rh = np.linalg.qr(Hw)
        beta = np.linalg.solve(Rh, Qh.T @ Yw)
        yb[i] = H[i + T] @ beta
        Xw = XE[i:i + T]; s_ = XE[i + T]
        QtX = Qh.T @ Xw
        G = (Xw.T @ Xw - QtX.T @ QtX) / T
        gg, V = np.linalg.eigh(G); gg = np.clip(gg, 0.0, None)
        w_ = V @ ((V.T @ s_) / (gg + z))
        a = ((Xw - Qh @ QtX) @ w_) / T
        yh[i] = yb[i] + a @ Yw
    yt = Y[T:T + N]
    e1 = yt - yb
    D = yh - yb
    dCW = 2 * e1 * D
    tCW = math.sqrt(N) * dCW.mean() / math.sqrt(dCW.var())
    # direct measurement of the centring failure, in units of its own standard error
    return tCW, float(dCW.mean()), float(np.mean(e1)), float(dCW.std())


def srad(scale):
    phi = np.zeros(22)
    phi[0] += 0.35 * scale; phi[:5] += 0.35 * scale / 5; phi[:22] += 0.25 * scale / 22
    C = np.zeros((22, 22)); C[0] = phi; C[1:, :-1] = np.eye(21)
    return float(max(abs(np.linalg.eigvals(C))))


if __name__ == "__main__":
    P, zt, n, reps = 25, 1.0, 700, 300
    res = {}
    print("=" * 88)
    print("Clark-West size under the SHARP NULL with a rolling-window HAR benchmark.")
    print("Extra features exogenous and signal-free, so the correct size is 0.05.")
    print("=" * 88)
    print(f"{'HAR persistence':>17}{'spec.radius':>13}{'T':>7}{'CW size':>11}"
          f"{'mean t_CW':>12}")
    print("-" * 88)
    for scale in (0.4, 0.7, 0.9, 1.0):
        for T in (250, 1000):
            rng = np.random.default_rng(4711)
            ts = [one(scale, T, P, zt, n, rng)[0] for _ in range(reps)]
            ts = np.array(ts)
            sz = float(np.mean(ts > 1.645))
            print(f"{scale:>17.2f}{srad(scale):>13.4f}{T:>7}{sz:>11.3f}{ts.mean():>12.2f}")
            res[f"scale{scale}_T{T}"] = dict(srad=srad(scale), T=T, cw_size=sz,
                                            mean_t=float(ts.mean()))
    with open(os.path.join(OUT, "cw_under_har.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"\n[saved] {OUT}/cw_under_har.json")
