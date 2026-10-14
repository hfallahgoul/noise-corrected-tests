"""kappa_a when the enlarged block holds other functionals of the same intraday path:
jump and semivariance terms, realized quarticity, or a second estimator of the same latent
variance (tab:pathfunc).

    python ldv_scope/kappa_pathfunc.py
Output: ldv_scope/output/kappa_pathfunc.json.  The penalty, window and jump sweeps in
kappa_pathfunc_{zt,robust,jumps}.json were produced with sweep() below; see README.
"""
import json, math, os
import numpy as np

OUT = os.environ.get("LDV_OUT",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"))
os.makedirs(OUT, exist_ok=True)

FUNCS = ("subsamp", "quartic", "bipower", "semivar")


def _path_functionals(g, m, M, sig):
    """Daily innovations and the within-day functionals sharing their path."""
    u = g.standard_normal((m, M))
    eps = sig * u.sum(1) / math.sqrt(M)
    half = M // 2
    f = {
        # level channel: a second estimator of the same latent quantity
        "subsamp": sig * u[:, :half].sum(1) / math.sqrt(half),
        # magnitude channel
        "quartic": (u ** 4).mean(1),
        "bipower": (np.abs(u[:, 1:]) * np.abs(u[:, :-1])).mean(1),
        "semivar": (u ** 2 * (u > 0)).mean(1),
    }
    # standardise so the ridge penalty means the same thing in every arm
    for k in f:
        f[k] = (f[k] - f[k].mean()) / f[k].std()
    return eps, f


def simulate(feat, T=250, P=25, zt=1.0, reps=40, n=1500, seed=1, M=78,
             har_lags=(1, 5, 22)):
    g = np.random.default_rng(seed)
    z = zt * P / T
    per = []
    maxlag = max(har_lags)
    for _ in range(reps):
        m = n + T + 80 + maxlag
        sig = 0.42
        eps, fun = _path_functionals(g, m, M, sig)

        Xex = np.empty((m, P)); Xex[0] = g.standard_normal(P)
        for t in range(1, m):
            Xex[t] = 0.95 * Xex[t - 1] + math.sqrt(1 - 0.95 ** 2) * g.standard_normal(P)

        y = np.zeros(m); b = (0.35, 0.35, 0.25)
        for t in range(maxlag, m):
            y[t] = (b[0] * y[t - 1] + b[1] * y[t - 5:t].mean()
                    + b[2] * y[t - 22:t].mean() + eps[t])

        L = P + maxlag + 4
        Y = y[L:]; M_ = len(Y)

        cols = [np.ones(M_), y[L - 1:L - 1 + M_]]
        cols.append(np.array([y[L + i - 5:L + i].mean() for i in range(M_)]))
        cols.append(np.array([y[L + i - 22:L + i].mean() for i in range(M_)]))
        H = np.column_stack(cols)

        if feat == "lags":
            XE = np.column_stack([y[L - 1 - k:L - 1 - k + M_] for k in range(P)])
        elif feat == "exog":
            XE = Xex[L:L + M_]
        else:
            v = fun[feat]
            XE = np.column_stack([v[L - 1 - k:L - 1 - k + M_] for k in range(P)])

        D2, SP = [], []
        for i in range(0, M_ - T - 1, 5):
            Xw, Yw, s = XE[i:i + T], Y[i:i + T], XE[i + T]
            Hw = H[i:i + T]
            Qh, _ = np.linalg.qr(Hw)
            QtX = Qh.T @ Xw
            G = (Xw.T @ Xw - QtX.T @ QtX) / T
            gg, V = np.linalg.eigh(G); gg = np.clip(gg, 0, None)
            w_ = V @ ((V.T @ s) / (gg + z))
            a = ((Xw - Qh @ QtX) @ w_) / T
            D2.append((a @ Yw) ** 2); SP.append(sig ** 2 * (a @ a))
        if D2:
            per.append(np.mean(D2) / np.mean(SP))
    per = np.array(per)
    return float(per.mean()), float(per.std(ddof=1) / math.sqrt(len(per)))


if __name__ == "__main__":
    res = {}
    print("=" * 94)
    print("kappa_a for features that are finer functionals of the target's own path.")
    print("HAR benchmark and HAR target throughout; sharp null by construction.")
    print("=" * 94)
    print(f"{'alt features':>12}{'channel':>12}{'kappa_a':>18}   verdict")
    print("-" * 94)
    arms = [
        ("lags",    "control",  "CONTROL: must reproduce tab:ldv-scope's admissible row"),
        ("exog",    "control",  "CONTROL: must reproduce the excluded row"),
        ("subsamp", "level",    "second estimator of the same latent quantity"),
        ("quartic", "magnitude","realized quarticity (HARQ input)"),
        ("bipower", "magnitude","bipower variation (Corsi jump input)"),
        ("semivar", "magnitude","upside semivariance"),
    ]
    for feat, chan, lab in arms:
        mu, se = simulate(feat, seed=20260917)
        ok = "OK" if abs(mu - 1) < 4 * se + 0.05 else "SCOPE FAILS"
        print(f"{feat:>12}{chan:>12}{mu:>11.3f} +/-{se:<5.3f}   {ok:<12} ({lab})")
        res[feat] = dict(kappa_a=mu, se=se, channel=chan, label=lab)

    with open(os.path.join(OUT, "kappa_pathfunc.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"\n[saved] {OUT}/kappa_pathfunc.json")


def sweep(arms, grid, key, reps=20, seed=20260917, **base):
    """Vary one design parameter over `grid` for each arm in `arms`."""
    out = {}
    hdr = f"{'feature':>10}" + "".join(f"{key}={v:<9}" for v in grid)
    print(hdr); print("-" * len(hdr))
    for feat in arms:
        row, cells = f"{feat:>10}", {}
        for v in grid:
            kw = dict(base); kw[key] = v
            mu, se = simulate(feat, reps=reps, seed=seed, **kw)
            flag = "" if abs(mu - 1) < 4 * se + 0.05 else "*"
            row += f"{mu:>7.3f}{flag:<1}   "
            cells[str(v)] = dict(kappa_a=mu, se=se)
        print(row)
        out[feat] = cells
    return out
