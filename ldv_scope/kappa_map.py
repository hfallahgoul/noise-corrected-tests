"""kappa_a = E[(a' eps)^2] / (sigma^2 E||a||^2) across benchmark/feature pairings
(tab:ldv-scope panel (a), rem:no-ldv).

Measured as the sharp-null sanity ratio R = mean(Delta^2) / mean(sigma^2 ||a||^2), which
lem:psi-t forces to one.  The benchmark is partialled out by QR and the ridge is fitted to
the residualised features, as in mc_har_diagnosis.py.

    python ldv_scope/kappa_map.py
Output: ldv_scope/output/kappa_map.json.
"""
import json, math, os, sys
import numpy as np

OUT = os.environ.get("LDV_OUT",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"))
os.makedirs(OUT, exist_ok=True)


def simulate(bench, target, feat, T=250, rho=0.85, P=25, zt=1.0, reps=40, n=1500,
             seed=1, split=0, har_lags=(1, 5, 22)):
    """R = mean(D^2)/mean(sigma^2 ||a||^2) under the sharp null.

    bench  : 'const' | 'exog' | 'ar' | 'har'   what the benchmark regresses on
    target : 'exog' | 'ar' | 'har'             how y is generated
    feat   : 'exog' | 'lags'                   what the alternative adds
    split  : if >0, estimate the benchmark projection on a window ending `split`
             observations BEFORE the ridge window, so M_t is not a function of the
             ridge window's own innovations.
    """
    g = np.random.default_rng(seed)
    z = zt * P / T
    per = []
    maxlag = max(har_lags) if (bench == "har" or target == "har") else 3
    for _ in range(reps):
        m = n + T + 80 + maxlag
        sig = 0.42
        eps = sig * g.standard_normal(m)
        # exogenous predictors used by the 'exog' benchmark
        Z = np.empty((m, 3)); Z[0] = g.standard_normal(3)
        for t in range(1, m):
            Z[t] = 0.9 * Z[t - 1] + math.sqrt(1 - 0.81) * g.standard_normal(3)
        # exogenous features for the alternative
        Xex = np.empty((m, P)); Xex[0] = g.standard_normal(P)
        for t in range(1, m):
            Xex[t] = 0.95 * Xex[t - 1] + math.sqrt(1 - 0.95 ** 2) * g.standard_normal(P)

        if target == "ar":
            y = np.zeros(m)
            for t in range(1, m):
                y[t] = rho * y[t - 1] + eps[t]
        elif target == "har":
            # HAR-style: y_t = b1*y_{t-1} + b5*mean(y_{t-5..t-1}) + b22*mean(y_{t-22..t-1}) + eps
            y = np.zeros(m)
            b = (0.35, 0.35, 0.25)
            for t in range(maxlag, m):
                d = y[t - 1]
                w = y[t - 5:t].mean()
                mo = y[t - 22:t].mean()
                y[t] = b[0] * d + b[1] * w + b[2] * mo + eps[t]
        else:
            y = Z @ np.array([0.5, 0.3, 0.2]) + eps

        L = P + maxlag + 4
        Y = y[L:]
        M_ = len(Y)

        def bench_matrix(offset):
            """benchmark regressor block aligned to rows L+offset ... L+offset+M_"""
            base = L + offset
            if bench == "exog":
                return np.column_stack([np.ones(M_), Z[base:base + M_]])
            if bench == "ar":
                return np.column_stack([np.ones(M_)] +
                                       [y[base - k:base - k + M_] for k in (1, 2, 3)])
            if bench == "har":
                cols = [np.ones(M_)]
                cols.append(y[base - 1:base - 1 + M_])
                wk = np.array([y[base + i - 5:base + i].mean() for i in range(M_)])
                mo = np.array([y[base + i - 22:base + i].mean() for i in range(M_)])
                cols += [wk, mo]
                return np.column_stack(cols)
            return np.ones((M_, 1))

        H = bench_matrix(0)
        XE = (np.column_stack([y[L - 1 - k:L - 1 - k + M_] for k in range(P)])
              if feat == "lags" else Xex[L:L + M_])

        D2, SP = [], []
        for i in range(0, M_ - T - 1 - split, 5):
            Xw, Yw = XE[i:i + T], Y[i:i + T]
            s = XE[i + T]
            if split == 0:
                Hw = H[i:i + T]
            else:
                # benchmark projection built from a DISJOINT earlier window
                j = i - split
                if j < 0:
                    continue
                Hw_far = H[j:j + T]
                # project the ridge window's rows onto the coefficient space fitted far away:
                # here we simply use the benchmark block of the far window to define the
                # residual-maker's column space evaluated at the current window's rows.
                Hw = H[i:i + T]
                Qf, _ = np.linalg.qr(Hw_far)
                # residualise using coefficients estimated on the far window
                bfar = np.linalg.lstsq(Hw_far, Y[j:j + T], rcond=None)[0]
                Yw = Yw - Hw @ bfar
                bx = np.linalg.lstsq(Hw_far, XE[j:j + T], rcond=None)[0]
                Xw = Xw - Hw @ bx
                G = (Xw.T @ Xw) / T
                gg, V = np.linalg.eigh(G); gg = np.clip(gg, 0, None)
                w_ = V @ ((V.T @ s) / (gg + z))
                a = (Xw @ w_) / T
                D2.append((a @ Yw) ** 2); SP.append(sig ** 2 * (a @ a))
                continue
            Qh, _ = np.linalg.qr(Hw)
            QtX = Qh.T @ Xw
            G = (Xw.T @ Xw - QtX.T @ QtX) / T
            gg, V = np.linalg.eigh(G); gg = np.clip(gg, 0, None)
            w_ = V @ ((V.T @ s) / (gg + z))
            a = ((Xw - Qh @ QtX) @ w_) / T
            D2.append((a @ Yw) ** 2); SP.append(sig ** 2 * (a @ a))
        if len(D2) == 0:
            continue
        per.append(np.mean(D2) / np.mean(SP))
    per = np.array(per)
    return float(per.mean()), float(per.std(ddof=1) / math.sqrt(len(per)))


if __name__ == "__main__":
    res = {}
    print("=" * 96)
    print("PANEL A. Separating the two channels.  kappa = 1 is what the lemma requires.")
    print("=" * 96)
    print(f"{'target':>8}{'benchmark':>12}{'alt features':>14}{'kappa':>18}   verdict")
    print("-" * 96)
    cases = [
        ("exog", "exog", "exog", "baseline: no lags anywhere"),
        ("exog", "const", "lags", "CAUSAL lag features, NO lagged-y benchmark"),
        ("ar",   "const", "lags", "AR target, mean benchmark, lag features"),
        ("ar",   "ar",    "exog", "lagged-y BENCHMARK, exogenous features"),
        ("ar",   "ar",    "lags", "lagged-y benchmark AND lag features"),
        ("har",  "har",   "lags", "HAR benchmark, lag-ridge alternative"),
        ("har",  "har",   "exog", "HAR benchmark, exogenous features"),
    ]
    for tgt, bch, ft, lab in cases:
        m_, s_ = simulate(bch, tgt, ft, seed=20260818)
        ok = "OK" if abs(m_ - 1) < 4 * s_ + 0.05 else "LEMMA FAILS"
        print(f"{tgt:>8}{bch:>12}{ft:>14}{m_:>11.3f} +/-{s_:<5.3f}   {ok:<12} ({lab})")
        res[f"A|{tgt}|{bch}|{ft}"] = dict(kappa=m_, se=s_, label=lab)

    with open(os.path.join(OUT, "kappa_map.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"\n[saved] {OUT}/kappa_map.json")
