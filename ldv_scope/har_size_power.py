"""Size and power of the three tests for HAR against a ridge on individual lags of the
target (tab:har-power, app:ldv-mc).

    benchmark    HAR: 1, RV_{t-1}, mean(RV_{t-5..t-1}), mean(RV_{t-22..t-1})
    alternative  ridge on P individual lags RV_{t-1}, ..., RV_{t-P}   (P >= 22, nests HAR)

    python ldv_scope/har_size_power.py [--seed 4711]
Output: ldv_scope/output/har_size_power[_seed4711].json and .log.
"""
import json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("LDV_OUT", os.path.join(HERE, "output"))
os.makedirs(OUT, exist_ok=True)
B_HAR = (0.35, 0.35, 0.25)


def local_var(r, k, half=10):
    """Smoothed local variance of the window residuals, rescaled so its mean matches the
    window estimate.  Same definition as Code/mc_har.py; vectorised, because this sits
    inside the innermost loop and a Python pass over T dominates everything else."""
    r2 = r ** 2
    T = len(r)
    cs = np.concatenate([[0.0], np.cumsum(r2)])
    s = np.arange(T)
    lo = np.maximum(0, s - half)
    hi = np.minimum(T, s + half + 1)
    out = (cs[hi] - cs[lo]) / (hi - lo)
    return out * (r2.sum() / max(T - k, 1)) / max(out.mean(), 1e-300)


def one_path(feat, lam, T, P, zt, n, rng, light=False):
    z = zt * P / T
    m = n + T + 140
    sig = 0.42
    eps = sig * rng.standard_normal(m)
    Xe = np.empty((m, P)); Xe[0] = rng.standard_normal(P)
    for t in range(1, m):
        Xe[t] = 0.95 * Xe[t - 1] + math.sqrt(1 - 0.95 ** 2) * rng.standard_normal(P)

    # extra-signal loading: for lag features, a combination of the FAR lags (6..P),
    # which HAR cannot span; for exogenous features, a random direction in Xe.
    if feat == "lags":
        # far lags only (HAR cannot span them), zero-sum so the extra term adds no
        # persistence at frequency zero and the process stays comfortably stationary
        w_ex = np.zeros(P)
        if P > 6:
            v = rng.standard_normal(P - 6)
            v -= v.mean()
            w_ex[6:] = v / np.linalg.norm(v)
    else:
        w_ex = rng.standard_normal(P); w_ex /= np.linalg.norm(w_ex)

    if feat == "lags" and lam > 0:              # guard: keep the AR polynomial stable
        phi = np.zeros(max(P, 22))
        phi[0] += B_HAR[0]
        phi[:5] += B_HAR[1] / 5
        phi[:22] += B_HAR[2] / 22
        phi[:P] += lam * w_ex
        comp = np.zeros((len(phi), len(phi)))
        comp[0] = phi; comp[1:, :-1] = np.eye(len(phi) - 1)
        srad = float(np.max(np.abs(np.linalg.eigvals(comp))))
        if srad > 0.995:
            raise RuntimeError(f"non-stationary DGP at lam={lam}: spectral radius {srad:.4f}")

    y = np.zeros(m)
    for t in range(30, m):
        mu = B_HAR[0] * y[t - 1] + B_HAR[1] * y[t - 5:t].mean() + B_HAR[2] * y[t - 22:t].mean()
        if lam > 0:
            extra = (np.dot(w_ex, y[t - P:t][::-1]) if feat == "lags"
                     else float(Xe[t - 1] @ w_ex))
            mu += lam * extra
        y[t] = mu + eps[t]

    L = P + 30
    M_ = len(y) - L
    wk = np.array([y[L + i - 5:L + i].mean() for i in range(M_)])
    mo = np.array([y[L + i - 22:L + i].mean() for i in range(M_)])
    H = np.column_stack([np.ones(M_), y[L - 1:L - 1 + M_], wk, mo])
    XE = (np.column_stack([y[L - 1 - k:L - 1 - k + M_] for k in range(P)])
          if feat == "lags" else Xe[L:L + M_])
    Y = y[L:L + M_]

    # half-squared first differences of the target, aligned to Y, for the weighted
    # correction.  hd[s] pairs Y[s] with its predecessor; hd[0] borrows hd[1] because
    # the first window element has no in-sample predecessor.
    hd = np.empty(M_)
    hd[1:] = 0.5 * np.diff(Y) ** 2
    hd[0] = hd[1]

    N = M_ - T - 1
    yb = np.empty(N); yh = np.empty(N); psi = np.empty(N); s2w = np.empty(N)
    psit = np.empty(N); psit_r = np.empty(N); psit_l = np.empty(N)
    k = H.shape[1]
    for i in range(N):
        Hw, Yw = H[i:i + T], Y[i:i + T]
        Qh, Rh = np.linalg.qr(Hw)
        beta = np.linalg.solve(Rh, Qh.T @ Yw)
        yb[i] = H[i + T] @ beta
        r = Yw - Hw @ beta
        s2w[i] = r @ r / max(T - k, 1)
        Xw = XE[i:i + T]; s_ = XE[i + T]
        QtX = Qh.T @ Xw
        G = (Xw.T @ Xw - QtX.T @ QtX) / T
        gg, V = np.linalg.eigh(G); gg = np.clip(gg, 0.0, None)
        w_ = V @ ((V.T @ s_) / (gg + z))
        a = ((Xw - Qh @ QtX) @ w_) / T
        yh[i] = yb[i] + a @ Yw
        psi[i] = a @ a
        if light:
            # the bandwidth sweep needs only DM/CW/CC, and the weighted corrections are
            # the expensive part of this loop
            continue
        a2 = a ** 2
        psit[i] = float(a2 @ hd[i:i + T])                 # (w1) differences of the target
        dr = np.empty(T)                                  # (w2) differences of the residual
        dr[1:] = 0.5 * np.diff(r) ** 2; dr[0] = dr[1]
        psit_r[i] = float(a2 @ dr)
        psit_l[i] = float(a2 @ local_var(r, k))           # (w3) smoothed local r^2

    yt = Y[T:T + N]
    e1, e2 = yt - yb, yt - yh
    D = yh - yb
    dDM = e1 ** 2 - e2 ** 2
    dCW = dDM + D ** 2
    dCCp = dDM + s2w * psi
    b = max(1, int(np.ceil(4 * (N / 100) ** (2 / 9))))

    def hac(x):
        x = x - x.mean(); g = np.mean(x * x)
        for h in range(1, b + 1):
            g += 2 * (1 - h / (b + 1)) * np.mean(x[h:] * x[:-h])
        return max(g, 1e-14)

    t = lambda d_, v: math.sqrt(N) * d_.mean() / math.sqrt(v)
    if light:
        return dict(t_CW=t(dCW, dCW.var()), t_CCp=t(dCCp, hac(dCCp)),
                    t_DM=t(dDM, hac(dDM)),
                    R_plug=float(np.mean(D ** 2) / np.mean(s2w * psi)),
                    Psihat=float(psi.mean()),
                    dMSPE=float(np.mean(e1 ** 2) - np.mean(e2 ** 2)))
    dCCw = dDM + psit; dCCwr = dDM + psit_r; dCCwl = dDM + psit_l
    return dict(t_CW=t(dCW, dCW.var()), t_CCp=t(dCCp, hac(dCCp)),
                t_CCw=t(dCCw, hac(dCCw)), t_CCwr=t(dCCwr, hac(dCCwr)),
                t_CCwl=t(dCCwl, hac(dCCwl)),
                t_DM=t(dDM, hac(dDM)),
                R_plug=float(np.mean(D ** 2) / np.mean(s2w * psi)),
                R_w=float(np.mean(D ** 2) / max(np.mean(psit), 1e-300)),
                Psihat=float(psi.mean()),
                Psitil=float(psit.mean()), Psitil_r=float(psit_r.mean()),
                Psitil_l=float(psit_l.mean()),
                sig2Psi=float(np.mean(s2w * psi)),
                dMSPE=float(np.mean(e1 ** 2) - np.mean(e2 ** 2)))


def run(feat, lams, T, P, zt, n, reps, seed=11):
    rows = []
    for lam in lams:
        rng = np.random.default_rng(seed)
        acc = {k: [] for k in ("t_CW", "t_CCp", "t_CCw", "t_CCwr", "t_CCwl", "t_DM",
                               "R_plug", "R_w", "Psihat", "Psitil", "Psitil_r",
                               "Psitil_l", "sig2Psi", "dMSPE")}
        for _ in range(reps):
            s = one_path(feat, lam, T, P, zt, n, rng)
            for k in acc: acc[k].append(s[k])
        rec = dict(feat=feat, lam=lam, T=T, P=P, zt=zt, n=n, reps=reps,
                   rej_DM=float(np.mean(np.array(acc["t_DM"]) > 1.645)),
                   rej_CW=float(np.mean(np.array(acc["t_CW"]) > 1.645)),
                   rej_CC=float(np.mean(np.array(acc["t_CCp"]) > 1.645)),
                   rej_CCw=float(np.mean(np.array(acc["t_CCw"]) > 1.645)),
                   rej_CCwr=float(np.mean(np.array(acc["t_CCwr"]) > 1.645)),
                   rej_CCwl=float(np.mean(np.array(acc["t_CCwl"]) > 1.645)),
                   R_plug=float(np.mean(acc["R_plug"])),
                   R_w=float(np.mean(acc["R_w"])),
                   Psihat=float(np.mean(acc["Psihat"])),
                   Psitil=float(np.mean(acc["Psitil"])),
                   Psitil_r=float(np.mean(acc["Psitil_r"])),
                   Psitil_l=float(np.mean(acc["Psitil_l"])),
                   sig2Psi=float(np.mean(acc["sig2Psi"])),
                   dMSPE=float(np.mean(acc["dMSPE"])),
                   sd_CC=float(np.std(acc["t_CCp"])),
                   mean_CC=float(np.mean(acc["t_CCp"])),
                   sd_CW=float(np.std(acc["t_CW"])),
                   mean_CW=float(np.mean(acc["t_CW"])),
                   sd_CCw=float(np.std(acc["t_CCw"])),
                   mean_CCw=float(np.mean(acc["t_CCw"])),
                   mean_CCwr=float(np.mean(acc["t_CCwr"])),
                   mean_CCwl=float(np.mean(acc["t_CCwl"])))
        rows.append(rec)
        print(f"  lam={lam:<5} Psihat={rec['Psihat']:.4f} R={rec['R_plug']:.3f} "
              f"dMSPE={rec['dMSPE']:+.5f} | rej  DM={rec['rej_DM']:.3f} "
              f"CW={rec['rej_CW']:.3f} CC={rec['rej_CC']:.3f} | "
              f"t_CC mean/sd = {rec['mean_CC']:+.2f}/{rec['sd_CC']:.2f}")
        print(f"        correction size:  sig2*Psi={rec['sig2Psi']:.5f}  "
              f"w1(dy)={rec['Psitil']:.5f}  w2(dr)={rec['Psitil_r']:.5f}  "
              f"w3(loc)={rec['Psitil_l']:.5f}")
        print(f"        rej:  CC={rec['rej_CC']:.3f}  w1={rec['rej_CCw']:.3f}  "
              f"w2={rec['rej_CCwr']:.3f}  w3={rec['rej_CCwl']:.3f}   "
              f"mean t: {rec['mean_CC']:+.2f} / {rec['mean_CCw']:+.2f} / "
              f"{rec['mean_CCwr']:+.2f} / {rec['mean_CCwl']:+.2f}")
    return rows


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    seed = 11
    for i, a_ in enumerate(sys.argv):
        if a_ == "--seed": seed = int(sys.argv[i + 1])
    tag = "" if seed == 11 else f"_seed{seed}"
    T, P, zt = 250, 25, 1.0
    n = 400 if quick else 700
    reps = 60 if quick else 300
    lams = [0.0, 0.05, 0.20] if quick else [0.0, 0.05, 0.10, 0.20]
    t0 = time.time(); rows = []
    print(f"# har_size_power.py  seed={seed}  reps={reps}  n={n}  T={T} P={P} ztil={zt}")
    print("# CC  = scalar plug-in  sigmahat^2_w * Psihat_t   (paper default)")
    print("# CCw = weighted        Psitilde_t = sum_s a_{t,s}^2 (1/2)(y_s-y_{s-1})^2")

    print("=" * 100)
    print("A. HAR benchmark vs ridge on P=25 LAGS of the target.  Nominal size 0.05 at lam=0.")
    print("   This is the configuration the paper admits (Remark: no-ldv).")
    print("=" * 100)
    rows += run("lags", lams, T, P, zt, n, reps, seed=seed)

    print("\n" + "=" * 100)
    print("B. HAR benchmark vs ridge on P=25 EXOGENOUS features -- the case that really fails.")
    print("=" * 100)
    rows += run("exog", lams, T, P, zt, n, reps, seed=seed)

    with open(os.path.join(OUT, f"har_size_power{tag}.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"\n[saved] {OUT}/har_size_power{tag}.json   ({(time.time()-t0)/60:.1f} min)")
