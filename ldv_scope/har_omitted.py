"""Size of the corrected test under H0-CC with an estimated innovation variance, HAR
benchmark with an omitted regressor (tab:har-omitted).

    python ldv_scope/har_omitted.py --run --T 250 [--seed 11] [--tag T250]
The T = 40 and T = 60 blocks of the table pool --seed 11 and --seed 4711.
Output: ldv_scope/output/har_omitted_T<T>[s<seed>].json.
"""
import argparse, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("LDV_OUT", os.path.join(HERE, "output"))
os.makedirs(OUT, exist_ok=True)
B_HAR = (0.35, 0.35, 0.25)
SIG = 0.42
SIG2 = SIG ** 2          # the true innovation variance, known to the generator


def local_var(r, k, half=10):
    """Smoothed local variance of the window residuals, rescaled to match the window
    estimate.  Copied from har_size_power.py so the two scripts compute the same object."""
    r2 = r ** 2
    T = len(r)
    cs = np.concatenate([[0.0], np.cumsum(r2)])
    s = np.arange(T)
    lo = np.maximum(0, s - half)
    hi = np.minimum(T, s + half + 1)
    out = (cs[hi] - cs[lo]) / (hi - lo)
    return out * (r2.sum() / max(T - k, 1)) / max(out.mean(), 1e-300)


def one_path(lam, T, P, Q, zt, n, rng, band="beyond"):
    """One replication.  band='beyond' puts the extra signal in lags P+1..Q, entirely
    outside the feature set; band='far' spreads it over 6..Q, so the ridge spans part of
    it directly.  'beyond' is the sharper misspecification and the one that should drive
    mu_CC negative first."""
    z = zt * P / T
    m = n + T + 40 + Q + 140

    # The loading is redrawn until the AR polynomial is stationary rather than raising on
    # an unlucky draw: the DGP is conditioned on stationarity, which is what is wanted, and
    # a hard failure would otherwise make the whole sweep depend on where the stream sits.
    lo = (P if band == "beyond" else 6)
    for _attempt in range(200):
        w_ex = np.zeros(Q)
        v = rng.standard_normal(Q - lo)
        v -= v.mean()                   # zero-sum: no extra persistence at frequency zero
        w_ex[lo:] = v / np.linalg.norm(v)
        if lam == 0:
            break
        phi = np.zeros(max(Q, 22))
        phi[0] += B_HAR[0]
        phi[:5] += B_HAR[1] / 5
        phi[:22] += B_HAR[2] / 22
        phi[:Q] += lam * w_ex
        comp = np.zeros((len(phi), len(phi)))
        comp[0] = phi; comp[1:, :-1] = np.eye(len(phi) - 1)
        if float(np.max(np.abs(np.linalg.eigvals(comp)))) <= 0.995:
            break
    else:
        raise RuntimeError(f"no stationary loading found at lam={lam}")

    eps = SIG * rng.standard_normal(m)
    y = np.zeros(m)
    for t in range(Q + 30, m):
        mu = B_HAR[0]*y[t-1] + B_HAR[1]*y[t-5:t].mean() + B_HAR[2]*y[t-22:t].mean()
        if lam > 0:
            mu += lam * float(np.dot(w_ex, y[t-Q:t][::-1]))
        y[t] = mu + eps[t]

    L = Q + 60
    M_ = len(y) - L
    wk = np.array([y[L+i-5:L+i].mean() for i in range(M_)])
    mo = np.array([y[L+i-22:L+i].mean() for i in range(M_)])
    H = np.column_stack([np.ones(M_), y[L-1:L-1+M_], wk, mo])
    XE = np.column_stack([y[L-1-k:L-1-k+M_] for k in range(P)])
    Y = y[L:L+M_]

    hd = np.empty(M_); hd[1:] = 0.5*np.diff(Y)**2; hd[0] = hd[1]

    N = M_ - T - 1
    yb = np.empty(N); yh = np.empty(N); psi = np.empty(N)
    s2w = np.empty(N); psit = np.empty(N); psit_l = np.empty(N)
    k = H.shape[1]
    for i in range(N):
        Hw, Yw = H[i:i+T], Y[i:i+T]
        Qh, Rh = np.linalg.qr(Hw)
        beta = np.linalg.solve(Rh, Qh.T @ Yw)
        yb[i] = H[i+T] @ beta
        r = Yw - Hw @ beta
        s2w[i] = r @ r / max(T - k, 1)
        Xw = XE[i:i+T]; s_ = XE[i+T]
        QtX = Qh.T @ Xw
        G = (Xw.T @ Xw - QtX.T @ QtX) / T
        gg, V = np.linalg.eigh(G); gg = np.clip(gg, 0.0, None)
        w_ = V @ ((V.T @ s_) / (gg + z))
        a = ((Xw - Qh @ QtX) @ w_) / T
        yh[i] = yb[i] + a @ Yw
        psi[i] = a @ a
        a2 = a ** 2
        psit[i] = float(a2 @ hd[i:i+T])
        psit_l[i] = float(a2 @ local_var(r, k))

    yt = Y[T:T+N]
    e1, e2 = yt - yb, yt - yh
    D = yh - yb
    dDM = e1**2 - e2**2
    dCW = dDM + D**2
    b = max(1, int(np.ceil(4*(N/100)**(2/9))))

    def hac(x):
        x = x - x.mean(); g = np.mean(x*x)
        for h in range(1, b+1):
            g += 2*(1 - h/(b+1))*np.mean(x[h:]*x[:-h])
        return max(g, 1e-14)

    t = lambda d_, v: math.sqrt(N)*d_.mean()/math.sqrt(v)
    dCC_w = dDM + s2w*psi          # the application's estimator
    dCC_o = dDM + SIG2*psi         # oracle, for the comparison the paper currently rests on
    dCC_t = dDM + psit             # weighted, prop:sigma-weighted
    dCC_l = dDM + psit_l           # weighted with the smoothed local residual variance
    return dict(
        t_CW=t(dCW, dCW.var()), t_DM=t(dDM, hac(dDM)),
        t_CC_w=t(dCC_w, hac(dCC_w)), t_CC_o=t(dCC_o, hac(dCC_o)),
        t_CC_t=t(dCC_t, hac(dCC_t)), t_CC_l=t(dCC_l, hac(dCC_l)),
        Psihat=float(psi.mean()), dMSPE=float(dDM.mean()),
        s2w=float(s2w.mean()),
        R_w=float(np.mean(D**2)/max(np.mean(s2w*psi), 1e-300)),
        R_o=float(np.mean(D**2)/max(np.mean(SIG2*psi), 1e-300)),
    )


def run(lams, T, P, Q, zt, n, reps, band, seed=11):
    rows = []
    for lam in lams:
        rng = np.random.default_rng(seed)
        acc = []
        t0 = time.time()
        for _ in range(reps):
            acc.append(one_path(lam, T, P, Q, zt, n, rng, band))
        A = {k: np.array([a[k] for a in acc]) for k in acc[0]}
        rj = lambda k: float(np.mean(A[k] > 1.645))
        # dMSPE = mu_CC - sigma^2 Psi is an identity, so the true sigma^2 recovers mu_CC
        # and the truth of H_cc is measured rather than assumed.
        mu_CC = float(A["dMSPE"].mean() + SIG2*A["Psihat"].mean())
        rec = dict(lam=lam, T=T, P=P, Q=Q, zt=zt, n=n, reps=reps, band=band,
                   Psihat=float(A["Psihat"].mean()), dMSPE=float(A["dMSPE"].mean()),
                   mu_CC=mu_CC, s2w=float(A["s2w"].mean()), s2_true=SIG2,
                   R_w=float(A["R_w"].mean()), R_o=float(A["R_o"].mean()),
                   rej_DM=rj("t_DM"), rej_CW=rj("t_CW"),
                   rej_CC_w=rj("t_CC_w"), rej_CC_o=rj("t_CC_o"),
                   rej_CC_t=rj("t_CC_t"), rej_CC_l=rj("t_CC_l"),
                   secs=round(time.time()-t0, 1))
        rec["seed"] = seed
        rows.append(rec)
        print(f"lam={lam:<5} mu_CC={mu_CC:+.5f}  Psi={rec['Psihat']:.4f}  "
              f"s2w/s2={rec['s2w']/SIG2:.3f}  R_w={rec['R_w']:.3f}  | "
              f"DM={rec['rej_DM']:.3f} CW={rec['rej_CW']:.3f}  "
              f"CC_w={rec['rej_CC_w']:.3f} CC_oracle={rec['rej_CC_o']:.3f} "
              f"CC_wtd={rec['rej_CC_t']:.3f}  [{rec['secs']}s]", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--T", type=int, default=250)
    ap.add_argument("--P", type=int, default=25)
    ap.add_argument("--Q", type=int, default=60)
    ap.add_argument("--zt", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=700)
    ap.add_argument("--band", default="beyond", choices=["beyond", "far"])
    ap.add_argument("--lams", default=None)
    ap.add_argument("--seed", type=int, default=11,
                    help="replication seed; the cells carrying a claim are run at more "
                         "than one, as har_size_power.py does")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    lams = ([float(x) for x in a.lams.split(",")] if a.lams
            else ([0.0, 0.05, 0.10, 0.20] if a.sweep
                  else [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]))
    reps = 60 if a.sweep else a.reps
    print(f"T={a.T} P={a.P} Q={a.Q} zt={a.zt} n={a.n} band={a.band} "
          f"reps={reps} seed={a.seed}\n")
    rows = run(lams, a.T, a.P, a.Q, a.zt, a.n, reps, a.band, seed=a.seed)
    if a.run:
        fn = f"har_omitted{'_'+a.tag if a.tag else ''}.json"
        json.dump(rows, open(os.path.join(OUT, fn), "w"), indent=1)
        print(f"\n[saved] {OUT}/{fn}")
