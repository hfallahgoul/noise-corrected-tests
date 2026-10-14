"""The design-conditional covariance kernel and the remainder Delta of the limiting variance
(tab:kernel-check, tab:delta; eq:kernel-decay, prop:omega).

Conditionally on the design the innovations are Gaussian, so Isserlis' theorem gives both
quantities in closed form in the weights w_{t-1} of eq:w.  With
rho_{t,s} = sigma^2 sum_overlap w_{t-1}[.] w_{s-1}[.]:

    V_off      = lim (1/n) sum_t sum_{0<|h|<T} 2 rho_{t,t+h}^2
    Cov(I,II)  = (4 sigma^2 / n) sum_t sum_{h=1}^{T-1} w_{t+h-1}[t] rho_{t,t+h}
    gamma_l    = 2 E[rho_l^2] - 4 sigma^2 E[w_{t-1}[t-l] rho_{t-l,t}]        (l >= 1)
    gamma_0    = 4 sigma^4 Psihat_t + 2 sigma^4 Psihat_t^2
    Delta      = V_off - 2 Cov(I,II)

The kernel table reports the root-mean-square autocovariance rms(rho_h)/rho_0 and
sum_l |gamma_l|; the script also checks gamma_0 + 2 sum_l gamma_l against omega^2 at every
design.

    python mc_kernel_delta.py kernel | delta | all     [--reps 24]

Output: output/mc_kernel/kernel_delta.json.
"""
import argparse, datetime, json, os, sys, zlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "output", "mc_kernel")
os.makedirs(OUT, exist_ok=True)


def seed_of(*parts):
    """crc32 of the repr -- stable across processes and machines (cf. mc_study.py)."""
    return zlib.crc32(repr(parts).encode()) % (2 ** 32)


def Psi(c, z):
    u = 1.0 + c + z
    return 0.5 * (u / np.sqrt(u * u - 4.0 * c) - 1.0)


def weights(c, z, T, n, rng, sig2=1.0):
    """W[i] = w_{t-1} for the (i+1)-th evaluation point, as a length-T vector.

    Index convention of eq:indexsets: to forecast y_t the design rows are
    Shat_{t-T-1}, ..., Shat_{t-2} and the fresh feature is Shat_{t-1}.  W[i][j] is the
    weight on the innovation at calendar time (t - T + j).
    """
    P = int(round(c * T))
    S = rng.standard_normal((n + T + 3, P))
    W = np.empty((n, T))
    zI = z * np.eye(P)
    for i in range(n):
        t = T + 2 + i
        X = S[t - T - 1:t - 1]
        A = np.linalg.inv(zI + X.T @ X / T)
        W[i] = (X @ (A @ S[t - 1])) / T
    return W, P


def pieces(W, T, sig2=1.0):
    """One vectorised pass: rho_0, rho_h, the cross term, V_off, Cov(I,II), gamma_l.

    For windows t and t+h the shared calendar times are [t+h-T, t-1], which is
    positions h..T-1 of W[t] against positions 0..T-1-h of W[t+h].  The weight the
    later forecast places on calendar time t is W[t+h][T-h].
    """
    rho0 = sig2 * np.sum(W * W, axis=1)
    Voff = 0.0
    covI2 = 0.0
    gam = np.zeros(T)                      # gam[l], l = 1..T-1
    rms = {}
    for h in range(1, T):
        rh = sig2 * np.sum(W[:-h, h:] * W[h:, :T - h], axis=1)     # rho_{t,t+h} over t
        m2 = float(np.mean(rh ** 2))
        cross = 4.0 * sig2 * float(np.mean(W[h:, T - h] * rh))     # 4 s^2 E[w_{t+h-1}[t] rho]
        Voff += 4.0 * m2                   # both signs of h
        covI2 += cross
        gam[h] = 2.0 * m2 - cross
        rms[h] = np.sqrt(m2)
    return dict(rho0=float(np.mean(rho0)), rho0_sq=float(np.mean(rho0 ** 2)),
                Voff=Voff, covI2=covI2, gamma=gam, rms=rms)


def run_design(c, z, T, n, reps, tag):
    acc = []
    for r in range(reps):
        rng = np.random.default_rng(seed_of(tag, c, z, T, n, r))
        W, P = weights(c, z, T, n, rng)
        acc.append(pieces(W, T))
    def col(k):
        return np.array([a[k] for a in acc])
    gam = np.stack([a["gamma"] for a in acc])                      # reps x T
    sumabs = np.sum(np.abs(gam[:, 1:]), axis=1)
    sumabs_rho = np.sum(2.0 * np.array([[2.0 * a["rms"][h] ** 2 / 2.0 for h in range(1, T)]
                                        for a in acc]), axis=1)     # sum_l 2 E[rho_l^2]
    lead = 4.0 * Psi(c, z)
    delta = col("Voff") - 2.0 * col("covI2")
    # consistency: Var((I)-(II)) two ways
    lhs = (4.0 * col("rho0") + 2.0 * col("rho0_sq")) + 2.0 * np.sum(gam[:, 1:], axis=1)
    rhs = 4.0 * col("rho0") + 2.0 * col("rho0_sq") + col("Voff") - 2.0 * col("covI2")
    def se(x):
        return float(np.std(x, ddof=1) / np.sqrt(len(x)))
    return dict(
        c=c, z=z, T=T, n=n, reps=reps, P=int(round(c * T)), Psi=float(Psi(c, z)),
        lead=float(lead),
        rho0=float(col("rho0").mean()), rho0_se=se(col("rho0")),
        Voff=float(col("Voff").mean()), Voff_se=se(col("Voff")),
        covI2=float(col("covI2").mean()), covI2_se=se(col("covI2")),
        delta=float(delta.mean()), delta_se=se(delta),
        delta_rel=float(delta.mean() / lead),
        sd_cw=float(np.sqrt((lead + delta.mean()) / lead)),
        sumabs_gamma=float(sumabs.mean()), sumabs_gamma_se=se(sumabs),
        sumabs_gamma_rho_only=float(sumabs_rho.mean()),
        rms_rel={str(h): float(np.mean([a["rms"][h] for a in acc]) / col("rho0").mean())
                 for h in (1, 2, max(1, T // 4), max(1, T // 2), T - 1)},
        identity_max_abs_diff=float(np.max(np.abs(lhs - rhs))),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["kernel", "delta", "all"], nargs="?", default="all")
    ap.add_argument("--reps", type=int, default=24)
    a = ap.parse_args()
    res = {"generated": datetime.datetime.now().isoformat(timespec="seconds"),
           "numpy": np.__version__, "script": os.path.basename(__file__)}

    if a.what in ("kernel", "all"):
        rows = []
        for T in (60, 120, 240):
            r = run_design(0.5, 0.1, T, 400, a.reps, "kernel")
            rows.append(r)
            print(f"[kernel] T={T:4d}  rho_0={r['rho0']:.4f}  "
                  f"rms(h=1)/rho_0={r['rms_rel']['1']:.4f}  "
                  f"sqrtT*rms={np.sqrt(T) * r['rms_rel']['1']:.3f}  "
                  f"sum|gamma_l|={r['sumabs_gamma']:.4f}  (identity {r['identity_max_abs_diff']:.1e})")
        res["kernel"] = rows

    if a.what in ("delta", "all"):
        rows = []
        for (c, z, T) in [(0.25, 0.1, 80), (0.5, 0.1, 60), (0.5, 0.1, 120),
                          (0.5, 1.0, 60), (0.9, 0.2, 80), (2.0, 0.5, 60)]:
            r = run_design(c, z, T, 400, a.reps, "delta")
            rows.append(r)
            print(f"[delta ] c={c:4} z={z:4} T={T:4d}  4s4Psi={r['lead']:.4f}  "
                  f"V_off={r['Voff']:.4f}({r['Voff_se']:.4f})  "
                  f"Cov(I,II)={r['covI2']: .4f}({r['covI2_se']:.4f})  "
                  f"Delta={r['delta']: .4f}({r['delta_se']:.4f})  "
                  f"rel={r['delta_rel']: .3f}  sd={r['sd_cw']:.3f}")
        res["delta"] = rows

    path = os.path.join(OUT, "kernel_delta.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
