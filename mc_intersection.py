"""Size and power of the three feasible corrected tests and the intersection rule
(tab:intersection, Section 5.8, sec:mc-intersection; cor:intersection).

Rolling-mean benchmark, AR(1) features with persistence RHO, dense signal, n = 4000, and the
window-sweep design of mc_window.py otherwise.  Four statistics are computed on the same
paths:

    oracle   true sigma^2 = 1, Bartlett HAC of the corrected differential
    diff     global difference estimator sigma-hat^2_Delta, HAC of the score with the
             influence-function term, as in mc_study.run_path (thm:feasible)
    window   window-residual estimator sigma-hat^2_{w,t}, HAC of the corrected differential
             itself (thm:feasible-w)
    both     reject only if diff and window both reject (cor:intersection)

Each design is run twice: on the boundary of H0-CC, with the candidate rescaled by
a = 2E[fg]/E[g^2] (estimated in a 20-path pilot) so that mu_CC = 0, and at the unscaled
alternative a = 1.

    python mc_intersection.py RHO R2 T REPS [SEED]
    python mc_intersection.py 0.95 0.10 12 200 1

The paper's table pools SEED = 1, ..., 5 at 200 replications each, for the six designs
(RHO, R2) in {0.3, 0.95} x {0.02, 0.10} at T = 12.
Output: output/mc_intersection/inter_rho<RHO>_R2<R2>_T<T>_s<SEED>.json.
"""
import json
import math
import os
import sys

import numpy as np

import mc_window as W

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output", "mc_intersection")
CRIT = 1.645


def stats(y, S, f, T, z, b, a):
    """Test statistics on one path, with the candidate's increment scaled by a."""
    N = len(y) - T - 1
    X, Yn, Fn = S[:N + T], y[1:N + T + 1], f[:N + T]
    sX, sF = W.roll(X, T)[:N], W.roll(Fn, T)[:N]
    sY, sYY = W.roll(Yn, T)[:N], W.roll(Yn ** 2, T)[:N]
    XX = W.roll(np.einsum("ti,tj->tij", X, X), T)[:N]
    XF = W.roll(X * Fn[:, None], T)[:N]
    XY = W.roll(X * Yn[:, None], T)[:N]
    s = S[T:T + N]
    w = np.linalg.solve(z * np.eye(W.P)[None] + XX / T, s[..., None])[..., 0]
    xb = sX / T
    # systematic part g, increment a'Y and Psihat_t = ||a_t||^2 of the demeaned ridge
    g = (np.einsum("ij,ij->i", w, XF) - np.einsum("ij,ij->i", xb, w) * sF) / T
    aY = (np.einsum("ij,ij->i", w, XY) - np.einsum("ij,ij->i", xb, w) * sY) / T
    psi = np.einsum("ti,tij,tj->t", w, XX - T * np.einsum("ti,tj->tij", xb, xb), w) / T ** 2
    y1, yt, ft = sY / T, y[T + 1:T + N + 1], f[T:T + N]
    D, ps, gs = a * aY, a * a * psi, a * g
    dDM = (yt - y1) ** 2 - (yt - y1 - D) ** 2
    rn = math.sqrt(N)

    def tstat(num, score):
        return rn * num.mean() / math.sqrt(max(W.bartlett(score, b), 1e-14))

    dO = dDM + ps                                             # oracle sigma^2 = 1
    tO = tstat(dO, dO)
    h = 0.5 * np.r_[0.0, np.diff(yt) ** 2]                    # difference estimator
    h[0] = h[1:].mean()
    s2d = 0.5 * np.mean(np.diff(yt) ** 2)
    dD = dDM + s2d * ps
    tD = tstat(dD, dD - dD.mean() + ps.mean() * (h - h.mean()))
    s2w = (sYY - sY ** 2 / T) / (T - 1)                       # window residual, q = 1
    dW = dDM + s2w * ps
    tW = tstat(dW, dW)
    b_t = (W.roll(Fn ** 2, T)[:N] - sF ** 2 / T) / (T - 1)    # lem:sigma-w bias
    return dict(tO=tO, tD=tD, tW=tW,
                Efg=float(np.mean(ft * g)), Eg2=float(np.mean(g * g)),
                mu=float(np.mean(2 * ft * gs - gs ** 2)),
                b_bias=float(np.mean(b_t * ps)), d_bias=float((s2d - 1) * ps.mean()))


def main(rho, R2, T, R, SEED=0):
    W.RHO = rho
    n, z, b = 4000, W.Z, W.bw(T)
    th = W.theta_for("dense", R2, np.random.default_rng(7))
    rng = np.random.default_rng(int(1e6 * rho + 1e3 * R2 + T) + 7919 * SEED)
    prng = np.random.default_rng(int(1e6 * rho + 1e3 * R2 + T))   # same pilot in every chunk
    pil = [stats(*W.gen(T, n, R2, th, prng), T, z, b, 1.0) for _ in range(20)]
    aCC = 2 * np.mean([p["Efg"] for p in pil]) / np.mean([p["Eg2"] for p in pil])
    out = {}
    for lab, a in (("boundary", aCC), ("unscaled", 1.0)):
        A = [stats(*W.gen(T, n, R2, th, rng), T, z, b, a) for _ in range(R)]

        def rej(k):
            return float(np.mean([x[k] > CRIT for x in A]))
        out.setdefault("raw", {})[lab] = [[x["tO"], x["tD"], x["tW"]] for x in A]
        both = float(np.mean([(x["tD"] > CRIT) and (x["tW"] > CRIT) for x in A]))
        out[lab] = dict(a=a, mu=float(np.mean([x["mu"] for x in A])), oracle=rej("tO"),
                        diff=rej("tD"), window=rej("tW"), both=both,
                        bias_w=float(np.mean([x["b_bias"] for x in A])),
                        bias_d=float(np.mean([x["d_bias"] for x in A])))
        o = out[lab]
        print(f"rho={rho} R2={R2} T={T} {lab:8s} a={a:.3f} mu_CC={o['mu']:+.4f} | "
              f"oracle {o['oracle']:.3f} diff {o['diff']:.3f} window {o['window']:.3f} "
              f"both {o['both']:.3f}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    fn = os.path.join(OUT, f"inter_rho{rho}_R2{R2}_T{T}_s{SEED}.json")
    json.dump(dict(rho=rho, R2=R2, T=T, reps=R, **out), open(fn, "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(float(a[0]), float(a[1]), int(a[2]), int(a[3]), int(a[4]) if len(a) > 4 else 0)
