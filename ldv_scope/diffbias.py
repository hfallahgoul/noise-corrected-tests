"""Bias of the difference estimator under a lagged-dependent-variable benchmark
(rem:three-sigmas, rem:no-ldv).

When the conditional mean loads on the target's own past, the signal is correlated with the
lagged innovation and the difference estimator is biased downward, unlike the upward bias
Var(f)(1 - rho_f) of prop:sigma-diff.

    python ldv_scope/diffbias.py
"""
import math
import numpy as np

B = (0.35, 0.35, 0.25)
SIG = 0.42
T, REPS, M = 250, 40, 6000


def one(scale, rng):
    b = tuple(scale * v for v in B)
    feedback = b[0] + b[1] / 5 + b[2] / 22          # d f_t / d y_{t-1}
    eps = SIG * rng.standard_normal(M)
    y = np.zeros(M)
    for t in range(30, M):
        y[t] = b[0]*y[t-1] + b[1]*y[t-5:t].mean() + b[2]*y[t-22:t].mean() + eps[t]
    y = y[500:]
    # HAR design on the same path
    n = len(y)
    d = y[21:n-1]
    w = np.array([y[t-4:t+1].mean() for t in range(21, n-1)])
    mo = np.array([y[t-21:t+1].mean() for t in range(21, n-1)])
    H = np.column_stack([np.ones(len(d)), d, w, mo])
    Y = y[22:n]
    # rolling-window residuals, matching how the test forms them
    res = []
    for i in range(0, len(Y) - T, 25):
        Hw, Yw = H[i:i+T], Y[i:i+T]
        Q, R = np.linalg.qr(Hw)
        res.append(Yw - Hw @ np.linalg.solve(R, Q.T @ Yw))
    res = np.concatenate(res)
    return (feedback,
            0.5 * np.mean(np.diff(Y) ** 2),          # w1: differences of the target
            0.5 * np.mean(np.diff(res) ** 2),        # w2: differences of the residual
            float(np.var(Y)))


if __name__ == "__main__":
    print("=" * 88)
    print(f"sigma^2 = {SIG**2:.4f}.  Each row averages {REPS} paths.")
    print("=" * 88)
    print(f"{'HAR scale':>10}{'df/dy_{t-1}':>13}{'Var(y)':>9}"
          f"{'.5E(dy)^2':>11}{'ratio':>8}{'.5E(dr)^2':>11}{'ratio':>8}"
          f"{'predicted':>11}")
    print("-" * 88)
    for scale in (0.2, 0.5, 0.8, 1.0):
        rng = np.random.default_rng(20260822)
        out = np.array([one(scale, rng)[1:] for _ in range(REPS)])
        fb = one(scale, np.random.default_rng(1))[0]
        dy, dr, vy = out.mean(0)
        pred = 1 - fb                      # leading term, ignoring the smoothness add-back
        print(f"{scale:>10.2f}{fb:>13.3f}{vy:>9.3f}{dy:>11.4f}{dy/SIG**2:>8.3f}"
              f"{dr:>11.4f}{dr/SIG**2:>8.3f}{pred:>11.3f}")
    print("\nColumn 'ratio' is the estimator divided by the true sigma^2; 1.000 is unbiased.")
    print("'predicted' is 1 - df/dy_{t-1}, the leading term of the downward bias.")
