"""ccpa.py against the vectorised engine of mc_intersection.py, on one simulated path.

    python tests/test_ccpa.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import ccpa                      # noqa: E402
import mc_intersection as MI     # noqa: E402
import mc_window as W            # noqa: E402


def main(T=12, n=400, R2=0.1, rho=0.95, a_scale=0.7, seed=3):
    W.RHO = rho
    z, b = W.Z, W.bw(T)
    th = W.theta_for("dense", R2, np.random.default_rng(7))
    y, S, f = W.gen(T, n, R2, th, np.random.default_rng(seed))
    ref = MI.stats(y, S, f, T, z, b, a_scale)

    N = len(y) - T - 1
    yt, yh1, yh2, psi, Ywin = [], [], [], [], []
    for i in range(N):
        X, Yw, s = S[i:i + T], y[i + 1:i + T + 1], S[T + i]
        a = a_scale * ccpa.ridge_increment_weights(X, s, z, H=np.ones(T))
        yt.append(y[T + 1 + i]); yh1.append(Yw.mean()); yh2.append(Yw.mean() + a @ Yw)
        psi.append(a @ a); Ywin.append(Yw)
    yt, yh1, yh2, psi = map(np.array, (yt, yh1, yh2, psi))

    oracle = ccpa.cc_test(yt, yh1, yh2, psi, 1.0, b)
    s2d, h = ccpa.sigma2_difference(yt)
    diff = ccpa.cc_test(yt, yh1, yh2, psi, s2d, b, influence=h)
    window = ccpa.cc_test(yt, yh1, yh2, psi, ccpa.sigma2_window(np.array(Ywin)), b)

    for name, mine, theirs in (("oracle", oracle["t"], ref["tO"]),
                               ("difference", diff["t"], ref["tD"]),
                               ("window", window["t"], ref["tW"])):
        print(f"{name:10s} ccpa {mine:+.12f}   engine {theirs:+.12f}")
        assert abs(mine - theirs) < 1e-8 * max(1.0, abs(theirs)), name
    print("ccpa agrees with the simulation engine")


if __name__ == "__main__":
    main()
