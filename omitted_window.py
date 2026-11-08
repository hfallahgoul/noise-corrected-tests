"""mu_CC against the window in the omitted-predictor design (numbers quoted in sec:mc-window).

The omitted block of tab:mc-signal (mc_study.arm_S, signal="omitted": the signal comes from a
variable outside the feature set, correlated 0.5 with the first predictor), at R^2 = 0.20,
rho_x = 0.3, P = 6, the T = 12 penalty z = 0.133 held fixed, sigma^2 known, and the window
lengthened.  Part of an unspanned signal enters the training targets the way noise does, so
mu_CC = 2E[fg] - E[g^2] depends on T and can change sign.

    python omitted_window.py            # about 5 minutes
Output: output/omitted_window.json
"""
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mc_study as M

RHO, R2, P, N, REPS = 0.3, 0.20, 6, 4000, 100
TS = (12, 24, 48, 96)


def main():
    z = M.solve_z(0.5, 0.5)                      # the T = 12 penalty of tab:mc-signal
    out = []
    for T in TS:
        g = np.random.default_rng(M.seed_of("omitted_window", T))
        acc = []
        for _ in range(REPS):
            y, S, f = M.gen_path(T, P, N, RHO, R2, g, theta=np.zeros(P), signal="omitted")
            acc.append(M.run_path(y, S, f, T, z, max(18, int(math.ceil(1.5 * T))), 1.0, 1.0))
        A = {k: np.array([a[k] for a in acc]) for k in ("mu_CC", "mspe_gap", "Psihat", "Efg", "Eg2")}
        rec = dict(T=T, c=P / T, z=z, n=N, reps=REPS,
                   mu_CC=float(A["mu_CC"].mean()), mu_CC_se=float(A["mu_CC"].std() / math.sqrt(REPS)),
                   mspe_gap=float(A["mspe_gap"].mean()), Psihat=float(A["Psihat"].mean()),
                   Efg2=float(2 * A["Efg"].mean()), Eg2=float(A["Eg2"].mean()))
        out.append(rec)
        print(f"T={T:3d} mu_CC={rec['mu_CC']:+.4f} ({rec['mu_CC_se']:.4f})  2E[fg]={rec['Efg2']:.4f} "
              f"E[g2]={rec['Eg2']:.4f}  gap={rec['mspe_gap']:+.4f}  Psihat={rec['Psihat']:.4f}", flush=True)
    json.dump(out, open(os.path.join(HERE, "output", "omitted_window.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
