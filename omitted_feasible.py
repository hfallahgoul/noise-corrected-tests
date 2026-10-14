"""The omitted-predictor design of tab:mc-signal with a feasible innovation variance
(app:signal-feasible).

Arm S of mc_study.py runs with the oracle sigma^2 = 1.  This script reruns the omitted block,
the one design that reaches mu_CC < 0 without degeneracy, and returns the oracle and the
difference-estimator statistics from identical paths, so the comparison carries no Monte
Carlo noise of its own.

    python omitted_feasible.py                 # all five R^2 values
    python omitted_feasible.py 0.05 0.1        # selected cells; each is seeded independently

Output: output/armS_feasible.json.
"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mc_study as M

T, n, b, rho, psit = 12, 8000, 18, 0.3, 0.5
c = 0.5; z = M.solve_z(c, psit); P = max(1, int(round(c * T)))
R = int(os.environ.get("R", "1000"))
GRID = [float(x) for x in sys.argv[1:]] or [0.0, 0.02, 0.05, 0.1, 0.2]


def cell(R2):
    g = np.random.default_rng(M.seed_of("omitted", R2))
    acc = []
    for _ in range(R):
        y, S, f = M.gen_path(T, P, n, rho, R2, g, theta=np.zeros(P), signal="omitted")
        acc.append(M.run_path(y, S, f, T, z, b, 1.0, 1.0, both=True))
    A = {k: np.array([a[k] for a in acc]) for k in acc[0]}
    rej = lambda k: float(np.mean(A[k] > 1.645))
    row = dict(R2=R2, Efg2=2*float(A["Efg"].mean()), Eg2=float(A["Eg2"].mean()),
               mu_CC=float(A["mu_CC"].mean()), mspe_gap=float(A["mspe_gap"].mean()),
               rej_CW=rej("t_CW"),
               rej_CC_oracle=rej("t_CC_hac__oracle"), rej_CC_diff=rej("t_CC_hac__diff"),
               ratio_oracle=float(np.nanmean(A["ratio__oracle"])),
               ratio_diff=float(np.nanmean(A["ratio__diff"])))
    print(f"R2={R2:<5} mu_CC={row['mu_CC']:+.5f}  CW={row['rej_CW']:.3f}  "
          f"CC(oracle)={row['rej_CC_oracle']:.3f}  CC(diff)={row['rej_CC_diff']:.3f}  "
          f"R_orc={row['ratio_oracle']:.3f} R_diff={row['ratio_diff']:.3f}", flush=True)
    return row


if __name__ == "__main__":
    fn = os.path.join(HERE, "output", "armS_feasible.json")
    rows = json.load(open(fn)) if os.path.exists(fn) else []
    for R2 in GRID:
        rows = [r for r in rows if r["R2"] != R2] + [cell(R2)]
        rows.sort(key=lambda r: r["R2"])
        json.dump(rows, open(fn, "w"), indent=1)
