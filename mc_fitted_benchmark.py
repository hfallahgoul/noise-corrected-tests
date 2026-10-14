"""The fitted-benchmark identity eq:wedge-general (Section 3.3, sec:wedge):

    E[e1^2 - e2^2 | D] = 2 f g - g^2 - sigma^2 ||a||^2 - 2 sigma^2 b'a,

with b and a the weights of the benchmark and of the increment on the training targets,
f = E[y_t | F_{t-1}] - b'F the systematic part of what the benchmark misses, g = a'F.
The same check is made for the Clark-West centring 2fg - 2 sigma^2 b'a and the corrected
centring 2fg - g^2 - 2 sigma^2 b'a.  T = 24, P = 6 AR(1) features, n = 600, ridge z = 0.3.

    A  rolling mean, increment residualised on it                 (b'a = 0)
    B  rolling mean, increment not residualised, features with mean one   (b'a != 0)
    C  fitted benchmark on [1, h_t] with h exogenous, increment residualised   (b'a = 0)
    D  AR(1) benchmark on lagged targets, exogenous features, increment residualised
       (b'a = 0, but the benchmark's weights are not D-measurable)

    python mc_fitted_benchmark.py CASE REPS SEED        # the paper: REPS = 300, SEED = 7
Output: output/mc_fitted_benchmark/<CASE>_s<SEED>.json.  Standard errors are across paths.
"""
import json
import os
import sys

import numpy as np
case = sys.argv[1]; reps = int(sys.argv[2]); seed = int(sys.argv[3])
T, n, P, z, sig = 24, 600, 6, 0.3, 1.0
rng = np.random.default_rng(seed)
out = []; pid = []
for r in range(reps):
    L = n + T + 50
    S = np.zeros((L, P)); h = np.zeros(L)
    for t in range(1, L):
        S[t] = 0.5 * S[t-1] + rng.standard_normal(P) * np.sqrt(1 - 0.25)
        h[t] = 0.8 * h[t-1] + rng.standard_normal() * 0.6
    if case == "B":
        S = S + 1.0                                    # features with nonzero mean
    theta = np.full(P, 0.25); eps = sig * rng.standard_normal(L)
    y = np.zeros(L); m = np.zeros(L)                   # m[t] = E[y[t+1] | F_t]
    for t in range(L - 1):
        if case == "D":
            m[t] = 0.3 + 0.9 * y[t] + 0.1 * (S[t] @ theta)
        elif case == "C":
            m[t] = 0.3 + 0.5 * h[t] + S[t] @ theta
        else:
            m[t] = 0.3 + S[t] @ theta
        y[t+1] = m[t] + eps[t+1]
    for t in range(L - n, L):
        s_idx = np.arange(t - T, t)                    # targets y[s], s = t-T..t-1
        Y, F, E = y[s_idx], m[s_idx - 1], eps[s_idx]
        X = S[s_idx - 1]; snow = S[t - 1]
        A = np.linalg.solve(z * np.eye(P) + X.T @ X / T, snow)
        w = X @ A / T
        if case in ("A", "B"):
            b = np.ones(T) / T
            a = w - w.mean() if case == "A" else w
        else:
            H = np.column_stack([np.ones(T), h[s_idx - 1] if case == "C" else y[s_idx - 1]])
            hnow = np.array([1.0, h[t-1] if case == "C" else y[t-1]])
            G = np.linalg.solve(H.T @ H, np.eye(2))
            b = H @ G @ hnow
            a = w - H @ (G @ (H.T @ w))
        e1 = y[t] - b @ Y; D = a @ Y
        f = m[t-1] - b @ F; g = a @ F; psi = a @ a; ba = b @ a
        out.append((e1**2 - (e1 - D)**2, 2*f*g - g*g - sig**2*psi - 2*sig**2*ba,
                    2*e1*D, 2*f*g - 2*sig**2*ba,
                    e1**2 - (e1 - D)**2 + sig**2*psi, 2*f*g - g*g - 2*sig**2*ba, ba)); pid.append(r)
O = np.array(out)
res = {}
for k, (i, j) in {"gap": (0, 1), "CW": (2, 3), "CC": (4, 5)}.items():
    d = O[:, i] - O[:, j]
    pm = np.array([d[np.array(pid) == r].mean() for r in range(reps)])   # path means
    res[k] = dict(realised=O[:, i].mean(), formula=O[:, j].mean(), diff=d.mean(),
                  se=pm.std(ddof=1) / np.sqrt(reps))
res["ba_term"] = -2 * sig**2 * O[:, 6].mean()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "mc_fitted_benchmark")
os.makedirs(OUT, exist_ok=True)
json.dump(res, open(os.path.join(OUT, f"{case}_s{seed}.json"), "w"), indent=1)
