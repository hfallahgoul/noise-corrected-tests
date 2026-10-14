"""The benchmark's degrees of freedom (prop:dof, app:dof) and the leverage residue of a
fitted benchmark (rem:psitil-residue).

A benchmark that projects the training targets on q regressors leaves the increment's weight
vector a_t = M_t w_{t-1} with M_t of rank T - q, so Psihat_n converges to (1 - q/T) Psi(c, z)
rather than Psi(c, z).  The script measures the error of the closed form with and without
that factor over q in {1, 2, 4, 8} and T in {12, 60, 250}, under the sharp null with i.i.d.
Gaussian features.  The 'leverage' block measures the innovation part of the differenced
residual, sigma^2 (2 - h_ss - h_{s-1,s-1} + 2 h_{s,s-1}), for a HAR benchmark.

    python mc_dof.py            # writes output/mc_dof/dof_check.json (about two minutes)
    python mc_dof.py --check    # re-run and compare with the stored file
"""
import argparse, json, math, os, zlib
import numpy as np

OUT = os.path.join("output", "mc_dof")


def seed_of(*parts):
    return zlib.crc32(repr(parts).encode()) & 0xFFFFFFFF


def psi_closed(c, z):
    u = 1.0 + c + z
    return 0.5 * (u / math.sqrt(u * u - 4.0 * c) - 1.0)


def cell(T, c, zt, q, reps=None, step=None, n=400):
    """Mean Psihat_t against the closed form, for a benchmark spending q df.

    The effect is large and low-variance, so long windows need far fewer draws.
    """
    if reps is None:
        reps = 150 if T <= 60 else 60
    if step is None:
        step = 5 if T <= 60 else 20
    P = max(2, round(c * T)); c = P / T; z = zt * P / T
    I = np.eye(T); acc = []
    for r in range(reps):
        g = np.random.default_rng(seed_of(T, P, zt, q, r))
        m = n + T + 2
        S = g.standard_normal((m, P))
        Z = g.standard_normal((m, q)); Z[:, 0] = 1.0        # benchmark design, w/ intercept
        for i in range(0, n, step):
            X = S[i:i + T]; s = S[i + T]; Zw = Z[i:i + T]
            M = I - Zw @ np.linalg.solve(Zw.T @ Zw, Zw.T)
            B = np.linalg.inv(z * I + X @ X.T / T)
            a = (M @ (B @ (X @ s))) / T
            acc.append(float(a @ a))
    ph = float(np.mean(acc)); pc = psi_closed(c, z)
    return dict(T=T, P=P, c=c, zt=zt, q=q, reps=reps, Psihat=ph, Psi_closed=pc,
                raw_err=ph / pc - 1.0, corrected_err=ph / ((1 - q / T) * pc) - 1.0)


def lev_cell(T, rho, q=4, reps=20):
    """Bias of the DIFFERENCED benchmark residual when the benchmark is fitted.

    prop:sigma-weighted writes r_s = f_{s-1} + eps_s, the residual of a benchmark whose
    coefficients are fixed.  The paper's residuals come from a fit on the same window,
    r = M Y, so E[r r' | D] = sigma^2 M is not diagonal and E[r_s^2 | D] = sigma^2(1-h_ss)
    with h the benchmark's hat matrix.  Differencing contains it: the innovation part of
    E[(r_s - r_{s-1})^2 | D] is

        sigma^2 (2 - h_ss - h_{s-1,s-1} + 2 h_{s,s-1}),

    a SECOND difference of the hat matrix rather than its diagonal.  For an intercept
    benchmark that is exactly zero -- the window mean cancels in the difference -- so this
    cell measures what Corsi's four-parameter HAR leaves, against the -q/T the
    undifferenced residuals carry.  The signal path is an AR(1) with correlation rho; the
    HAR regressors are a constant and the daily, weekly and monthly averages, as in
    rv_application.py.
    """
    dif = []
    for r in range(reps):
        g = np.random.default_rng(seed_of("lev", T, rho, q, r))
        m = T + 400
        y = np.zeros(m)
        for i in range(1, m):
            y[i] = rho * y[i - 1] + g.standard_normal()
        Z = np.array([[1.0, y[i - 1], y[i - 5:i].mean(), y[i - 22:i].mean()]
                      for i in range(m - T, m)])
        Pz = Z @ np.linalg.solve(Z.T @ Z, Z.T)
        h = np.diag(Pz)
        off = np.array([Pz[s, s - 1] for s in range(1, T)])
        dif.append(float(((2 - h[1:] - h[:-1] + 2 * off) / 2.0 - 1.0).mean()))
    return dict(T=T, rho=rho, q=q, reps=reps,
                differenced_bias=float(np.mean(dif)), undifferenced_bias=-q / T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    rows = [cell(T, 1.0, zt, q)
            for q in (1, 2, 4, 8) for T in (12, 60, 250) for zt in (0.1, 1.0, 10.0)]
    lev = [lev_cell(T, rho) for T in (60, 250) for rho in (0.5, 0.9, 0.95, 0.99)]
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "dof_check.json")
    if a.check and os.path.exists(path):
        old = json.load(open(path))
        bad = [(o, n) for o, n in zip(old["rows"], rows)
               if abs(o["Psihat"] - n["Psihat"]) > 1e-12]
        bad += [(o, n) for o, n in zip(old.get("leverage", lev), lev)
                if abs(o["differenced_bias"] - n["differenced_bias"]) > 1e-12]
        print("MISMATCH" if bad else "mc_dof: reproduces exactly")
        return
    json.dump(dict(rows=rows, leverage=lev), open(path, "w"), indent=1)
    raw = [abs(r["raw_err"]) for r in rows]; cor = [abs(r["corrected_err"]) for r in rows]
    print(f"cells: {len(rows)}")
    print(f"  |raw error|        max {max(raw)*100:6.2f}%   min {min(raw)*100:5.2f}%")
    print(f"  |corrected error|  max {max(cor)*100:6.2f}%   min {min(cor)*100:5.2f}%")
    for T in (12, 60, 250):
        sub = [abs(r["corrected_err"]) for r in rows if r["T"] == T]
        print(f"  T={T:3d}: corrected max {max(sub)*100:5.2f}%")
    print("fitted-benchmark bias in the differenced residual (rem:psitil-residue):")
    for T in (60, 250):
        d = [abs(r["differenced_bias"]) for r in lev if r["T"] == T]
        print(f"  T={T:3d}: differenced {min(d)*100:5.2f}% to {max(d)*100:5.2f}%"
              f"   undifferenced {4/T*100:5.2f}%")


if __name__ == "__main__":
    main()
