"""The 3x3 p-value grid: which null each statistic tests (tab:mc-grid, figures pgrid_*).

Rows are data generated under the three nulls of tab:three-nulls, columns the three
statistics, and each panel the histogram of p-values with the uniform reference and 95%
bands.  Under its own null a statistic's p-values are uniform (the diagonal); mass at zero
off the diagonal is over-rejection, mass at one under-rejection.  Scale factors are fixed in
a pilot, so no replication uses its own evaluation sample.  mc_study.py calls this for
Psi* in {0.2, 0.5, 1.0} with the oracle and the difference estimator.
"""
import numpy as np, math
import matplotlib.pyplot as plt

# ----------------------------------------------------------------- settings
T, c, z, n = 8, 0.5, 0.5, 2000     # inside every condition: T << b << sqrt(n/T)
b          = 12                    # bandwidth; needs T << b << sqrt(n/T)
R2         = 0.10                  # signal strength for the two composite rows
R          = 1000                  # replications per row
SEED       = 20260816
RHO_X      = 0.90    # AR(1) persistence of the predictors.  MUST be > 0: with iid rows
                     # the signal has rho_f = 0, the difference estimator gives NO bias
                     # reduction (delta_f = var(f)), and sigma^2 is overstated by R^2.
SIGMA2     = None    # None -> difference estimator; 1.0 -> oracle, to isolate its effect

P   = int(c * T)
rng = np.random.default_rng(SEED)
I   = np.eye(T); one = np.ones(T); M = I - np.outer(one, one) / T
Psi = lambda c, z: 0.5*((1+c+z)/np.sqrt((1+c+z)**2 - 4*c) - 1)
Phi = lambda t: 0.5*(1 + math.erf(t/math.sqrt(2)))

def bartlett(x, bw):
    x = x - x.mean(); m = len(x); s = x @ x / m
    for l in range(1, bw+1):
        s += 2*(1 - l/(bw+1)) * (x[l:] @ x[:-l]) / m
    return s

def ar1_rows(m, P, rho, gen):
    S = np.empty((m, P)); S[0] = gen.standard_normal(P)
    for t in range(1, m):
        S[t] = rho*S[t-1] + np.sqrt(1-rho**2)*gen.standard_normal(P)
    return S

def one_path(theta, scale, gen):
    """Simulate one path; return the three t-statistics and the realised mu_CC."""
    S   = ar1_rows(n+T+1, P, RHO_X, gen)
    eps = gen.standard_normal(n+T+1)
    f   = S @ theta
    y   = np.roll(f, 1) + eps
    g = np.empty(n); v = np.empty(n); psi = np.empty(n); y1 = np.empty(n)
    for i in range(n):
        X = S[i:i+T]; Y = y[i+1:i+T+1]; F = f[i:i+T]; s = S[i+T]
        B = np.linalg.inv(z*I + X @ X.T / T)
        a = (M @ (B @ (X @ s))) / T
        y1[i] = Y.mean(); g[i] = a @ F; v[i] = a @ (Y - F); psi[i] = a @ a
    yt = y[T+1:T+n+1]; ft = f[T:T+n]
    gs, vs, ps_ = scale*g, scale*v, scale**2*psi        # FIXED scale, no look-ahead
    yh   = y1 + gs + vs
    s2   = SIGMA2 if SIGMA2 is not None else 0.5*np.mean(np.diff(yt)**2)
    dDM  = (yt-y1)**2 - (yt-yh)**2
    dCW  = dDM + (yh-y1)**2
    dCC  = dDM + s2*ps_
    h    = 0.5*np.r_[0., np.diff(yt)]; h[0] = h[1:].mean()
    sc_  = dCC - dCC.mean() + ps_.mean()*(h - h.mean())
    return (np.sqrt(n)*dDM.mean()/dDM.std(),
            np.sqrt(n)*dCW.mean()/dCW.std(),
            np.sqrt(n)*dCC.mean()/np.sqrt(max(bartlett(sc_, b), 1e-14)),
            np.mean(2*ft*gs - gs**2),
            np.mean(2*ft*gs - gs**2) - s2*np.mean(ps_))

# ------------------------------------------------- pilot: population moments
print("pilot ...")
th1 = rng.standard_normal(P); th1 /= np.linalg.norm(th1); th1 *= np.sqrt(R2)
Efg = Eg2 = Ev2 = 0.0
for _ in range(60):
    S = ar1_rows(n+T+1, P, RHO_X, rng); eps = rng.standard_normal(n+T+1)
    f = S @ th1; y = np.roll(f, 1) + eps
    for i in range(0, n, 7):
        X = S[i:i+T]; Y = y[i+1:i+T+1]; F = f[i:i+T]; s = S[i+T]
        B = np.linalg.inv(z*I + X @ X.T / T); a = (M @ (B @ (X @ s))) / T
        Efg += f[i+T]*(a @ F); Eg2 += (a @ F)**2; Ev2 += (a @ (Y-F))**2
K = 60*len(range(0, n, 7)); Efg, Eg2, Ev2 = Efg/K, Eg2/K, Ev2/K
a_CC = 2*Efg/Eg2                 # mu_CC = 0        -> the CC null
a_DM = 2*Efg/(Eg2 + Ev2)         # MSPE_2 = MSPE_1  -> the DM null
print(f"  E[fg]={Efg:.5f}  E[g^2]={Eg2:.5f}  E[v^2]={Ev2:.5f}  (Psi={Psi(c,z):.4f})")
print(f"  fixed scales:  a_CC = {a_CC:.4f}   a_DM = {a_DM:.4f}")

ROWS = [("H0-cov:  f = 0",            np.zeros(P), 1.0),
        ("H0-CC:   2E[fg] = E[g^2]",  th1,         a_CC),
        ("H0-MSPE: MSPE2 = MSPE1",    th1,         a_DM)]

# ------------------------------------------------------------------- run
res = {}
for lab, th, sc in ROWS:
    g2 = np.random.default_rng(abs(hash(lab)) % 2**32)
    out = np.array([one_path(th, sc, g2) for _ in range(R)])
    res[lab] = out
    print(f"\n{lab}   realised mu_CC={out[:,3].mean():+.5f}  MSPE gap={out[:,4].mean():+.5f}")
    print(f"{'':10}{'reject 5%':>11}{'sqrt(m)D':>10}{'  uniform?':>12}")
    for j, nm in enumerate(['DM','CW','CC']):
        pv = np.array([1-Phi(t) for t in out[:, j]])
        x = np.sort(pv); m = len(x)
        D = max(np.max(np.arange(1,m+1)/m - x), np.max(x - np.arange(0,m)/m))
        print(f"{nm:>10}{np.mean(pv<0.05):11.3f}{math.sqrt(m)*D:10.2f}"
              f"{('  UNIFORM' if math.sqrt(m)*D < 1.36 else '  not uniform'):>12}")

# ------------------------------------------------------------------ figure
fig, ax = plt.subplots(3, 3, figsize=(11, 9), sharex=True)
band = 1.96*math.sqrt(0.1*0.9/R)*R/10
for i, (lab, _, _) in enumerate(ROWS):
    for j, nm in enumerate(['DM','CW','CC']):
        pv = np.array([1-Phi(t) for t in res[lab][:, j]])
        A = ax[i, j]
        A.hist(pv, bins=10, range=(0,1),
               color=['#7f7f7f','#c0392b','#2c6fbb'][j], edgecolor='white')
        A.axhline(R/10, ls='--', c='k', lw=1)
        A.axhline(R/10+band, ls=':', c='k', lw=1); A.axhline(R/10-band, ls=':', c='k', lw=1)
        if i == 0: A.set_title(nm, fontsize=12)
        if j == 0: A.set_ylabel(lab.split(':')[0], fontsize=10)
        if i == 2: A.set_xlabel('p-value')
fig.suptitle('Which null is each statistic testing?  '
             'p-values under each of the three nulls', fontsize=13)
fig.tight_layout(); fig.savefig('pvalue_grid.png', dpi=150)
print("\nsaved pvalue_grid.png")
