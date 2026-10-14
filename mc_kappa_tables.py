"""Both panels of tab:kappa, regenerated from one function with one seed.

Panel (a): kappa_ldv against target persistence rho and the number k of own lags in the
autoregressive benchmark, at T=250, P=25, ztilde=1.  Panel (b): against the effective
penalty and the feature count at rho=0.85, k=3.  The extra features are exogenous and the
benchmark is correctly specified, so every departure from one is the failure of lem:psi-t
in a lagged-target design.  Both panels call mc_har_diagnosis.ratio with the same seed, so
the cell they share (rho=0.85, k=3, P=25, ztilde=1) is the same number in both.

    python mc_kappa_tables.py          # a few minutes

Updates the two panel keys of output/mc_har/kappa_exploration.json and records the run.
"""
import json, os, sys, datetime
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mc_har_diagnosis import ratio, OUT

fn = os.path.join(OUT, "kappa_exploration.json")
res = json.load(open(fn)) if os.path.exists(fn) else {}
a = {}
for rho in (0.0, 0.5, 0.85, 0.95):
    a[f"rho{rho:g}"] = {}
    for k in (1, 3, 6):
        m, se = ratio("ar", "ar", "exog", T=250, rho=rho, P=25, zt=1.0, k=k)
        a[f"rho{rho:g}"][f"k{k}"] = round(m, 3); print(f"rho={rho} k={k}: {m:.3f} ({se:.3f})", flush=True)
b = {}
for zt, key in ((0.1, "zt0.1"), (1.0, "zt1.0"), (10.0, "zt10")):
    b[key] = {}
    for P in (10, 25, 60):
        m, se = ratio("ar", "ar", "exog", T=250, rho=0.85, P=P, zt=zt, k=3)
        b[key][f"P{P}"] = round(m, 3); print(f"zt={zt} P={P}: {m:.3f} ({se:.3f})", flush=True)
res["by_rho_and_k_T250_P25_zt1"] = a
res["by_penalty_and_P_T250_rho085_k3"] = b
res["panels_run"] = dict(script="mc_kappa_tables.py", seed=1, reps=25, n=1500,
                         when=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
json.dump(res, open(fn, "w"), indent=1)
print("wrote", fn)
