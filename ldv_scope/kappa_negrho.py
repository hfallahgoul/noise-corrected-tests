"""kappa_a at negative persistence, rho in [-0.95, 0) (rem:no-ldv).

Uses simulate() from kappa_map.py unchanged.

    python ldv_scope/kappa_negrho.py
Output: ldv_scope/output/kappa_negrho.json and .log.
"""
import json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kappa_map import simulate

OUT = os.environ.get("LDV_OUT", os.path.dirname(os.path.abspath(__file__)))
SEED = 20260910
REPS = 120

DESIGNS = [
    ("ar", "ar",    "lags", "A  lagged-y bench, lag features",  "passing"),
    ("ar", "ar",    "exog", "B  lagged-y bench, exog features", "failing"),
    ("ar", "const", "lags", "C  const bench (MISSPECIFIED)",    "control"),
]
RHOS = [-0.95, -0.85, -0.70, -0.50, -0.25, 0.0, 0.25, 0.50, 0.85, 0.95]
CONFIGS = [
    dict(T=250, P=25, zt=1.0),
    dict(T=125, P=25, zt=0.1),
    dict(T=125, P=60, zt=0.1),
    dict(T=500, P=60, zt=0.01),
    dict(T=250, P=25, zt=1e-3),
]


def verdict(m, s, status):
    if status == "passing":
        return "OK" if abs(m - 1) < 4 * s + 0.05 else "LEMMA FAILS"
    if status == "failing":
        return "kappa>1 (conservative)" if m - 1 > 4 * s else "NOT above one"
    return "signal, not cancellation"


def main():
    res = {"meta": dict(seed=SEED, reps=REPS, script="kappa_negrho.py")}
    t0 = time.time()

    print("=" * 112)
    print(f"PANEL N1.  kappa_a against persistence, including rho < 0.  "
          f"T=250, P=25, ztil=1, reps={REPS}")
    print("=" * 112)
    hdr = f"{'design':>34}{'status':>10}" + "".join(f"{r:>7}" for r in RHOS)
    print(hdr)
    print("-" * len(hdr))
    for tgt, bch, ft, lab, status in DESIGNS:
        cells = {}
        for r in RHOS:
            m, s = simulate(bch, tgt, ft, rho=r, reps=REPS, seed=SEED)
            cells[str(r)] = dict(kappa=m, se=s, verdict=verdict(m, s, status))
        print(f"{lab:>34}{status:>10}"
              + "".join(f"{cells[str(r)]['kappa']:>7.3f}" for r in RHOS))
        print(f"{'(se)':>34}{'':>10}"
              + "".join(f"{cells[str(r)]['se']:>7.3f}" for r in RHOS))
        res[f"N1|{tgt}|{bch}|{ft}"] = dict(label=lab, status=status, by_rho=cells)

    print()
    print("=" * 112)
    print(f"PANEL N2.  Robustness at rho = -0.85 across window, feature count and penalty. "
          f"reps={REPS}")
    print("=" * 112)
    print(f"{'design':>34}{'T':>6}{'P':>5}{'ztil':>9}{'kappa':>18}   verdict")
    print("-" * 112)
    for tgt, bch, ft, lab, status in DESIGNS:
        if status == "control":
            continue
        for kw in CONFIGS:
            m, s = simulate(bch, tgt, ft, rho=-0.85, reps=REPS, seed=SEED + 1, **kw)
            print(f"{lab:>34}{kw['T']:>6}{kw['P']:>5}{kw['zt']:>9}"
                  f"{m:>11.3f} +/-{s:<5.3f}   {verdict(m, s, status)}")
            res[f"N2|{tgt}|{bch}|{ft}|T{kw['T']}|P{kw['P']}|z{kw['zt']}"] = dict(
                label=lab, status=status, rho=-0.85, kappa=m, se=s,
                verdict=verdict(m, s, status), **kw)

    res["meta"]["elapsed_sec"] = round(time.time() - t0, 1)
    path = os.path.join(OUT, "kappa_negrho.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"\n[saved] {path}   ({res['meta']['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
