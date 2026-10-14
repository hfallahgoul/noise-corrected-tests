"""The discriminating cell of tab:ldv-scope and the robustness of the two admissible cases.

    python ldv_scope/kappa_map2.py
Output: ldv_scope/output/kappa_map2.json.
"""
import json, os
from kappa_map import simulate

OUT = os.environ.get("LDV_OUT",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"))
res = {}

print("=" * 100)
print("PANEL B. The discriminating cell: lag FEATURES with a non-lagged benchmark.")
print("  target='exog' means y_t = b'Z_{t-1} + eps_t, so the benchmark spans the")
print("  conditional mean exactly and lags of y carry NO incremental signal.")
print("=" * 100)
print(f"{'target':>8}{'benchmark':>12}{'alt features':>14}{'kappa':>18}   verdict")
print("-" * 100)
cases = [
    ("exog", "exog", "exog", "no lags anywhere (baseline)"),
    ("exog", "exog", "lags", "lag FEATURES, exogenous benchmark  <-- discriminating"),
    ("ar",   "ar",   "exog", "lagged-y benchmark, exogenous features"),
    ("ar",   "ar",   "lags", "lagged-y benchmark AND lag features"),
]
for tgt, bch, ft, lab in cases:
    m_, s_ = simulate(bch, tgt, ft, seed=771, reps=40)
    ok = "OK" if abs(m_ - 1) < 4 * s_ + 0.05 else "LEMMA FAILS"
    print(f"{tgt:>8}{bch:>12}{ft:>14}{m_:>11.3f} +/-{s_:<5.3f}   {ok:<12} ({lab})")
    res[f"B|{tgt}|{bch}|{ft}"] = dict(kappa=m_, se=s_, label=lab)

print()
print("=" * 100)
print("PANEL C. Robustness of the two passing lagged-y cases across the regime the")
print("  paper cares about: light shrinkage, high persistence, high complexity.")
print("=" * 100)
print(f"{'case':>22}{'rho':>7}{'T':>6}{'P':>5}{'ztil':>7}{'kappa':>18}   verdict")
print("-" * 100)
grid = [
    ("ar",  "ar",  "lags", dict(rho=0.85, T=250, P=25, zt=1.0)),
    ("ar",  "ar",  "lags", dict(rho=0.95, T=250, P=25, zt=1.0)),
    ("ar",  "ar",  "lags", dict(rho=0.85, T=125, P=25, zt=0.1)),
    ("ar",  "ar",  "lags", dict(rho=0.85, T=125, P=60, zt=0.1)),
    ("ar",  "ar",  "lags", dict(rho=0.95, T=125, P=60, zt=0.01)),
    ("har", "har", "lags", dict(T=250, P=25, zt=1.0)),
    ("har", "har", "lags", dict(T=125, P=25, zt=0.1)),
    ("har", "har", "lags", dict(T=125, P=60, zt=0.1)),
    ("har", "har", "lags", dict(T=500, P=60, zt=0.01)),
    ("ar",  "ar",  "exog", dict(rho=0.85, T=125, P=60, zt=0.1)),
    ("har", "har", "exog", dict(T=125, P=60, zt=0.1)),
]
for tgt, bch, ft, kw in grid:
    m_, s_ = simulate(bch, tgt, ft, seed=9001, reps=40, **kw)
    ok = "OK" if abs(m_ - 1) < 4 * s_ + 0.05 else "LEMMA FAILS"
    tag = f"{bch} bench / {ft}"
    print(f"{tag:>22}{kw.get('rho','-'):>7}{kw['T']:>6}{kw['P']:>5}{kw['zt']:>7}"
          f"{m_:>11.3f} +/-{s_:<5.3f}   {ok}")
    res[f"C|{tgt}|{bch}|{ft}|{kw}"] = dict(kappa=m_, se=s_, **{k: v for k, v in kw.items()})

with open(os.path.join(OUT, "kappa_map2.json"), "w") as fh:
    json.dump(res, fh, indent=1, default=str)
print(f"\n[saved] {OUT}/kappa_map2.json")
