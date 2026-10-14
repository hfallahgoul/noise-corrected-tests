"""Size of the corrected test as the bandwidth condition T = o(n^{1/3}) is violated by a
growing margin (rem:bandwidth-window, rem:T3-condition).

    python ldv_scope/bandwidth_cost.py
Output: ldv_scope/output/bandwidth_cost.json and .log.
"""
import json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from har_size_power import one_path          # identical estimator to tab:har-power

OUT = os.environ.get("LDV_OUT", os.path.join(HERE, "output"))
os.makedirs(OUT, exist_ok=True)
CRIT = 1.645


def cell(T, P, n, reps, zt=1.0, seed=707):
    """Size of the corrected and Clark--West statistics under the sharp null."""
    rng = np.random.default_rng(seed)
    cc, cw, dm, ratio = [], [], [], []
    for _ in range(reps):
        s = one_path("lags", 0.0, T, P, zt, n, rng, light=True)
        cc.append(s["t_CCp"]); cw.append(s["t_CW"]); dm.append(s["t_DM"])
        ratio.append(s["R_plug"])
    cc = np.array(cc); cw = np.array(cw)
    return dict(T=T, P=P, n=n, zt=zt, c=P / T, reps=reps,
                F2_need=4 * T ** 3, F2_margin=n / (4.0 * T ** 3),
                F2_ok=bool(n >= 4 * T ** 3),
                size_CC=float(np.mean(cc > CRIT)), size_CW=float(np.mean(cw > CRIT)),
                mean_CC=float(cc.mean()), sd_CC=float(cc.std()),
                mean_CW=float(cw.mean()), sd_CW=float(cw.std()),
                ratio=float(np.mean(ratio)))


def panel(name, rows_spec, reps):
    print("=" * 96)
    print(name)
    print("=" * 96)
    hdr = (f"{'T':>5}{'P':>4}{'c':>7}{'n':>8}{'4T^3':>11}{'n/4T^3':>9}{'F2':>5}"
           f"{'size CC':>9}{'size CW':>9}{'mean t_CC':>11}{'sd':>6}{'R':>7}")
    print(hdr); print("-" * len(hdr))
    out = []
    for (T, P, n) in rows_spec:
        r = cell(T, P, n, reps)
        out.append(r)
        print(f"{T:>5}{P:>4}{r['c']:>7.2f}{n:>8}{r['F2_need']:>11,}{r['F2_margin']:>9.4f}"
              f"{'yes' if r['F2_ok'] else 'no':>5}{r['size_CC']:>9.3f}{r['size_CW']:>9.3f}"
              f"{r['mean_CC']:>11.2f}{r['sd_CC']:>6.2f}{r['ratio']:>7.2f}")
    return out


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    reps = 60 if quick else 300
    t0 = time.time()
    print(f"# bandwidth_cost.py  reps={reps}  sharp null, own-lag features, one-sided 5%\n")

    A = panel("A. F2 isolated: c = 0.10 and ztilde = 1 fixed, only n moves.",
              [(20, 2, 500), (20, 2, 2000), (20, 2, 8000),
               (20, 2, 32000), (20, 2, 64000)], reps)

    B = panel("B. The applied calibration: T = 250, c = 0.10, at the applications' n.",
              [(250, 25, 1200), (250, 25, 5000), (250, 25, 7400)], reps)

    json.dump(dict(panelA=A, panelB=B, reps=reps),
              open(os.path.join(OUT, "bandwidth_cost.json"), "w"), indent=1)
    print(f"\n[saved] {OUT}/bandwidth_cost.json   ({(time.time()-t0)/60:.1f} min)")
