"""Size in the admissible own-lag HAR design at the applications' sample sizes (the bandwidth
discussion, rem:bandwidth-window).

    python ldv_scope/panelB.py
Output: ldv_scope/output/bandwidth_cost.json (panel B).
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bandwidth_cost import panel

if __name__ == "__main__":
    t0 = time.time()
    B = panel("B. The applied calibration: T = 250, c = 0.10, own-lag features, sharp null.",
              [(250, 25, 1200), (250, 25, 5000), (250, 25, 7400)], 300)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    json.dump(dict(panelB=B, reps=300),
              open(os.path.join(out, "bandwidth_cost.json"), "w"), indent=1)
    print(f"\n[saved] ({(time.time()-t0)/60:.1f} min)")
