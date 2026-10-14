"""Independent-seed replicates for tab:cw-har (seeds 13579 and 24680) and tab:har-power.

    python ldv_scope/verify.py
Output: ldv_scope/output/verify.json and .log.
"""
import numpy as np, math, json, sys
import os
from cw_under_har import one as cw_one, srad
from har_size_power import run as hsp_run

print("="*88)
print("V1. Clark-West size under a rolling HAR benchmark, SHARP NULL, independent seeds")
print("="*88)
out={}
for seed in (13579, 24680):
    for scale,T in ((1.0,250),(1.0,1000),(0.4,250)):
        rng=np.random.default_rng(seed)
        ts=np.array([cw_one(scale,T,25,1.0,700,rng)[0] for _ in range(300)])
        sz=float(np.mean(ts>1.645)); se=math.sqrt(sz*(1-sz)/300)
        print(f"  seed={seed} HAR scale={scale} (srad={srad(scale):.4f}) T={T:>4}: "
              f"CW size = {sz:.3f} +/-{se:.3f}   mean t = {ts.mean():+.2f}")
        out[f"V1|{seed}|{scale}|{T}"]=dict(size=sz,se=se,mean_t=float(ts.mean()))

print()
print("="*88)
print("V2. HAR bench vs lag-ridge: size and power of CC, independent seed")
print("="*88)
rows=hsp_run("lags",[0.0,0.10,0.20],250,25,1.0,700,300,seed=98765)
out["V2"]=rows
json.dump(out,open(os.path.join(os.environ.get("LDV_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")), "verify.json"),"w"),indent=1)
print("\n[saved] verify.json")
