"""Whether lag features must nest the benchmark's regressors or only share its filtration:
features are lags s, ..., s + P - 1 (caption of tab:ldv-scope).

    python ldv_scope/nest_check.py
Output: ldv_scope/output/nest_check.json and .log.
"""
import numpy as np, math, json
import os

def kappa(start, T=250, rho=0.85, P=25, zt=1.0, reps=40, n=1200, seed=1, bench="ar"):
    g=np.random.default_rng(seed); z=zt*P/T; per=[]
    for _ in range(reps):
        m=n+T+140; sig=0.42; eps=sig*g.standard_normal(m)
        y=np.zeros(m)
        if bench=="ar":
            for t in range(1,m): y[t]=rho*y[t-1]+eps[t]
        else:
            for t in range(30,m):
                y[t]=0.35*y[t-1]+0.35*y[t-5:t].mean()+0.25*y[t-22:t].mean()+eps[t]
        L=P+start+40; M_=len(y)-L
        if bench=="ar":
            H=np.column_stack([np.ones(M_)]+[y[L-k:L-k+M_] for k in (1,2,3)])
        else:
            wk=np.array([y[L+i-5:L+i].mean() for i in range(M_)])
            mo=np.array([y[L+i-22:L+i].mean() for i in range(M_)])
            H=np.column_stack([np.ones(M_),y[L-1:L-1+M_],wk,mo])
        XE=np.column_stack([y[L-start-k:L-start-k+M_] for k in range(P)])
        Y=y[L:L+M_]; D2=[];SP=[]
        for i in range(0,M_-T-1,5):
            Hw,Xw,Yw=H[i:i+T],XE[i:i+T],Y[i:i+T]; s=XE[i+T]
            Qh,_=np.linalg.qr(Hw); QtX=Qh.T@Xw
            G=(Xw.T@Xw-QtX.T@QtX)/T
            gg,V=np.linalg.eigh(G); gg=np.clip(gg,0,None)
            w_=V@((V.T@s)/(gg+z)); a=((Xw-Qh@QtX)@w_)/T
            D2.append((a@Yw)**2); SP.append(sig**2*(a@a))
        per.append(np.mean(D2)/np.mean(SP))
    per=np.array(per); return float(per.mean()), float(per.std(ddof=1)/math.sqrt(len(per)))

print("="*84)
print("Does the lag-feature result survive when the features do NOT nest the benchmark?")
print("  features = lags (start+1) ... (start+P) of y;  start=1 means the usual nested set")
print("="*84)
print(f"{'first lag used':>16}{'nests benchmark?':>20}{'AR(3) bench':>18}{'HAR bench':>18}")
print("-"*84)
res={}
for start,nest in ((1,"yes"),(4,"no (AR)"),(8,"no"),(23,"no")):
    a_,sa=kappa(start,seed=606,bench="ar")
    h_,sh=kappa(start,seed=606,bench="har")
    print(f"{start+1:>16}{nest:>20}{a_:>11.3f} +/-{sa:<5.3f}{h_:>11.3f} +/-{sh:<5.3f}")
    res[f"start{start}"]=dict(ar=a_,ar_se=sa,har=h_,har_se=sh,nests=nest)
json.dump(res,open(os.path.join(os.environ.get("LDV_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")), "nest_check.json"),"w"),indent=1)
print("\n[saved] nest_check.json")
