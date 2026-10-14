"""Figure fig:rv-data: the realized-variance series of Section 6 and its autocorrelation.

Reads the stored target series for SPY (Risk Lab 84398, 5-minute subsampled realized
volatility from trades, column c5) from output/rv_series/84398.npz, a copy of the
Risk Lab-derived series stored by rv_window_store.py, keeps the
paper's sample (to 19 August 2026), and writes figures/fig-rv-data.pdf next to the paper.
    python fig_rv_data.py
"""
import os, numpy as np, tempfile
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplcache-rvdata"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "output", "rv_series", "84398.npz")
OUTS = [os.path.join(HERE, "output", "rv_figures", "fig-rv-data.pdf"),
        os.path.abspath(os.path.join(HERE, "..", "Latex", "figures", "fig-rv-data.pdf"))]
END, T, MAXLAG = 20260819, 250, 100

d = np.load(SRC, allow_pickle=True)
date, y = np.asarray(d["date"]).astype(int), np.asarray(d["y"], float)
keep = (date <= END) & np.isfinite(y)
date, y = date[keep], y[keep]
t = [datetime.strptime(str(x), "%Y%m%d") for x in date]

yc = y - y.mean()
acf = np.array([np.dot(yc[:len(yc)-k], yc[k:]) / np.dot(yc, yc) for k in range(1, MAXLAG + 1)])
band = 1.96 / np.sqrt(len(y))

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.6, 1]})
ax[0].plot(t, y, lw=0.4, color="0.25")
i1 = int(np.searchsorted(date, 20081231)); i0 = i1 - T
ax[0].axvspan(t[i0], t[i1], color="0.85", lw=0)
ax[0].set_ylabel("log daily realized variance")
ax[0].set_title("(a) SPY, 5-minute realized variance", fontsize=9)
ax[1].bar(np.arange(1, MAXLAG + 1), acf, width=0.8, color="0.35")
ax[1].axhspan(-band, band, color="0.85", lw=0)
ax[1].set_xlim(0, MAXLAG + 1); ax[1].set_ylim(0, 1)
ax[1].set_xlabel("lag (days)"); ax[1].set_title("(b) autocorrelation", fontsize=9)
fig.tight_layout()
for o in OUTS:
    os.makedirs(os.path.dirname(o), exist_ok=True); fig.savefig(o)
print(f"n={len(y)}, {date[0]}-{date[-1]}, acf lag1={acf[0]:.3f} lag22={acf[21]:.3f} lag100={acf[99]:.3f}")
