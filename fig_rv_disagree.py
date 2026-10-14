"""Candidate figure: per-asset t_CW against t_CC in the five-estimator panel (sec:rv-disagree).

Reads output/rv_cross_multiest/panel.json, keeps the middle penalty (ztilde = 1) and the
series passing the sanity ratio, and plots one point per asset. Writes
output/rv_figures/fig-rv-disagree.pdf (and a PNG preview).
    python fig_rv_disagree.py
"""
import os, json, tempfile, numpy as np
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplcache-rvdis"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, "output", "rv_cross_multiest", "panel.json")))["rows"]
own = json.load(open(os.path.join(HERE, "output", "rv_cross", "panel.json")))
own = own["rows"] if isinstance(own, dict) else own
ZT, CRIT = 1.0, 1.645
# Same sample as tab:rv-multiest: assets present in both panels and passing the sanity
# ratio in both, at the middle penalty.
OWN = {r["id"]: r for r in own if r["zt"] == ZT and r.get("ratio", 0) >= 1}
R = [r for r in rows if r["zt"] == ZT and r["ratio"] >= 1 and r["id"] in OWN]
xo = np.array([OWN[r["id"]]["t_CW"] for r in R]); yo = np.array([OWN[r["id"]]["t_CC"] for r in R])
x = np.array([r["t_CW"] for r in R]); y = np.array([r["t_CC"] for r in R])
spy = [r for r in R if r["id"] == 84398]
n = len(R)
q = dict(cw_only=np.mean((x > CRIT) & (y <= CRIT)), both=np.mean((x > CRIT) & (y > CRIT)),
         cc_only=np.mean((x <= CRIT) & (y > CRIT)), neither=np.mean((x <= CRIT) & (y <= CRIT)))

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
BLUE, RED, SHADE = "#3B75AF", "#D1495B", "#FBE3D6"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(5.0, 4.3))
lo = min(x.min(), y.min(), xo.min(), yo.min()) - 0.3
hi = max(x.max(), y.max(), xo.max(), yo.max(), 5) + 0.3
ax.fill_between([CRIT, hi], lo, CRIT, color=SHADE, lw=0, zorder=0)
ax.plot([lo, hi], [lo, hi], color="0.7", lw=0.6, zorder=1)
ax.axvline(CRIT, color="0.35", lw=0.7, ls="--", zorder=1)
ax.axhline(CRIT, color="0.35", lw=0.7, ls="--", zorder=1)
ax.scatter(xo, yo, s=9, facecolor="none", edgecolor=BLUE, lw=0.6, alpha=0.8, zorder=2)
ax.scatter(x, y, s=9, color=RED, alpha=0.75, lw=0, zorder=3)
if spy:
    ax.scatter([spy[0]["t_CW"]], [spy[0]["t_CC"]], s=70, marker="*", color="k", zorder=4)
    ax.annotate("SPY", (spy[0]["t_CW"], spy[0]["t_CC"]), xytext=(7, 4),
                textcoords="offset points", fontsize=8, fontweight="bold")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel(r"CW statistic $t_{\mathrm{CW}}$")
ax.set_ylabel(r"corrected statistic $t_{\mathrm{CC}}$")
qo = np.mean((xo > CRIT) & (yo <= CRIT))
handles = [Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=BLUE,
                  label="own lags"),
           Line2D([], [], marker="o", ls="", color=RED,
                  label="five other estimators"),
           Patch(facecolor=SHADE, label="CW rejects, corrected does not")]
ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8, handletextpad=0.4)
fig.tight_layout()
out = os.path.join(HERE, "output", "rv_figures", "fig-rv-disagree")
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=130)
paper = os.path.abspath(os.path.join(HERE, "..", "Latex", "figures", "fig-rv-disagree.pdf"))
if os.path.isdir(os.path.dirname(paper)):
    fig.savefig(paper)
print(f"n={n}", {k: round(v, 3) for k, v in q.items()},
      "spy", (spy[0]["t_CW"], spy[0]["t_CC"]) if spy else None,
      "x range", round(x.min(), 2), round(x.max(), 2), "y range", round(y.min(), 2), round(y.max(), 2))
