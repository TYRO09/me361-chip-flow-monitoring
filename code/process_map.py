"""Process maps: predicted chip ejection angle (fitted model) and rake-face flow angle (theory) over feed x depth."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from physics_flow_angle import flow_angles

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = pd.read_csv(os.path.join(ROOT, "results", "run_summary.csv"))
X = np.c_[np.ones(len(S)), S.feed, S.ap]
b = np.linalg.lstsq(X, S.psi_mean, rcond=None)[0]
f = np.linspace(0.05, 0.25, 41); a = np.linspace(0.2, 1.2, 41); F, AP = np.meshgrid(f, a)
PSI = b[0] + b[1] * F + b[2] * AP
fc, ac = np.linspace(0.05, 0.25, 11), np.linspace(0.2, 1.2, 11)
ETA = np.array([[flow_angles(x, y)["eta_colwell"] for x in fc] for y in ac])

fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
for ax, (XX, YY, ZZ), title, cm in [(axs[0], (F, AP, PSI), f"Measured-data model: ψ = {b[0]:.1f} + {b[1]:.0f}·f + {b[2]:.1f}·ap", "viridis"),
                                     (axs[1], (*np.meshgrid(fc, ac), ETA), "Theory (Colwell, rε = 0.4 mm): rake-face flow angle η", "magma")]:
    cs = ax.contourf(XX, YY, ZZ, 14, cmap=cm); ax.contour(XX, YY, ZZ, 14, colors="k", linewidths=0.3)
    plt.colorbar(cs, ax=ax, label="degrees")
    ax.add_patch(plt.Rectangle((0.08, 0.3), 0.07, 0.6, fill=False, ec="w", lw=1.5, ls="--"))
    ax.scatter(S.feed, S.ap, c="w", ec="k", s=60, zorder=3)
    for r in S.itertuples():
        ax.annotate(f"{r.run}\n{r.psi_mean:.0f}°" if ax is axs[0] else r.run, (r.feed, r.ap), xytext=(5, 4), textcoords="offset points", fontsize=8, color="w")
    ax.set_xlabel("feed f (mm/rev)"); ax.set_ylabel("depth of cut ap (mm)"); ax.set_title(title, fontsize=10)
axs[0].text(0.155, 0.23, "outside the dashed box = extrapolation", color="w", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "results", "figures", "fig9_process_map.png"), dpi=150)
print("coeffs", b.round(2))
