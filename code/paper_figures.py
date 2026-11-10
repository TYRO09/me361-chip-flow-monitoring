"""Figures for the paper version of the report (results/paper/)."""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from physics_flow_angle import flow_angles

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results"); OUT = os.path.join(RES, "figures")
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight"})
C = {"meas": "#1b7837", "theory": "#2166ac", "eject": "#d6a100", "grey": "#888888", "bad": "#b2182b"}


def rake_vs_theory():
    rf = pd.read_csv(os.path.join(RES, "rake_face_flow.csv"))
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = np.arange(len(rf))
    for re, mk_ in ((0.4, "v"), (0.8, "_"), (1.2, "^")):
        th = [180 - flow_angles(f, a, r_eps=re)["eta_colwell"] for f, a in zip(rf.feed, rf.ap)]
        ax.scatter(x, th, marker=mk_, s=28 if mk_ == "_" else 12, color=C["theory"], linewidths=1.5,
                   label=f"theory (Colwell), $r_\\varepsilon$={re} mm")
    ax.scatter(x, rf.psi_eject_med, color=C["eject"], marker="s", s=18, label="ejection $\\psi$ (in air)", zorder=3)
    det = rf.detected / rf.windows
    ok = det >= 0.4
    ax.scatter(x[ok], rf.psi_face_med[ok], color=C["meas"], s=30, label="rake face, measured", zorder=4)
    ax.scatter(x[~ok & rf.psi_face_med.notna()], rf.psi_face_med[~ok & rf.psi_face_med.notna()], facecolor="none",
               edgecolor=C["meas"], s=30, label="rake face, <40% of windows", zorder=4)
    ax.set_xticks(x, [f"{r}\n{f:.2f}/{a:.1f}" for r, f, a in zip(rf.run, rf.feed, rf.ap)])
    ax.set_ylabel("direction in top view (deg)\n0 = feed, 90 = outward")
    ax.axhline(90, color=C["grey"], lw=0.5)
    ax.set_ylim(20, 175)
    ax.legend(fontsize=5.5, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.savefig(os.path.join(OUT, "fig13_rake_vs_theory.pdf"))


def augmentation():
    d = pd.read_csv(os.path.join(RES, "augment_leakage.csv"))
    d = d[d.model != "random forest f+ap"] if False else d
    fig, ax = plt.subplots(figsize=(3.4, 3.3))
    labels, loro, rnd = [], [], []
    for _, r in d.iterrows():
        labels.append(f"{r.model.replace(' f+ap', '').replace('+time', ' + time')}\n{r.data} (n={r.samples})")
        loro.append(r.loro_run_rmse); rnd.append(r.random_split_run_rmse)
    y = np.arange(len(d))[::-1]
    ax.barh(y + 0.2, loro, 0.38, color=C["theory"], label="leave-one-run-out (honest)")
    ax.barh(y - 0.2, np.nan_to_num(rnd), 0.38, color=C["bad"], alpha=0.7, label="random split (leaky)")
    ax.axvline(d.baseline_same_target.iloc[0], color=C["grey"], ls="--", lw=0.8)
    ax.text(d.baseline_same_target.iloc[0] + 0.1, y.max() + 0.6, "predict the mean", fontsize=5.5, color=C["grey"])
    ax.set_yticks(y, labels, fontsize=6)
    ax.set_xlabel("run-level RMSE of chip angle (deg)")
    for yy, v in zip(y - 0.2, rnd):
        if np.isfinite(v):
            ax.text(v + 0.08, yy, f"{v:.1f}", va="center", fontsize=5, color=C["bad"])
    ax.legend(fontsize=5.5, frameon=False, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2)
    fig.savefig(os.path.join(OUT, "fig14_augmentation_leakage.pdf"))


def cnn_v3():
    p = os.path.join(RES, "cnn_v3_metrics.json")
    if not os.path.exists(p):
        return
    m = json.load(open(p))
    order = [v for v in ("hist_ridge", "motion", "motion_rot", "motion_rotm", "clip_rot") if v in m]
    names = {"hist_ridge": "histogram + ridge", "motion": "motion CNN (v2)", "motion_rot": "motion CNN\n+ rotate all",
             "motion_rotm": "motion CNN\n+ rotate motion",
             "clip_rot": "5-frame clip\n+ rotate all"}
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axs[0]
    lim = (30, 75)
    ax.plot(lim, lim, color=C["grey"], lw=0.6)
    mk = {"hist_ridge": "D", "motion": "o", "motion_rot": "s", "motion_rotm": "P", "clip_rot": "^"}
    for v in order:
        f = pd.DataFrame(m[v]["folds"])
        ax.scatter(f.true, f.pred, marker=mk[v], s=22, label=f"{names[v].replace(chr(10), ' ')} (slope {m[v]['slope']:.2f})")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel("measured run mean (deg)"); ax.set_ylabel("predicted, run held out (deg)")
    ax.legend(fontsize=5.5, frameon=False)
    ax = axs[1]
    x = np.arange(len(order))
    ax.bar(x - 0.2, [m[v]["run_rmse"] for v in order], 0.4, color=C["theory"], label="run RMSE")
    ax.bar(x + 0.2, [m[v]["frame_mae"] for v in order], 0.4, color=C["eject"], label="frame MAE")
    ax.axhline(m[order[0]]["run_rmse_base"], color=C["grey"], ls="--", lw=0.8)
    short = {"hist_ridge": "hist.\n+ ridge", "motion": "motion\n(v2)", "motion_rot": "motion\nrot. all",
             "motion_rotm": "motion\nrot. motion", "clip_rot": "clip\nrot. all"}
    ax.set_xticks(x, [short[v] for v in order], fontsize=6)
    ax.text(x[-1] + 0.45, m[order[0]]["run_rmse_base"] + 0.15, "predict the mean", fontsize=5.5, color=C["grey"], ha="right")
    ax.set_ylabel("error (deg)"); ax.set_ylim(0, 10); ax.legend(fontsize=6, frameon=False, loc="upper left", ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig15_cnn_v3.pdf"))


def old_clips():
    p = os.path.join(RES, "old_clips.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    clips = list(d["clip"].unique())
    fig, axs = plt.subplots(1, len(clips), figsize=(7.0, 1.9), sharey=False)
    for ax, c in zip(np.atleast_1d(axs), clips):
        v = d[d["clip"] == c]
        ax.plot(v.t, v.psi_peak, ".", color=C["grey"], ms=2, label="measured, frame")
        ax.plot(v.t, v.psi_smooth, color="k", lw=1, label="measured, 0.5-s median")
        ax.plot(v.t, v.psi_cnn.rolling(15, center=True, min_periods=5).median(), color=C["theory"], lw=1,
                label="CNN trained on Nov. runs")
        ax.plot(v.t, v.psi_hist.rolling(15, center=True, min_periods=5).median(), color=C["eject"], lw=1,
                label="histogram + ridge")
        sm = pd.read_csv(os.path.join(RES, "old_clips_summary.csv")).set_index("clip")
        ttl = c if sm.loc[c, "usable"] else f"{c}: almost no chip motion (rejected)"
        ax.set_title(ttl, fontsize=7); ax.set_xlabel("time in clip (s)")
    np.atleast_1d(axs)[0].set_ylabel("$\\psi$ (deg)")
    np.atleast_1d(axs)[0].legend(fontsize=5.5, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig16_old_clips.pdf"))


if __name__ == "__main__":
    for f in sys.argv[1:] or ["rake_vs_theory", "augmentation", "cnn_v3", "old_clips"]:
        globals()[f]()
