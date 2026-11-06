"""Run-level chip ejection angle, physics comparison and parameter -> angle models.

Reads results/angles/R*_frames.csv (from measure_chip_angle.py) and session runs.csv.
Writes results/run_summary.csv, results/model_comparison.csv and figures in results/figures/.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from physics_flow_angle import flow_angles, DEFAULT_TOOL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

# Steady cutting: tool on the bar and chips streaming; excludes the entry and the exit,
# where the chip stops and the angle becomes meaningless (picked from the angle/activity traces).
STEADY = {"R1": (10, 34), "R2": (10, 20), "R3": (10, 30), "R4": (8, 18), "R5": (13, 28), "R6": (8, 17)}

runs = pd.read_csv(os.path.join(ROOT, "session_2026-11-04", "runs.csv"))
runs = runs.set_index("run_id")


def frame_table():
    out = []
    for r in STEADY:
        d = pd.read_csv(os.path.join(RES, "angles", f"{r}_frames.csv"))
        d = d[d.valid == 1].copy()
        t0, t1 = STEADY[r]
        d["steady"] = (d.t >= t0) & (d.t <= t1) & (d.activity > 0.25 * d.activity.median())
        out.append(d)
    return pd.concat(out, ignore_index=True)


def block_bootstrap(x, t, block=1.0, n=2000, seed=0):
    """CI for the mean resampling 1-s blocks (neighbouring frames are strongly correlated)."""
    rng = np.random.default_rng(seed)
    b = np.floor((t - t.min()) / block).astype(int)
    groups = [x[b == k] for k in np.unique(b)]
    means = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


def loo(Xf, y):
    """Leave-one-run-out predictions for an ordinary least squares model (X includes intercept)."""
    pred = np.zeros_like(y)
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        beta, *_ = np.linalg.lstsq(Xf[m], y[m], rcond=None)
        pred[i] = Xf[i] @ beta
    return pred


def main():
    fr = frame_table()
    fr.to_csv(os.path.join(RES, "frames_all.csv"), index=False)
    rows = []
    for r, d in fr[fr.steady].groupby("run"):
        lo, hi = block_bootstrap(d.psi_peak.values, d.t.values)
        f, ap = runs.loc[r, "feed_mm_per_rev"], runs.loc[r, "depth_of_cut_mm"]
        th = flow_angles(f, ap, **DEFAULT_TOOL)
        th8 = flow_angles(f, ap, r_eps=0.8, kappa_r=95, eps_r=80)
        rows.append(dict(run=r, feed=f, ap=ap, dia_after=runs.loc[r, "dia_after_mm"], rpm=runs.loc[r, "spindle_rpm_calc"],
                         n_frames=len(d), psi_mean=d.psi_peak.mean(), psi_median=d.psi_peak.median(),
                         psi_ci_lo=lo, psi_ci_hi=hi, psi_sd=d.psi_peak.std(),
                         eta_colwell=th["eta_colwell"], eta_ymo=th["eta_ymo"], eta_colwell_r08=th8["eta_colwell"],
                         h_eq=th["h_eq"]))
    s = pd.DataFrame(rows)
    s.round(3).to_csv(os.path.join(RES, "run_summary.csv"), index=False)
    print(s.round(2).to_string(index=False))

    y = s.psi_mean.values
    one = np.ones(len(y))
    models = {
        "M0 constant (mean of other runs)": np.c_[one],
        "M1 linear: feed + depth": np.c_[one, s.feed, s.ap],
        "M2 physics: Colwell eta (r=0.4)": np.c_[one, s.eta_colwell],
        "M3 physics: YMO eta (r=0.4)": np.c_[one, s.eta_ymo],
        "M4 physics: Colwell eta (r=0.8)": np.c_[one, s.eta_colwell_r08],
        "M5 depth only": np.c_[one, s.ap],
        "M6 feed only": np.c_[one, s.feed],
        "M7 hybrid: Colwell eta + feed": np.c_[one, s.eta_colwell, s.feed],
    }
    comp = []
    preds = {}
    for name, X in models.items():
        p = loo(X, y); preds[name] = p
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        fit = X @ beta
        r2 = 1 - ((y - fit) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        comp.append(dict(model=name, loo_rmse_deg=np.sqrt(((y - p) ** 2).mean()), loo_mae_deg=np.abs(y - p).mean(),
                         train_r2=r2, coefficients=np.round(beta, 3).tolist()))
    comp = pd.DataFrame(comp)
    comp.round(3).to_csv(os.path.join(RES, "model_comparison.csv"), index=False)
    print(comp.round(2).to_string(index=False))
    r_c = np.corrcoef(s.eta_colwell, y)[0, 1]
    print(f"correlation psi vs Colwell eta: r = {r_c:.2f}")

    # ---------- figures ----------
    colors = dict(zip(STEADY, plt.cm.tab10.colors))
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.5), sharey=True)
    for ax, r in zip(axs.flat, STEADY):
        d = fr[fr.run == r]
        ax.plot(d.t, d.psi_peak, ".", ms=2, color="0.7", label="every frame")
        sm = d.set_index("t").psi_peak.rolling(15, center=True, min_periods=5).median()
        ax.plot(sm.index, sm.values, color=colors[r], lw=2, label="0.5 s rolling median")
        ax.axvspan(*STEADY[r], color=colors[r], alpha=0.12, label="steady cut used")
        f, ap = runs.loc[r, "feed_mm_per_rev"], runs.loc[r, "depth_of_cut_mm"]
        ax.set_title(f"{r}: f={f} mm/rev, ap={ap} mm"); ax.set_xlabel("time in clip (s)")
        ax.set_ylim(-20, 140); ax.grid(alpha=0.3)
    axs[0, 0].set_ylabel("ejection angle ψ (deg)"); axs[1, 0].set_ylabel("ejection angle ψ (deg)")
    axs[0, 0].legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_angle_timeseries.png"), dpi=150); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axs[0]
    for ap, mk in zip(sorted(s.ap.unique()), "osD"):
        q = s[s.ap == ap]
        ax.errorbar(q.feed, q.psi_mean, yerr=[q.psi_mean - q.psi_ci_lo, q.psi_ci_hi - q.psi_mean],
                    marker=mk, ms=8, capsize=4, lw=1.5, label=f"ap = {ap} mm")
        for _, row in q.iterrows():
            ax.annotate(row.run, (row.feed, row.psi_mean), textcoords="offset points", xytext=(8, -3), fontsize=9)
    ax.set_xlabel("feed f (mm/rev)"); ax.set_ylabel("measured ejection angle ψ (deg)")
    ax.set_title("Measured angle per run (mean, 95% CI)"); ax.grid(alpha=0.3); ax.legend()
    ax.set_xlim(0.05, 0.18)
    ax = axs[1]
    ax.errorbar(s.eta_colwell, s.psi_mean, yerr=[s.psi_mean - s.psi_ci_lo, s.psi_ci_hi - s.psi_mean], fmt="o", capsize=4, color="C3")
    for _, row in s.iterrows():
        ax.annotate(row.run, (row.eta_colwell, row.psi_mean), textcoords="offset points", xytext=(6, 4), fontsize=9)
    b = np.polyfit(s.eta_colwell, s.psi_mean, 1); xx = np.linspace(s.eta_colwell.min() - 3, s.eta_colwell.max() + 3, 10)
    ax.plot(xx, np.polyval(b, xx), "--", color="C3", alpha=0.6, label=f"fit: ψ = {b[1]:.1f} {'+' if b[0] >= 0 else '−'} {abs(b[0]):.2f}·η  (r = {r_c:.2f})")
    ax.set_xlabel("theoretical rake-face flow angle η, Colwell (deg)"); ax.set_ylabel("measured ejection angle ψ (deg)")
    ax.set_title("Does theory explain the measurement?"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_runs_and_theory.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    lim = (y.min() - 8, y.max() + 8)
    for name, mk in zip(["M1 linear: feed + depth", "M2 physics: Colwell eta (r=0.4)", "M0 constant (mean of other runs)"], "os^"):
        ax.plot(y, preds[name], mk, ms=8, label=name, alpha=0.8)
    ax.plot(lim, lim, "k--", lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("measured ψ (deg)"); ax.set_ylabel("predicted ψ, run left out (deg)")
    ax.set_title("Leave-one-run-out predictions"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_loo_predictions.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
