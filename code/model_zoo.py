"""Variations: measurement sensitivity, a zoo of parameter -> angle models, a curl-physics hybrid,
frame-level models with time-in-cut, and inverse models (cutting parameters from video/audio).

Run after: measure_chip_angle.py (base + variants), analyze.py, audio_sensor.py.
"""
import json, os, sys, warnings
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from analyze import STEADY
from physics_flow_angle import flow_angles

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
runs = pd.read_csv(os.path.join(ROOT, "session_2026-11-04", "runs.csv")).set_index("run_id")
RUN_IDS = list(STEADY)


def steady_frames(folder):
    out = []
    for r in RUN_IDS:
        d = pd.read_csv(os.path.join(folder, f"{r}_frames.csv"))
        d = d[d.valid == 1]
        t0, t1 = STEADY[r]
        d = d[(d.t >= t0) & (d.t <= t1) & (d.activity > 0.25 * d.activity.median())].copy()
        d["t_in_cut"] = (d.t - t0) / (t1 - t0)
        out.append(d)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- 1. sensitivity
def sensitivity():
    variants = {"base (r=220 px, thr=18)": os.path.join(RES, "angles")}
    for tag, lab in [("rout150", "annulus r=150 px"), ("rout300", "annulus r=300 px"), ("thr12", "motion thr=12"), ("thr28", "motion thr=28")]:
        p = os.path.join(RES, "angles_variants", tag)
        if os.path.isdir(p):
            variants[lab] = p
    tab = {}
    for lab, p in variants.items():
        d = steady_frames(p)
        tab[lab] = d.groupby("run").psi_peak.mean()
    base = steady_frames(os.path.join(RES, "angles"))
    tab["estimator: weighted mean"] = base.groupby("run").psi_mean.mean()
    tab["estimator: streak orientation"] = base.groupby("run").psi_streak.mean()
    t = pd.DataFrame(tab).loc[RUN_IDS]
    t.round(1).to_csv(os.path.join(RES, "sensitivity.csv"))
    feed_effect = pd.DataFrame({c: [t.loc["R2", c] - t.loc["R1", c], t.loc["R4", c] - t.loc["R3", c], t.loc["R6", c] - t.loc["R5", c]] for c in t},
                               index=["R2-R1", "R4-R3", "R6-R5"])
    feed_effect.round(1).to_csv(os.path.join(RES, "sensitivity_feed_effect.csv"))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    tp = t.drop(columns=["estimator: streak orientation"])  # streak estimate is ~noise (SD 40-50 deg), shown in text only
    x = np.arange(len(RUN_IDS)); w = 0.8 / tp.shape[1]
    for i, c in enumerate(tp.columns):
        ax.bar(x + (i - tp.shape[1] / 2 + 0.5) * w, tp[c], w, label=c)
    ax.set_xticks(x); ax.set_xticklabels([f"{r}\nf={runs.loc[r,'feed_mm_per_rev']} ap={runs.loc[r,'depth_of_cut_mm']}" for r in RUN_IDS], fontsize=8)
    ax.set_ylabel("mean ψ in steady cut (deg)"); ax.set_ylim(20, 80); ax.grid(alpha=0.3, axis="y")
    ax.set_title("Measurement robustness: same videos, different settings / estimators"); ax.legend(fontsize=7, ncol=4, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig6_sensitivity.png"), dpi=150); plt.close(fig)
    print("\nSENSITIVITY (run means)\n", t.round(1).to_string(), "\nfeed effect\n", feed_effect.round(1).to_string())
    return t


# ---------------------------------------------------------------- 2. run-level features
def run_features():
    fr = steady_frames(os.path.join(RES, "angles"))
    au = pd.read_csv(os.path.join(RES, "audio_sensor.csv")).set_index("run")
    rows = []
    for r in RUN_IDS:
        d = fr[fr.run == r]
        f, ap = runs.loc[r, "feed_mm_per_rev"], runs.loc[r, "depth_of_cut_mm"]
        th = flow_angles(f, ap)
        rows.append(dict(run=r, f=f, ap=ap, D=runs.loc[r, "dia_after_mm"], N=runs.loc[r, "spindle_rpm_calc"],
                         area=f * ap, h_eq=th["h_eq"], eta=th["eta_colwell"], eta_ymo=th["eta_ymo"],
                         psi=d.psi_peak.mean(), psi_streak=d.psi_streak.mean(), spread=d.spread.mean(),
                         reach=d.reach.mean(), coherence=d.coherence.mean(), clutter=d.clutter.mean(),
                         activity=d.activity.median(), noise_db=au.loc[r, "cut_noise_db"]))
    return pd.DataFrame(rows).set_index("run")


# ---------------------------------------------------------------- 3. model zoo (run level, LORO)
def curl_model(params, eta_deg, h):
    a, b, off = params
    phi = np.radians(a + b * h * 100)               # curl angle grows/shrinks with chip thickness
    eta = np.radians(eta_deg)
    return np.degrees(np.arctan2(np.sin(eta), -np.cos(eta) * np.cos(phi))) + off


def fit_curl(eta, h, y):
    res = least_squares(lambda p: curl_model(p, eta, h) - y, x0=[120, 0, 0], bounds=([0, -50, -40], [180, 50, 40]))
    return res.x


def loro(make, X, y):
    p = np.zeros(len(y))
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        mdl = make(); mdl.fit(X[m], y[m]); p[i] = mdl.predict(X[i:i + 1])[0]
    return p


class Const:
    def fit(self, X, y): self.c = y.mean(); return self
    def predict(self, X): return np.full(len(X), self.c)


def model_zoo(F):
    y = F.psi.values
    fa = F[["f", "ap"]].values
    zoo = {
        "Constant (baseline)": (Const, fa),
        "Linear: f + ap": (lambda: make_pipeline(StandardScaler(), Ridge(alpha=1e-6)), fa),
        "Ridge: f, ap, f·ap": (lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)), np.c_[fa, fa[:, 0] * fa[:, 1]]),
        "Quadratic ridge": (lambda: make_pipeline(StandardScaler(), PolynomialFeatures(2), Ridge(alpha=3.0)), fa),
        "Gaussian process (RBF)": (lambda: make_pipeline(StandardScaler(), GaussianProcessRegressor(
            ConstantKernel(50.0) * RBF([1.0, 1.0]) + WhiteKernel(5.0), normalize_y=True)), fa),
        "k-nearest neighbours (k=2)": (lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(2)), fa),
        "Random forest": (lambda: RandomForestRegressor(200, min_samples_leaf=1, random_state=0), fa),
        "Linear: chip area + h_eq": (lambda: make_pipeline(StandardScaler(), Ridge(alpha=1e-6)), F[["area", "h_eq"]].values),
        "Linear: f + ap + rpm (confounded)": (lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)), F[["f", "ap", "N"]].values),
        "Physics: Colwell η only": (lambda: make_pipeline(StandardScaler(), Ridge(alpha=1e-6)), F[["eta"]].values),
    }
    rows, preds = [], {}
    for name, (mk, X) in zoo.items():
        p = loro(mk, X, y); preds[name] = p
        rows.append(dict(model=name, loo_rmse=np.sqrt(((p - y) ** 2).mean()), loo_mae=np.abs(p - y).mean()))
    # curl-physics hybrid: rake-face flow (Colwell) + chip up-curl that depends on chip thickness
    p = np.zeros(len(y))
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        prm = fit_curl(F.eta.values[m], F.h_eq.values[m], y[m])
        p[i] = curl_model(prm, F.eta.values[i:i + 1], F.h_eq.values[i:i + 1])[0]
    preds["Physics hybrid: Colwell η + curl(h)"] = p
    rows.append(dict(model="Physics hybrid: Colwell η + curl(h)", loo_rmse=np.sqrt(((p - y) ** 2).mean()), loo_mae=np.abs(p - y).mean()))
    full = fit_curl(F.eta.values, F.h_eq.values, y)
    z = pd.DataFrame(rows).sort_values("loo_rmse")
    z.round(2).to_csv(os.path.join(RES, "model_zoo.csv"), index=False)
    print("\nMODEL ZOO (leave-one-run-out)\n", z.round(2).to_string(index=False))
    print("curl hybrid params on all runs (phi = a + b*h*100, offset):", np.round(full, 2))
    # infeasibility check for the pure curl model (no offset): needs tan(psi) >= tan(eta)
    feas = np.tan(np.radians(F.psi)) >= np.tan(np.radians(F.eta))
    print("pure geometric curl feasible per run:", dict(zip(F.index, feas)))

    fig, ax = plt.subplots(figsize=(9, 4.4))
    c = ["0.6" if n.startswith("Constant") else ("C3" if n.startswith("Physics") else ("C1" if "confounded" in n else "C0")) for n in z.model]
    ax.barh(z.model, z.loo_rmse, color=c)
    base = z[z.model.str.startswith("Constant")].loo_rmse.iloc[0]
    ax.axvline(base, color="k", ls="--", lw=1, label="baseline (predict the average)")
    ax.invert_yaxis(); ax.set_xlabel("leave-one-run-out RMSE (deg) — lower is better")
    ax.set_title("Parameter → chip angle: 11 model variants"); ax.grid(alpha=0.3, axis="x"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig7_model_zoo.png"), dpi=150); plt.close(fig)
    return z, full


# ---------------------------------------------------------------- 4. frame-level models
def frame_level():
    fr = steady_frames(os.path.join(RES, "angles"))
    fr["f"] = fr.run.map(runs.feed_mm_per_rev); fr["ap"] = fr.run.map(runs.depth_of_cut_mm)
    fr["psi_s"] = fr.groupby("run").psi_peak.transform(lambda s: s.rolling(15, center=True, min_periods=5).median())
    res = {}
    for name, cols in [("f, ap", ["f", "ap"]), ("f, ap, time-in-cut", ["f", "ap", "t_in_cut"])]:
        errs = []
        for r in RUN_IDS:
            tr, te = fr[fr.run != r], fr[fr.run == r]
            m = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, random_state=0)
            m.fit(tr[cols], tr.psi_s); errs.append(np.abs(m.predict(te[cols]) - te.psi_s).values)
        res[name] = float(np.concatenate(errs).mean())
    within = {}
    for r in RUN_IDS:  # how much of the within-run variation is a drift with time?
        d = fr[fr.run == r].dropna(subset=["psi_s"])
        b = np.polyfit(d.t_in_cut, d.psi_s, 1); within[r] = round(float(b[0]), 1)
    print("\nFRAME-LEVEL MAE (GBM, LORO):", {k: round(v, 2) for k, v in res.items()})
    print("drift over the cut (deg from start to end of steady window):", within)
    return res, within


# ---------------------------------------------------------------- 5. inverse models
def inverse(F):
    out = {}
    yc = (F.f > 0.1).astype(int).values
    for name, cols in [("video: ψ, spread, reach", ["psi", "spread", "reach"]),
                       ("video: activity, clutter", ["activity", "clutter"]),
                       ("audio: cutting noise level", ["noise_db"])]:
        X = F[cols].values; ok = 0
        for i in range(len(yc)):
            m = np.arange(len(yc)) != i
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0)).fit(X[m], yc[m])
            ok += int(clf.predict(X[i:i + 1])[0] == yc[i])
        out[f"feed class from {name}"] = f"{ok}/6"
    au = pd.read_csv(os.path.join(RES, "audio_sensor.csv")).set_index("run")
    out["feed from audio (cut time + tachometer)"] = f"6/6 correct, mean |error| {au.feed_error_pct.abs().mean():.1f}%"
    # depth of cut from audio level once feed and speed are known (physics: noise ~ removal rate)
    X = np.c_[F.noise_db.values, np.log(F.f.values)]
    p = loro(lambda: Ridge(alpha=1e-6), X, F.ap.values)
    out["depth from audio level + known feed (LORO MAE, mm)"] = round(float(np.abs(p - F.ap.values).mean()), 3)
    out["depth predictions"] = dict(zip(F.index, np.round(p, 2)))
    print("\nINVERSE MODELS\n", json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    sens = sensitivity()
    F = run_features()
    F.round(4).to_csv(os.path.join(RES, "run_features.csv"))
    print("\nRUN FEATURES\n", F.round(3).to_string())
    z, curl = model_zoo(F)
    fl = frame_level()
    inv = inverse(F)
    json.dump(dict(curl_params=list(map(float, curl)), frame_level=fl[0], drift=fl[1], inverse=inv),
              open(os.path.join(RES, "model_zoo_extra.json"), "w"), indent=1, default=str)
