"""Chip build-up / entanglement indicator from the top-view video features.

Idea: a continuous chip that is ejected cleanly gives a narrow, steady spray. When chips start to
snarl around the tool or workpiece, the moving area grows (activity), the spray widens (spread) and
static chip mass appears in front of the tool (clutter). Each feature is turned into a robust z-score
against the first 4 s of the same cut (so no training and no other run is needed); the score is the
mean of the three; an alarm is raised when the 1-s rolling score stays above 3 for 1 s.
"""
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
VID = os.path.join(ROOT, "session_2026-11-04", "videos_top")
CUT = {"R1": (10, 36), "R2": (10, 22), "R3": (8, 33), "R4": (8, 20), "R5": (12, 30), "R6": (8, 18)}
BASE_S, THR, HOLD_S = 4.0, 3.0, 1.0


def score_run(r):
    d = pd.read_csv(os.path.join(RES, "angles", f"{r}_frames.csv"))
    t0, t1 = CUT[r]
    d = d[(d.valid == 1) & (d.t >= t0) & (d.t <= t1)].copy()
    d["log_act"] = np.log(d.activity + 1)
    base = d[d.t < t0 + BASE_S]
    zs = []
    for c in ["log_act", "spread", "clutter"]:
        med = base[c].median(); mad = 1.4826 * (base[c] - med).abs().median() + 1e-9
        d["z_" + c] = (d[c] - med) / mad
        zs.append("z_" + c)
    d["score"] = d[zs].mean(axis=1)
    d["score_s"] = d.score.rolling(30, center=True, min_periods=10).median()
    above = d.score_s > THR
    alarm_t = None
    run_len = 0
    for t, a in zip(d.t, above):
        run_len = run_len + 1 if a else 0
        if run_len >= 30 * HOLD_S:
            alarm_t = t - HOLD_S; break
    return d, alarm_t


def frame_at(r, t, size=420):
    f = [x for x in os.listdir(VID) if x.startswith(r + "_")][0]
    cap = cv2.VideoCapture(os.path.join(VID, f)); cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, im = cap.read(); cap.release()
    h, w = im.shape[:2]; s = size / h
    return cv2.cvtColor(cv2.resize(im, (int(w * s), size)), cv2.COLOR_BGR2RGB)


def main():
    out, rows = {}, []
    for r in CUT:
        d, a = score_run(r)
        out[r] = (d, a)
        rows.append(dict(run=r, alarm_at_s=None if a is None else round(a, 2), max_score=round(d.score_s.max(), 2)))
        d[["run", "t", "z_log_act", "z_spread", "z_clutter", "score", "score_s"]].round(3).to_csv(
            os.path.join(RES, "angles", f"{r}_entangle.csv"), index=False)
    tab = pd.DataFrame(rows); tab.to_csv(os.path.join(RES, "entangle_alarms.csv"), index=False)
    print(tab.to_string(index=False))

    fig = plt.figure(figsize=(12, 7.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])
    ax = fig.add_subplot(gs[0, :])
    for k, r in enumerate(CUT):
        d, a = out[r]
        ax.plot(d.t - CUT[r][0], d.score_s, lw=1.6 if r == "R2" else 1, color=f"C{k}", label=r + (f"  alarm at {a:.1f} s" if a else ""))
        if a: ax.axvline(a - CUT[r][0], color=f"C{k}", ls=":", lw=1)
    ax.axhline(THR, color="k", ls="--", lw=1, label="alarm threshold")
    ax.set_xlabel("time since start of cut (s)"); ax.set_ylabel("build-up score (robust z)")
    ax.set_title("Chip build-up / entanglement indicator (no training: each run is compared with its own first 4 s)")
    ax.legend(fontsize=8, ncol=4); ax.grid(alpha=0.3)
    d2, a2 = out["R2"]
    for j, (t, lab) in enumerate([(12.0, "R2 at 12 s: clean chip stream"), (a2 + 1.5 if a2 else 19.0, "R2 after alarm: chips snarl at the tool"), ]):
        ax = fig.add_subplot(gs[1, j]); ax.imshow(frame_at("R2", t)); ax.set_title(lab, fontsize=9); ax.axis("off")
    ax = fig.add_subplot(gs[1, 2])
    for c, lab in [("z_log_act", "moving area"), ("z_spread", "spray width"), ("z_clutter", "static chips")]:
        ax.plot(d2.t, d2[c].rolling(30, center=True, min_periods=10).median(), label=lab)
    ax.set_title("R2: which signals rise", fontsize=9); ax.set_xlabel("time in clip (s)"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig8_entanglement.png"), dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
