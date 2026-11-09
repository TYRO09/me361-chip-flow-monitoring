"""Numbers for the report's web-app section.

1. Replay regression: every run through the live engine vs the offline results (median angle, valid frames, alarm).
2. Calibration sensitivity: the tool tip is mis-clicked by 5 and 10 px along and across the feed direction; how far
   does the measured median angle move? (first 200 frames after calibration, runs R2, R3, R6)
Writes results/live_engine_regression.csv and results/tip_sensitivity.csv.
"""
import os
import sys
import itertools

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "code"))
from app import replay  # noqa: E402
from app.engine import ChipEngine, Calibration  # noqa: E402

RES = os.path.join(ROOT, "results")


def run_engine(run, cal, max_frames=None):
    eng, out, started, ended = ChipEngine(cal), [], False, False
    for bgr, t in itertools.islice(replay.frames(run), max_frames):
        if not started and t >= replay.CUT[run][0]:
            eng.reset_cut(); started = True
        if not ended and t >= replay.CUT[run][1]:
            eng.end_cut(); ended = True
        s = eng.push(bgr, t)
        if s is not None:
            out.append(s)
    return pd.DataFrame(out)


def regression():
    rows = []
    for r in replay.CUT:
        st = run_engine(r, replay.calibration_for(r))
        off = pd.read_csv(os.path.join(RES, "angles", f"{r}_frames.csv")); off = off[off.valid == 1]
        al = st[st.alarm].alarm_t
        rows.append(dict(run=r, engine_median=st[st.valid].psi_raw.median(), offline_median=off.psi_peak.median(),
                         engine_valid=int(st.valid.sum()), offline_valid=len(off), alarm_t=None if al.empty else round(al.iloc[0], 1)))
        print(rows[-1], flush=True)
    d = pd.DataFrame(rows); d["diff_deg"] = d.engine_median - d.offline_median
    d.round(2).to_csv(os.path.join(RES, "live_engine_regression.csv"), index=False)


def sensitivity():
    rows = []
    for r in ("R2", "R3", "R6"):
        base = replay.calibration_for(r)
        f, o = np.array(base.feed_dir), np.array(base.out_dir)
        ref = run_engine(r, base, 200)
        m0 = ref[ref.valid].psi_raw.median()
        for axis, vec in (("along feed", f), ("across feed", o)):
            for px in (-10, -5, 5, 10):
                d = px * vec
                cal = Calibration(tip=tuple(np.array(base.tip) + d), feed_dir=base.feed_dir, out_dir=base.out_dir,
                                  ref_frame=base.ref_frame, start_tip=tuple(np.array(base.start_tip) + d))
                st = run_engine(r, cal, 200)
                m = st[st.valid].psi_raw.median() if st.valid.any() else np.nan
                rows.append(dict(run=r, shift=axis, px=px, median=m, change_deg=m - m0, valid_frames=int(st.valid.sum())))
                print(rows[-1], flush=True)
    pd.DataFrame(rows).round(2).to_csv(os.path.join(RES, "tip_sensitivity.csv"), index=False)


if __name__ == "__main__":
    which = sys.argv[1:] or ["regression", "sensitivity"]
    if "regression" in which:
        regression()
    if "sensitivity" in which:
        sensitivity()
