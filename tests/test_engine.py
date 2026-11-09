"""Task 2: causal chip engine + replay must reproduce the offline results."""
import os
import sys
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, ROOT)

from app import replay  # noqa: E402  (RED until app/ exists)
from app.engine import ChipEngine  # noqa: E402

RUNS = ["R1", "R2", "R3", "R4", "R5", "R6"]
OFFLINE_ALARM_R2 = 20.1          # causal alarm time of the validated demo (s)


@lru_cache(maxsize=None)
def replayed(run):
    """Feed one recorded run through the engine; return list of state dicts."""
    eng = ChipEngine(replay.calibration_for(run))
    out, started, ended = [], False, False
    for bgr, t in replay.frames(run):
        if not started and t >= replay.CUT[run][0]:
            eng.reset_cut(); started = True
        if not ended and t >= replay.CUT[run][1]:
            eng.end_cut(); ended = True
        s = eng.push(bgr, t)
        if s is not None:
            out.append(s)
    return out


@pytest.mark.parametrize("run", RUNS)
def test_replay_median_matches_offline(run):
    states = replayed(run)
    mine = np.array([s["psi_raw"] for s in states if s["valid"]])
    off = pd.read_csv(os.path.join(ROOT, "results", "angles", f"{run}_frames.csv"))
    off = off[off.valid == 1].psi_peak.values
    assert len(mine) > 0.8 * len(off), (len(mine), len(off))
    assert abs(np.median(mine) - np.median(off)) <= 1.0, (np.median(mine), np.median(off))


def test_r2_alarm_only():
    alarms = {r: next((s["alarm_t"] for s in replayed(r) if s["alarm"]), None) for r in RUNS}
    assert alarms["R2"] is not None and abs(alarms["R2"] - OFFLINE_ALARM_R2) <= 1.0, alarms
    assert all(alarms[r] is None for r in RUNS if r != "R2"), alarms


def _short_run(n=60, run="R2"):
    eng = ChipEngine(replay.calibration_for(run))
    frames = []
    for i, (bgr, t) in enumerate(replay.frames(run)):
        frames.append((bgr, t))
        if i >= n:
            break
    return eng, frames


def test_tracking_lost_does_not_crash_and_recovers():
    eng, frames = _short_run(40)
    states = []
    for bgr, t in frames[:15]:
        states.append(eng.push(bgr, t))
    for k in range(6):                                   # tool vanishes: black frames
        states.append(eng.push(np.zeros_like(frames[0][0]), frames[15][1] + k / 30))
    for bgr, t in frames[21:40]:                         # and comes back
        states.append(eng.push(bgr, t))
    got = [s for s in states if s is not None]
    assert any(s["tracking_lost"] for s in got)
    assert all(not np.isfinite(s["psi"]) or s["valid"] for s in got if s["tracking_lost"])
    assert not got[-1]["tracking_lost"], "engine did not recover when the tool re-appeared"


def test_irregular_timing_is_flagged_not_crashing():
    eng, frames = _short_run(60)
    kept = [(b, t) for i, (b, t) in enumerate(frames) if not (20 <= i < 30)]   # a 0.33-s hole
    rng = np.random.default_rng(0)
    states = [eng.push(b, t + rng.uniform(-0.02, 0.02)) for b, t in kept]
    got = [s for s in states if s is not None]
    gap = [s for s in got if s.get("gap")]
    assert gap, "a 0.33 s hole must be reported"
    assert all(not s["valid"] for s in gap)
    assert any(s["valid"] for s in got)


def test_orientation_change_flagged():
    eng, frames = _short_run(10)
    states = [eng.push(np.ascontiguousarray(np.rot90(b)), t) for b, t in frames[:6]]
    got = [s for s in states if s is not None]
    assert got and all(s["orientation_changed"] and not s["valid"] for s in got)
