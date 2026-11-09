"""Task 3: streaming audio engine must agree with the batch audio sensor."""
import os
import sys

import numpy as np
from scipy import signal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, ROOT)

import audio_sensor as A  # noqa: E402
from app.audio_engine import AudioEngine  # noqa: E402  (RED until it exists)

CHUNK = 1024


def stream(x, eng, idle_at=3.0, every=0.05):
    """Push x in chunks; calibrate idle once `idle_at` seconds are in; sample the state every `every` s."""
    rows, next_t, calibrated = [], 2.0, False
    for i in range(0, len(x) - CHUNK, CHUNK):
        eng.push(x[i:i + CHUNK])
        t = (i + CHUNK) / A.SR
        if not calibrated and t >= idle_at:
            eng.calibrate_idle(); calibrated = True
        if t >= next_t:
            s = eng.state()
            if s is not None:
                rows.append((t, s))
            next_t = t + every
    return rows


def test_no_audio_returns_none():
    eng = AudioEngine(16000)
    assert eng.state() is None
    eng.push(np.zeros(16000, np.float32))          # 1 s is not enough for a spindle-speed estimate
    assert eng.state() is None


def test_rpm_matches_batch():
    x = A.load("R1")
    th = A.tacho(x)
    rows = stream(x, AudioEngine(16000), every=0.0)
    ts = np.array([t for t, _ in rows]); rpm = np.array([s["rpm"] for _, s in rows])
    checked = 0
    for tc, ref_rpm, _ in th[4:-2]:
        k = np.searchsorted(ts, tc + 1.0)             # first chunk at/after the end of the batch window
        if k < len(ts) and ts[k] - (tc + 1.0) < 0.1 and np.isfinite(rpm[k]):
            assert abs(rpm[k] - ref_rpm) / ref_rpm < 0.01, (tc, rpm[k], ref_rpm)
            checked += 1
    assert checked > 20


def test_rpm_from_48k_input():
    x = A.load("R1")[: 16000 * 12]
    x48 = signal.resample_poly(x, 3, 1).astype(np.float32)
    eng = AudioEngine(48000)
    for i in range(0, len(x48) - 3072, 3072):
        eng.push(x48[i:i + 3072])
    s = eng.state()
    n = eng.next_tacho - A.SR // 2                        # end of the last rpm window (16 kHz samples)
    ref = A.f0_estimate(x[n - 2 * A.SR:n])[0] * 60
    assert abs(s["rpm"] - ref) / ref < 0.01, (s["rpm"], ref)


def test_cutting_state_matches_batch():
    for run in ("R1", "R2", "R4"):
        x = A.load(run)
        t, e = A.cut_envelope(x)
        lo, hi = np.percentile(e, 10), np.percentile(e, 90)
        batch_on = e > (lo + hi) / 2
        rows = stream(x, AudioEngine(16000))
        agree = []
        for tt, s in rows:
            k = int(round((tt - 0.2) / 0.05)) - 1       # engine's median filter lags 4 bins
            if tt > 3.5 and 0 <= k < len(batch_on) and s["cutting"] is not None:
                agree.append(bool(s["cutting"]) == bool(batch_on[k]))
        assert np.mean(agree) > 0.95, (run, np.mean(agree))


def test_calibrate_idle_reports_noise_floor_and_rpm():
    x = A.load("R1")
    eng = AudioEngine(16000)
    eng.push(x[: 16000 * 4])
    r = eng.calibrate_idle()
    assert -45 < r["idle_db"] < -36
    ref = A.f0_estimate(x[16000 * 2:16000 * 4])[0] * 60
    assert abs(r["rpm_detected"] - ref) / ref < 0.01
