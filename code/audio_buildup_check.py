"""Does the microphone see chip build-up? (Dornfeld & Pan 1985 report that acoustic emission responds to
chip congestion / entangling; we only have an audible microphone, so this checks whether a usable signal is there.)

Per run, inside the cut: 2-6 kHz band level (50-ms bins, dB) and its 1-s standard deviation. Each is compared with the
run's own first 4 s of cut. 'Change' = largest 1-s rolling median above that baseline over the rest of the cut, in
units of the baseline's robust spread. R2 is the run that tangles (vision alarm at ~18.7-20 s); a useful
audio indicator must rise on R2 clearly more than on the five clean runs.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audio_sensor as A  # noqa: E402
from entangle_indicator import CUT  # noqa: E402

BIN = 0.05


def run_stats(run):
    x = A.load(run)
    t, e = A.cut_envelope(x)                     # 50-ms band-level bins, dB
    t0, t1 = CUT[run]
    m = (t >= t0) & (t <= t1)
    tt, ee = t[m], e[m]
    sd = pd.Series(ee).rolling(20, min_periods=10).std().values
    out = {}
    for name, v in (("band_db", ee), ("band_sd_db", sd)):
        base = v[tt < t0 + 4]
        base = base[np.isfinite(base)]
        med, mad = np.median(base), 1.4826 * np.median(np.abs(base - np.median(base))) + 0.3   # 0.3 dB floor
        roll = pd.Series(v).rolling(20, min_periods=10).median().values
        late = np.isfinite(roll) & (tt >= t0 + 4)
        z = (roll - med) / mad
        k = int(np.nanargmax(np.where(late, z, -np.inf)))
        out[name] = dict(change_z=float(z[k]), at_s=float(tt[k]), rise_db=float(roll[k] - med))
    return out


def main():
    rows = []
    for r in CUT:
        s = run_stats(r)
        rows.append(dict(run=r, band_change_z=round(s["band_db"]["change_z"], 1), band_rise_db=round(s["band_db"]["rise_db"], 2),
                         band_at_s=round(s["band_db"]["at_s"], 1), sd_change_z=round(s["band_sd_db"]["change_z"], 1),
                         sd_at_s=round(s["band_sd_db"]["at_s"], 1)))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(ROOT, "results", "audio_buildup_check.csv"), index=False)
    print(d.to_string(index=False))
    for col in ("band_change_z", "sd_change_z"):
        others = d[d.run != "R2"][col].max()
        r2 = float(d[d.run == "R2"][col].iloc[0])
        verdict = "USABLE" if r2 > 1.5 * others and r2 > 3 else "NOT USABLE"
        print(f"{col}: R2 {r2:.1f} vs largest clean run {others:.1f} -> {verdict}")


if __name__ == "__main__":
    main()
