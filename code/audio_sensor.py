"""Audio as a virtual sensor: spindle speed (tachometer), cut start/stop, and feed estimate.

- Spindle speed: harmonic-sum pitch search. For each 2-s window, every candidate rotation
  frequency f0 (15-45 Hz) is scored by the summed spectral prominence at k*f0, k = 1..8.
- Cutting on/off: energy in the 2-6 kHz band (chip/tool contact noise), smoothed and thresholded
  half-way between the quiet and the cutting level.
- Feed estimate: the program cuts a known length (Z+1 -> Z-60 = 61 mm), so
  f [mm/rev] = 61 / (cut duration [s] * N [rev/s]).
Needs results/audio/R*.f32 (16 kHz mono float32, extracted with ffmpeg).
"""
import os
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUD = os.path.join(ROOT, "results", "audio")
FIG = os.path.join(ROOT, "results", "figures")
SR = 16000
CUT_LENGTH_MM = 61.0


def load(r):
    return np.fromfile(os.path.join(AUD, f"{r}.f32"), np.float32)


def prominence_spectrum(x):
    f, P = signal.welch(x, SR, nperseg=SR * 2, noverlap=SR)
    m = (f > 10) & (f < 1000)
    db = 10 * np.log10(P[m] + 1e-15)
    return f[m], db - signal.medfilt(db, 61)


def f0_estimate(x, kmax=8):
    f, pr = prominence_spectrum(x)
    pr = np.clip(pr, 0, None)
    cands = np.arange(15, 45, 0.02)
    score = [sum(np.interp(k * c, f, pr) for k in range(1, kmax + 1)) for c in cands]
    i = int(np.argmax(score))
    return cands[i], score[i]


def tacho(x, win=2.0, hop=0.5):
    out = []
    for s in np.arange(0, len(x) / SR - win, hop):
        f0, sc = f0_estimate(x[int(s * SR): int((s + win) * SR)])
        out.append((s + win / 2, f0 * 60, sc))
    return np.array(out)


def cut_envelope(x, hop=0.05):
    sos = signal.butter(4, [2000, 6000], "bandpass", fs=SR, output="sos")
    y = signal.sosfiltfilt(sos, x) ** 2
    n = int(hop * SR)
    e = 10 * np.log10(y[: len(y) // n * n].reshape(-1, n).mean(1) + 1e-12)
    e = signal.medfilt(e, 9)
    t = (np.arange(len(e)) + 0.5) * hop
    return t, e


def cut_interval(t, e):
    lo, hi = np.percentile(e, 10), np.percentile(e, 90)
    on = e > (lo + hi) / 2
    # longest continuous "on" stretch (bridging gaps < 0.5 s)
    idx = np.flatnonzero(on)
    if not len(idx):
        return None
    segs, start = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if t[b] - t[a] > 0.5:
            segs.append((start, a)); start = b
    segs.append((start, idx[-1]))
    s = max(segs, key=lambda q: t[q[1]] - t[q[0]])
    return t[s[0]], t[s[1]]


def main():
    runs = pd.read_csv(os.path.join(ROOT, "session_2026-11-04", "runs.csv")).set_index("run_id")
    rows = []
    fig, axs = plt.subplots(6, 2, figsize=(13, 14))
    for k, r in enumerate(runs.index):
        x = load(r)
        th = tacho(x)
        t, e = cut_envelope(x)
        iv = cut_interval(t, e)
        dur = iv[1] - iv[0]
        cut = (th[:, 0] > iv[0] + 1) & (th[:, 0] < iv[1] - 1)
        n_audio = float(np.median(th[cut, 1]))
        f_est = CUT_LENGTH_MM / (dur * n_audio / 60)
        n_prog = runs.loc[r, "spindle_rpm_calc"]
        level = float(np.median(e[(t > iv[0] + 1) & (t < iv[1] - 1)]))
        rows.append(dict(run=r, rpm_program=n_prog, rpm_audio=round(n_audio, 1), cut_noise_db=round(level, 2),
                         mrr_mm3_per_s=round(runs.loc[r, "feed_mm_per_rev"] * runs.loc[r, "depth_of_cut_mm"] * 150e3 / 60, 1),
                         rpm_error_pct=round(100 * (n_audio - n_prog) / n_prog, 2),
                         cut_start_s=round(iv[0], 2), cut_end_s=round(iv[1], 2), cut_duration_s=round(dur, 2),
                         cut_duration_expected_s=runs.loc[r, "expected_cut_time_s"],
                         feed_true=runs.loc[r, "feed_mm_per_rev"], feed_audio=round(f_est, 4),
                         feed_error_pct=round(100 * (f_est - runs.loc[r, "feed_mm_per_rev"]) / runs.loc[r, "feed_mm_per_rev"], 1)))
        ax = axs[k, 0]
        ax.plot(th[:, 0], th[:, 1], ".-", ms=3, label="audio tachometer")
        ax.axhline(n_prog, color="C3", ls="--", label=f"program (G96): {n_prog} rpm")
        ax.axvspan(*iv, color="C2", alpha=0.12)
        ax.set_ylim(600, 2600); ax.set_ylabel("rpm"); ax.set_title(f"{r}: spindle speed from sound", fontsize=10)
        ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=0.3)
        ax = axs[k, 1]
        ax.plot(t, e, lw=0.8)
        ax.axvspan(*iv, color="C2", alpha=0.12, label=f"cutting: {dur:.1f} s → f ≈ {f_est:.3f} mm/rev (true {runs.loc[r,'feed_mm_per_rev']})")
        ax.set_ylabel("2–6 kHz energy (dB)"); ax.set_title(f"{r}: cut detection from sound", fontsize=10)
        ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=0.3)
    axs[-1, 0].set_xlabel("time in clip (s)"); axs[-1, 1].set_xlabel("time in clip (s)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_audio_sensor.png"), dpi=130); plt.close(fig)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(ROOT, "results", "audio_sensor.csv"), index=False)
    print(d.to_string(index=False))


if __name__ == "__main__":
    main()
