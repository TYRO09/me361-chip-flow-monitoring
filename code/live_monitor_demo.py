"""Live-monitor demo video: replays a run with everything the system computes, frame by frame.

Left: the top-view video with the tracked tool tip, the measured chip direction (raw and smoothed).
Right: a dashboard: chip-angle dial, the angle trace, spindle speed + cutting state from the
microphone, the chip build-up score and the entanglement alarm.
Every number shown is computed causally (only from frames up to the current one), except the 0.5-s
display smoothing. Usage: python3 code/live_monitor_demo.py R2 [t_start t_end]
"""
import math, os, subprocess, sys
import cv2
import numpy as np
import pandas as pd
import imageio_ffmpeg

sys.path.insert(0, os.path.dirname(__file__))
import audio_sensor as A

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
VID = os.path.join(ROOT, "session_2026-11-04", "videos_top")
W_PANEL, H = 520, 960
DARK, FG, ACC, OK_C, BAD = (28, 28, 30), (235, 235, 235), (60, 200, 255), (90, 200, 90), (60, 60, 230)


def vec(s):
    return np.array([float(v) for v in s.strip("[]").split(",")])


def put(img, txt, xy, sc=0.6, col=FG, th=1):
    cv2.putText(img, txt, xy, cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)


def dial(p, cx, cy, R, psi, psi_pred):
    cv2.ellipse(p, (cx, cy), (R, R), 0, -90, 0, (90, 90, 90), 2, cv2.LINE_AA)
    for a in range(0, 91, 15):
        r = math.radians(a); x, y = cx + R * math.cos(r), cy - R * math.sin(r)
        cv2.line(p, (int(x), int(y)), (int(cx + (R - 10) * math.cos(r)), int(cy - (R - 10) * math.sin(r))), (150, 150, 150), 1)
    put(p, "feed", (cx + R - 20, cy + 20), 0.45, (170, 170, 170)); put(p, "outward", (cx - 30, cy - R - 8), 0.45, (170, 170, 170))
    if psi_pred is not None:
        r = math.radians(psi_pred)
        cv2.line(p, (cx, cy), (int(cx + R * math.cos(r)), int(cy - R * math.sin(r))), (200, 120, 255), 2, cv2.LINE_AA)
    if psi is not None and not np.isnan(psi):
        r = math.radians(np.clip(psi, -5, 95))
        cv2.arrowedLine(p, (cx, cy), (int(cx + (R - 4) * math.cos(r)), int(cy - (R - 4) * math.sin(r))), ACC, 4, cv2.LINE_AA, tipLength=0.12)


def trace(p, x0, y0, w, h, ts, ys, t_now, span, lo, hi, col, ref=None, ref_col=(200, 120, 255)):
    cv2.rectangle(p, (x0, y0), (x0 + w, y0 + h), (70, 70, 70), 1)
    if ref is not None:
        yy = int(y0 + h - (ref - lo) / (hi - lo) * h); cv2.line(p, (x0, yy), (x0 + w, yy), ref_col, 1, cv2.LINE_AA)
    m = (ts > t_now - span) & (ts <= t_now) & ~np.isnan(ys)
    if m.sum() > 1:
        px = (x0 + (ts[m] - (t_now - span)) / span * w).astype(int)
        py = (y0 + h - (np.clip(ys[m], lo, hi) - lo) / (hi - lo) * h).astype(int)
        cv2.polylines(p, [np.c_[px, py].reshape(-1, 1, 2)], False, col, 2, cv2.LINE_AA)


def main(run="R2", t_a=None, t_b=None):
    runs = pd.read_csv(os.path.join(ROOT, "session_2026-11-04", "runs.csv")).set_index("run_id")
    f, ap = runs.loc[run, "feed_mm_per_rev"], runs.loc[run, "depth_of_cut_mm"]
    psi_pred = 26.8 + 101 * f + 17.7 * ap                       # M1 model prediction for these settings
    fr = pd.read_csv(os.path.join(RES, "angles", f"{run}_frames.csv"))
    ent = pd.read_csv(os.path.join(RES, "angles", f"{run}_entangle.csv"))
    au = pd.read_csv(os.path.join(RES, "audio_sensor.csv")).set_index("run")
    t_a = t_a if t_a is not None else max(0, au.loc[run, "cut_start_s"] - 2)
    t_b = t_b if t_b is not None else au.loc[run, "cut_end_s"] + 1
    src = [v for v in os.listdir(VID) if v.startswith(run + "_")][0]
    os.makedirs(os.path.join(RES, "demo"), exist_ok=True)
    out = os.path.join(RES, "demo", f"{run}_live_monitor.mp4")
    alarm_t = render_dashboard(os.path.join(VID, src), out, fr, ent, A.load(run), t_a, t_b,
                               f"{run}   f = {f} mm/rev   ap = {ap} mm   Vc = 150 m/min", psi_pred)
    print(out, "alarm at", alarm_t)


def render_dashboard(video, out, fr, ent, audio, t_a, t_b, subtitle="", psi_pred=None, progress=None):
    """Write the side-by-side monitor video for video[t_a..t_b] to out (mp4).
    fr: per-frame table from measure_chip_angle; ent: scored table from entangle_indicator.score_frames
    (may be empty); audio: 16 kHz mono float32 samples of the clip, or None (then the cut state is taken
    as 'cutting' and no rpm is shown). Works for portrait and landscape clips. Returns the alarm time or None."""
    fr = fr.copy(); ent = ent.copy()
    fr["psi_disp"] = fr.psi_peak.where(fr.valid == 1).rolling(15, min_periods=3).median()   # causal 0.5 s
    if len(ent):
        ent["score_c"] = ent.score.rolling(30, min_periods=10).median()                     # causal 1 s
    if audio is not None and len(audio) > 3 * A.SR:
        th = A.tacho(audio); te, ee = A.cut_envelope(audio)
        lo, hi = np.percentile(ee, 10), np.percentile(ee, 90); cutting = ee > (lo + hi) / 2
    else:
        th, cutting = None, None

    cap = cv2.VideoCapture(video); fps = cap.get(cv2.CAP_PROP_FPS)
    ok, im = cap.read()
    if not ok:
        raise FileNotFoundError(f"cannot read {video}")
    s = 960 / max(im.shape[:2])
    w_left = int(round(im.shape[1] * s)); w_left += w_left % 2       # libx264 needs even sizes
    ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{w_left + W_PANEL}x{H}", "-r", f"{fps:.3f}", "-i", "-", "-ss", f"{t_a:.3f}", "-i", video,
                           "-map", "0:v", "-map", "1:a:0?", "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                           "-c:a", "aac", out], stdin=subprocess.PIPE)
    cap.set(cv2.CAP_PROP_POS_MSEC, t_a * 1000)
    alarm_on, alarm_t = False, None
    run_len = 0
    while True:
        ok, im = cap.read()
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 - 1 / fps
        if not ok or t > t_b:
            break
        if progress:
            progress(min(1.0, max(0.0, (t - t_a) / max(t_b - t_a, 1e-6))))
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        im = cv2.copyMakeBorder(im[:H, :w_left], 0, max(0, H - im.shape[0]), 0, max(0, w_left - im.shape[1]),
                                cv2.BORDER_CONSTANT, value=DARK)
        k = int(np.argmin(np.abs(fr.t.values - t))); row = fr.iloc[k]
        if cutting is not None:
            ci = min(int(t / 0.05), len(cutting) - 1); cut_now = bool(cutting[ci])
        else:
            cut_now = True
        psi_now = row.psi_disp if (abs(row.t - t) < 0.05 and cut_now) else np.nan
        if abs(row.t - t) < 0.05 and row.valid == 1 and cut_now and not np.isnan(row.tip_x):
            tip = np.array([row.tip_x, row.tip_y]); uf, uo = vec(row.u_f), vec(row.u_o)
            cv2.circle(im, tuple(tip.astype(int)), 25, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.arrowedLine(im, tuple(tip.astype(int)), tuple((tip + 70 * uf).astype(int)), (255, 160, 60), 2, cv2.LINE_AA, tipLength=0.2)
            for ang, col, L, w in [(row.psi_peak, (120, 120, 255), 150, 1), (psi_now, ACC, 200, 3)]:
                if not np.isnan(ang):
                    d = math.cos(math.radians(ang)) * uf + math.sin(math.radians(ang)) * uo
                    cv2.arrowedLine(im, tuple(tip.astype(int)), tuple((tip + L * d).astype(int)), col, w, cv2.LINE_AA, tipLength=0.1)
        # ---------------- dashboard ----------------
        p = np.full((H, W_PANEL, 3), DARK, np.uint8)
        put(p, "CHIP FLOW MONITOR", (20, 40), 0.85, FG, 2)
        put(p, subtitle, (20, 70), 0.5, (190, 190, 190))
        put(p, f"t = {t:5.1f} s", (W_PANEL - 120, 40), 0.55, (190, 190, 190))
        dial(p, 60, 300, 190, psi_now, psi_pred)
        put(p, "chip angle", (270, 140), 0.55, (190, 190, 190))
        put(p, "--" if np.isnan(psi_now) else f"{psi_now:4.0f} deg", (270, 190), 1.3, ACC, 3)
        if psi_pred is not None:
            put(p, f"model predicts {psi_pred:.0f} deg", (270, 225), 0.5, (200, 120, 255))
        if th is not None:                       # microphone section only when the sound was analysed
            put(p, "from the microphone", (20, 360), 0.5, (170, 170, 170))
            ti = int(np.argmin(np.abs(th[:, 0] - t))); rpm = th[ti, 1]
            put(p, f"spindle  {rpm:5.0f} rpm", (20, 395), 0.75, FG, 2)
            put(p, "CUTTING" if cut_now else "not cutting", (300, 395), 0.75, OK_C if cut_now else (150, 150, 150), 2)
        put(p, "chip angle, last 8 s (deg)", (20, 445), 0.5, (170, 170, 170))
        trace(p, 20, 455, W_PANEL - 40, 150, fr.t.values, fr.psi_disp.values, t, 8, 0, 100, ACC, ref=psi_pred)
        put(p, "chip build-up score, last 8 s", (20, 640), 0.5, (170, 170, 170))
        sc = np.nan
        if len(ent):
            ei = int(np.argmin(np.abs(ent.t.values - t))); sc = ent.score_c.values[ei] if abs(ent.t.values[ei] - t) < 0.05 else np.nan
            trace(p, 20, 650, W_PANEL - 40, 120, ent.t.values, ent.score_c.values, t, 8, -2, 5, (90, 200, 250), ref=3.0, ref_col=(60, 60, 230))
        if not np.isnan(sc):
            run_len = run_len + 1 if sc > 3 else 0
            if run_len >= 30 and not alarm_on:
                alarm_on, alarm_t = True, t
        if alarm_on:
            cv2.rectangle(p, (20, 800), (W_PANEL - 20, 880), BAD, -1)
            put(p, "ALARM: CHIP BUILD-UP / TANGLE", (35, 835), 0.7, (255, 255, 255), 2)
            put(p, f"since t = {alarm_t:.1f} s - stop and clear chips", (35, 865), 0.5, (255, 255, 255))
            cv2.rectangle(im, (2, 2), (w_left - 3, H - 3), BAD, 6)
        else:
            cv2.rectangle(p, (20, 800), (W_PANEL - 20, 880), (45, 70, 45), -1)
            put(p, "chip flow normal", (35, 848), 0.7, OK_C, 2)
        put(p, "yellow arrow = chip direction (thin: raw frame), blue = feed", (20, 920), 0.42, (150, 150, 150))
        ff.stdin.write(np.hstack([im, p]).tobytes())
    ff.stdin.close(); ff.wait(); cap.release()
    return alarm_t


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "R2", *(float(v) for v in a[1:3]))
