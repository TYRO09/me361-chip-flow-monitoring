"""End-to-end analysis of one new top-view clip: chip angle, build-up alarm, audio sensor, dashboard video.

The user supplies three points on one reference frame (in the 960-px working frame used by
measure_chip_angle): the tool tip, any point further along the bar in the feed direction (towards the
chuck), and any point on the outward side of the bar (away from the workpiece). Everything else is the
same method as for the six lab runs.

    from pipeline import analyze
    res = analyze("clip.mp4", tip=(222, 238), feed_pt=(400, 238), out_pt=(222, 100),
                  ref_t=16, window=(10, 24), out_dir="uploads/clip")
"""
import json
import math
import os
import subprocess
import sys

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import audio_sensor as A
from entangle_indicator import score_frames
from live_monitor_demo import render_dashboard
from measure_chip_angle import R_IN, R_OUT, SCALE_MAX, TPL, measure_run, write_frames_csv

MAX_WINDOW_S = 30.0     # all frames of the window are held in memory (~2.5 GB for 30 s at 960 px)


def video_info(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, im = cap.read(); cap.release()
    if not ok:
        raise ValueError(f"cannot read video {path}")
    s = SCALE_MAX / max(im.shape[:2])
    return dict(fps=fps, frames=n, duration=n / fps, width=im.shape[1], height=im.shape[0],
                work_w=int(round(im.shape[1] * s)), work_h=int(round(im.shape[0] * s)))


def read_frame(path, t):
    """RGB frame at time t (s), scaled like measure_chip_angle (longer side 960 px)."""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, im = cap.read(); cap.release()
    if not ok:
        raise ValueError(f"cannot read frame at {t:.2f} s")
    s = SCALE_MAX / max(im.shape[:2])
    im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def extract_audio(video, out_f32):
    """16 kHz mono float32 sound track (same format as results/audio/*.f32). None if the clip has no audio."""
    r = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", video, "-vn", "-ac", "1",
                        "-ar", str(A.SR), "-f", "f32le", out_f32], capture_output=True)
    if r.returncode != 0 or not os.path.exists(out_f32) or os.path.getsize(out_f32) < 4 * A.SR:
        return None
    return np.fromfile(out_f32, np.float32)


def audio_summary(x, cut_length_mm=None):
    """Spindle speed trace, cutting interval and (if the cut length is known) a feed estimate."""
    if x is None or len(x) < 3 * A.SR:
        return None
    th = A.tacho(x)
    t, e = A.cut_envelope(x)
    iv = A.cut_interval(t, e)
    out = dict(tacho=th, cut=iv, rpm=None, feed=None)
    if iv is not None:
        m = (th[:, 0] > iv[0] + 1) & (th[:, 0] < iv[1] - 1)
        if m.any():
            out["rpm"] = float(np.median(th[m, 1]))
            if cut_length_mm:
                out["feed"] = cut_length_mm / ((iv[1] - iv[0]) * out["rpm"] / 60)
    return out


def directions(tip, feed_pt, out_pt):
    """Unit feed vector (tip -> feed point) and the outward unit vector perpendicular to it,
    on the side of out_pt. Image coordinates (x right, y down)."""
    tip, feed_pt, out_pt = (np.asarray(p, float) for p in (tip, feed_pt, out_pt))
    f = feed_pt - tip
    if np.hypot(*f) < 5:
        raise ValueError("the feed-direction point is too close to the tool tip")
    f /= np.hypot(*f)
    o = np.array([-f[1], f[0]])
    side = float(np.dot(out_pt - tip, o))
    if abs(side) < 5:
        raise ValueError("the outward point lies on the bar axis; click clearly to one side of the bar")
    return f, (o if side > 0 else -o)


def template_box(tip, feed, out):
    """The insert patch the tracker locks onto (same placement as measure_chip_angle)."""
    c = np.asarray(tip, float) + 70 * out - 45 * feed
    return c, (c - TPL / 2, c + TPL / 2)


def check_setup(info, tip, feed, out, ref_t, window):
    t0, t1 = window
    if not 0 <= t0 < t1 <= info["duration"] + 1e-6:
        raise ValueError(f"analysis window must lie inside the clip (0-{info['duration']:.1f} s)")
    if t1 - t0 > MAX_WINDOW_S:
        raise ValueError(f"analysis window is {t1 - t0:.0f} s; keep it to {MAX_WINDOW_S:.0f} s or less (memory)")
    if t1 - t0 < 3:
        raise ValueError("analysis window must be at least 3 s")
    if not t0 <= ref_t <= t1:
        raise ValueError("the reference frame must lie inside the analysis window")
    _, (lo, hi) = template_box(tip, feed, out)
    m = 90  # tracker search margin
    if lo[0] < 0 or lo[1] < 0 or hi[0] > info["work_w"] or hi[1] > info["work_h"]:
        raise ValueError("the tracking patch behind the tool tip runs off the image; the tip is too close to the edge "
                         "or the directions are wrong")
    if lo[0] < m or lo[1] < m or hi[0] > info["work_w"] - m or hi[1] > info["work_h"] - m:
        return "The tool is close to the image edge; tracking may be lost if it moves further out."
    return None


def analyze(video, tip, feed_pt, out_pt, ref_t, window, out_dir, f=None, ap=None, cut_length_mm=None,
            render=True, progress=None, use_audio=True):
    """Run the whole chain on one clip. Writes into out_dir:
    frames.csv, entangle.csv, overlays/*.jpg, summary.json and (if render) dashboard.mp4.
    progress: callback(stage: str, fraction: float).
    use_audio=False skips the sound analysis (no rpm, cut detection or feed estimate; the dashboard
    then has no microphone section and the build-up baseline is the start of the analysis window)."""
    p = progress or (lambda stage, frac: None)
    os.makedirs(out_dir, exist_ok=True)
    info = video_info(video)
    feed, out = directions(tip, feed_pt, out_pt)
    warning = check_setup(info, tip, feed, out, ref_t, window)

    if use_audio:
        p("audio", 0.0)
        x = extract_audio(video, os.path.join(out_dir, "audio.f32"))
        au = audio_summary(x, cut_length_mm)
    else:
        x, au = None, None

    cfg = dict(ref_t=float(ref_t), tip=tuple(map(float, tip)), feed=tuple(feed), out=tuple(out),
               window=(float(window[0]), float(window[1])))
    for old in os.listdir(os.path.join(out_dir, "overlays")) if os.path.isdir(os.path.join(out_dir, "overlays")) else []:
        os.remove(os.path.join(out_dir, "overlays", old))
    rows = measure_run("U", cfg, save_overlays=6, video_path=video, out_dir=out_dir,
                       progress=lambda fr: p("measure", fr))
    write_frames_csv(rows, os.path.join(out_dir, "frames.csv"))
    fr = pd.read_csv(os.path.join(out_dir, "frames.csv"))
    valid = fr[fr.valid == 1]
    if valid.empty:
        raise RuntimeError("no valid frames: the tool was not tracked or no moving chips were found "
                           "(check the tool tip position and the analysis window)")

    # build-up score: baseline = first 4 s of steady cutting, so start 1 s after the sound says the cut began
    e0, e1 = window
    if au and au["cut"]:
        e0, e1 = max(e0, au["cut"][0] + 1), min(e1, au["cut"][1] - 1)
    if e1 - e0 < 6:
        e0, e1 = window
    ent, alarm = score_frames(fr, e0, e1)
    ent[["run", "t", "z_log_act", "z_spread", "z_clutter", "score", "score_s"]].round(3).to_csv(
        os.path.join(out_dir, "entangle.csv"), index=False)

    psi_pred = 26.8 + 101 * f + 17.7 * ap if (f and ap) else None    # linear model fitted on the lab runs
    summary = dict(
        video=os.path.basename(video), duration_s=round(info["duration"], 2), fps=round(info["fps"], 2),
        window=list(window), ref_t=ref_t, tip=list(tip), feed_dir=[round(v, 4) for v in feed],
        out_dir=[round(v, 4) for v in out], frames=len(fr), valid_frames=int(len(valid)),
        tracking_ok_pct=round(100 * float((fr.track >= 0.5).mean()), 1) if "track" in fr else None,
        psi_median=round(float(valid.psi_peak.median()), 1), psi_mean=round(float(valid.psi_peak.mean()), 1),
        psi_iqr=[round(float(v), 1) for v in valid.psi_peak.quantile([0.25, 0.75])],
        psi_pred=None if psi_pred is None else round(psi_pred, 1), feed_mm_rev=f, ap_mm=ap,
        score_window=[round(e0, 2), round(e1, 2)], alarm_s=None if alarm is None else round(float(alarm), 2),
        max_score=None if ent.empty else round(float(ent.score_s.max()), 2),
        audio=au is not None, cut_s=None if not au or not au["cut"] else [round(v, 2) for v in au["cut"]],
        rpm_audio=None if not au or au["rpm"] is None else round(au["rpm"], 1),
        feed_audio=None if not au or au["feed"] is None else round(au["feed"], 4),
        cut_length_mm=cut_length_mm, warning=warning)
    if au:
        pd.DataFrame(au["tacho"][:, :2], columns=["t", "rpm"]).to_csv(os.path.join(out_dir, "tacho.csv"), index=False)

    if render:
        sub = os.path.basename(video)[:28]
        if f and ap:
            sub += f"   f = {f} mm/rev   ap = {ap} mm"
        render_dashboard(video, os.path.join(out_dir, "dashboard.mp4"), fr, ent, x,
                         max(0.0, window[0] - 1), min(info["duration"], window[1] + 1), sub, psi_pred,
                         progress=lambda fr_: p("render", fr_))
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    p("done", 1.0)
    return summary


def draw_setup(img, tip=None, feed_pt=None, out_pt=None):
    """Annotate the reference frame with the picked points, derived directions, tracking patch and ring."""
    im = img.copy()
    if tip is not None:
        t = tuple(int(v) for v in tip)
        cv2.circle(im, t, R_IN, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(im, t, R_OUT, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.drawMarker(im, t, (255, 230, 0), cv2.MARKER_CROSS, 22, 2)
    if feed_pt is not None:
        cv2.circle(im, tuple(int(v) for v in feed_pt), 6, (60, 140, 255), -1)
    if out_pt is not None:
        cv2.circle(im, tuple(int(v) for v in out_pt), 6, (60, 220, 60), -1)
    if tip is not None and feed_pt is not None and out_pt is not None:
        try:
            f, o = directions(tip, feed_pt, out_pt)
        except ValueError:
            return im
        tp = np.asarray(tip, float)
        cv2.arrowedLine(im, tuple((tp).astype(int)), tuple((tp + 120 * f).astype(int)), (60, 140, 255), 3, cv2.LINE_AA, tipLength=0.15)
        cv2.arrowedLine(im, tuple((tp).astype(int)), tuple((tp + 120 * o).astype(int)), (60, 220, 60), 3, cv2.LINE_AA, tipLength=0.15)
        _, (lo, hi) = template_box(tip, f, o)
        cv2.rectangle(im, tuple(lo.astype(int)), tuple(hi.astype(int)), (255, 80, 200), 2)
        for a in (0, 45, 90):
            d = math.cos(math.radians(a)) * f + math.sin(math.radians(a)) * o
            e = np.clip(tp + (R_OUT + 18) * d, [5, 18], [im.shape[1] - 30, im.shape[0] - 5])
            cv2.putText(im, f"{a}", tuple(e.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return im
