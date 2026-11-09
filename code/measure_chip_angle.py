"""Measure chip flow angle frame-by-frame from the top-view videos.

Angle convention (psi, degrees), measured in the image plane (= rake face seen from above):
    psi = 0   -> chip leaves the tool tip along the feed direction (towards the chuck)
    psi = 90  -> chip leaves radially outward, perpendicular to the workpiece axis
Positive psi rotates from the feed direction towards "radially outward".
Using workpiece-relative directions makes the number independent of how the phone was held.

Method per frame:
  1. track the tool tip with template matching on the insert (phone is handheld and follows the tool)
  2. re-estimate the workpiece axis direction from long straight edges (corrects phone rotation)
  3. three-frame differencing after aligning neighbours by the tip motion -> fast-moving chip pixels
  4. weighted histogram of the direction (tip -> moving pixel) in an annulus around the tip;
     the peak is the chip flow angle for that frame
"""
import csv, json, math, os, sys
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSION = os.path.join(ROOT, "session_2026-11-04")
OUT = os.path.join(ROOT, "results", "angles")

# Hand-read from one reference frame per run, frame scaled so its longer side is 960 px.
# feed / out: unit directions in image coords (x right, y down) at the reference frame.
RUNS = {
    "R1": dict(file="R1_f0.08_ap0.9_top.MOV", ref_t=18, tip=(555, 355), feed=(0, 1), out=(1, 0), window=(10, 37)),
    "R2": dict(file="R2_f0.15_ap0.9_top.MOV", ref_t=16, tip=(222, 238), feed=(1, 0), out=(0, -1), window=(10, 24)),
    "R3": dict(file="R3_f0.08_ap0.3_top.MOV", ref_t=20, tip=(348, 330), feed=(0, -1), out=(-1, 0), window=(8, 34)),
    "R4": dict(file="R4_f0.15_ap0.3_top.MOV", ref_t=14, tip=(378, 325), feed=(1, 0), out=(0, -1), window=(8, 22)),
    "R5": dict(file="R5_f0.08_ap0.6_top.MOV", ref_t=22, tip=(295, 595), feed=(-1, 0), out=(0, 1), window=(8, 32)),
    "R6": dict(file="R6_f0.15_ap0.6_top.MOV", ref_t=14, tip=(245, 318), feed=(1, 0), out=(0, -1), window=(9, 20)),
}

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chip_kernel import (SCALE_MAX, TPL, SEARCH, R_IN, R_OUT, DIFF_THR, PSI_RANGE, rot, axis_rotation,  # noqa: E402
                         track_step, blur16, frame_features)


def load_frames(path, t0, t1):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t0 * fps))
    frames, times = [], []
    for i in range(int((t1 - t0) * fps)):
        ok, fr = cap.read()
        if not ok:
            break
        s = SCALE_MAX / max(fr.shape[:2])
        frames.append(cv2.resize(fr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
        times.append(t0 + i / fps)
    return frames, np.array(times), fps


def track(frames, ref_idx, tip0, tpl_center):
    """Template-track the insert forwards and backwards from the reference frame."""
    g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    cx, cy = int(tpl_center[0]), int(tpl_center[1])
    tpl = g[ref_idx][cy - TPL // 2: cy + TPL // 2, cx - TPL // 2: cx + TPL // 2]
    off = np.array(tip0, float) - np.array([cx, cy], float)
    pos = [None] * len(frames)
    score = [0.0] * len(frames)
    pos[ref_idx], score[ref_idx] = np.array([cx, cy], float), 1.0
    for rng in (range(ref_idx + 1, len(frames)), range(ref_idx - 1, -1, -1)):
        prev = pos[ref_idx].copy()
        for i in rng:
            st = track_step(g[i], tpl, prev, SEARCH)
            if st is None:
                break
            cur, mx = st
            pos[i], score[i] = cur, mx
            if mx > 0.5:
                prev = cur
    tips = [None if p is None else p + off for p in pos]
    return tips, score, g


def measure_run(rid, cfg, save_overlays=6):
    path = os.path.join(SESSION, "videos_top", cfg["file"])
    t0, t1 = cfg["window"]
    frames, times, fps = load_frames(path, t0, t1)
    ref_idx = int(round((cfg["ref_t"] - t0) * fps))
    feed0, out0 = np.array(cfg["feed"], float), np.array(cfg["out"], float)
    ref_axis = math.degrees(math.atan2(feed0[1], feed0[0])) % 180
    tpl_center = np.array(cfg["tip"]) + 70 * out0 - 45 * feed0
    tips, score, g = track(frames, ref_idx, cfg["tip"], tpl_center)
    g_blur = [blur16(x) for x in g]

    rows = []
    os.makedirs(os.path.join(OUT, "overlays"), exist_ok=True)
    overlay_idx = set(np.linspace(1, len(frames) - 2, save_overlays).astype(int)) if save_overlays else set()
    for i in range(1, len(frames) - 1):
        if any(tips[j] is None for j in (i - 1, i, i + 1)) or min(score[i - 1:i + 2]) < 0.5:
            rows.append(dict(run=rid, t=round(times[i], 3), valid=0)); continue
        tip = tips[i]
        drot = axis_rotation(g[i], tip, ref_axis)
        u_f, u_o = rot(feed0, drot), rot(out0, drot)
        ft = frame_features(g_blur[i - 1], g_blur[i], g_blur[i + 1], g[i], tips[i - 1], tips[i], tips[i + 1], u_f, u_o,
                            thr=DIFF_THR, r_out=R_OUT)
        if ft is None:
            rows.append(dict(run=rid, t=round(times[i], 3), valid=0, n_px=0, track=round(score[i], 3))); continue
        sel, peak = ft["sel"], ft["psi_peak"]
        rows.append(dict(run=rid, t=round(times[i], 3), valid=1, psi_peak=round(peak, 2), psi_mean=round(ft["psi_mean"], 2),
                         psi_streak=round(ft["psi_streak"], 2), coherence=round(ft["coherence"], 3), spread=round(ft["spread"], 2),
                         reach=round(ft["reach"], 3), clutter=round(ft["clutter"], 4),
                         n_px=ft["n_px"], activity=round(ft["activity"], 1), track=round(score[i], 3),
                         axis_rot=round(drot, 2), tip_x=round(tip[0], 1), tip_y=round(tip[1], 1),
                         u_f=[round(u_f[0], 4), round(u_f[1], 4)], u_o=[round(u_o[0], 4), round(u_o[1], 4)]))
        if i in overlay_idx:
            ov = frames[i].copy()
            mk = np.zeros_like(ov); mk[sel] = (0, 0, 255)
            ov = cv2.addWeighted(ov, 1, mk, 0.8, 0)
            p = tuple(int(v) for v in tip)
            cv2.circle(ov, p, R_IN, (255, 255, 0), 1); cv2.circle(ov, p, R_OUT, (255, 255, 0), 1)
            cv2.arrowedLine(ov, p, tuple(int(v) for v in tip + 120 * u_f), (255, 0, 0), 2)
            d = math.cos(math.radians(peak)) * u_f + math.sin(math.radians(peak)) * u_o
            cv2.arrowedLine(ov, p, tuple(int(v) for v in tip + 200 * d), (0, 255, 0), 3)
            cv2.putText(ov, f"{rid} t={times[i]:.1f}s psi={peak:.0f} deg", (10, 30), 0, 0.8, (0, 255, 255), 2)
            cv2.imwrite(os.path.join(OUT, "overlays", f"{rid}_t{times[i]:05.1f}.jpg"), ov)
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--rout", type=int, default=R_OUT, help="outer radius of the measuring annulus (px)")
    ap.add_argument("--thr", type=int, default=DIFF_THR, help="motion threshold (grey levels)")
    ap.add_argument("--tag", default="", help="write to results/angles_variants/<tag> instead of results/angles")
    args = ap.parse_args()
    R_OUT, DIFF_THR = args.rout, args.thr
    if args.tag:
        OUT = os.path.join(ROOT, "results", "angles_variants", args.tag)
    os.makedirs(OUT, exist_ok=True)
    which = args.runs or list(RUNS)
    for rid in which:
        rows = measure_run(rid, RUNS[rid], save_overlays=0 if args.tag else 6)
        keys = ["run", "t", "valid", "psi_peak", "psi_mean", "psi_streak", "coherence", "spread", "reach", "clutter",
                "n_px", "activity", "track", "axis_rot", "tip_x", "tip_y", "u_f", "u_o"]
        with open(os.path.join(OUT, f"{rid}_frames.csv"), "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=keys); wr.writeheader()
            for r in rows:
                wr.writerow({k: (json.dumps(r[k]) if isinstance(r.get(k), list) else r.get(k, "")) for k in keys})
        v = [r for r in rows if r.get("valid")]
        pk = np.array([r["psi_peak"] for r in v]) if v else np.array([np.nan])
        print(f"{rid}: frames={len(rows)} valid={len(v)} psi median={np.nanmedian(pk):.1f} IQR=({np.nanpercentile(pk,25):.1f},{np.nanpercentile(pk,75):.1f})")
