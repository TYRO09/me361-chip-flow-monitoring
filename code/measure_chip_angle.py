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

SCALE_MAX = 960
TPL = 120            # template size (px)
SEARCH = 90          # search radius around previous position (px)
R_IN, R_OUT = 25, 220
DIFF_THR = 18        # grey-level threshold for "moving" pixels
PSI_RANGE = (-60, 150)


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


def rot(v, deg):
    a = math.radians(deg)
    return np.array([v[0] * math.cos(a) - v[1] * math.sin(a), v[0] * math.sin(a) + v[1] * math.cos(a)])


def axis_angle(gray, tip, ref_deg):
    """Orientation (deg, mod 180) of the workpiece axis from long straight edges near the tip."""
    h, w = gray.shape
    x0, y0 = int(tip[0]), int(tip[1])
    m = np.zeros_like(gray)
    cv2.circle(m, (x0, y0), 400, 255, -1)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120) & m
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, 80, minLineLength=180, maxLineGap=10)
    if lines is None:
        return None
    angs, wts = [], []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):  # OpenCV 4 returns (N,1,4), OpenCV 5 returns (N,4)
        a = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
        d = (a - ref_deg + 90) % 180 - 90
        if abs(d) < 15:
            angs.append(ref_deg + d)
            wts.append(math.hypot(x2 - x1, y2 - y1))
    if not angs:
        return None
    return float(np.average(angs, weights=wts))


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
            h, w = g[i].shape
            x0 = int(max(0, prev[0] - TPL // 2 - SEARCH)); x1 = int(min(w, prev[0] + TPL // 2 + SEARCH))
            y0 = int(max(0, prev[1] - TPL // 2 - SEARCH)); y1 = int(min(h, prev[1] + TPL // 2 + SEARCH))
            win = g[i][y0:y1, x0:x1]
            if win.shape[0] < TPL or win.shape[1] < TPL:
                break
            res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            cur = np.array([x0 + loc[0] + TPL // 2, y0 + loc[1] + TPL // 2], float)
            pos[i], score[i] = cur, mx
            if mx > 0.5:
                prev = cur
    tips = [None if p is None else p + off for p in pos]
    return tips, score, g


def shift(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)


def measure_run(rid, cfg, save_overlays=6, video_path=None, out_dir=None, progress=None):
    """Measure one clip. cfg holds ref_t, tip, feed, out, window (and file, used when video_path is None).
    out_dir: where overlay images go (default OUT). progress: optional callback(fraction 0..1)."""
    path = video_path or os.path.join(SESSION, "videos_top", cfg["file"])
    out_dir = out_dir or OUT
    t0, t1 = cfg["window"]
    frames, times, fps = load_frames(path, t0, t1)
    if not frames:
        raise FileNotFoundError(f"no frames read from {path} (missing file or window {t0}-{t1} s outside the clip)")
    ref_idx = int(round((cfg["ref_t"] - t0) * fps))
    if not 0 <= ref_idx < len(frames):
        raise ValueError(f"reference time {cfg['ref_t']} s is outside the analysis window {t0}-{t1} s")
    feed0, out0 = np.array(cfg["feed"], float), np.array(cfg["out"], float)
    ref_axis = math.degrees(math.atan2(feed0[1], feed0[0])) % 180
    tpl_center = np.array(cfg["tip"]) + 70 * out0 - 45 * feed0
    tips, score, g = track(frames, ref_idx, cfg["tip"], tpl_center)
    g_blur = [cv2.GaussianBlur(x, (3, 3), 0).astype(np.int16) for x in g]

    h, w = g[0].shape
    yy, xx = np.mgrid[0:h, 0:w]
    rows = []
    os.makedirs(os.path.join(out_dir, "overlays"), exist_ok=True)
    overlay_idx = set(np.linspace(1, len(frames) - 2, save_overlays).astype(int)) if save_overlays else set()
    for i in range(1, len(frames) - 1):
        if progress and i % 15 == 0:
            progress(i / (len(frames) - 1))
        if any(tips[j] is None for j in (i - 1, i, i + 1)) or min(score[i - 1:i + 2]) < 0.5:
            rows.append(dict(run=rid, t=round(times[i], 3), valid=0)); continue
        tip = tips[i]
        ax = axis_angle(g[i], tip, ref_axis)
        drot = 0.0 if ax is None else ((ax - ref_axis + 90) % 180 - 90)
        u_f, u_o = rot(feed0, drot), rot(out0, drot)
        a = shift(g_blur[i - 1], *(tips[i] - tips[i - 1]))
        b = shift(g_blur[i + 1], *(tips[i] - tips[i + 1]))
        c = g_blur[i]
        motion = np.minimum(np.abs(c - a), np.abs(c - b)).astype(np.float32)
        dx, dy = xx - tip[0], yy - tip[1]
        r = np.hypot(dx, dy)
        along_f = dx * u_f[0] + dy * u_f[1]
        along_o = dx * u_o[0] + dy * u_o[1]
        psi = np.degrees(np.arctan2(along_o, along_f))
        sel = (r > R_IN) & (r < R_OUT) & (along_o > -5) & (psi > PSI_RANGE[0]) & (psi < PSI_RANGE[1]) & (motion > DIFF_THR)
        wts = motion[sel] - DIFF_THR
        n_px = int(sel.sum())
        if n_px < 40:
            rows.append(dict(run=rid, t=round(times[i], 3), valid=0, n_px=n_px, track=round(score[i], 3))); continue
        hist, edges = np.histogram(psi[sel], bins=np.arange(PSI_RANGE[0], PSI_RANGE[1] + 1, 2), weights=wts)
        k = np.exp(-0.5 * (np.arange(-7, 8) * 2 / 6.0) ** 2)
        hs = np.convolve(hist, k / k.sum(), mode="same")
        peak = float(edges[np.argmax(hs)] + 1)
        mean = float(np.average(psi[sel], weights=wts))
        # extra features: angular spread of the spray, reach, and a second, independent angle
        # estimate from the orientation of the motion streaks (structure tensor) instead of position
        a = np.radians(psi[sel])
        R = np.hypot(np.average(np.cos(a), weights=wts), np.average(np.sin(a), weights=wts))
        spread = float(np.degrees(np.sqrt(-2 * np.log(max(R, 1e-6)))))
        reach = float(np.average(r[sel], weights=wts) / R_OUT)
        gy, gx = np.gradient(cv2.GaussianBlur(motion, (5, 5), 0))
        jxx, jyy, jxy = (np.sum(wts * gx[sel] ** 2), np.sum(wts * gy[sel] ** 2), np.sum(wts * gx[sel] * gy[sel]))
        streak = 0.5 * math.atan2(2 * jxy, jxx - jyy) + math.pi / 2           # streak axis in image coords
        sv = np.array([math.cos(streak), math.sin(streak)])
        ps = math.degrees(math.atan2(sv @ u_o, sv @ u_f))
        psi_streak = ps if ps >= 0 else ps + 180                               # axis -> outward-going direction
        coherence = float(math.hypot(jxx - jyy, 2 * jxy) / (jxx + jyy + 1e-9))
        # static clutter in front of the tool: edge density of the current frame in the same annulus,
        # regardless of motion (rises when chips pile up / tangle near the tool)
        ring = (r > R_IN) & (r < R_OUT) & (along_o > -5) & (psi > PSI_RANGE[0]) & (psi < 90)
        edges = cv2.Canny(g[i], 60, 160) > 0
        clutter = float(edges[ring].mean())
        rows.append(dict(run=rid, t=round(times[i], 3), valid=1, psi_peak=round(peak, 2), psi_mean=round(mean, 2),
                         psi_streak=round(psi_streak, 2), coherence=round(coherence, 3), spread=round(spread, 2),
                         reach=round(reach, 3), clutter=round(clutter, 4),
                         n_px=n_px, activity=round(float(wts.sum()), 1), track=round(score[i], 3),
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
            cv2.imwrite(os.path.join(out_dir, "overlays", f"{rid}_t{times[i]:05.1f}.jpg"), ov)
    return rows


FRAME_KEYS = ["run", "t", "valid", "psi_peak", "psi_mean", "psi_streak", "coherence", "spread", "reach", "clutter",
              "n_px", "activity", "track", "axis_rot", "tip_x", "tip_y", "u_f", "u_o"]


def write_frames_csv(rows, path):
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=FRAME_KEYS); wr.writeheader()
        for r in rows:
            wr.writerow({k: (json.dumps(r[k]) if isinstance(r.get(k), list) else r.get(k, "")) for k in FRAME_KEYS})


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
        write_frames_csv(rows, os.path.join(OUT, f"{rid}_frames.csv"))
        v = [r for r in rows if r.get("valid")]
        pk = np.array([r["psi_peak"] for r in v]) if v else np.array([np.nan])
        print(f"{rid}: frames={len(rows)} valid={len(v)} psi median={np.nanmedian(pk):.1f} IQR=({np.nanpercentile(pk,25):.1f},{np.nanpercentile(pk,75):.1f})")
