"""Per-frame chip-flow maths shared by the offline script and the live engine.

Angle convention (psi, degrees): 0 = feed direction (towards the chuck), 90 = radially outward.
All functions are pure: they take already-scaled grey frames (longer side 960 px) and tip positions.
"""
import math
from functools import lru_cache

import cv2
import numpy as np

SCALE_MAX = 960
TPL = 120            # template size (px)
SEARCH = 90          # search radius around previous position (px)
R_IN, R_OUT = 25, 220
DIFF_THR = 18        # grey-level threshold for "moving" pixels
PSI_RANGE = (-60, 150)
MIN_PIXELS = 40


def rot(v, deg):
    a = math.radians(deg)
    return np.array([v[0] * math.cos(a) - v[1] * math.sin(a), v[0] * math.sin(a) + v[1] * math.cos(a)])


def axis_angle(gray, tip, ref_deg):
    """Orientation (deg, mod 180) of the workpiece axis from long straight edges near the tip."""
    x0, y0 = int(tip[0]), int(tip[1])
    m = np.zeros_like(gray)
    cv2.circle(m, (x0, y0), 400, 255, -1)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120) & m
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, 80, minLineLength=180, maxLineGap=10)
    if lines is None:
        return None
    angs, wts = [], []
    for x1, y1, x2, y2 in lines[:, 0]:
        a = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
        d = (a - ref_deg + 90) % 180 - 90
        if abs(d) < 15:
            angs.append(ref_deg + d)
            wts.append(math.hypot(x2 - x1, y2 - y1))
    if not angs:
        return None
    return float(np.average(angs, weights=wts))


def axis_rotation(g, tip, ref_deg):
    """Rotation (deg) of the workpiece axis relative to the calibration frame; 0.0 if no line is found."""
    ax = axis_angle(g, tip, ref_deg)
    return 0.0 if ax is None else float(((ax - ref_deg + 90) % 180) - 90)


def track_step(g, tpl, prev_xy, search=SEARCH):
    """One template-matching step. Returns (centre_xy, score), or None if the search window leaves the frame."""
    h, w = g.shape
    x0 = int(max(0, prev_xy[0] - TPL // 2 - search)); x1 = int(min(w, prev_xy[0] + TPL // 2 + search))
    y0 = int(max(0, prev_xy[1] - TPL // 2 - search)); y1 = int(min(h, prev_xy[1] + TPL // 2 + search))
    win = g[y0:y1, x0:x1]
    if win.shape[0] < TPL or win.shape[1] < TPL:
        return None
    res = np.nan_to_num(cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED), nan=0.0, posinf=0.0, neginf=0.0)
    _, mx, _, loc = cv2.minMaxLoc(res)
    return np.array([x0 + loc[0] + TPL // 2, y0 + loc[1] + TPL // 2], float), float(mx)


def shift(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)


@lru_cache(maxsize=4)
def _grid(h, w):
    yy, xx = np.mgrid[0:h, 0:w]
    return yy, xx


def blur16(g):
    return cv2.GaussianBlur(g, (3, 3), 0).astype(np.int16)


def frame_features(g_prev, g_cur, g_next, g_raw, tip_prev, tip_cur, tip_next, u_f, u_o,
                   thr=DIFF_THR, r_in=R_IN, r_out=R_OUT):
    """Chip-direction features for the middle frame, or None if fewer than MIN_PIXELS moving pixels.

    g_prev/g_cur/g_next: blurred int16 grey frames (see blur16); g_raw: uint8 grey of the middle frame.
    """
    h, w = g_cur.shape
    yy, xx = _grid(h, w)
    tip = tip_cur
    a = shift(g_prev, *(tip_cur - tip_prev))
    b = shift(g_next, *(tip_cur - tip_next))
    motion = np.minimum(np.abs(g_cur - a), np.abs(g_cur - b)).astype(np.float32)
    dx, dy = xx - tip[0], yy - tip[1]
    r = np.hypot(dx, dy)
    along_f = dx * u_f[0] + dy * u_f[1]
    along_o = dx * u_o[0] + dy * u_o[1]
    psi = np.degrees(np.arctan2(along_o, along_f))
    sel = (r > r_in) & (r < r_out) & (along_o > -5) & (psi > PSI_RANGE[0]) & (psi < PSI_RANGE[1]) & (motion > thr)
    wts = motion[sel] - thr
    n_px = int(sel.sum())
    if n_px < MIN_PIXELS:
        return None
    hist, edges = np.histogram(psi[sel], bins=np.arange(PSI_RANGE[0], PSI_RANGE[1] + 1, 2), weights=wts)
    k = np.exp(-0.5 * (np.arange(-7, 8) * 2 / 6.0) ** 2)
    hs = np.convolve(hist, k / k.sum(), mode="same")
    peak = float(edges[np.argmax(hs)] + 1)
    mean = float(np.average(psi[sel], weights=wts))
    ang = np.radians(psi[sel])
    R = np.hypot(np.average(np.cos(ang), weights=wts), np.average(np.sin(ang), weights=wts))
    spread = float(np.degrees(np.sqrt(-2 * np.log(max(R, 1e-6)))))
    reach = float(np.average(r[sel], weights=wts) / r_out)
    gy, gx = np.gradient(cv2.GaussianBlur(motion, (5, 5), 0))
    jxx, jyy, jxy = (np.sum(wts * gx[sel] ** 2), np.sum(wts * gy[sel] ** 2), np.sum(wts * gx[sel] * gy[sel]))
    streak = 0.5 * math.atan2(2 * jxy, jxx - jyy) + math.pi / 2
    sv = np.array([math.cos(streak), math.sin(streak)])
    ps = math.degrees(math.atan2(sv @ u_o, sv @ u_f))
    psi_streak = ps if ps >= 0 else ps + 180
    coherence = float(math.hypot(jxx - jyy, 2 * jxy) / (jxx + jyy + 1e-9))
    ring = (r > r_in) & (r < r_out) & (along_o > -5) & (psi > PSI_RANGE[0]) & (psi < 90)
    clutter = float((cv2.Canny(g_raw, 60, 160) > 0)[ring].mean())
    return dict(psi_peak=peak, psi_mean=mean, psi_streak=psi_streak, coherence=coherence, spread=spread,
                reach=reach, clutter=clutter, n_px=n_px, activity=float(wts.sum()), sel=sel)
