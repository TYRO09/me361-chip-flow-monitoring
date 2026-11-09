"""Replay of the recorded runs R1-R6 through the same engine as the live phone stream."""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
import measure_chip_angle as M  # noqa: E402
from chip_kernel import track_step, TPL, SEARCH  # noqa: E402
from entangle_indicator import CUT  # noqa: E402  (cut start/end of each run, seconds)
from app.engine import Calibration, scale_frame  # noqa: E402

VID = os.path.join(M.SESSION, "videos_top")


_CAL = {}


def _calibrate(run):
    """(Calibration, index of the first streamed frame inside the window).

    The hand-read tip/feed/outward directions belong to the reference frame (ref_t). They are carried back by
    tracking the insert backwards until the match stays confident (score > 0.5); the earliest such frame is the
    calibration frame. This mirrors what
    the user does in the live app: tap the tip and the feed direction on a preview frame. The template is cut
    from the reference frame itself (exact hand-read tip); `start_tip` tells the engine where the tool is in the
    first streamed frame.
    """
    if run in _CAL:
        return _CAL[run]
    cfg = M.RUNS[run]
    t0, _ = cfg["window"]
    cap = cv2.VideoCapture(os.path.join(VID, cfg["file"]))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t0 * fps))
    ref_idx = int(round((cfg["ref_t"] - t0) * fps))
    raw, grey = [], []
    for i in range(ref_idx + 1):
        ok, fr = cap.read()
        if not ok:
            break
        raw.append(fr)
        grey.append(cv2.cvtColor(scale_frame(fr), cv2.COLOR_BGR2GRAY))
    cap.release()
    feed, out = np.array(cfg["feed"], float), np.array(cfg["out"], float)
    c = np.array(cfg["tip"]) + 70 * out - 45 * feed
    cx, cy = int(c[0]), int(c[1])
    tpl = grey[-1][cy - TPL // 2: cy + TPL // 2, cx - TPL // 2: cx + TPL // 2]
    off = np.array(cfg["tip"], float) - np.array([cx, cy], float)
    prev, pos, i_cal = np.array([cx, cy], float), {len(grey) - 1: np.array([cx, cy], float)}, len(grey) - 1
    for i in range(len(grey) - 2, -1, -1):
        st = track_step(grey[i], tpl, prev, SEARCH)
        if st is None or st[1] <= 0.5:
            break
        prev = st[0]
        pos[i], i_cal = st[0], i
    tip0 = pos[i_cal] + off
    # directions stay as hand-read at the reference frame: per-frame workpiece rotation is measured against them,
    # exactly as in the offline script (so the replay is a like-for-like regression check).
    cal = Calibration(tip=tuple(cfg["tip"]), feed_dir=tuple(feed), out_dir=tuple(out), ref_frame=raw[-1], start_tip=tuple(tip0))
    _CAL[run] = (cal, i_cal)
    return _CAL[run]


def calibration_for(run):
    return _calibrate(run)[0]


def frames(run):
    """Yield (raw BGR frame, time in s) from the calibration frame to the end of the measurement window."""
    cfg = M.RUNS[run]
    t0, t1 = cfg["window"]
    i_cal = _calibrate(run)[1]
    cap = cv2.VideoCapture(os.path.join(VID, cfg["file"]))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t0 * fps) + i_cal)
    for i in range(i_cal, int((t1 - t0) * fps)):
        ok, fr = cap.read()
        if not ok:
            break
        yield fr, t0 + i / fps
    cap.release()
