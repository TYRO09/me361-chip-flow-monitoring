"""Causal chip-angle engine: calibrate once, then push frames one by one.

The three-frame difference needs a neighbour on each side, so the state returned by push() describes the
frame one step behind (about 33 ms at 30 fps). Everything uses only frames seen so far.
"""
import math
import os
import sys
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))
from chip_kernel import (SCALE_MAX, TPL, SEARCH, rot, axis_rotation, track_step, blur16, frame_features)  # noqa: E402

MAX_GAP_S = 0.12          # neighbours further apart than this (dropped frames) -> no motion estimate
MIN_TRACK = 0.5           # template-match score below this = tool lost
BASE_S, THR, HOLD = 4.0, 3.0, 30     # baseline length (s); alarm threshold; 30 samples = 1 s at 30 fps
REFRESH_SCORE, REFRESH_ALPHA = 0.8, 0.03   # template refresh: only on very confident matches, slowly
FEATS = ("log_act", "spread", "clutter")


def scale_frame(bgr):
    s = SCALE_MAX / max(bgr.shape[:2])
    return cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


@dataclass
class Calibration:
    tip: tuple            # tool-tip pixel in the 960-px-scaled calibration frame
    feed_dir: tuple       # unit vector along the feed direction (image coords)
    out_dir: tuple        # unit vector radially outward
    ref_frame: np.ndarray  # the calibration frame (any size; it is scaled like the stream)
    mm_per_px: float = None
    start_tip: tuple = None   # where the tip is in the FIRST stream frame, if that differs from `tip` (replay only)


def _unit(v):
    v = np.array(v, float)
    return v / np.linalg.norm(v)


class ChipEngine:
    def __init__(self, calib: Calibration):
        ref = scale_frame(calib.ref_frame)
        self.landscape = ref.shape[1] >= ref.shape[0]
        self.feed0, self.out0 = _unit(calib.feed_dir), _unit(calib.out_dir)
        self.ref_axis = math.degrees(math.atan2(self.feed0[1], self.feed0[0])) % 180
        tip = np.array(calib.tip, float)
        c = tip + 70 * self.out0 - 45 * self.feed0
        cx, cy = int(c[0]), int(c[1])
        g = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        self.tpl = g[cy - TPL // 2: cy + TPL // 2, cx - TPL // 2: cx + TPL // 2].copy()
        self.off = tip - np.array([cx, cy], float)
        self.prev = (np.array(calib.start_tip, float) - self.off) if calib.start_tip is not None else np.array([cx, cy], float)
        self.buf = deque(maxlen=3)
        self.psi_hist = deque(maxlen=15)
        self.reset_cut(active=False)

    # ---------------- cut bookkeeping (build-up score) ----------------
    def reset_cut(self, active=True):
        self.cut_active, self.cut_t0 = active, None
        self.base, self.stats = [], None
        self.scores = deque(maxlen=30)
        self.run_len, self.alarm, self.alarm_t = 0, False, None

    def end_cut(self):
        self.cut_active = False

    def _build_up(self, t, ft):
        if not self.cut_active or ft is None:
            return float("nan")
        if self.cut_t0 is None:
            self.cut_t0 = t
        x = np.array([math.log(ft["activity"] + 1), ft["spread"], ft["clutter"]])
        if t < self.cut_t0 + BASE_S:
            self.base.append(x)
            return float("nan")
        if self.stats is None:
            if len(self.base) < 10:
                return float("nan")
            b = np.array(self.base)
            med = np.median(b, 0)
            self.stats = (med, 1.4826 * np.median(np.abs(b - med), 0) + 1e-9)
        z = (x - self.stats[0]) / self.stats[1]
        self.scores.append(float(z.mean()))
        if len(self.scores) < 10:
            return float("nan")
        sc = float(np.median(self.scores))
        self.run_len = self.run_len + 1 if sc > THR else 0
        if self.run_len >= HOLD and not self.alarm:
            self.alarm, self.alarm_t = True, t
        return sc

    def _refresh(self, g, cur):
        """Let the template follow slow appearance change (chips, lighting) without drifting off the insert."""
        cx, cy = int(cur[0]), int(cur[1])
        patch = g[cy - TPL // 2: cy + TPL // 2, cx - TPL // 2: cx + TPL // 2]
        if patch.shape == self.tpl.shape:
            self.tpl = cv2.addWeighted(self.tpl, 1 - REFRESH_ALPHA, patch, REFRESH_ALPHA, 0)

    # ---------------- per frame ----------------
    def _state(self, t, **kw):
        s = dict(t=t, valid=False, tracking_lost=False, gap=False, orientation_changed=False, psi_raw=float("nan"),
                 psi=float("nan"), tip=None, u_f=None, u_o=None, activity=0.0, spread=float("nan"),
                 clutter=float("nan"), n_px=0, score=float("nan"), alarm=self.alarm, alarm_t=self.alarm_t, drot=0.0)
        s.update(kw)
        return s

    def push(self, bgr, t):
        im = scale_frame(bgr)
        if (im.shape[1] >= im.shape[0]) != self.landscape:
            return self._state(t, orientation_changed=True)
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        st = track_step(g, self.tpl, self.prev, SEARCH)
        if st is None:
            tip, score = None, 0.0
        else:
            cur, score = st
            if score > MIN_TRACK:
                self.prev = cur
            if score > REFRESH_SCORE:
                self._refresh(g, cur)
            tip = cur + self.off
        self.buf.append((t, g, blur16(g), tip, score))
        if len(self.buf) < 3:
            return None
        (t0, _, b0, tp0, s0), (t1, g1, b1, tp1, s1), (t2, _, b2, tp2, s2) = self.buf
        lost = tp1 is None or s1 < MIN_TRACK
        if t2 - t0 > MAX_GAP_S:
            return self._state(t1, gap=True, tracking_lost=lost)
        if tp0 is None or tp1 is None or tp2 is None or min(s0, s1, s2) < MIN_TRACK:
            self.psi_hist.append(float("nan"))
            return self._state(t1, tracking_lost=lost, score=self._build_up(t1, None))
        drot = axis_rotation(g1, tp1, self.ref_axis)
        u_f, u_o = rot(self.feed0, drot), rot(self.out0, drot)
        ft = frame_features(b0, b1, b2, g1, tp0, tp1, tp2, u_f, u_o)
        if ft is None:
            self.psi_hist.append(float("nan"))
            return self._state(t1, score=self._build_up(t1, None))
        self.psi_hist.append(ft["psi_peak"])
        h = np.array(self.psi_hist)
        psi = float(np.nanmedian(h)) if np.isfinite(h).sum() >= 3 else float("nan")
        sc = self._build_up(t1, ft)
        return self._state(t1, valid=True, psi_raw=ft["psi_peak"], psi=psi, tip=(float(tp1[0]), float(tp1[1])),
                           u_f=u_f.tolist(), u_o=u_o.tolist(), activity=ft["activity"], spread=ft["spread"],
                           clutter=ft["clutter"], n_px=ft["n_px"], score=sc, drot=drot)
