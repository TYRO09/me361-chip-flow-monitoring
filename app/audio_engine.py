"""Streaming audio sensor: spindle speed, cutting on/off and band-energy features from a microphone stream.

Same maths as code/audio_sensor.py, run causally: a 2-s window every 0.5 s for the rpm (harmonic-sum pitch
search), and 50-ms bins of 2-6 kHz band level for the cutting state. Audio arrives as float32 mono at an integer
multiple of 16 kHz and is decimated with a stateful filter, so there are no chunk-edge clicks in the band.
"""
import os
import sys
from collections import deque

import numpy as np
from scipy import signal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))
import audio_sensor as A  # noqa: E402

SR = A.SR                       # 16000
BIN = int(0.05 * SR)            # 50-ms bins
WIN = 2 * SR                    # rpm window
HOP = SR // 2                   # rpm every 0.5 s
CUT_MARGIN_DB = 5.5             # cutting = band level above the idle floor by this much (idle -41..-38 dB, cut -24..-31 dB)


class AudioEngine:
    def __init__(self, sr_in=SR):
        if sr_in % SR:
            raise ValueError("sample rate must be a multiple of 16000 (the phone page sends 16000 or 48000)")
        self.q = sr_in // SR
        self.aa = signal.butter(6, 0.45 * SR, "low", fs=sr_in, output="sos") if self.q > 1 else None
        self.aa_zi = np.zeros((self.aa.shape[0], 2)) if self.q > 1 else None
        self.phase = 0
        self.bp = signal.butter(4, [2000, 6000], "bandpass", fs=SR, output="sos")
        self.bp_zi = np.zeros((self.bp.shape[0], 2))
        self.ring = np.zeros(0, np.float32)
        self.total = 0                       # samples at 16 kHz received so far
        self.left = np.zeros(0)              # band-filtered samples not yet binned
        self.bins = deque(maxlen=400)        # last 20 s of 50-ms band levels (dB)
        self.all_bins = []
        self.idle_db = None
        self.rpm = float("nan")
        self.next_tacho = WIN

    def push(self, pcm):
        x = np.asarray(pcm, np.float64)
        if self.q > 1:
            x, self.aa_zi = signal.sosfilt(self.aa, x, zi=self.aa_zi)
            x = x[(-self.phase) % self.q::self.q]
            self.phase = (self.phase + len(pcm)) % self.q
        self.ring = np.concatenate([self.ring, x.astype(np.float32)])[-8 * SR:]
        self.total += len(x)
        y, self.bp_zi = signal.sosfilt(self.bp, x, zi=self.bp_zi)
        y = np.concatenate([self.left, y ** 2])
        n = len(y) // BIN
        if n:
            db = 10 * np.log10(y[: n * BIN].reshape(n, BIN).mean(1) + 1e-12)
            self.bins.extend(db); self.all_bins.extend(db)
        self.left = y[n * BIN:]
        while self.total >= self.next_tacho and len(self.ring) >= WIN:
            # window ends at sample `next_tacho`; the ring ends at `total`
            end = len(self.ring) - (self.total - self.next_tacho)
            f0, _ = A.f0_estimate(self.ring[end - WIN:end])
            self.rpm = f0 * 60
            self.next_tacho += HOP

    def _floor(self):
        if self.idle_db is not None:
            return self.idle_db
        return float(np.percentile(self.all_bins, 10)) if len(self.all_bins) >= 100 else None

    def calibrate_idle(self):
        """Fix the noise floor from the last ~3 s (spindle running, not cutting) and report the detected rpm."""
        b = list(self.bins)[-60:]
        if len(b) < 20 or len(self.ring) < WIN:
            return dict(idle_db=None, rpm_detected=None)
        self.idle_db = float(np.median(b))
        f0, _ = A.f0_estimate(self.ring[-WIN:])
        return dict(idle_db=self.idle_db, rpm_detected=float(f0 * 60))

    def state(self):
        if self.total < WIN or len(self.bins) < 9:
            return None
        b = np.array(self.bins)
        band = float(np.median(b[-9:]))          # causal version of the batch 9-bin median filter
        floor = self._floor()
        return dict(rpm=self.rpm, cutting=None if floor is None else bool(band > floor + CUT_MARGIN_DB),
                    band_db=band, band_var_db=float(np.std(b[-20:])), buildup_audio=None)
