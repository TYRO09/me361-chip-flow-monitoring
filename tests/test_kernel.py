"""Task 1: the shared per-frame kernel must reproduce the saved offline results exactly."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))

import chip_kernel  # noqa: E402  (RED until the kernel exists)
import measure_chip_angle as M  # noqa: E402


def test_kernel_exports_interface():
    for name in ("frame_features", "track_step", "axis_rotation"):
        assert callable(getattr(chip_kernel, name))


def test_kernel_matches_saved_csv():
    rows = pd.DataFrame(M.measure_run("R2", M.RUNS["R2"], save_overlays=0))
    saved = pd.read_csv(os.path.join(ROOT, "results", "angles", "R2_frames.csv"))
    a = rows[rows.valid == 1].set_index("t")
    b = saved[saved.valid == 1].set_index("t")
    assert list(a.index) == list(b.index)
    for col in ("psi_peak", "activity", "clutter", "spread"):
        assert np.allclose(a[col].astype(float), b[col].astype(float), atol=0.011), col
