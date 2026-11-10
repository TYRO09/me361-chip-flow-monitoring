"""Unit checks for the iteration-3 methods: rake-face band direction and the label-changing rotation."""
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))

import rake_face_flow as R  # noqa: E402


def _band(angle_deg, width=6, value=40.0):
    img = np.zeros((R.S, R.S), np.float32)
    u = np.array([math.cos(math.radians(angle_deg)), -math.sin(math.radians(angle_deg))])
    n = np.array([-u[1], u[0]])
    for r in range(10, 100):
        for o in range(-width // 2, width // 2 + 1):
            x, y = (R.TIP + u * r + n * o).round().astype(int)
            img[y, x] = value
    return img


def test_band_direction_is_found_on_the_insert_side():
    rng = np.random.default_rng(0)
    for ang in (110, 129, 150):
        psi, z = R.peak(_band(ang) + rng.normal(0, 3, (R.S, R.S)).astype(np.float32))
        assert abs(psi - ang) <= 2, (ang, psi)
        assert z > 2


def test_band_in_the_air_is_not_reported_as_rake_face():
    psi, _ = R.peak(_band(55))          # ejection side only: no insert-side band
    assert not (abs(psi - 55) <= 2)


def test_rotation_augmentation_adds_delta_to_the_chip_angle():
    import torch
    import cnn_v3_local as C
    W = 160
    img = torch.zeros(1, 1, W, W)
    tip = (C.TIP[0] * W, C.TIP[1] * W)
    for r in range(15, 60):
        x = int(tip[0] + r * math.cos(math.radians(40))); y = int(tip[1] - r * math.sin(math.radians(40)))
        img[0, 0, y, x] = 1
    out = C.rotate(img, torch.tensor([25.0]))[0, 0]
    yy, xx = torch.nonzero(out > 0.2, as_tuple=True)
    ang = np.degrees(np.arctan2(-(yy.float() - tip[1]).numpy(), (xx.float() - tip[0]).numpy()))
    assert abs(np.median(ang) - 65) < 3
