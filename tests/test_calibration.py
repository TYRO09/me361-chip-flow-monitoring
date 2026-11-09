"""Task 5: calibration, setup, idle-noise check and run summary endpoints."""
import json
import os
import struct
import sys

import cv2
import numpy as np
import pytest
from starlette.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, ROOT)

from app import replay, server  # noqa: E402
from app.engine import scale_frame  # noqa: E402
import audio_sensor as A  # noqa: E402
from tests.test_server import jpeg_msgs, read_until  # noqa: E402


@pytest.fixture()
def client():
    server.session.reset()
    with TestClient(server.app) as c:
        yield c
    server.session.reset()


def phone_with_frames(client, n=3, run="R1", audio_sr=16000):
    ws = client.websocket_connect("/ws/phone")
    p = ws.__enter__()
    p.send_text(json.dumps({"hello": "phone", "audio_sr": audio_sr, "fps": 30}))
    for m in jpeg_msgs(run, n):
        p.send_bytes(m)
    for _ in range(n):
        p.receive()                                   # acks: the frames have been processed
    return ws, p


def body_for(run="R1", feed_len=0.1, shift=(0.0, 0.0)):
    cal = replay.calibration_for(run)
    h, w = scale_frame(cal.ref_frame).shape[:2]
    tip = np.array(cal.tip) / [w, h] + shift
    f, o = np.array(cal.feed_dir), np.array(cal.out_dir)
    return dict(tip=tip.tolist(), feed_pt=(tip + feed_len * f).tolist(), work_pt=(tip - 0.2 * o).tolist()), cal


def test_calibrate_normalised_points(client):
    ws, p = phone_with_frames(client)
    try:
        body, cal = body_for()
        r = client.post("/api/calibrate", json=body)
        assert r.status_code == 200 and r.json()["ok"], r.text
        c = server.session.calibration
        assert np.allclose(c.tip, cal.tip, atol=1.0)
        assert np.allclose(c.feed_dir, cal.feed_dir, atol=0.01) and np.allclose(c.out_dir, cal.out_dir, atol=0.01)
        assert client.get("/api/config").json()["calibrated"] is True
    finally:
        ws.__exit__(None, None, None)


def test_calibrate_needs_a_camera_frame(client):
    body, _ = body_for()
    r = client.post("/api/calibrate", json=body)
    assert r.status_code == 409 and "frame" in r.json()["error"]


def test_calibrate_warns_tip_near_edge_and_rejects_template_off_frame(client):
    ws, p = phone_with_frames(client)                 # R1 frames are landscape, 960 x 540 after scaling
    try:
        near = dict(tip=[0.93, 0.5], feed_pt=[0.96, 0.5], work_pt=[0.93, 0.7])   # travelling right, 7% from the edge
        r = client.post("/api/calibrate", json=near)
        assert r.status_code == 200 and "tip_near_edge" in r.json()["warnings"], r.text
        mid = dict(tip=[0.4, 0.5], feed_pt=[0.45, 0.5], work_pt=[0.4, 0.7])      # plenty of room: no warning
        assert "tip_near_edge" not in client.post("/api/calibrate", json=mid).json()["warnings"]
        corner = dict(tip=[0.01, 0.01], feed_pt=[0.1, 0.01], work_pt=[0.01, 0.2])
        r3 = client.post("/api/calibrate", json=corner)
        assert r3.status_code == 422 and "edge" in r3.json()["error"]
    finally:
        ws.__exit__(None, None, None)


def test_rotation_after_calibration_flagged(client):
    ws, p = phone_with_frames(client)
    try:
        body, _ = body_for()
        assert client.post("/api/calibrate", json=body).status_code == 200
        with client.websocket_connect("/ws/dash") as dash:
            for bgr, t in list(__import__("itertools").islice(replay.frames("R1"), 8)):
                ok, buf = cv2.imencode(".jpg", scale_frame(np.ascontiguousarray(np.rot90(bgr))))
                p.send_bytes(b"\x01" + struct.pack(">d", t * 1000) + buf.tobytes())
            s = read_until(dash, lambda j: j.get("type") == "state")
            assert s["chip"]["orientation_changed"] is True and "orientation_changed" in s["warnings"]
    finally:
        ws.__exit__(None, None, None)


def test_idle_noise_rpm_check(client):
    x = A.load("R1")
    with client.websocket_connect("/ws/phone") as p:
        p.send_text(json.dumps({"hello": "phone", "audio_sr": 16000, "fps": 30}))
        for i in range(0, 16000 * 8, 1600):               # first 8 s of R1: the spindle has settled at ~1500 rpm
            p.send_bytes(b"\x02" + x[i:i + 1600].astype("<f4").tobytes())
        p.send_bytes(jpeg_msgs("R1", 1)[0])
        assert json.loads(p.receive()["text"])["type"] == "ack"      # messages are handled in order: the audio is in
        ref = A.f0_estimate(x[16000 * 6:16000 * 8])[0] * 60
        client.post("/api/setup", json=dict(d_mm=20, f=0.1, ap=3.0, vc=150))     # big ap makes the pre-cut diameter matter
        # idle sound is heard BEFORE the cut, on the larger diameter d + 2 ap (G96 speed is set by the current diameter)
        d_ok = 150000 / (np.pi * ref) - 2 * server.session.params["ap"]
        r = client.post("/api/idle_noise", json=dict(d_mm=d_ok, vc=150))
        assert r.status_code == 200 and r.json()["ok"] and abs(r.json()["rpm_detected"] - ref) / ref < 0.02, r.text
        r = client.post("/api/idle_noise", json=dict(d_mm=2 * (d_ok + 6.0), vc=150))   # commanded rpm half of what is heard
        assert not r.json()["ok"] and "rpm_mismatch" in r.json()["warnings"]
        assert server.session.idle_db is not None


def test_idle_noise_without_mic(client):
    with client.websocket_connect("/ws/phone") as p:
        p.send_text(json.dumps({"hello": "phone", "audio_sr": None, "fps": 30}))
        r = client.post("/api/idle_noise", json=dict(d_mm=24.8, vc=150))
        assert r.status_code == 409 and "microphone" in r.json()["error"]


def test_setup_returns_prediction_and_extrapolation(client):
    r = client.post("/api/setup", json=dict(d_mm=24.8, f=0.15, ap=0.6, vc=150, material="AISI 1020", r_eps=0.4)).json()
    assert r["predicted_psi"] == pytest.approx(26.8 + 101 * 0.15 + 17.7 * 0.6, abs=0.05)
    assert r["extrapolation"] is False and r["rpm_expected"] == pytest.approx(150000 / (np.pi * 24.8), rel=1e-3)
    r = client.post("/api/setup", json=dict(d_mm=24.8, f=0.4, ap=2.0, vc=150, material="AISI 1020", r_eps=0.4)).json()
    assert r["extrapolation"] is True


def test_summary_has_measured_and_predicted(client):
    client.post("/api/setup", json=dict(d_mm=24.8, f=0.15, ap=0.6, vc=150, material="AISI 1020", r_eps=0.4))
    with client.websocket_connect("/ws/dash") as dash:
        client.post("/api/replay/R6?speed=0")
        seen = False
        for _ in range(3000):
            m = dash.receive()
            if m.get("text"):
                j = json.loads(m["text"])
                seen = seen or j["type"] == "state"
                if seen and j["type"] == "status" and j["mode"] is None:
                    break
    s = client.get("/api/summary").json()
    assert s["predicted_psi"] == pytest.approx(52.6, abs=0.2)
    assert s["measured_psi_median"] is not None and 20 < s["measured_psi_median"] < 90
    assert s["valid_frames"] > 100 and s["alarm_t"] is None
    csv = client.get("/api/summary.csv").text
    assert csv.splitlines()[0].startswith("t,") and len(csv.splitlines()) > 100


def test_calibrate_accepts_template_that_fits_even_close_to_the_left_edge(client):
    ws, p = phone_with_frames(client)                 # 960 x 540 after scaling
    try:
        # template centre = tip + 70*out - 45*feed = (100, 270): 100 px from the edge, template (+-60) still fits
        tip = np.array([145.0, 340.0]) / [960, 540]
        body = dict(tip=tip.tolist(), feed_pt=(tip + [0.1, 0]).tolist(), work_pt=(tip + [0, 0.2]).tolist())
        r = client.post("/api/calibrate", json=body)
        assert r.status_code == 200 and r.json()["ok"], r.text
    finally:
        ws.__exit__(None, None, None)


def test_setup_and_config_reject_bad_numbers(client):
    assert client.post("/api/setup", json=dict(d_mm="abc")).status_code == 422
    assert client.post("/api/setup", json=dict(d_mm=0)).status_code == 422
    assert client.post("/api/setup", json=dict(f=-0.1)).status_code == 422
    assert client.get("/api/config?d=0").status_code == 422
    assert client.post("/api/idle_noise", json=dict(d_mm=0)).status_code in (409, 422)
