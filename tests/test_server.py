"""Task 4: server, phone/dashboard websockets, replay."""
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

from app import replay, server  # noqa: E402  (RED until server exists)
from app.engine import scale_frame  # noqa: E402


@pytest.fixture()
def client():
    server.session.reset()
    with TestClient(server.app) as c:
        yield c
    server.session.reset()


def jpeg_msgs(run, n):
    out = []
    for i, (bgr, t) in enumerate(replay.frames(run)):
        ok, buf = cv2.imencode(".jpg", scale_frame(bgr), [cv2.IMWRITE_JPEG_QUALITY, 70])
        out.append(b"\x01" + struct.pack(">d", t * 1000) + buf.tobytes())
        if i + 1 >= n:
            break
    return out


def read_until(ws, pred, limit=400, seen=None):
    for _ in range(limit):
        m = ws.receive()
        if m.get("text"):
            j = json.loads(m["text"])
            if seen is not None:
                seen.append(j)
            if pred(j):
                return j
    raise AssertionError("expected message never arrived")


def test_replay_streams_states(client):
    with client.websocket_connect("/ws/dash") as dash:
        assert client.post("/api/replay/R2?speed=8").status_code == 200
        s = read_until(dash, lambda j: j.get("type") == "state" and j.get("chip", {}).get("valid"))
        assert s["mode"] == "replay" and s["chip"]["psi_raw"] is not None
        assert s["audio"] is not None
        client.post("/api/stop")


def test_phone_frames_roundtrip(client):
    server.session.set_calibration(replay.calibration_for("R1"))
    with client.websocket_connect("/ws/dash") as dash, client.websocket_connect("/ws/phone") as phone:
        phone.send_text(json.dumps({"hello": "phone", "audio_sr": 16000, "fps": 30, "rtt_ms": 20}))
        for m in jpeg_msgs("R1", 60):
            phone.send_bytes(m)
        s = read_until(dash, lambda j: j.get("type") == "state" and j["chip"]["valid"])
        assert s["mode"] == "live" and s["latency_ms"] >= 10 and s["fps"] > 0


def test_second_phone_refused(client):
    with client.websocket_connect("/ws/phone") as p1:
        p1.send_text(json.dumps({"hello": "phone", "audio_sr": 16000}))
        with client.websocket_connect("/ws/phone") as p2:
            p2.send_text(json.dumps({"hello": "phone", "audio_sr": 16000}))
            j = json.loads(p2.receive()["text"])
            assert j["type"] == "refused"
            m = p2.receive()
            assert m["type"] == "websocket.close" and m["code"] == 4001


def test_phone_reconnect_keeps_calibration(client):
    cal = replay.calibration_for("R1")
    server.session.set_calibration(cal)
    with client.websocket_connect("/ws/phone") as p:
        p.send_text(json.dumps({"hello": "phone", "audio_sr": 16000}))
        for m in jpeg_msgs("R1", 5):
            p.send_bytes(m)
    engine_before = server.session.live.engine
    with client.websocket_connect("/ws/phone") as p:           # phone reconnects
        p.send_text(json.dumps({"hello": "phone", "audio_sr": 16000}))
        for m in jpeg_msgs("R1", 5):
            p.send_bytes(m)
    assert server.session.calibration is cal and server.session.live.engine is engine_before


def test_mic_off_gives_null_audio(client):
    server.session.set_calibration(replay.calibration_for("R1"))
    with client.websocket_connect("/ws/dash") as dash, client.websocket_connect("/ws/phone") as phone:
        phone.send_text(json.dumps({"hello": "phone", "audio_sr": None, "fps": 30}))
        for m in jpeg_msgs("R1", 60):
            phone.send_bytes(m)
        seen = []
        s = read_until(dash, lambda j: j.get("type") == "state" and j["chip"]["valid"], seen=seen)
        assert s["audio"] is None
        assert any(j.get("type") == "status" and j["mic"] is False for j in seen)


def test_qr_and_config(client):
    r = client.get("/api/qr")
    assert r.status_code == 200 and b"<svg" in r.content
    c = client.get("/api/config?d=24.8&vc=150").json()
    assert c["rpm_expected"] == pytest.approx(150000 / (np.pi * 24.8), rel=1e-3) or c["rpm_expected"] == 2500
    assert c["phone_url"].startswith("https://")
