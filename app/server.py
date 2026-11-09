"""ME361 live chip-flow monitor: one Starlette app serving the dashboard (/), the phone page (/phone) and websockets.

Phone  -> /ws/phone : text JSON (hello, rtt, tilt) and binary messages
                      0x01 + float64 big-endian capture time (ms) + JPEG      (video frame)
                      0x02 + float32 little-endian mono PCM at 16 kHz         (microphone)
Server -> /ws/phone : {"type": "ack"} after each processed frame (flow control), or {"type": "refused"}
Server -> /ws/dash  : text JSON {"type": "state" | "status" | "tilt"} and binary 0x01 + JPEG preview frames
"""
import asyncio
import io
import json
import math
import os
import socket
import struct
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "code"))
import audio_sensor  # noqa: E402
from app import replay as replay_mod  # noqa: E402
from app.audio_engine import AudioEngine  # noqa: E402
from app.engine import Calibration, ChipEngine, scale_frame  # noqa: E402
import measure_chip_angle as M  # noqa: E402
import physics_flow_angle  # noqa: E402

PORT = int(os.environ.get("APP_PORT", "8443"))
PREVIEW_FPS = 15
CUT_ON_S, CUT_OFF_S = 0.3, 1.0      # audio must say "cutting" this long to start a cut / "not cutting" to end it
PHONE_STALE_S = 5.0
CUT_LENGTH_MM = 61.0                # Z+1 -> Z-60
MODEL = (26.8, 101.0, 17.7)         # fitted M1: psi = 26.8 + 101 f + 17.7 ap
TESTED = dict(f=(0.08, 0.15), ap=(0.3, 0.9))
DEFAULT_PARAMS = dict(d_mm=24.8, f=0.15, ap=0.6, vc=150.0, material="AISI 1020 (assumed)", r_eps=0.4)


def _pos(v):
    x = float(v)
    if not math.isfinite(x) or x <= 0:
        raise ValueError("must be a positive number")
    return x


def rpm_for(d_mm, vc):
    return min(2500.0, 1000.0 * vc / (math.pi * d_mm))


def predict_psi(f, ap):
    return MODEL[0] + MODEL[1] * f + MODEL[2] * ap


def extrapolating(f, ap):
    eps = 1e-6
    return not (TESTED["f"][0] - eps <= f <= TESTED["f"][1] + eps and TESTED["ap"][0] - eps <= ap <= TESTED["ap"][1] + eps)


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))        # no packet is sent; this only picks the outgoing interface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def clean(o):
    """JSON-safe copy: numpy -> python, NaN/inf -> None."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


class Pipeline:
    """Engine + audio + cut tracking for one source (the live phone, or a replay)."""

    def __init__(self, engine, audio):
        self.engine, self.audio = engine, audio
        self.cut_on, self.on_since, self.off_since = False, None, None

    def _cut_edges(self, t, cutting, valid):
        if cutting is None:                     # mic off: assume the cut starts with the first valid frame
            if not self.cut_on and valid:
                self.engine.reset_cut(); self.cut_on = True
            return
        if cutting:
            self.off_since = None
            self.on_since = t if self.on_since is None else self.on_since
            if not self.cut_on and t - self.on_since >= CUT_ON_S:
                self.engine.reset_cut(); self.cut_on = True
        else:
            self.on_since = None
            self.off_since = t if self.off_since is None else self.off_since
            if self.cut_on and t - self.off_since >= CUT_OFF_S:
                self.engine.end_cut(); self.cut_on = False

    def step(self, img, t):
        chip = self.engine.push(img, t)
        aud = self.audio.state() if self.audio is not None else None
        if chip is None:
            return None
        cutting = None if aud is None else aud["cutting"]
        self._cut_edges(t, cutting, chip["valid"])
        chip = dict(chip)
        chip["psi_display"] = chip["psi"] if (chip["valid"] and cutting is not False) else float("nan")
        warnings = [k for k in ("orientation_changed", "tracking_lost") if chip[k]] + (["frame_gap"] if chip["gap"] else [])
        return dict(chip=chip, audio=aud, cut_active=self.cut_on, warnings=warnings)


class Session:
    def __init__(self):
        self.exec = ThreadPoolExecutor(1)       # engine and audio state are only ever touched from this thread
        self.dashes = set()
        self.replay_task = None
        self.reset()

    def reset(self):
        self._cancel_replay()
        self.calibration, self.live = None, None
        self.audio, self.mic = None, False
        self.idle_db = None
        self.phone, self.phone_seen = None, 0.0
        self.rtt, self.mode = 0.0, None
        self.arrivals = deque(maxlen=60)
        self.tilt = None
        self.last_jpeg = None
        self.params = dict(DEFAULT_PARAMS)
        self.history = []

    def _cancel_replay(self):
        t = self.replay_task
        if t is not None and not t.done():
            t.get_loop().call_soon_threadsafe(t.cancel)
        self.replay_task = None

    # ---- calibration / pipelines ----
    def set_calibration(self, cal):
        self.calibration = cal
        self.live = Pipeline(ChipEngine(cal), self.audio)
        self.history = []

    def set_audio(self, sr):
        if sr is None:
            self.audio, self.mic = None, False
        else:
            self.audio, self.mic = AudioEngine(int(sr)), True
            if self.idle_db is not None:
                self.audio.idle_db = self.idle_db
        if self.live is not None:
            self.live.audio = self.audio

    def record(self, out):
        c = out["chip"]
        if out["cut_active"]:
            self.history.append(dict(t=out["t"], valid=c["valid"], psi_raw=c["psi_raw"], score=c["score"],
                                     alarm=c["alarm"], alarm_t=c["alarm_t"]))

    # ---- fan-out to dashboards ----
    def status(self):
        return dict(type="status", phone="connected" if self.phone is not None else "none", mic=self.mic,
                    mode=self.mode, calibrated=self.calibration is not None)

    async def broadcast(self, msg=None, jpeg=None):
        for ws in list(self.dashes):
            try:
                if msg is not None:
                    await ws.send_text(json.dumps(clean(msg)))
                if jpeg is not None:
                    await ws.send_bytes(b"\x01" + jpeg)
            except Exception:
                self.dashes.discard(ws)

    def fps(self):
        a = self.arrivals
        return 0.0 if len(a) < 2 or a[-1] == a[0] else (len(a) - 1) / (a[-1] - a[0])


session = Session()


# ----------------------------- websockets -----------------------------
async def ws_dash(ws: WebSocket):
    await ws.accept()
    session.dashes.add(ws)
    try:
        await ws.send_text(json.dumps(clean(session.status())))
        while True:
            m = await ws.receive()
            if m["type"] == "websocket.disconnect":
                break
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        session.dashes.discard(ws)


async def ws_phone(ws: WebSocket):
    await ws.accept()
    now = time.time()
    if session.phone is not None and now - session.phone_seen < PHONE_STALE_S:
        await ws.send_text(json.dumps(dict(type="refused", reason="another phone is already connected")))
        await ws.close(code=4001)
        return
    if session.mode == "replay":
        session._cancel_replay(); session.mode = None
    session.phone, session.phone_seen = ws, now
    loop = asyncio.get_running_loop()
    last_prev = 0.0
    try:
        while True:
            m = await ws.receive()
            if m["type"] == "websocket.disconnect":
                break
            session.phone_seen = time.time()
            if m.get("text"):
                j = json.loads(m["text"])
                if j.get("hello"):
                    session.set_audio(j.get("audio_sr"))
                    session.rtt = float(j.get("rtt_ms") or 0.0)
                    session.mode = "live"
                    await session.broadcast(session.status())
                elif "rtt_ms" in j:
                    session.rtt = float(j["rtt_ms"])
                elif "tilt" in j:
                    session.tilt = j["tilt"]
                    await session.broadcast(dict(type="tilt", **j))
            elif m.get("bytes"):
                b = m["bytes"]
                if b[0] == 2 and session.audio is not None:
                    pcm = np.frombuffer(b[1:], dtype="<f4")
                    await loop.run_in_executor(session.exec, session.audio.push, pcm)
                elif b[0] == 1:
                    t_recv = time.time()
                    session.arrivals.append(t_recv)
                    jpeg = b[9:]
                    session.last_jpeg = jpeg
                    t = struct.unpack(">d", b[1:9])[0] / 1000.0
                    if session.live is not None:
                        def work(jpeg=jpeg, t=t):
                            img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                            return session.live.step(img, t)
                        out = await loop.run_in_executor(session.exec, work)
                        if out is not None:
                            out.update(type="state", mode="live", t=t, fps=session.fps(),
                                       latency_ms=session.rtt / 2 + (time.time() - t_recv) * 1000)
                            session.record(out)
                            await session.broadcast(out)
                    if t_recv - last_prev >= 1.0 / PREVIEW_FPS:
                        last_prev = t_recv
                        await session.broadcast(jpeg=jpeg)
                    await ws.send_text('{"type":"ack"}')
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        if session.phone is ws:
            session.phone = None
            await session.broadcast(session.status())


# ----------------------------- replay -----------------------------
async def run_replay(run, speed):
    loop = asyncio.get_running_loop()
    cal = await loop.run_in_executor(session.exec, replay_mod.calibration_for, run)
    x = audio_sensor.load(run)
    audio = AudioEngine(16000)
    audio.push(x[:3 * 16000]); audio.calibrate_idle()
    pipe = Pipeline(ChipEngine(cal), audio)
    session.history = []
    row = pd.read_csv(os.path.join(M.SESSION, "runs.csv")).set_index("run_id").loc[run]
    session.params = dict(DEFAULT_PARAMS, d_mm=float(row.dia_after_mm), f=float(row.feed_mm_per_rev),
                          ap=float(row.depth_of_cut_mm), vc=float(row.Vc_m_per_min))
    session.mode = "replay"
    await session.broadcast(session.status())
    it = iter(replay_mod.frames(run))
    state = dict(apos=3 * 16000)
    wall0 = t_first = None

    def work(bgr, t):
        end = int(t * 16000)
        if end > state["apos"]:
            audio.push(x[state["apos"]:end]); state["apos"] = end
        out = pipe.step(bgr, t)
        ok, buf = cv2.imencode(".jpg", scale_frame(bgr), [cv2.IMWRITE_JPEG_QUALITY, 70])
        return out, buf.tobytes()

    try:
        while True:
            item = await loop.run_in_executor(session.exec, next, it, None)
            if item is None:
                break
            bgr, t = item
            t_first = t if t_first is None else t_first
            wall0 = time.time() if wall0 is None else wall0
            if speed > 0:
                delay = wall0 + (t - t_first) / speed - time.time()
                if delay > 0:
                    await asyncio.sleep(delay)
            t_recv = time.time()
            session.arrivals.append(t_recv)
            out, jpeg = await loop.run_in_executor(session.exec, work, bgr, t)
            if out is not None:
                out.update(type="state", mode="replay", t=t, fps=session.fps(), latency_ms=(time.time() - t_recv) * 1000)
                session.record(out)
                await session.broadcast(out, jpeg)
    finally:
        if session.mode == "replay":
            session.mode = None
            await session.broadcast(session.status())


async def api_replay(request):
    run = request.path_params["run"]
    if run not in replay_mod.CUT:
        return JSONResponse({"error": "unknown run"}, status_code=404)
    speed = float(request.query_params.get("speed", "1"))
    session._cancel_replay()
    session.replay_task = asyncio.create_task(run_replay(run, speed))
    return JSONResponse({"ok": True, "run": run})


async def api_stop(request):
    session._cancel_replay()
    return JSONResponse({"ok": True})


# ----------------------------- plain routes -----------------------------
def phone_url():
    return f"https://{lan_ip()}:{PORT}/phone"


async def api_qr(request):
    import qrcode
    import qrcode.image.svg
    buf = io.BytesIO()
    qrcode.make(phone_url(), image_factory=qrcode.image.svg.SvgPathImage).save(buf)
    return Response(buf.getvalue(), media_type="image/svg+xml")


async def api_config(request):
    try:
        d, vc = _pos(request.query_params.get("d", "30")), _pos(request.query_params.get("vc", "150"))
    except ValueError:
        return JSONResponse(dict(error="d and vc must be positive numbers"), status_code=422)
    return JSONResponse(dict(lan_ip=lan_ip(), phone_url=phone_url(), rpm_expected=rpm_for(d, vc),
                             runs=sorted(replay_mod.CUT), calibrated=session.calibration is not None,
                             params=session.params))


async def api_setup(request):
    b = await request.json()
    p = dict(DEFAULT_PARAMS)
    try:
        for k in p:
            if k in b:
                p[k] = str(b[k])[:60] if k == "material" else _pos(b[k])
    except (ValueError, TypeError):
        return JSONResponse(dict(error=f"{k} must be a positive number"), status_code=422)
    session.params = p
    return JSONResponse(dict(predicted_psi=predict_psi(p["f"], p["ap"]), rpm_expected=rpm_for(p["d_mm"], p["vc"]),
                             extrapolation=extrapolating(p["f"], p["ap"]), tested=TESTED, params=p))


def _ray_to_edge(tip, d, w, h):
    t = []
    for axis, size in ((0, w), (1, h)):
        if abs(d[axis]) > 1e-9:
            t.append(((size if d[axis] > 0 else 0) - tip[axis]) / d[axis])
    return min(t)


async def api_calibrate(request):
    if session.last_jpeg is None:
        return JSONResponse(dict(ok=False, error="no camera frame yet: connect the phone first"), status_code=409)
    b = await request.json()
    img = cv2.imdecode(np.frombuffer(session.last_jpeg, np.uint8), cv2.IMREAD_COLOR)
    h, w = scale_frame(img).shape[:2]
    P = lambda q: np.array([q[0] * w, q[1] * h], float)       # normalised 0-1 -> pixels of the 960-px frame
    tip, fp, wp = P(b["tip"]), P(b["feed_pt"]), P(b["work_pt"])
    feed = fp - tip
    if np.linalg.norm(feed) < 10:
        return JSONResponse(dict(ok=False, error="feed direction points are too close together"), status_code=422)
    feed /= np.linalg.norm(feed)
    n = np.array([-feed[1], feed[0]])
    out = n if np.dot(n, tip - wp) > 0 else -n                # outward = away from the workpiece
    c = tip + 70 * out - 45 * feed                            # centre of the tracking template (see ChipEngine)
    if not (60 <= c[0] <= w - 60 and 60 <= c[1] <= h - 60):
        return JSONResponse(dict(ok=False, error="tool tip is too close to the frame edge: re-aim the phone so the tip and "
                                                 "the area around it are well inside the picture"), status_code=422)
    mm_per_px = None
    if b.get("scale_pts") and b.get("scale_mm"):
        dpx = np.linalg.norm(P(b["scale_pts"][0]) - P(b["scale_pts"][1]))
        mm_per_px = float(b["scale_mm"]) / dpx if dpx > 5 else None
    warnings = []
    room = _ray_to_edge(tip, feed, w, h)
    need = CUT_LENGTH_MM / mm_per_px if mm_per_px else 0.15 * max(w, h)
    if room < need:
        warnings.append("tip_near_edge")
    cal = Calibration(tip=tuple(tip), feed_dir=tuple(feed), out_dir=tuple(out), ref_frame=img, mm_per_px=mm_per_px)
    session.set_calibration(cal)
    await session.broadcast(session.status())
    return JSONResponse(dict(ok=True, warnings=warnings, mm_per_px=mm_per_px, room_px=float(room)))


async def api_idle_noise(request):
    b = await request.json()
    try:
        d_mm, vc = _pos(b.get("d_mm", session.params["d_mm"])), _pos(b.get("vc", session.params["vc"]))
    except (ValueError, TypeError):
        return JSONResponse(dict(ok=False, error="d_mm and vc must be positive numbers"), status_code=422)
    if session.audio is None:
        return JSONResponse(dict(ok=False, error="microphone is off: allow microphone access on the phone"), status_code=409)
    loop = asyncio.get_running_loop()
    r = await loop.run_in_executor(session.exec, session.audio.calibrate_idle)
    if r["idle_db"] is None:
        return JSONResponse(dict(ok=False, error="need about 3 s of audio first"), status_code=409)
    session.idle_db = r["idle_db"]
    # the idle sound is heard before the cut, on the larger diameter (after-cut diameter + 2 ap); G96 sets rpm from it
    expected = rpm_for(d_mm + 2 * session.params["ap"], vc)
    ok = abs(r["rpm_detected"] - expected) / expected <= 0.05
    return JSONResponse(dict(ok=ok, warnings=[] if ok else ["rpm_mismatch"], idle_db=r["idle_db"],
                             rpm_detected=r["rpm_detected"], rpm_expected=expected))


async def api_theory(request):
    """Rake-face flow direction from the literature (Colwell 1954; Young, Mathew and Oxley 1987)."""
    q = request.query_params
    try:
        f, ap, r = float(q["f"]), float(q["ap"]), float(q.get("r_eps", "0.4"))
        if min(f, ap, r) <= 0:
            raise ValueError
        t = physics_flow_angle.flow_angles(f, ap, r_eps=r, kappa_r=95.0, eps_r=80.0)
    except (KeyError, ValueError):
        return JSONResponse(dict(error="f, ap and r_eps must be positive numbers"), status_code=422)
    # eta is measured from -z (straight back over the insert); psi from +z (feed): psi = 180 - eta
    return JSONResponse(dict(eta_colwell=t["eta_colwell"], eta_ymo=t["eta_ymo"], psi_theory_colwell=180 - t["eta_colwell"],
                             psi_theory_ymo=180 - t["eta_ymo"], tool=dict(r_eps=r, kappa_r=95.0, eps_r=80.0)))


async def api_chip_forms(request):
    d = pd.read_csv(os.path.join(os.path.dirname(HERE), "code", "observed_chip_forms.csv"))
    return JSONResponse(dict(runs=clean(d.to_dict("records")), axes=dict(f=[0.025, 2.5], ap=[0.1, 16.0])))


def make_summary():
    h = session.history
    valid = [r for r in h if r["valid"]]
    psi = np.array([r["psi_raw"] for r in valid]) if valid else np.array([])
    p = session.params
    med = float(np.median(psi)) if len(psi) else None
    pred = predict_psi(p["f"], p["ap"])
    return clean(dict(params=p, predicted_psi=pred, measured_psi_median=med,
                      measured_psi_iqr=[float(np.percentile(psi, 25)), float(np.percentile(psi, 75))] if len(psi) else None,
                      error_deg=None if med is None else med - pred, frames=len(h), valid_frames=len(valid),
                      alarm_t=next((r["alarm_t"] for r in h if r["alarm"]), None),
                      cut_duration_s=(h[-1]["t"] - h[0]["t"]) if h else 0.0, extrapolation=extrapolating(p["f"], p["ap"])))


async def api_summary(request):
    return JSONResponse(make_summary())


async def api_summary_csv(request):
    rows = ["t,psi_raw,valid,score,alarm"] + [
        f'{r["t"]:.3f},{"" if not math.isfinite(r["psi_raw"]) else r["psi_raw"]},{int(r["valid"])},'
        f'{"" if not math.isfinite(r["score"]) else round(r["score"], 3)},{int(r["alarm"])}' for r in session.history]
    return Response("\n".join(rows) + "\n", media_type="text/csv")


async def page(name):
    return FileResponse(os.path.join(HERE, "static", name))


async def index(request):
    return await page("dashboard.html")


async def phone_page(request):
    return await page("phone.html")


app = Starlette(routes=[
    Route("/", index), Route("/phone", phone_page),
    Route("/api/qr", api_qr), Route("/api/config", api_config), Route("/api/setup", api_setup, methods=["POST"]),
    Route("/api/calibrate", api_calibrate, methods=["POST"]), Route("/api/idle_noise", api_idle_noise, methods=["POST"]),
    Route("/api/theory", api_theory), Route("/api/chip_forms", api_chip_forms),
    Route("/api/summary", api_summary), Route("/api/summary.csv", api_summary_csv),
    Route("/api/replay/{run}", api_replay, methods=["POST"]), Route("/api/stop", api_stop, methods=["POST"]),
    WebSocketRoute("/ws/phone", ws_phone), WebSocketRoute("/ws/dash", ws_dash),
    Mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static"),
])


def main():
    import uvicorn
    key, cert = os.path.join(HERE, "certs", "key.pem"), os.path.join(HERE, "certs", "cert.pem")
    if os.environ.get("APP_HTTP"):               # development only: plain http on localhost (the phone camera needs https)
        print(f"\n  Dashboard (http, local only):  http://localhost:{PORT}/\n")
        uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
        return
    if not (os.path.exists(key) and os.path.exists(cert)):
        sys.exit("No certificate yet: run  bash app/make_cert.sh  first (phones only allow the camera over HTTPS).")
    print(f"\n  Dashboard (laptop):  https://localhost:{PORT}/\n  Phone page:          {phone_url()}\n")
    uvicorn.run(app, host="0.0.0.0", port=PORT, ssl_keyfile=key, ssl_certfile=cert, log_level="warning")


if __name__ == "__main__":
    main()
