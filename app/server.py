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
from app.engine import ChipEngine, scale_frame  # noqa: E402

PORT = int(os.environ.get("APP_PORT", "8443"))
PREVIEW_FPS = 15
CUT_ON_S, CUT_OFF_S = 0.3, 1.0      # audio must say "cutting" this long to start a cut / "not cutting" to end it
PHONE_STALE_S = 5.0


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
        return dict(chip=chip, audio=aud, cut_active=self.cut_on)


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

    def _cancel_replay(self):
        t = self.replay_task
        if t is not None and not t.done():
            t.get_loop().call_soon_threadsafe(t.cancel)
        self.replay_task = None

    # ---- calibration / pipelines ----
    def set_calibration(self, cal):
        self.calibration = cal
        self.live = Pipeline(ChipEngine(cal), self.audio)

    def set_audio(self, sr):
        if sr is None:
            self.audio, self.mic = None, False
        else:
            self.audio, self.mic = AudioEngine(int(sr)), True
            if self.idle_db is not None:
                self.audio.idle_db = self.idle_db
        if self.live is not None:
            self.live.audio = self.audio

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
                    t = struct.unpack(">d", b[1:9])[0] / 1000.0
                    if session.live is not None:
                        def work(jpeg=jpeg, t=t):
                            img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                            return session.live.step(img, t)
                        out = await loop.run_in_executor(session.exec, work)
                        if out is not None:
                            out.update(type="state", mode="live", t=t, fps=session.fps(),
                                       latency_ms=session.rtt / 2 + (time.time() - t_recv) * 1000)
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
    d = float(request.query_params.get("d", "30"))
    vc = float(request.query_params.get("vc", "150"))
    rpm = min(2500.0, 1000.0 * vc / (math.pi * d))
    return JSONResponse(dict(lan_ip=lan_ip(), phone_url=phone_url(), rpm_expected=rpm,
                             runs=sorted(replay_mod.CUT)))


async def page(name):
    return FileResponse(os.path.join(HERE, "static", name))


async def index(request):
    return await page("dashboard.html")


async def phone_page(request):
    return await page("phone.html")


app = Starlette(routes=[
    Route("/", index), Route("/phone", phone_page),
    Route("/api/qr", api_qr), Route("/api/config", api_config),
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
