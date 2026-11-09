"""Stand-in for the phone: streams a recorded top-view video (with its sound) to the server, like phone.js does.

Use it to rehearse the whole live path (calibration clicks, idle-noise check, alarm) without a phone:
    APP_HTTP=1 python3 app/server.py          # terminal 1
    python3 app/fake_phone.py R1 --start 18 --hold 40   # terminal 2, then open http://localhost:8443/
Frames go out at the video's own rate; audio is 16 kHz float32 in 100-ms chunks; the stream loops.
"""
import argparse
import asyncio
import json
import os
import struct
import sys
import time

import cv2
import numpy as np
import websockets

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, ROOT)
import audio_sensor  # noqa: E402
import measure_chip_angle as M  # noqa: E402
from app.engine import scale_frame  # noqa: E402


async def main(run, start, port, https, hold, idle_audio):
    x = audio_sensor.load(run)
    path = os.path.join(M.SESSION, "videos_top", M.RUNS[run]["file"])
    scheme = "wss" if https else "ws"
    import ssl
    ctx = ssl._create_unverified_context() if https else None
    async with websockets.connect(f"{scheme}://localhost:{port}/ws/phone", max_size=None, ssl=ctx) as ws:
        await ws.send(json.dumps(dict(hello="phone", audio_sr=16000, fps=30, rtt_ms=5)))

        async def drain():
            async for m in ws:
                j = json.loads(m)
                if j.get("type") == "refused":
                    print("refused:", j["reason"]); os._exit(1)
        asyncio.create_task(drain())
        while True:
            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
            wall0, a_pos, i = time.time(), (0 if idle_audio else int(start * 16000)), 0
            frozen = None
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                if hold and i == 0:
                    frozen = fr
                if frozen is not None and i / fps < hold:       # keep the picture still while the user calibrates
                    fr = frozen
                t = start + i / fps
                await asyncio.sleep(max(0.0, wall0 + i / fps - time.time()))
                if i / fps < idle_audio:                        # first seconds of the clip: spindle running, no cutting
                    a_end = int(i / fps * 16000)
                else:
                    if a_pos < int(start * 16000) and i / fps >= idle_audio and idle_audio:
                        a_pos = int(t * 16000)                  # then jump to the sound that matches the picture
                    a_end = int(t * 16000)
                for k in range(a_pos, min(a_end, len(x)), 1600):
                    await ws.send(b"\x02" + x[k:min(k + 1600, a_end)].astype("<f4").tobytes())
                a_pos = a_end
                ok, buf = cv2.imencode(".jpg", scale_frame(fr), [cv2.IMWRITE_JPEG_QUALITY, 70])
                await ws.send(b"\x01" + struct.pack(">d", wall0 * 1000 + i * 1000 / fps) + buf.tobytes())
                i += 1
            cap.release()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="R1")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--https", action="store_true")
    ap.add_argument("--idle-audio", type=float, default=0.0, help="seconds of idle (pre-cut) sound to play first, for the idle-noise step")
    ap.add_argument("--hold", type=float, default=0.0, help="seconds to show the first frame as a still picture (for calibration)")
    a = ap.parse_args()
    asyncio.run(main(a.run, a.start, a.port, a.https, a.hold, a.idle_audio))
