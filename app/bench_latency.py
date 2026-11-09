"""Server-side latency of the live pipeline: capture-to-state time as the dashboard sees it.

Streams the recorded run R6 through a running dev server (APP_HTTP=1 python3 app/server.py) at real-time speed
and reports the median / 95th percentile of the `latency_ms` field and the achieved frame rate.
Usage: python3 app/bench_latency.py [port]
"""
import asyncio
import json
import sys
import urllib.request

import numpy as np
import websockets


async def main(port):
    lat, fps = [], []
    async with websockets.connect(f"ws://localhost:{port}/ws/dash", max_size=None) as ws:
        urllib.request.urlopen(urllib.request.Request(f"http://localhost:{port}/api/replay/R6", method="POST"))
        while True:
            m = await asyncio.wait_for(ws.recv(), 60)
            if isinstance(m, bytes):
                continue
            j = json.loads(m)
            if j["type"] == "state":
                lat.append(j["latency_ms"]); fps.append(j["fps"])
            elif j["type"] == "status" and j["mode"] is None and lat:
                break
    lat = np.array(lat)
    print(f"states: {len(lat)}  latency median {np.median(lat):.0f} ms  p95 {np.percentile(lat, 95):.0f} ms  "
          f"max {lat.max():.0f} ms  fps median {np.median(fps[10:]):.1f}")


asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 8765))
