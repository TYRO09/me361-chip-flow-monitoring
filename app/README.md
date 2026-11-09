# Live chip-flow monitor: run it

A laptop runs the analysis; a phone is only the camera and microphone.

## First time

```bash
pip install -r requirements-app.txt        # one extra package (qrcode); the rest is already used by the project
bash app/make_cert.sh                      # self-signed certificate for this laptop's IP (phones need https for the camera)
python3 app/server.py                      # prints the dashboard and phone URLs
```

Open `https://localhost:8443/` on the laptop (accept the certificate warning once).
Re-run `make_cert.sh` if the laptop's IP changes (new Wi-Fi) or after 30 days.

## At the machine (the 4 steps on the dashboard)

1. **Setup**: type the diameter after the cut, f, ap, Vc. The page shows the expected rpm and the model's chip angle.
2. **Calibrate**
   - Scan the QR code with the phone, tap "continue" past the certificate warning, tap *Start camera and microphone*.
   - Stand the phone on a mount or the machine bed, camera looking down at the tool, tool tip well inside the picture. Do not rotate it afterwards.
   - On the laptop: click the tool tip, a point toward the chuck, and a point on the workpiece. Optional: two ends of a known length and its size in mm.
   - With the spindle running and **no cutting**, press *Record 3 s idle sound*. It must report the expected rpm (it checks against the diameter before the cut).
   - *Apply calibration*, then *Start live monitoring*.
3. **Live**: start the cut. The angle, rpm, cutting state and build-up alarm update by themselves.
4. **Summary**: measured vs predicted angle and the alarm time; CSV download.

The angle is sensitive to where you click the tool tip: a 5-10 px error moves the angle by several degrees (see the report, section 13). Click the very tip of the insert, on a frozen picture if possible.

## Replay mode (no lathe, no phone)

On step 1 press **R1-R6**. The recorded video and sound go through exactly the same code, and the screen shows a REPLAY badge.
**R2** is the run whose chips tangle: the alarm fires at about 20 s.

## Rehearsal checklist (10 minutes, the day before)

- [ ] `python3 app/server.py` starts; dashboard opens; replay **R2** shows the alarm.
- [ ] Phone and laptop on the same Wi-Fi (or the phone's hotspot). University Wi-Fi often blocks device-to-device traffic: if the phone cannot open the page, use a hotspot.
- [ ] Phone opens the page, camera and microphone allowed, the dashboard shows `phone: connected`, `mic: on`, 25-30 fps, latency well under 300 ms.
- [ ] Calibration done in under 2 minutes by someone who has not done it before.
- [ ] Laptop charger plugged in, screen sleep off, phone screen sleep off (the page asks for a wake lock).
- [ ] Fallback ready: replay R2 and R5 work without the phone.

## Without a phone: stand-in for rehearsal

```bash
APP_HTTP=1 python3 app/server.py               # plain http on localhost only
python3 app/fake_phone.py R1 --start 17 --hold 12 --idle-audio 10   # streams a recorded video like the phone does
```

## Safety note

The server has no password: anyone on the same network who knows the address can watch the dashboard or connect as the phone. Use your own hotspot or a trusted network, not a public one, and stop the server after the demo.

## Tests

```bash
pip install pytest httpx
python3 -m pytest tests -q                     # about 3 minutes; replays all six runs through the engine
python3 app/bench_latency.py                   # server-side latency, with the dev server running
```
