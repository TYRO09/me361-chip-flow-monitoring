# ME361 live demo web app: design spec (2026-11-09)

## Goal
A demo for the professor: enter the cutting parameters, set a phone on the lathe, calibrate in under two
minutes, press Start, and watch chip angle, spindle rpm, cutting state and the tangle alarm update live.
The report also gets a literature section, and at least one result from the papers is used inside the system.

## Decisions already agreed
- Laptop does all processing; the phone is only a camera + microphone (opens a web page, no app).
- A **replay mode** feeds the recorded R1-R6 runs through the *same* pipeline, so the demo works with no
  lathe, no phone and no Wi-Fi. It is always labelled "REPLAY" on screen.
- References are given as full citations with DOI links. No PDFs are downloaded or bundled.

## Architecture
```
phone /phone  --wss: JPEG frames (~960 px, 20-30 fps) + PCM audio-->  laptop server  --wss: state JSON + preview-->  laptop /  (dashboard)
recorded R1-R6 (replay.py) -------------------------------------->  same engine
```
Starlette + uvicorn (already installed), HTTPS with a self-signed certificate made by `openssl` for the laptop's
LAN IP (phone browsers only allow the camera on HTTPS). The dashboard shows a QR code of the phone URL.
The only new pip dependency is `qrcode` (pure Python).

| File | Job |
|---|---|
| `app/server.py` | routes `/`, `/phone`, `/ws/phone`, `/ws/dash`, `/api/qr`, `/api/replay/<run>` |
| `app/engine.py` | causal chip-angle engine: `calibrate(frame, tip, feed_dir, out_dir)`, then `push(frame, t) -> state` |
| `app/audio_engine.py` | ring buffer + streaming rpm / cutting-state / audio build-up feature, built on `code/audio_sensor.py` |
| `app/replay.py` | streams a recorded run (video + audio) into the engine at real time |
| `app/static/*` | dashboard and phone pages (vanilla JS, canvas, no build step) |
| `app/make_cert.sh` | self-signed cert for the LAN IP |
| `tests/test_engine.py` | regression test against the offline results |

Per-frame maths is **moved** out of `code/measure_chip_angle.py` into one shared function used by both the
offline script and the engine, so the live app cannot drift from what the report validated. The only change is
causality: the three-frame difference uses frames i-2, i-1, i, so the output lags by one frame (about 33 ms).

## Dashboard flow
1. **Setup**: diameter, feed, depth of cut, Vc, material, nose radius. Shows the model's predicted angle,
   Colwell/YMO theory angle, expected rpm and the f-ap point.
2. **Calibration** (each step shows a green tick when done):
   - Place the phone: live preview with a level indicator and a "tool tip in frame" box.
   - Tap the tool tip, then two points along the feed direction (sets the feed axis; outward is perpendicular).
     The tip patch becomes the tracking template.
   - Tap the two ends of a known length (e.g. the chuck diameter) to get mm per pixel (used for on-screen scale only).
   - 3 s of idle sound sets the noise floor; the detected rpm is checked against the commanded rpm.
3. **Live**: video with tip, feed and chip arrows; angle dial (measured vs model vs theory); angle and
   build-up traces; rpm and CUTTING/idle from the microphone; alarm banner.
4. **Summary**: measured vs predicted angle, alarm time, JSON/CSV export.

## Using the literature
- **Colwell (1954), Young-Mathew-Oxley (1987)**: already used in the report; the dashboard now draws the theory
  direction next to the fitted and measured ones.
- **Dornfeld & Pan (1985), Sukvittayawong & Inasaki (1991)**: acoustic emission responds to chip congestion/entangling
  and to chip form. We add a microphone feature (2-6 kHz band level and its short-term variance) as a second,
  independent build-up indicator, fused with the vision score. First step of the build is a check on R2 (the run
  that tangles). If audio shows no usable signal it is reported as a negative result and left out of the alarm.
- **ISO 10910 / ISO 3685**: an f-ap "chip control chart" panel in the ISO 10910 axis format, showing our six runs with
  the ISO 3685 chip-form code we observed in each video. No class boundaries are drawn, since they depend on
  the insert and material and we have no data for them.
- Every number on screen is tagged *measured*, *our model* or *literature*.

## Report changes
New sections "Live web application and calibration" and "Related work and how it was used", and a References list at the
end (verified citations only, listed below). README gets run instructions.

## Success criteria (checked by the regression test)
1. Replaying R1-R6 through the engine reproduces the offline per-run median angle within 1 degree.
2. R2 replay raises the alarm within 1 s of the offline causal alarm time (20.1 s); no alarm on R1, R3-R6.
3. Phone to dashboard latency under 300 ms on the local network (measured and reported).
4. Calibration is possible in under 2 minutes by someone who has not seen it.
5. Replay mode runs with no phone, no network.

## Risks and how they are handled
- **Phone self-signed HTTPS prompt**: one-time "continue" tap; rehearsal instructions in the README.
- **Tool leaves the frame during the 61 mm pass if the phone is fixed**: the tracker follows the tool; the calibration
  screen warns if the tip would leave the frame, and says a carriage-mounted phone is the alternative.
- **Frame rate**: the thresholds were tuned at 30 fps. The phone page requests 30 fps; the dashboard shows the
  achieved fps and flags a degraded reading when it falls below 20.
- **Not validated on a fixed-phone lathe cut**: all validation is on the six handheld videos. The report will say so.

## Verified references
1. Colwell, L. V. (1954). Predicting the angle of chip flow for single-point cutting tools. *Trans. ASME* 76, 199-203.
2. Stabler, G. V. (1951). The fundamental geometry of cutting tools. *Proc. IMechE* 165.
3. Young, H. T., Mathew, P., Oxley, P. L. B. (1987). Allowing for nose radius effects in predicting the chip flow
   direction and cutting forces in bar turning. *Proc. IMechE* 201. DOI 10.1243/PIME_PROC_1987_201_108_02
   (page numbers differ between sources, so none are given).
4. Jawahir, I. S., van Luttervelt, C. A. (1993). Recent developments in chip control research and applications.
   *CIRP Annals* 42(2), 659-693.
5. Dornfeld, D. A., Pan, C. S. (1985). A study of continuous/discontinuous chip formation using acoustic emission.
   *J. Applied Metalworking* 4, 18-29. DOI 10.1007/BF02833673
6. Sukvittayawong, S., Inasaki, I. (1991). Identification of chip form in turning process. *JSME Int. J. Ser. III*
   34(4), 553-560.
7. Kishawy, H. A. et al. (2018). Application of acoustic emissions in machining processes: analysis and critical
   review. *Int. J. Adv. Manuf. Technol.* DOI 10.1007/s00170-018-2341-y
8. ISO 10910:1995 and ISO 3685 (chip control classification; chip-form codes).
