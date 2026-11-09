# ME361 Live Demo Web App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A laptop-hosted web app where a phone streams camera + microphone and a dashboard shows live chip angle, rpm, cutting state and the tangle alarm, with a calibration wizard, a replay mode, and literature-based panels.

**Architecture:** One shared per-frame kernel (moved out of `code/measure_chip_angle.py`) is used by both the offline script and a causal `ChipEngine`. A Starlette server receives phone frames/audio over WebSocket, runs the engines, and pushes state JSON to the dashboard. Replay mode feeds the recorded R1-R6 runs through the same engines.

**Tech Stack:** Python 3 (starlette, uvicorn, websockets, opencv, numpy, scipy, pandas already installed; `qrcode` new), vanilla JS + canvas, `openssl` for a self-signed cert, pytest.

**Spec:** `docs/superpowers/specs/2026-11-09-live-demo-webapp-design.md`

## Global Constraints

- Angle convention unchanged: psi 0 = feed direction (toward chuck), 90 = radially outward.
- Frames are scaled so the longer side is 960 px; motion threshold 18 grey levels; annulus 25-220 px; tuned for 30 fps.
- Audio: 16 kHz mono internally; rpm by harmonic-sum search over 15-45 Hz; cutting state from 2-6 kHz band energy; 2-s windows, 0.5-s hop.
- Build-up alarm: robust z of log activity, spread, clutter against the first 4 s of the cut; alarm when the 1-s rolling median of the mean score stays above 3 for 1 s.
- HTTPS only (self-signed for the laptop LAN IP). Only new pip dependency: `qrcode`.
- REPLAY mode is always labelled on screen. Every on-screen quantity is tagged measured / our model / literature.
- Cutting data: Vc 150 m/min (G96 S150), rpm cap 2500, cut length 61 mm, AISI 1020, CCMT 09T304 (r_eps 0.4), kappa_r 95 deg.
- Fitted model: psi = 26.8 + 101 f + 17.7 ap (valid f 0.08-0.15, ap 0.3-0.9; outside that show an "extrapolation" flag).
- Commits are local only. Nothing is pushed to GitHub until the user says so.

## Review Focus

1. **Tool lost / leaves frame** (occlusion, fast carriage move): engine reports `tracking_lost`, draws no arrow, does not crash, and recovers when the template matches again (score > 0.5).
2. **Irregular frame timing** (Wi-Fi jitter, dropped or duplicate frames): the three-frame difference is skipped when neighbours are more than 0.1 s apart, and the dashboard shows the achieved fps.
3. **Portrait vs landscape, other resolutions**: calibration points are stored normalised (0-1) and the engine rescales, so a phone rotated after calibration is detected and flagged rather than silently wrong.
4. **Microphone denied or silent**: video-only mode works, audio fields are `null`, no audio-driven alarm, banner says "mic off".
5. **Phone connects/reconnects/second phone**: dashboard may open first; a phone reconnecting mid-run keeps calibration; a second phone is refused with a message.

---

### Task 1: Shared frame kernel

**Files:**
- Create: `code/chip_kernel.py`
- Modify: `code/measure_chip_angle.py` (call the kernel instead of the inline code at lines 134-183)
- Test: `tests/test_kernel.py`

**Interfaces:**
- Produces: `frame_features(g_prev, g_cur, g_next, g_raw, tip_prev, tip_cur, tip_next, u_f, u_o, thr=18, r_in=25, r_out=220) -> dict | None` (None when fewer than 40 moving pixels; dict has the same keys as the current CSV row: psi_peak, psi_mean, spread, reach, clutter, n_px, activity, coherence, psi_streak). Inputs `g_*` are blurred int16 grey, `g_raw` is uint8 grey.
- Produces: `track_step(g: ndarray, tpl: ndarray, prev_xy, search=90) -> (xy, score)`.
- Produces: `axis_rotation(g, tip, ref_deg) -> float` (degrees, 0.0 when no line found).

- [ ] **Step 1: Write failing test** `test_kernel_matches_saved_csv`: for R2, rerun `measure_run("R2", RUNS["R2"], save_overlays=0)` after the refactor and assert every `psi_peak`, `activity`, `clutter` equals `results/angles/R2_frames.csv` (to 0.01) for all valid rows.
- [ ] **Step 2: Run** `pytest tests/test_kernel.py -v`; expected FAIL (module missing).
- [ ] **Step 3: Move** the per-frame maths from `measure_run` into `code/chip_kernel.py` with the signatures above; `measure_run` becomes loops + kernel calls. Keep the numerics identical.
- [ ] **Step 4: Run** the test; expected PASS.
- [ ] **Step 5: Commit** `refactor: shared per-frame kernel for offline and live use`.

### Task 2: Causal chip engine and replay

**Files:**
- Create: `app/__init__.py`, `app/engine.py`, `app/replay.py`
- Test: `tests/test_engine.py`

**Interfaces:**
- Consumes: Task 1 kernel.
- Produces: `@dataclass Calibration(tip: (float,float), feed_dir: (float,float), out_dir: (float,float), ref_frame: ndarray, mm_per_px: float | None)`; points are in the 960-px-scaled frame.
- Produces: `class ChipEngine`: `__init__(self, calib: Calibration)`; `push(self, bgr: ndarray, t: float) -> dict | None` returning `{t, valid, tracking_lost, psi, psi_raw, tip, activity, spread, clutter, score, alarm, alarm_t}` for the frame one step behind (None while fewer than 3 frames are buffered). `reset_cut()` restarts the 4-s baseline.
- Produces: `replay.calibration_for(run: str) -> Calibration` (uses the hand-read tip/feed/out in `RUNS`, tracks backward to the window start), and `replay.frames(run) -> Iterator[(ndarray, float)]`.

- [ ] **Step 1: Write failing tests**
  - `test_replay_median_matches_offline` (parametrised R1-R6): feed `replay.frames(run)` into `ChipEngine`; assert `abs(median(psi) - offline_median) <= 1.0` where offline median comes from `results/angles/{run}_frames.csv` over valid frames.
  - `test_r2_alarm_only`: alarm on R2 within 1 s of 20.1 s; no alarm on R1, R3-R6.
  - `test_tracking_lost_does_not_crash`: push 10 frames, then 5 black frames, then real frames; assert `tracking_lost` True during black and no exception; assert recovery.
  - `test_irregular_timing`: drop every third frame and jitter timestamps by 20 ms; assert engine still returns states and `valid` is False only where neighbours are more than 0.1 s apart.
- [ ] **Step 2: Run** `pytest tests/test_engine.py -v`; expected FAIL.
- [ ] **Step 3: Implement** `ChipEngine` with a 3-frame ring buffer, forward template tracking via `track_step`, axis re-estimation every 3rd frame, the build-up scorer (4-s robust baseline, causal 1-s rolling median, hold 1 s) and `psi` = causal 0.5-s rolling median of `psi_peak`. Normalise incoming frames to a 960-px long side.
- [ ] **Step 4: Run** the tests; expected PASS. If a run is outside 1 degree, report the number rather than loosening the tolerance.
- [ ] **Step 5: Commit** `feat: causal chip engine with replay`.

### Task 3: Audio engine and the audio build-up check

**Files:**
- Create: `app/audio_engine.py`, `code/audio_buildup_check.py`
- Test: `tests/test_audio_engine.py`

**Interfaces:**
- Consumes: `code/audio_sensor.py` (`f0_estimate`, `cut_envelope` logic).
- Produces: `class AudioEngine`: `__init__(self, sr_in: int)`; `push(self, pcm: ndarray) -> None` (float32 mono, any rate, resampled to 16 kHz); `calibrate_idle(self) -> dict` (noise floor dB, detected rpm); `state(self) -> dict | None` with `{rpm, cutting, band_db, band_var_db, buildup_audio}`; returns None until 2 s of audio.

- [ ] **Step 1: Write failing tests**: `test_rpm_matches_batch` (stream R1.f32 in 1024-sample chunks; rpm within 1% of `audio_sensor.tacho` values at the same times); `test_cutting_state_matches_batch` (agreement above 95% of 50-ms bins); `test_no_audio_returns_none`.
- [ ] **Step 2: Run** `pytest tests/test_audio_engine.py -v`; expected FAIL.
- [ ] **Step 3: Implement** the streaming wrapper (ring buffer, tacho every 0.5 s, 2-6 kHz band level and its 1-s variance).
- [ ] **Step 4: Run** the tests; expected PASS.
- [ ] **Step 5: Audio build-up check** `code/audio_buildup_check.py`: for R1-R6, compute `band_db` and `band_var_db` over the cut and compare the 4 s after the vision alarm on R2 with the first 4 s of the cut; write `results/audio_buildup_check.csv` and a one-line verdict. If the R2 change is not larger than the largest change seen on R1/R3-R6, set `buildup_audio` to display-only and say so in the report; do not feed it into the alarm.
- [ ] **Step 6: Commit** `feat: streaming audio engine and R2 audio check`.

### Task 4: Server, phone page, dashboard skeleton

**Files:**
- Create: `app/server.py`, `app/make_cert.sh`, `app/static/phone.html`, `app/static/dashboard.html`, `app/static/app.js`, `requirements-app.txt` (contains `qrcode`)
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `ChipEngine`, `AudioEngine`, `replay`.
- Produces: routes `GET /`, `GET /phone`, `GET /api/qr` (SVG of the phone URL), `GET /api/config` (LAN IP, expected rpm for given D/Vc), `WS /ws/phone` (binary frames: 1 byte type + payload; type 1 = JPEG, type 2 = float32 PCM; first text message = `{sr, fps}`), `WS /ws/dash` (JSON states), `POST /api/replay/{run}` and `POST /api/stop`.
- Message protocol on `/ws/dash`: `{type: "state", mode: "live"|"replay", chip: {...}, audio: {...}|null, fps, latency_ms}` and `{type: "status", phone: "connected"|"none", mic: bool}`.

- [ ] **Step 1: Write failing tests** with Starlette `TestClient`: `test_replay_streams_states` (POST replay R2, expect state messages on `/ws/dash`); `test_phone_frames_roundtrip` (send 60 JPEG frames of R1 over `/ws/phone`, expect states, `latency_ms` present); `test_second_phone_refused`; `test_phone_reconnect_keeps_calibration`; `test_mic_off_gives_null_audio`.
- [ ] **Step 2: Run** `pytest tests/test_server.py -v`; expected FAIL.
- [ ] **Step 3: Implement** the server (single active phone, calibration held server-side), `make_cert.sh` (openssl, SAN = LAN IP and localhost), and the phone page (`getUserMedia` with echoCancellation/noiseSuppression/autoGainControl off, 960x540 at 30 fps, JPEG quality 0.7, audio via AudioWorklet, wake lock). Dashboard shows the QR code, connection status and achieved fps.
- [ ] **Step 4: Run** the tests; expected PASS. Then run `app/make_cert.sh` and `uvicorn app.server:app --ssl-keyfile ... --ssl-certfile ...`, open `https://localhost:8443/phone` on the laptop webcam as a rehearsal, and record the measured latency.
- [ ] **Step 5: Commit** `feat: server, phone page and dashboard skeleton`.

### Task 5: Dashboard wizard (setup, calibration, live, summary)

**Files:**
- Modify: `app/static/dashboard.html`, `app/static/app.js`, `app/server.py` (calibration endpoints)
- Create: `app/static/dashboard.css`
- Test: `tests/test_calibration.py`

**Interfaces:**
- Produces: `POST /api/calibrate` body `{tip:[x,y], feed_pt:[x,y], scale_pts:[[x,y],[x,y]]|null, scale_mm: number|null}` with coordinates normalised 0-1; response `{ok, warnings: [str]}`. `POST /api/idle_noise` runs `AudioEngine.calibrate_idle` and returns `{rpm_detected, rpm_expected, ok}`. `GET /api/summary` returns the run summary; `GET /api/summary.csv` the export.
- Warnings include: `tip_near_edge` (tip within 15% of the frame edge in the feed direction), `portrait_landscape_changed`, `rpm_mismatch` (more than 5% off).

- [ ] **Step 1: Write failing tests**: `test_calibrate_normalised_points`, `test_calibrate_warns_tip_near_edge`, `test_rotation_after_calibration_flagged`, `test_idle_noise_rpm_check` (feed 3 s of R1 idle audio), `test_summary_has_measured_and_predicted`.
- [ ] **Step 2: Run** `pytest tests/test_calibration.py -v`; expected FAIL.
- [ ] **Step 3: Implement** the four steps in the dashboard: setup form (D, f, ap, Vc, material, r_eps) with predicted angle, expected rpm = min(2500, 150000/(pi D)) and the extrapolation flag; calibration with a level indicator (phone sends DeviceOrientation), tap-to-set tip / feed axis / scale on the live preview, idle-noise recording with a countdown; live view with video, tip and chip arrows, angle dial (measured vs model vs theory), traces, rpm and CUTTING/idle, alarm banner and a "REPLAY" badge; summary with CSV export.
- [ ] **Step 4: Run** the tests; expected PASS. Then walk the whole wizard in the browser pane in replay mode and at phone width, and check the console for errors.
- [ ] **Step 5: Commit** `feat: dashboard wizard with calibration and summary`.

### Task 6: Literature panels

**Files:**
- Create: `code/observed_chip_forms.csv` (run, f, ap, ISO 3685 code, note), `app/static/chart.js`
- Modify: `app/static/dashboard.html`, `app/server.py` (`GET /api/theory?f=&ap=&r_eps=`)
- Test: `tests/test_theory.py`

**Interfaces:**
- Consumes: `code/physics_flow_angle.flow_angles(f, ap, r_eps, kappa_r, eps_r)`.
- Produces: `/api/theory` returns `{eta_colwell, eta_ymo, psi_theory_colwell, psi_theory_ymo}` where `psi_theory = 180 - eta` (the frame conversion used in the report).

- [ ] **Step 1: Write failing tests**: `test_theory_matches_report_values` (for the six runs, `/api/theory` equals the values in `results/process_map` / report table to 0.1 degree); `test_theory_limit` (ap = 20 mm gives eta near 90 - kappa_r = -5).
- [ ] **Step 2: Run** `pytest tests/test_theory.py -v`; expected FAIL.
- [ ] **Step 3: Implement** the endpoint and a dial overlay for the theory direction tagged "literature".
- [ ] **Step 4: Label the six videos** with ISO 3685 chip-form codes by viewing representative frames of each run; record in `code/observed_chip_forms.csv`. Draw the f-ap chart (ISO 10910 axis format, feed 0.025-2.5 on x, depth 0.1-16 on y, log scales) with only these six points and the current setting; no class boundaries.
- [ ] **Step 5: Run** the tests; expected PASS.
- [ ] **Step 6: Commit** `feat: literature panels (theory direction, f-ap chip chart)`.

### Task 7: Report, README, final checks

**Files:**
- Modify: `results/REPORT.md` (regenerate `REPORT.pdf` with `code/md_to_pdf.py`), `README.md`
- Create: `app/README.md` (rehearsal checklist), `results/figures/fig11_webapp.png`

- [ ] **Step 1: Report**: add "Live web application and calibration" (architecture, calibration, regression results, measured latency, limits) and "Related work and how it was used" (the verified references from the spec, one paragraph each saying what was taken and what was not), then a "References" list as the last section. State plainly that validation was on handheld videos only and the audio build-up verdict from Task 3.
- [ ] **Step 2: README and `app/README.md`**: setup, `pip install -r requirements-app.txt`, cert, run command, phone steps, and a 10-minute rehearsal checklist with the replay fallback.
- [ ] **Step 3: Screenshot** the dashboard in replay mode into `fig11_webapp.png` and include it in the report.
- [ ] **Step 4: Run the full suite** `pytest tests -v`; expected all PASS. Rebuild the PDF and check page count and that the references render.
- [ ] **Step 5: Commit** `docs: report sections, references and run instructions`. Ask the user before pushing.
