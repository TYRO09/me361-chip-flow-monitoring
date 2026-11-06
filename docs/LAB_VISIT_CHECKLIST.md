# Lab visit checklist: chip flow angle dataset

You get one visit, so this is ordered by priority. If you run short of time, the top of each section matters most.

---

## 1. Ask the lab staff first (5 min, high value)

- [ ] **The CNC programs (G-code) for the videos already shot on 12–13 Oct.** Photograph the program on the controller screen. It contains spindle speed (S), feed (F), and depths per pass. With it, the videos you already have can get parameters attached.
- [ ] **Workpiece material grade**, e.g. Al 6061-T6, 6063, 2024. "Aluminium" is not enough.
- [ ] **Insert ISO code** (e.g. `CCMT 09T304-xx`) and **chip-breaker name**. Photograph the insert box or label.
- [ ] **Tool holder ISO code** (e.g. `SCLCR 2020K09`). Photograph the marking on the holder. It gives the entering/approach angle, rake and inclination angle.
- [ ] Machine make/model and max RPM.
- [ ] Whether the controller shows **spindle load %**. If yes, film the screen during each run too (free extra feature).
- [ ] Permission and the safe place to mount a phone.

---

## 2. Parameters to record for EVERY run

| Field | Notes |
|---|---|
| Run ID | R01, R02, … (also shown on a paper slate in the video) |
| Video filename(s) | e.g. IMG_1410.MOV. Note it right after recording |
| Operation | OD turning / facing / grooving / profiling |
| Feed direction | towards chuck / away / radial in / radial out |
| Speed | Write **N (rpm)** AND whether the program uses **G96 constant surface speed (Vc, m/min)** or **G97 fixed rpm** |
| Diameter at cut | start and end (mm). Needed to compute Vc = π·D·N/1000 |
| Feed f | mm/rev |
| Depth of cut ap | mm (radial for OD turning, axial for facing) |
| Insert + nose radius rε | from the ISO code (last digits, e.g. 04 = 0.4 mm) |
| Chip breaker | name/code |
| Holder | ISO code (entering angle κr) |
| Cutting edge | new / used (which corner) |
| Coolant | off / flood / mist, and type |
| Chip form seen | continuous, helical, C-shaped, broken, **tangled** |
| Entangled? | yes / no, and around what (workpiece, tool, chuck) |
| Notes | anything odd |

**The 3 most important are feed, depth of cut and speed**, but chip *flow angle* also depends strongly on **nose radius, entering angle and inclination angle**. Without the tool geometry, the model cannot generalise.

---

## 3. Experiment plan (what to vary)

The theory says depth of cut and feed (relative to the nose radius) drive the flow angle most, and speed less. So:

**Minimum plan (~13 runs, same tool):**
- 3 feeds × 3 depths of cut at one speed = 9 runs (e.g. f = 0.08 / 0.15 / 0.25 mm/rev, ap = 0.3 / 1.0 / 2.0 mm, chosen with the operator so they're safe for the machine)
- 2 extra speeds at the middle feed/depth = 2 runs
- Repeat the middle point 2 more times = 2 runs (shows how noisy the measurement is)
- Do the runs in **shuffled order**, not sorted.

**If there's more time:**
- Repeat the 3×3 at a second speed.
- Change the insert to a different **nose radius** or holder (different entering angle) and repeat a few points. This is the most valuable addition.
- Try with and without coolant at the middle point.
- Try a depth of cut *smaller* than the nose radius. The flow angle changes most here.

**Keep each run long enough:** at least 10–15 s of steady cutting in the video.

---

## 3a. Cut length, recording time and depth of cut

**Cut length: 10 cm per run (prof's recommendation).** That's a good choice: it gives a long stretch of steady cutting, and long continuous chips have time to show whether they tangle.

**Cutting time for 10 cm (OD turning):**

> time (s) = 60 × 100 mm ÷ (feed f [mm/rev] × spindle speed N [rpm]) = **6000 ÷ (f × N)**

| Spindle speed N | f = 0.08 mm/rev | f = 0.15 mm/rev | f = 0.25 mm/rev |
|---|---|---|---|
| 800 rpm | 94 s | 50 s | 30 s |
| 1200 rpm | 63 s | 33 s | 20 s |
| 1600 rpm | 47 s | 25 s | 15 s |
| 2000 rpm | 38 s | 20 s | 12 s |

(If the operator sets cutting speed instead: N = 1000 × Vc ÷ (π × D). For example, Vc = 250 m/min on a 50 mm bar gives about 1600 rpm.)

**Recording time per run = cutting time + about 5 s before the tool touches + about 5 s after it leaves.** In practice that's **about 20 s to 100 s per run**. Start recording before the cycle start button is pressed, and stop after the tool retracts.

**Depth of cut: what I suggest**
- Use 3 levels, set **relative to the insert's nose radius rε** (read it from the insert code):
  - **Low: just below rε**, e.g. **0.3 mm** for a 0.4 mm nose radius. The chip flow angle changes most in this range, so it's the most informative level.
  - **Medium: about 1.0 mm.**
  - **High: about 2.0 mm.**
- Check the insert box for its recommended ap and feed range (the chip breaker's working window), and stay inside it.
- The operator has the final say on what's safe for the machine and workpiece.

**Stock budget (check this with the operator before starting):** each 10 cm pass removes 2 × ap from the diameter. The minimum plan (13 runs) removes about **13–14 mm of radius, i.e. ~27 mm off the diameter.** A 50 mm bar would end up near 23 mm, which is thin for a long overhang. So either:
- use a bar of **≥ 60 mm** diameter with the tailstock centre engaged, or
- use **two bars**, or
- cut different 10 cm zones of a long bar.

**Speed tip:** because the diameter keeps shrinking, ask for **G96 (constant surface speed)** so the cutting speed Vc stays the same across runs. Still write down the diameter for every run.

**Total time estimate:** 13 runs × (~1 min of cutting + ~3–5 min for setting up, measuring and logging) ≈ **1–1.5 hours**, plus the camera setup and calibration photo.

---

## 4. How to film (this decides whether the angle can be measured at all)

The current videos are handheld, with the camera angle, zoom and orientation changing. A 2D angle in the image then depends on where the phone was, so those angles **can't be compared across clips**. For the new data:

- [ ] **Phone fixed on a tripod/clamp/magnetic mount. Same position for ALL runs.** Don't touch it between runs. If it must move, start a new "camera setup" in the log.
- [ ] **View:** looking down onto the **rake face of the insert** (top of the insert), so the chip leaving the cutting edge is seen from above. If you have a **second phone**, add a side view along the workpiece axis.
- [ ] **Landscape**, no zoom changes, **slow-motion 120 or 240 fps** if the phone supports it (otherwise 60 fps, 1080p or 4K).
- [ ] **Lock focus and exposure** (long-press on the iPhone screen → AE/AF LOCK) on the tool tip.
- [ ] Turn off Cinematic / Action mode and auto macro switching.
- [ ] **Dark background** behind the chip if possible (black card/cloth out of the chip path). Shiny aluminium on shiny metal is hard to segment.
- [ ] Extra light from the side if it's dim, to avoid strong glare on the insert.
- [ ] **Calibration shot (machine STOPPED, with permission):** with the camera in its final position, photograph a **ruler** and a **protractor or printed angle grid** laid at the insert's height, plus the workpiece axis visible. This lets the image angle be converted to a real angle.
- [ ] **Slate:** before each run, hold a paper with the run ID in view for 1 s (or say it aloud).
- [ ] Film from **before the tool engages until after it exits**.
- [ ] Keep the phone safe from chips and coolant: behind the door glass or with a clear bag over it. Never reach in while the spindle is running.

---

## 5. Physical chip samples (cheap, very useful)

- [ ] After each run, collect a few chips into a **labeled zip bag** (run ID).
- [ ] Photograph them on a dark sheet with a ruler.
- [ ] Later: measure **chip thickness with calipers/micrometer**. Chip thickness ÷ uncut thickness = chip compression ratio, another feature and a check on the physics.

---

## 6. What to bring

- [ ] Tripod, clamp, or magnetic phone mount (+ a second phone if possible)
- [ ] Power bank and **at least 15 GB free storage**. iPhone Settings → Camera → Record Slo-mo shows the size per minute for each mode. The plan is ~15–20 min of footage.
- [ ] Black card/cloth, small torch or LED light
- [ ] Ruler, protractor (or printed angle grid), marker, tape
- [ ] Zip bags for chips
- [ ] This checklist + a printed copy of the log table in section 2
- [ ] Safety glasses, closed shoes; follow the lab's rules

---

## 7. Before you leave the lab

- [ ] Every video filename is matched to a run ID in the log.
- [ ] You have photos of the insert/holder labels and the G-code (old and new runs).
- [ ] The calibration photo is taken from the final camera position.
- [ ] AirDrop/upload the videos right away (WhatsApp compresses video, so don't send through it).
