# ME361 demo script (about 10 minutes + questions)

Say the quoted parts in your own words; the numbers are the ones in the report. Screen = laptop dashboard.

## Before you go in (15 minutes earlier)

- [ ] Laptop charged and plugged in, screen sleep off. Phone charged, screen sleep off.
- [ ] Same Wi-Fi for both, or the phone's hotspot (university Wi-Fi often blocks device-to-device traffic).
- [ ] Terminal: `cd Me361` then `bash app/make_cert.sh` (only if the Wi-Fi changed since the last run), then `python3 app/server.py`. Open `https://localhost:8443/` and accept the warning.
- [ ] Press **R2** once as a test: the red alarm must appear at about 20 s. Press **Stop**.
- [ ] Phone: open the address printed in the terminal (or scan the QR on the Calibrate step), tap past the warning, **Start camera and microphone**. The dashboard must say `phone: connected`, `mic: on`.
- [ ] Open `results/paper/paper.pdf` on a second window, at Fig. 2 (chip on the rake face) and Table III (models).
- [ ] Decide now: **live phone** (only if it worked in the test) or **replay only**. Replay is a perfectly good demo.

## 0:00 The problem (1 min)

"On a lathe, long chips can wrap around the tool and the workpiece. That ruins the surface, can break the insert, and the operator has to stop the machine. We asked: can a phone, a camera and a microphone, tell us where the chip goes and warn us when chips start to build up?"

## 1:00 The data (1 min)

"We ran six turning cuts on a FANUC lathe at a constant cutting speed of 150 metres per minute. Two feeds, 0.08 and 0.15 millimetres per revolution, and three depths of cut, 0.3, 0.6 and 0.9 millimetres. A phone filmed the tool from above, with sound."

Be upfront: "Six runs, one tool, one material. The insert code and steel grade were not recorded, so we assumed a standard insert and mild steel."

## 2:00 How the angle is measured (1.5 min)

"We track the tool tip in the video, subtract neighbouring frames to see only what moves, and take the direction in which the moving pixels leave the tip. Zero degrees is along the feed, ninety degrees is straight outward. We get an angle for every frame, about 3,100 frames in total. We checked that it is not an artefact of our settings: the effect of feed has the same sign in all 18 variants we tried."

## 3:30 What we found (2 min)

1. "**A simple model works best.** Chip angle = 26.8 + 101 times the feed + 17.7 times the depth of cut. Tested by leaving one run out each time, the error is 6.6 degrees against 7.9 for just guessing the average. With six runs we do not trust anything fancier, and we tried eleven models."
2. "**There are two angles, and we measure both.** The textbook chip-flow angle (Stabler, Colwell, Young-Mathew-Oxley) is on the rake face. Where the chip *goes* is the ejection angle, after it leaves the face. Our first measurement was the ejection angle, and against theory it ran the wrong way (correlation minus 0.52). So we zoomed in on the insert at full resolution and averaged over time: the chip lying on the face shows up as a standing bright band. On run R2 it sits at 129 degrees, on the insert side of the tip as theory says, which is a flow angle of 51 degrees. Colwell's model gives that value if the nose radius is about 0.8 to 1.2 mm, not the 0.4 mm we assumed, so the next step is to photograph the insert. It is clear on one run and faint on two others, so we present it as strong evidence on one run, not a full validation."
3. "**A neural network on the raw images learned which video clip a frame came from, not the chip direction.** Showing it only the moving pixels improved it from 8.8 to 6.8 degrees. The data set is public on Kaggle (link below)."
4. "**Sound alone gives the spindle speed within 0.4 percent and the feed within 2.6 percent**, because we know the cut length."
5. "**The build-up alarm fires only on R2**, the run where the chips really tangled."

## 5:30 Live demo (3 min)

**Step 1, Setup.** "I type the cutting parameters. The page tells me the expected spindle speed and our model's chip angle, and warns me if I am outside the six runs we tested." Change the feed to 0.4 to show the yellow warning, then set it back to 0.15. "The chart is in the standard ISO 10910 format: feed against depth of cut, with our six runs."

**Step 2, Calibrate** (live phone only). "The phone is a camera and a microphone, nothing is installed. I scan this code." Click the tool tip, a point toward the chuck, a point on the workpiece. "Three clicks tell the system where the tip is and which way the feed goes. The spindle is running but not cutting, so I record three seconds of idle sound and it checks the speed against the diameter." Press **Apply calibration**, then **Start live monitoring**.

Say honestly: "How accurately I click the tip matters. In our test, a 5-pixel error moved the angle by up to 5 degrees."

**Step 3, Live** (or replay). If you have no live cut, say: "I do not have a lathe in the room, so I will replay a recorded run through exactly the same code, and the amber REPLAY label stays on screen." Go back to **1 Setup**, press **R2**.

Point out while it runs:
- the yellow arrow is the chip direction, the blue arrow is the feed direction;
- "the angle, the spindle speed and CUTTING come from the microphone and the video, one frame behind, so about 30 milliseconds";
- "the score at the bottom compares each moment with the first four seconds of the same cut. Watch it climb."

At about 20 s the red banner appears: "That is the tangle alarm. The chips are snarling at the tool, and you can see them." Then press **Show summary**: "Measured against predicted angle, and the time of the alarm."

If time allows, press **R5**: "A normal cut: no alarm."

## 9:00 What we read and what changed (1 min)

"We checked eight sources. Colwell and Young-Mathew-Oxley give us the theory direction, shown on the page and labelled as literature. ISO 10910 gives the axes of the chart. Dornfeld and Pan found that acoustic emission responds to tangling chips, so we tested whether a phone microphone shows it. It does not: a phone hears ordinary sound, not the ultrasonic acoustic emission those papers use. That is a negative result and we kept it in the report."

## 9:45 Limits and next steps (1 min)

"What we have not done: every number comes from six handheld videos. We have not yet run the phone on a fixed stand during a real cut, and the alarm is validated on one real tangle. Next visit: a tripod and a 120 to 240 frames per second camera, a randomised run order, a calibration photo with a ruler, and measuring the chips and the insert. The theory question needs the insert code and the chip breaker."

## Questions you will probably get

| Question | Honest answer |
|---|---|
| Is the measured angle ground truth? | No, it is our computer-vision measurement. The absolute value shifts by 3-6 degrees with the measuring ring we use; the comparison between runs does not. |
| Why only six runs? | One lab session. That is why we use leave-one-run-out testing and keep the model to two inputs. |
| Does the alarm generalise? | We only have one real tangle, and no false alarm on the other five. We would need more tangles to claim more. |
| Which angle are you measuring, the chip-flow angle or something else? | Both. The ejection angle (in the air, which matters for tangling) on every frame, and the rake-face flow angle (the textbook one) from the standing chip on the insert. The rake-face one is clear on R2 (flow angle 51 degrees) and faint on R3 and R6. |
| Why does theory not match the ejection angle? | Theory predicts the rake-face direction, and that one does agree on the insert side. The chip is then turned forward by about 70-85 degrees by curl, the chip breaker and the shoulder. |
| Why not just make the dataset bigger with augmentation? | We tried: cutting runs into 1-2 s clips helps a little (6.6 to 4.6 degrees), mostly from a cleaner target. A random split instead looks great (down to 0.05 degrees) but that is leakage, not accuracy. More runs are needed, not more copies. |
| Why not the CNN? | With six runs it memorises the clip. The motion-only version is better, and the one to retrain with about 30 runs. |
| How fast is it? | About 26 ms per frame on the laptop, 29 frames per second. We did not measure the real phone-to-laptop delay yet. |
| Why does the angle depend on where I click the tip? | The angle is measured from the tip. A 5-pixel error moved it by up to 5 degrees, a 10-pixel error up to 7 degrees. |
| Can sound detect tangling? | Not with a phone microphone, in our test. Spindle speed and cut on/off yes. |
| Where is the code and data? | Code: GitHub repository (private, ask us for access). Data: Kaggle, link below. |

## If something goes wrong

| Problem | Do this |
|---|---|
| Phone will not connect | Say "network is blocking it", press **R2** replay. Nothing else changes. |
| Page shows a certificate warning | Click Advanced, then continue. Expected, it is a self-signed certificate. |
| Calibration says tip too close to the edge | Click again further from the edge, or skip live and use replay. |
| Alarm does not appear in replay | Press **Stop**, press **R2** again, wait until about 20 s. |
| Server dies | `python3 app/server.py` again, reload the page. |

## Links to have ready

- Paper (hand this to the professor): `results/paper/paper.pdf` (7 pages, two columns).
- Lab-notebook report: `results/REPORT.pdf` (section 13 is the web app, 14 the literature).
- Dataset (public): https://www.kaggle.com/datasets/tyro09/me361-chip-angle-frames
