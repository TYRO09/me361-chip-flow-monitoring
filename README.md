# ME361: Live monitoring of chip entangling (chip flow angle)

**Main results:** [`results/REPORT.pdf`](results/REPORT.pdf) (iteration 2) · interactive predictor: [`results/chip_direction_predictor.html`](results/chip_direction_predictor.html) · demo videos: [`results/demo/`](results/demo/)

> **Not in this repository** (too large for GitHub, kept on the project laptop): the raw videos (`session_2026-11-04/videos_top`, `videos_side`, `session_2026-10_unlabeled`), extracted audio (`results/audio`), the CNN training crops (on Kaggle as the private dataset `tyro09/me361-chip-angle-frames`) and model weights (`*.pt`).

## Folder layout

```
Me361/
├── README.md                     this file
├── file_map.csv                  original filename → new path (nothing was deleted)
├── docs/                         lab visit checklist (md + pdf)
├── code/                         measurement, physics, analysis, model_zoo, audio_sensor,
│                                 entangle_indicator, live_monitor_demo, process_map, crops, md→pdf
├── results/                      REPORT.md/.pdf (v2; v1 kept as REPORT_v1.*), angles, figures,
│                                 demo/ (live-monitor videos), angles_variants/ (sensitivity runs)
├── kaggle/                       dataset (crops + labels), kernel/ (CNN v1), kernel_v2/ (input variants), outputs
├── session_2026-11-04/           6 labelled runs (the real dataset)
│   ├── runs.csv                  one row per run: parameters, files, blanks to fill
│   ├── videos_top/               R1–R6, top view looking down on the insert (iPhone, 1080p 30 fps)
│   ├── videos_side/              side view: full continuous recording + R1–R4 clips cut to match
│   ├── photos/                   CNC controller screens, named by the run they belong to
│   └── notes/                    handwritten run table
└── session_2026-10_unlabeled/    first batch (Oct): no parameters, handheld, kept as-is
```

File names follow `R<run>_f<feed mm/rev>_ap<depth of cut mm>_<view>`, e.g. `R2_f0.15_ap0.9_top.MOV`.

## Session 2026-11-04: what's known

| Run | Feed (mm/rev) | Depth of cut (mm) | Diameter (mm) | Spindle rpm* | Top view | Side view |
|---|---|---|---|---|---|---|
| R1 | 0.08 | 0.9 | 32.0 → 30.2 | ~1580 | ✓ | ✓ |
| R2 | 0.15 | 0.9 | 30.2 → 28.4 | ~1680 | ✓ | ✓ (chips tangle at ~20 s) |
| R3 | 0.08 | 0.3 | 28.4 → 27.8 | ~1720 | ✓ | ✓ |
| R4 | 0.15 | 0.3 | 27.8 → 27.2 | ~1755 | ✓ | ✓ |
| R5 | 0.08 | 0.6 | 27.2 → 26.0 | ~1835 | ✓ | ✗ (side recording ended before R5) |
| R6 | 0.15 | 0.6 | 26.0 → 24.8 | ~1925 | ✓ | ✗ |

\*The program uses **G96 S150**: constant cutting speed Vc = 150 m/min, with a 2500 rpm cap (G92 S2500). So rpm = 150000 / (π·D) and rises as the bar gets thinner. Your "1600 rpm" was the approximate reading at the start.

**From the CNC program photos (program O2001, tool T0001):**
`G00 X65 Z1 → G01 X<dia> F0.2 → G01 Z1 F1.0 → G01 Z-60 F<feed> → G01 X45 F0.1 → retract`
- **Cut length is 60 mm (Z1 → Z−60), not 10 cm.**
- X30.2 (R1) and X28.4 (R2) are visible in the photos and match depths of 0.9 mm per side. The other diameters are calculated from your notes, assuming a starting diameter of 32.0 mm.
- The controller clock runs about 13 minutes behind the phones.

**How the files were matched:**
- Top view: the 6 iPhone clips are in time order, and their lengths alternate long/short, matching feeds 0.08/0.15.
- Side view: placed on the side recording by matching the audio. Offsets are in `runs.csv`. The match was checked visually (tool position and the R2 chip tangle line up).

## Still missing (fill in `runs.csv` when you have it)
- [ ] Material grade (only "MS / mild steel" known)
- [ ] Insert ISO code + nose radius, holder code (entering angle)
- [ ] Machine make/model (controller is FANUC 0i Mate-TD)
- [ ] Coolant (looks dry)
- [ ] Chip thickness per run (you have the chips: measure with a micrometer)
- [ ] Starting diameter of the bar (32.0 mm is inferred)
- [ ] Is there a side-view recording of R5–R6?
- [ ] Calibration photo (ruler/protractor) for the camera views
