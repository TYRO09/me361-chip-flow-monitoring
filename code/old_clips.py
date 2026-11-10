"""Out-of-domain test on the October 2026 clips (different part, different operation, unknown f/ap, 60 fps).

Only constant-zoom stretches are used (zoom found from ORB feature matches between frames; within +-4%).
Frames are sub-sampled to 30 fps (as the November data) and rescaled so the insert has the same size in
pixels as in November (screw centre to cutting corner ~100 px at the 960 scale). Tip, feed axis and the
"outward" side were read by hand from one reference frame (as in the app's calibration).

There is no measured feed or depth, so this is NOT an accuracy test against settings. It tests:
  1. does the tracker and the hand-crafted measurement keep working on new footage (valid fraction);
     A clip whose median chip-motion activity is below 10% of the quietest November run has no chip flow in
     the measuring ring (its "angle" is noise from the insert outline) and is marked usable=False.
  2. does the image model trained on November (cnn_v3_final.pt) agree with the hand-crafted measurement on
     footage it has never seen (both are measurements of the same thing; agreement = transfer).
Writes results/old_clips.csv (per frame) and results/old_clips_summary.csv.
"""
import json, math, os, sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chip_kernel import TPL, SEARCH, R_OUT, DIFF_THR, rot, axis_rotation, track_step, blur16, frame_features  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD = os.path.join(ROOT, "session_2026-10_unlabeled")
RES = os.path.join(ROOT, "results")
# tip/feed/out at the reference frame, in the 960-scale picture; scale = extra factor so the insert matches November
CLIPS = {
    "IMG_1303": dict(window=(4.0, 10.6), ref_t=7.9, tip=(243, 640), feed=(0, 1), out=(-1, 0), scale=1.15),
    "IMG_1306": dict(window=(3.3, 8.6), ref_t=6.4, tip=(225, 655), feed=(0, -1), out=(-1, 0), scale=1.5),
}


def load(name, cfg):
    cap = cv2.VideoCapture(os.path.join(OLD, name + ".MOV"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = int(round(fps / 30))
    t0, t1 = cfg["window"]
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t0 * fps))
    frames, times, i = [], [], 0
    while True:
        ok, fr = cap.read()
        t = t0 + i / fps
        if not ok or t > t1:
            break
        if i % step == 0:
            s = 960 * cfg["scale"] / max(fr.shape[:2])
            frames.append(cv2.resize(fr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
            times.append(t)
        i += 1
    return frames, np.array(times)


def measure(name, cfg):
    frames, times = load(name, cfg)
    k = cfg["scale"]
    tip0 = np.array(cfg["tip"], float) * k
    feed0, out0 = np.array(cfg["feed"], float), np.array(cfg["out"], float)
    ref_idx = int(np.argmin(np.abs(times - cfg["ref_t"])))
    ref_axis = math.degrees(math.atan2(feed0[1], feed0[0])) % 180
    g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    c = tip0 + 70 * out0 - 45 * feed0
    cx, cy = int(c[0]), int(c[1])
    tpl = g[ref_idx][cy - TPL // 2: cy + TPL // 2, cx - TPL // 2: cx + TPL // 2]
    off = tip0 - np.array([cx, cy], float)
    pos, score = [None] * len(g), [0.0] * len(g)
    pos[ref_idx], score[ref_idx] = np.array([cx, cy], float), 1.0
    for rng in (range(ref_idx + 1, len(g)), range(ref_idx - 1, -1, -1)):
        prev = pos[ref_idx].copy()
        for i in rng:
            st = track_step(g[i], tpl, prev, SEARCH)
            if st is None:
                break
            pos[i], score[i] = st
            if score[i] > 0.5:
                prev = pos[i]
    tips = [None if p is None else p + off for p in pos]
    gb = [blur16(x) for x in g]
    rows, crops = [], []
    for i in range(1, len(g) - 1):
        r = dict(clip=name, t=round(float(times[i]), 3), valid=0, track=round(score[i], 3))
        if any(tips[j] is None for j in (i - 1, i, i + 1)) or min(score[i - 1:i + 2]) < 0.5:
            rows.append(r); continue
        d = axis_rotation(g[i], tips[i], ref_axis)
        uf, uo = rot(feed0, d), rot(out0, d)
        ft = frame_features(gb[i - 1], gb[i], gb[i + 1], g[i], tips[i - 1], tips[i], tips[i + 1], uf, uo,
                            thr=DIFF_THR, r_out=R_OUT)
        if ft is not None:
            r.update(valid=1, psi_peak=ft["psi_peak"], spread=ft["spread"], activity=ft["activity"])
        rows.append(r)
        # canonical crop, exactly as make_crops.py (440 px source -> 224, tip at (0.35, 0.65))
        kk = 224 / 440
        M = np.array([[kk * uf[0], kk * uf[1], 0.35 * 224 - kk * tips[i] @ uf],
                      [-kk * uo[0], -kk * uo[1], 0.65 * 224 + kk * tips[i] @ uo]], np.float32)
        crops.append((len(rows) - 1, cv2.cvtColor(cv2.warpAffine(frames[i], M, (224, 224), flags=cv2.INTER_AREA),
                                                   cv2.COLOR_BGR2GRAY)))
    return pd.DataFrame(rows), crops


def cnn_predict(df, crops):
    """Run the November-trained motion CNN (cnn_v3_final.pt) and the histogram-ridge model on the crops."""
    import torch
    from PIL import Image
    import cnn_v3_local as C
    from sklearn.linear_model import RidgeCV
    W = C.W
    idx = [i for i, _ in crops]
    grey = np.stack([np.asarray(Image.fromarray(c).resize((W, W), Image.BILINEAR)) for _, c in crops]).astype(np.float32)
    t = df.t.values[idx]
    n = len(grey)
    prv = np.array([j - 1 if j > 0 and abs(t[j] - t[j - 1] - 1 / 30) < 0.02 else j for j in range(n)])
    nxt = np.array([j + 1 if j < n - 1 and abs(t[j + 1] - t[j] - 1 / 30) < 0.02 else j for j in range(n)])
    d1, d2 = np.abs(grey - grey[prv]), np.abs(grey - grey[nxt])
    X = torch.from_numpy(np.stack([grey, np.clip(d1 * 4, 0, 255), np.clip(d2 * 4, 0, 255)], 1).astype(np.uint8))
    m = C.net(3)
    m.load_state_dict(torch.load(os.path.join(RES, "cnn_v3_final.pt"), map_location=C.DEV))
    p = C.predict(m, X, np.arange(n))
    # histogram + ridge, trained on all November frames
    lab, _, mag = C.load()
    H = C.hist_features(mag)
    rg = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(H, lab.psi.values)
    ph = rg.predict(C.hist_features(np.maximum(d1, d2)))
    df.loc[df.index[idx], "psi_cnn"] = p
    df.loc[df.index[idx], "psi_hist"] = ph
    return df


def main():
    out = []
    for name, cfg in CLIPS.items():
        df, crops = measure(name, cfg)
        df = cnn_predict(df, crops)
        df["psi_smooth"] = df.psi_peak.rolling(15, center=True, min_periods=5).median()
        out.append(df)
        v = df[df.valid == 1]
        print(f"{name}: frames {len(df)}, tracked {int((df.track > 0.5).sum())}, valid {len(v)}, "
              f"psi median {v.psi_peak.median():.1f}, CNN median {v.psi_cnn.median():.1f}")
    df = pd.concat(out)
    df.to_csv(os.path.join(RES, "old_clips.csv"), index=False)
    nov = pd.read_csv(os.path.join(RES, "frames_all.csv"))
    gate = 0.1 * nov[(nov.valid == 1) & nov.steady].groupby("run").activity.median().min()
    summ = []
    for name, d in df.groupby("clip"):
        v = d[(d.valid == 1) & d.psi_smooth.notna() & d.psi_cnn.notna()]
        sec = v.assign(s=(v.t // 1)).groupby("s")[["psi_smooth", "psi_cnn", "psi_hist"]].median()
        act = float(d[d.valid == 1].activity.median())
        summ.append(dict(clip=name, seconds=len(sec), frames=len(d), valid_frac=round(float((d.valid == 1).mean()), 3),
                         activity_median=round(act), usable=bool(act > gate),
                         psi_meas_median=round(float(v.psi_peak.median()), 1),
                         psi_cnn_median=round(float(v.psi_cnn.median()), 1),
                         psi_hist_median=round(float(v.psi_hist.median()), 1),
                         frame_mae_cnn=round(float((v.psi_cnn - v.psi_smooth).abs().mean()), 1),
                         frame_mae_hist=round(float((v.psi_hist - v.psi_smooth).abs().mean()), 1),
                         frame_mae_const50=round(float((50 - v.psi_smooth).abs().mean()), 1),
                         r_sec_cnn=round(float(sec.psi_smooth.corr(sec.psi_cnn)), 2) if len(sec) > 2 else np.nan,
                         r_sec_hist=round(float(sec.psi_smooth.corr(sec.psi_hist)), 2) if len(sec) > 2 else np.nan))
    pd.DataFrame(summ).to_csv(os.path.join(RES, "old_clips_summary.csv"), index=False)
    print(pd.DataFrame(summ).to_string())


if __name__ == "__main__":
    main()
