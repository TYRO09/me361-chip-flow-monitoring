"""Chip flow on the rake face (the textbook chip-flow angle), measured from the same top-view videos.

The ejection angle psi (measure_chip_angle.py) is taken 25-220 px (960 scale) from the tip, where the chip
is already in the air. Here we look at the first ~3 mm next to the tip at full resolution (1920 px), where the
chip still lies on the insert.

In steady cutting the chip on the rake face is a *standing* bright ribbon (it slides along itself at ~1 m/s,
so frame differencing hardly sees it), while flying chips come and go. Method:
  1. track the insert (same tracker as the main method) and cut a tip-centred, axis-aligned crop from every
     native-resolution frame of the steady cut (feed -> right, radially outward -> up);
  2. median over time -> the standing image of the run (insert + standing chip);
  3. there is no chip-free frame of the insert in the right place (before/after the cut the tool is off the
     bar), so the reference is the median of the *other five runs'* standing images, each registered to this
     run with an affine ECC fit on the insert pattern and bar edges away from the tip (same insert, same holder);
  4. ribbon = this run minus reference; its direction is the ray from the tip (15-95 px, ~0.5-3.3 mm) with
     the highest mean contrast on the insert side (psi 95-175 deg; psi > 175 is the bar edge / minor edge);
  5. the same in 1-s windows; a window counts as a detection if its peak is > 2 SD above the ray median.

psi_face uses the ejection-angle convention (0 = feed, 90 = outward). The theory's flow angle is
eta = 180 - psi_face (physics_flow_angle.py: from straight back over the insert towards outward).

Writes results/rake_face_flow.csv and results/figures/fig12_rake_face.jpg.
"""
import math, os, sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure_chip_angle as M  # noqa: E402
from chip_kernel import TPL, SEARCH, track_step, axis_rotation, rot  # noqa: E402
from physics_flow_angle import flow_angles  # noqa: E402

RES = os.path.join(M.ROOT, "results")
RUNS = ["R1", "R2", "R3", "R4", "R5", "R6"]
FA = {"R1": (.08, .9), "R2": (.15, .9), "R3": (.08, .3), "R4": (.15, .3), "R5": (.08, .6), "R6": (.15, .6)}
S = 280                          # crop side, native px
TIP = np.array([170.0, 175.0])   # tip inside the crop
TH = np.arange(60, 200, 1.0)
INSERT = (TH >= 95) & (TH <= 175)
_yy, _xx = np.mgrid[0:S, 0:S]
_R = np.hypot(_xx - TIP[0], TIP[1] - _yy)
REG_MASK = np.where((_R < 125) & ((TIP[1] - _yy) > -8), 0, 1).astype(np.uint8)


def track_run(rid, t_start, t_end):
    """[(frame index, t, tip in native px, u_feed, u_out)] for confidently tracked frames."""
    cfg = M.RUNS[rid]
    cap = cv2.VideoCapture(os.path.join(M.SESSION, "videos_top", cfg["file"]))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_start * fps))
    g, times, scale = [], [], 1.0
    for i in range(int((t_end - t_start) * fps)):
        ok, fr = cap.read()
        if not ok:
            break
        scale = M.SCALE_MAX / max(fr.shape[:2])
        g.append(cv2.cvtColor(cv2.resize(fr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY))
        times.append(t_start + i / fps)
    ref_idx = int(round((cfg["ref_t"] - t_start) * fps))
    feed0, out0 = np.array(cfg["feed"], float), np.array(cfg["out"], float)
    ref_axis = math.degrees(math.atan2(feed0[1], feed0[0])) % 180
    c = np.array(cfg["tip"]) + 70 * out0 - 45 * feed0
    cx, cy = int(c[0]), int(c[1])
    tpl = g[ref_idx][cy - TPL // 2: cy + TPL // 2, cx - TPL // 2: cx + TPL // 2]
    off = np.array(cfg["tip"], float) - np.array([cx, cy], float)
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
    out = []
    for i, p in enumerate(pos):
        if p is None or score[i] < 0.6:
            continue
        tip = p + off
        d = axis_rotation(g[i], tip, ref_axis)
        out.append((int(round(times[i] * fps)), times[i], tip / scale, rot(feed0, d), rot(out0, d)))
    return out


def crops(rid, track):
    """[(t, grey crop float32)] at native resolution, tip at TIP, feed -> +x, outward -> -y."""
    cap = cv2.VideoCapture(os.path.join(M.SESSION, "videos_top", M.RUNS[rid]["file"]))
    want = {fi: rest for fi, *rest in track}
    idx = min(want)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    out = []
    while idx <= max(want):
        ok, img = cap.read()
        if not ok:
            break
        w = want.get(idx)
        idx += 1
        if w is None:
            continue
        t, tip, uf, uo = w
        A = np.array([[uf[0], uf[1], TIP[0] - tip @ uf], [-uo[0], -uo[1], TIP[1] + tip @ uo]], np.float32)
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out.append((t, cv2.warpAffine(grey, A, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    .astype(np.float32)))
    return out


def register(src, dst):
    """src warped onto dst (affine ECC on the insert pattern), brightness matched; None if ECC fails."""
    w = np.eye(2, 3, dtype=np.float32)
    try:
        _, w = cv2.findTransformECC(cv2.GaussianBlur(dst, (5, 5), 0), cv2.GaussianBlur(src, (5, 5), 0), w,
                                    cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5),
                                    REG_MASK, 5)
    except cv2.error:
        return None
    o = cv2.warpAffine(src, w, (S, S), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    a, b = o[REG_MASK > 0], dst[REG_MASK > 0]
    return (o - a.mean()) / a.std() * b.std() + b.mean()


def ray_profile(d, r0=15, r1=95, half=4):
    """Mean of d along a 9-px-wide ray from the tip, for every direction in TH."""
    rr, oo = np.arange(r0, r1), np.arange(-half, half + 1)
    out = []
    for t in TH:
        u = np.array([math.cos(math.radians(t)), -math.sin(math.radians(t))])
        n = np.array([-u[1], u[0]])
        P = TIP + u * rr[:, None, None] + n * oo[None, :, None]
        x = np.clip(np.round(P[..., 0]).astype(int), 0, S - 1)
        y = np.clip(np.round(P[..., 1]).astype(int), 0, S - 1)
        out.append(d[y, x].mean())
    return np.array(out)


def peak(d):
    p = ray_profile(d)
    k = int(np.argmax(np.where(INSERT, p, -1e9)))
    z = (p[k] - np.median(p[INSERT])) / (np.std(p[INSERT]) + 1e-6)
    return float(TH[k]), float(z)


def main():
    fr = pd.read_csv(os.path.join(RES, "frames_all.csv"))
    data, med = {}, {}
    for rid in RUNS:
        st = fr[(fr.run == rid) & fr.steady]
        s0, s1 = float(st.t.min()), float(st.t.max())
        cr = [(t, c) for t, c in crops(rid, track_run(rid, s0 - 0.05, s1 + 0.05)) if s0 <= t <= s1]
        data[rid] = (np.array([t for t, _ in cr]), np.stack([c for _, c in cr]))
        med[rid] = np.median(data[rid][1], 0)
        print(rid, "frames", len(cr))
    rows, panels = [], []
    for rid in RUNS:
        regs = [r for r in (register(med[o], med[rid]) for o in RUNS if o != rid) if r is not None]
        ref = np.median(np.stack(regs), 0)
        psi_all, z_all = peak(med[rid] - ref)
        t, c = data[rid]
        wins = []
        for w0 in np.arange(t.min(), t.max() - 1.0, 1.0):
            m = (t >= w0) & (t < w0 + 1.0)
            if m.sum() >= 10:
                wins.append(peak(np.median(c[m], 0) - ref))
        wins = np.array(wins)
        det = wins[wins[:, 1] > 2.0]
        f, a = FA[rid]
        th = flow_angles(f, a)
        rows.append(dict(run=rid, feed=f, ap=a, n_ref=len(regs), psi_face_all=psi_all, z_all=round(z_all, 2),
                         windows=len(wins), detected=len(det),
                         psi_face_med=float(np.median(det[:, 0])) if len(det) else np.nan,
                         psi_face_iqr=f"{np.percentile(det[:, 0], 25):.0f}-{np.percentile(det[:, 0], 75):.0f}" if len(det) else "",
                         eta_meas=180 - float(np.median(det[:, 0])) if len(det) else np.nan,
                         eta_colwell_r04=round(th["eta_colwell"], 1), eta_ymo_r04=round(th["eta_ymo"], 1),
                         eta_colwell_r08=round(flow_angles(f, a, r_eps=0.8)["eta_colwell"], 1),
                         eta_colwell_r12=round(flow_angles(f, a, r_eps=1.2)["eta_colwell"], 1),
                         psi_eject_med=float(fr[(fr.run == rid) & fr.steady & (fr.valid == 1)].psi_peak.median())))
        r = rows[-1]
        print(f"{rid}: windows {len(wins)}, detected {len(det)}, psi_face {r['psi_face_med']} ({r['psi_face_iqr']}), "
              f"theory psi_face {180 - r['eta_colwell_r04']:.0f} (r=0.4) / {180 - r['eta_colwell_r08']:.0f} (r=0.8)")
        if rid in ("R2", "R3", "R6"):
            panels.append(_panel(rid, med[rid], ref, r))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "rake_face_flow.csv"), index=False)
    leg = np.zeros((34, panels[0].shape[1], 3), np.uint8)
    for x, col, s in ((10, (0, 220, 255), "ejection psi (in the air)"), (330, (255, 170, 60), "theory, Colwell, r=0.4 mm"),
                      (680, (60, 255, 60), "measured on the rake face")):
        cv2.line(leg, (x, 17), (x + 40, 17), col, 4); cv2.putText(leg, s, (x + 50, 23), 0, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(RES, "figures", "fig12_rake_face.jpg"), np.vstack(panels + [leg]), [cv2.IMWRITE_JPEG_QUALITY, 90])


def _panel(rid, m, ref, r):
    y0, y1, x0, x1, k = 50, 215, 60, 240, 2
    def big(im):
        return cv2.resize(im[y0:y1, x0:x1], None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    g = cv2.cvtColor(np.clip(m, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    rf = cv2.cvtColor(np.clip(ref, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    d = cv2.applyColorMap(np.clip(np.clip(m - ref, 0, None) * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    tiles = [big(g), big(rf), big(d)]
    p0 = (int((TIP[0] - x0) * k), int((TIP[1] - y0) * k))
    arrows = [(r["psi_eject_med"], (0, 220, 255)), (180 - r["eta_colwell_r04"], (255, 170, 60))]
    if np.isfinite(r["psi_face_med"]):
        arrows.append((r["psi_face_med"], (60, 255, 60)))
    for tl in (tiles[0], tiles[2]):
        for ang, col in arrows:
            e = (int(p0[0] + 150 * math.cos(math.radians(ang))), int(p0[1] - 150 * math.sin(math.radians(ang))))
            cv2.arrowedLine(tl, p0, e, col, 3, tipLength=0.12)
    labels = (f"{rid} f={r['feed']} ap={r['ap']}: standing image", "reference: other 5 runs", "difference = standing chip")
    for tl, s in zip(tiles, labels):
        cv2.rectangle(tl, (0, 0), (tl.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(tl, s, (6, 19), 0, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.hstack(tiles)


if __name__ == "__main__":
    main()
