"""Cut a canonical crop around the tool tip for every steady-cut frame (CNN training data).

Each crop is rotated (and mirrored if needed) so that the feed direction points right and
"radially outward" points up. The tip sits at (35%, 65%) of the crop. Labels: per-frame chip
ejection angle (0.5 s rolling median of the hand-crafted measurement) and the feed class.
"""
import json, os, sys
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from measure_chip_angle import RUNS, SESSION, SCALE_MAX

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "kaggle", "dataset")
SRC = 440          # crop side in (960-scaled) source pixels
DST = 224


def main():
    os.makedirs(os.path.join(OUT, "crops"), exist_ok=True)
    fr = pd.read_csv(os.path.join(ROOT, "results", "frames_all.csv"))
    runs = pd.read_csv(os.path.join(SESSION, "runs.csv")).set_index("run_id")
    rows = []
    for rid, d in fr.groupby("run"):
        d = d.sort_values("t").copy()
        d["psi_smooth"] = d.set_index("t").psi_peak.rolling(15, center=True, min_periods=5).median().values
        d = d[d.steady]
        cap = cv2.VideoCapture(os.path.join(SESSION, "videos_top", RUNS[rid]["file"]))
        fps = cap.get(cv2.CAP_PROP_FPS)
        # read sequentially and crop on the fly (seeking in HEVC .MOV files is very slow)
        wanted = {int(round(row.t * fps)): row for _, row in d.iterrows()}
        idx = 0
        while idx <= max(wanted):
            ok, img = cap.read()
            if not ok:
                break
            row = wanted.get(idx)
            idx += 1
            if row is None:
                continue
            s = SCALE_MAX / max(img.shape[:2])
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            uf, uo = np.array(json.loads(row.u_f)), np.array(json.loads(row.u_o))
            tip = np.array([row.tip_x, row.tip_y])
            k = DST / SRC
            c = np.array([0.35 * DST, 0.65 * DST])
            # dst = k * [ (p - tip).uf , -(p - tip).uo ] + c
            M = np.array([[k * uf[0], k * uf[1], c[0] - k * tip @ uf],
                          [-k * uo[0], -k * uo[1], c[1] + k * tip @ uo]], np.float32)
            crop = cv2.warpAffine(img, M, (DST, DST), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)
            name = f"{rid}_{row.t:07.3f}.jpg"
            cv2.imwrite(os.path.join(OUT, "crops", name), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            rows.append(dict(file=f"crops/{name}", run=rid, t=row.t, psi=round(row.psi_smooth, 2),
                             psi_raw=row.psi_peak, feed=runs.loc[rid, "feed_mm_per_rev"],
                             ap=runs.loc[rid, "depth_of_cut_mm"]))
        print(rid, len([r for r in rows if r["run"] == rid]))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "labels.csv"), index=False)


if __name__ == "__main__":
    main()
