"""CNN v3 (local, Apple GPU): can augmentation and temporal clips fix the "squeezed to the average" problem?

Same 2,336 tip-centred crops and the same leave-one-run-out (LORO) test as v1/v2 (kaggle/kernel_v2).
The v2 diagnosis: with 6 runs the training labels only span ~40-65 deg, so the network predicts ~the mean.

Variants
  motion        : v2 winner again (3 ch: grey, |d prev|, |d next|), shift/brightness jitter only
  motion_rotm   : as motion_rot, but only the two motion channels are rotated; the grey picture of tool and
                  bar stays put (otherwise the network can read delta from the tilted bar instead of the chips)
  motion_rot    : + label-changing rotation: rotate the crop about the tool tip by delta ~ U(-35, 35) deg and
                  add delta to the label. Crops are axis-aligned (feed -> right), so the chip direction
                  relative to the image x-axis really changes by delta. This creates angles the six runs never
                  produced (5-100 deg) without inventing a new physical cut.
  clip_rot      : 5-frame clip: [grey, |d t-2|, |d t-1|, |d t+1|, |d t+2|] (5 channels) + rotation
  hist_ridge    : no CNN. 72-bin histogram of motion-pixel directions around the tip (+ radius weighting)
                  then ridge regression. This is close to re-learning the hand-crafted measurement.
  leak_demo     : hist_ridge and motion_rot scored with a *random 5-fold split of frames* instead of LORO.
                  Shows how "more data + random split" fakes a good score (neighbouring frames are near copies).

Writes results/cnn_v3_metrics.json, results/cnn_v3_preds.csv, and (with --final) results/cnn_v3_final.pt.
"""
import argparse, json, math, os, time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from torchvision import models

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "kaggle", "dataset")
RES = os.path.join(ROOT, "results")
DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
W = 160                                   # network input size (crops are 224, tip at (0.35, 0.65))
TIP = (0.35, 0.65)
PSI_MU, PSI_SD = 50.0, 20.0
ROT = 35.0


def load():
    lab = pd.read_csv(os.path.join(DATA, "labels.csv")).dropna(subset=["psi"]).sort_values(["run", "t"]).reset_index(drop=True)
    grey = np.stack([np.asarray(Image.open(os.path.join(DATA, f)).convert("L").resize((W, W), Image.BILINEAR))
                     for f in lab.file]).astype(np.float32)
    n = len(lab)

    def nb(k):   # index of the frame k steps away in the same run (~k/30 s), else self
        j = np.arange(n) + k
        ok = (j >= 0) & (j < n)
        j = np.where(ok, j, np.arange(n))
        ok &= (lab.run.values[j] == lab.run.values) & (np.abs(lab.t.values[j] - lab.t.values - k / 30) < 0.02)
        return np.where(ok, j, np.arange(n))
    d = {k: np.abs(grey - grey[nb(k)]) for k in (-2, -1, 1, 2)}
    sc = lambda a: np.clip(a * 4, 0, 255)
    inputs = {"motion": np.stack([grey, sc(d[-1]), sc(d[1])], 1),
              "clip": np.stack([grey, sc(d[-2]), sc(d[-1]), sc(d[1]), sc(d[2])], 1)}
    return lab, {k: torch.from_numpy(v.astype(np.uint8)) for k, v in inputs.items()}, np.maximum(d[-1], d[1])


def rotate(x, deg):
    """Rotate a batch about the tool tip, counter-clockwise on screen: the chip angle psi grows by deg."""
    a = torch.deg2rad(deg.float()).view(-1)
    tx, ty = TIP[0] * 2 - 1, TIP[1] * 2 - 1
    c, s = torch.cos(a), torch.sin(a)
    th = torch.zeros(len(a), 2, 3, device=x.device)
    th[:, 0, 0], th[:, 0, 1], th[:, 1, 0], th[:, 1, 1] = c, -s, s, c
    th[:, 0, 2] = tx - (c * tx - s * ty)
    th[:, 1, 2] = ty - (s * tx + c * ty)
    return F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), align_corners=False)


def net(ch):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if ch != 3:   # spread the pretrained RGB filters over the extra channels
        w = m.conv1.weight.data.mean(1, keepdim=True).repeat(1, ch, 1, 1) * 3 / ch
        m.conv1 = nn.Conv2d(ch, 64, 7, 2, 3, bias=False); m.conv1.weight.data = w
    m.fc = nn.Linear(512, 1)
    return m.to(DEV)


def prep(xb, train, rot):
    x = xb.to(DEV).float() / 255
    y_shift = torch.zeros(len(x), device=DEV)
    if train:
        g = 0.7 + 0.6 * torch.rand(len(x), 1, 1, 1, device=DEV)
        x = (x * g).clamp(0, 1)
        dx, dy = np.random.randint(-8, 9, 2)
        x = torch.roll(x, (int(dy), int(dx)), (2, 3))
        if rot:
            y_shift = (torch.rand(len(x), device=DEV) * 2 - 1) * ROT
            if rot == "motion_only":   # keep the scene (grey channel) fixed: the tilt of the bar cannot give delta away
                x = torch.cat([x[:, :1], rotate(x[:, 1:], y_shift)], 1)
            else:
                x = rotate(x, y_shift)
    return (x - 0.45) / 0.25, y_shift


def fit(X, y, idx, rot, epochs=8, bs=64, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    m = net(X.shape[1])
    steps = epochs * (len(idx) // bs)
    opt = torch.optim.AdamW(m.parameters(), lr=4e-4, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=4e-4, total_steps=steps)
    yt = torch.tensor(y, dtype=torch.float32)
    for ep in range(epochs):
        m.train()
        perm = np.random.permutation(idx)
        for k in range(len(idx) // bs):
            b = perm[k * bs:(k + 1) * bs]
            x, sh = prep(X[b], True, rot)
            target = ((yt[b].to(DEV) + sh) - PSI_MU) / PSI_SD
            loss = F.smooth_l1_loss(m(x)[:, 0], target)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return m


@torch.no_grad()
def predict(m, X, idx):
    m.eval()
    out = [m(prep(X[idx[k:k + 256]], False, False)[0])[:, 0].cpu() for k in range(0, len(idx), 256)]
    return torch.cat(out).numpy() * PSI_SD + PSI_MU


def hist_features(motion_mag):
    h, w = motion_mag.shape[1:]
    yy, xx = np.mgrid[0:h, 0:w]
    dx, dy = xx - TIP[0] * w, TIP[1] * h - yy
    r, ang = np.hypot(dx, dy), np.degrees(np.arctan2(dy, dx))
    ring = (r > 8) & (r < 0.7 * w)
    bins = np.clip(((ang[ring] + 180) / 5).astype(int), 0, 71)
    rb = np.clip((r[ring] / (0.7 * w) * 3).astype(int), 0, 2)
    feats = []
    for mm in motion_mag:
        v = np.clip(mm[ring] - 12, 0, None)
        H = np.zeros((3, 72)); np.add.at(H, (rb, bins), v)
        feats.append((H / (H.sum() + 1e-6)).ravel())
    return np.array(feats)


def score(lab, idx_list, preds, name):
    folds = []
    for run, (te, p) in zip(sorted(lab.run.unique()), zip(idx_list, preds)):
        y = lab.psi.values[te]
        folds.append(dict(run=run, n=len(te), true=float(y.mean()), pred=float(np.mean(p)), mae=float(np.abs(p - y).mean())))
    f = pd.DataFrame(folds)
    base = np.array([lab.psi.values[lab.run.values != r].mean() for r in f.run])
    slope = float(np.polyfit(f.true, f.pred, 1)[0])
    return dict(variant=name, run_rmse=float(np.sqrt(((f.pred - f.true) ** 2).mean())),
                run_rmse_base=float(np.sqrt(((base - f.true) ** 2).mean())),
                frame_mae=float((f.mae * f.n).sum() / f.n.sum()), r=float(np.corrcoef(f.true, f.pred)[0, 1]),
                slope=slope, folds=folds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="hist_ridge,motion,motion_rot,clip_rot,leak_demo")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--final", action="store_true", help="also train the --final-variant model on all six runs and save it")
    ap.add_argument("--final-variant", default="motion_rot", choices=["motion", "motion_rot", "motion_rotm"])
    a = ap.parse_args()
    t0 = time.time()
    lab, X, mag = load()
    y = lab.psi.values
    runs = sorted(lab.run.unique())
    loro = [(np.flatnonzero(lab.run != r), np.flatnonzero(lab.run == r)) for r in runs]
    print("device", DEV, "frames", len(lab), "label range", y.min(), y.max(), f"load {time.time()-t0:.0f}s")
    path = os.path.join(RES, "cnn_v3_metrics.json")
    metrics = json.load(open(path)) if os.path.exists(path) else {}
    allp = []
    H = None
    for v in a.variants.split(","):
        t1 = time.time()
        if v == "hist_ridge" or v == "leak_demo":
            H = hist_features(mag) if H is None else H
        if v == "hist_ridge":
            ps = [RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(H[tr], y[tr]).predict(H[te]) for tr, te in loro]
        elif v in ("motion", "motion_rot", "clip_rot", "motion_rotm", "clip_rotm"):
            Xv = X["clip" if v.startswith("clip") else "motion"]
            mode = "motion_only" if v.endswith("rotm") else v.endswith("rot")
            ps = []
            for tr, te in loro:
                ps.append(predict(fit(Xv, y, tr, rot=mode, epochs=a.epochs), Xv, te))
                print(f"  {v} fold done {time.time()-t1:.0f}s")
        elif v == "leak_demo":
            kf = list(KFold(5, shuffle=True, random_state=0).split(y))
            pr = np.zeros(len(y)); pc = np.zeros(len(y))
            for tr, te in kf:
                pr[te] = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(H[tr], y[tr]).predict(H[te])
            tr, te = kf[0]
            pc[te] = predict(fit(X["motion"], y, tr, rot=True, epochs=a.epochs), X["motion"], te)
            metrics["leak_demo"] = dict(hist_ridge_random5fold_frame_mae=float(np.abs(pr - y).mean()),
                                        motion_rot_random_fold1_frame_mae=float(np.abs(pc[te] - y[te]).mean()),
                                        frame_mae_predict_mean=float(np.abs(y - y.mean()).mean()))
            print(v, metrics["leak_demo"]); continue
        else:
            raise SystemExit(f"unknown variant {v}")
        res = score(lab, [te for _, te in loro], ps, v)
        metrics[v] = res
        for (tr, te), p in zip(loro, ps):
            allp.append(pd.DataFrame(dict(variant=v, run=lab.run.values[te], t=lab.t.values[te], psi=y[te], pred=p)))
        print(v, {k: round(val, 2) for k, val in res.items() if isinstance(val, float)}, f"{time.time()-t1:.0f}s")
        json.dump(metrics, open(path, "w"), indent=1)
    if allp:
        out = os.path.join(RES, "cnn_v3_preds.csv")
        new = pd.concat(allp)
        if os.path.exists(out):
            old = pd.read_csv(out)
            new = pd.concat([old[~old.variant.isin(new.variant.unique())], new])
        new.to_csv(out, index=False)
    if a.final:
        fv = a.final_variant
        mode = "motion_only" if fv.endswith("rotm") else fv.endswith("rot")
        m = fit(X["motion"], y, np.arange(len(y)), rot=mode, epochs=a.epochs)
        torch.save(m.state_dict(), os.path.join(RES, "cnn_v3_final.pt"))
        json.dump({"final_variant": fv}, open(os.path.join(RES, "cnn_v3_final.json"), "w"))
        print("saved final model", fv)
    print(f"total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
