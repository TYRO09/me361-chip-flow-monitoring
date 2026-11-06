"""ME361 CNN v2: does changing what the network SEES fix the v1 failure (it learned clip identity)?

Same 2,336 tip-centred crops, same leave-one-run-out test. Four input/model variants:
  rgb_finetune   : v1 baseline again (ImageNet ResNet-18, fine-tuned, colour jitter)
  motion         : 3 channels = [grey(t), |grey(t)-grey(t-1)|, |grey(t+1)-grey(t)|]  -> sees only what moves
  masked_rgb     : RGB with static background blacked out (pixels that do not move are zeroed)
  linear_probe   : frozen ImageNet features of the motion input + ridge regression (cannot memorise much)
Consecutive crops are already aligned on the tool tip, so frame differences show chip motion only.
Outputs (/kaggle/working): metrics_v2.json, preds_v2.csv
"""
import glob, json, os, time, zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import RidgeCV, LogisticRegression
from torch.utils.data import DataLoader, Dataset
from torchvision import models

torch.manual_seed(0); np.random.seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "")

labels_path = glob.glob("/kaggle/input/**/labels.csv", recursive=True)[0]
base = os.path.dirname(labels_path)
if not os.path.isdir(os.path.join(base, "crops")):
    z = glob.glob(os.path.join(base, "**", "crops.zip"), recursive=True)[0]
    zipfile.ZipFile(z).extractall("/kaggle/working/data"); base = "/kaggle/working/data"
lab = pd.read_csv(labels_path).dropna(subset=["psi"]).sort_values(["run", "t"]).reset_index(drop=True)
lab["feed_cls"] = (lab.feed > 0.1).astype(int)

# ---- load every crop once (2,336 x 224 x 224 x 3 uint8 ~ 350 MB) ----
imgs = np.stack([np.asarray(Image.open(os.path.join(base, f)).convert("RGB")) for f in lab.file])
grey = imgs.astype(np.float32).mean(-1)
prev_i = np.arange(len(lab)) - 1; next_i = np.arange(len(lab)) + 1
for i in range(len(lab)):  # neighbour must be the same run and ~1 frame away, else use self
    if prev_i[i] < 0 or lab.run[prev_i[i]] != lab.run[i] or lab.t[i] - lab.t[prev_i[i]] > 0.07: prev_i[i] = i
    if next_i[i] >= len(lab) or lab.run[next_i[i]] != lab.run[i] or lab.t[next_i[i]] - lab.t[i] > 0.07: next_i[i] = i
d1 = np.abs(grey - grey[prev_i]); d2 = np.abs(grey[next_i] - grey)
motion = np.stack([grey, np.clip(d1 * 4, 0, 255), np.clip(d2 * 4, 0, 255)], -1).astype(np.uint8)
mask = (np.maximum(d1, d2) > 12).astype(np.float32)
mask = torch.nn.functional.max_pool2d(torch.from_numpy(mask)[:, None], 9, 1, 4)[:, 0].numpy()  # dilate
masked = (imgs * mask[..., None]).astype(np.uint8)
INPUTS = {"rgb_finetune": imgs, "motion": motion, "masked_rgb": masked}
print("inputs ready", {k: v.shape for k, v in INPUTS.items()}, "mean moving fraction", float(mask.mean()))

PSI_MU, PSI_SD = 50.0, 15.0
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class DS(Dataset):
    def __init__(self, arr, idx, train):
        self.arr, self.idx, self.train = arr, np.asarray(idx), train
    def __len__(self): return len(self.idx)
    def __getitem__(self, k):
        i = self.idx[k]
        x = torch.from_numpy(self.arr[i]).permute(2, 0, 1).float() / 255
        if self.train:   # brightness/contrast jitter + small shift; no flips/rotations (they change the angle)
            x = (x - x.mean()) * (0.7 + 0.6 * torch.rand(1)) + x.mean() * (0.7 + 0.6 * torch.rand(1))
            dx, dy = np.random.randint(-12, 13, 2); x = torch.roll(x, (int(dy), int(dx)), (1, 2))
        x = (x.clamp(0, 1) - MEAN) / STD
        return x, torch.tensor((lab.psi[i] - PSI_MU) / PSI_SD, dtype=torch.float32), torch.tensor(lab.feed_cls[i], dtype=torch.float32)


def resnet(head=2):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, head); return m.to(DEV)


def train(arr, idx, epochs=6):
    m = resnet(); dl = DataLoader(DS(arr, idx, True), batch_size=64, shuffle=True, num_workers=2, drop_last=True)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=epochs * len(dl))
    for ep in range(epochs):
        m.train(); tot = 0
        for x, y, c in dl:
            x, y, c = x.to(DEV), y.to(DEV), c.to(DEV); o = m(x)
            loss = nn.functional.smooth_l1_loss(o[:, 0], y) + 0.5 * nn.functional.binary_cross_entropy_with_logits(o[:, 1], c)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step(); tot += loss.item()
    print(f"   last-epoch loss {tot/len(dl):.4f}")
    return m


@torch.no_grad()
def predict(m, arr, idx):
    m.eval(); out = []
    for x, _, _ in DataLoader(DS(arr, idx, False), batch_size=128, num_workers=2):
        out.append(m(x.to(DEV)).cpu())
    o = torch.cat(out); return (o[:, 0] * PSI_SD + PSI_MU).numpy(), torch.sigmoid(o[:, 1]).numpy()


@torch.no_grad()
def features(arr):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1); m.fc = nn.Identity(); m = m.to(DEV).eval()
    out = []
    for x, _, _ in DataLoader(DS(arr, np.arange(len(arr)), False), batch_size=128, num_workers=2):
        out.append(m(x.to(DEV)).cpu())
    return torch.cat(out).numpy()


t0 = time.time(); RUNS = sorted(lab.run.unique()); allp, metrics = [], {}
for name in ["motion", "masked_rgb", "rgb_finetune", "linear_probe"]:
    print("=== variant", name)
    if name == "linear_probe":
        F = features(motion)
    folds = []
    for run in RUNS:
        tr = np.flatnonzero(lab.run != run); te = np.flatnonzero(lab.run == run)
        if name == "linear_probe":
            rg = RidgeCV(alphas=np.logspace(0, 5, 11)).fit(F[tr], lab.psi[tr]); p = rg.predict(F[te])
            fp = LogisticRegression(C=0.01, max_iter=2000).fit(F[tr], lab.feed_cls[tr]).predict_proba(F[te])[:, 1]
        else:
            m = train(INPUTS[name], tr); p, fp = predict(m, INPUTS[name], te)
        y = lab.psi.values[te]; b = lab.psi.values[tr].mean()
        folds.append(dict(run=run, n=len(te), true=float(y.mean()), pred=float(p.mean()), base=float(b),
                          mae=float(np.abs(p - y).mean()), mae_base=float(np.abs(b - y).mean()),
                          feed_acc=float(((fp > 0.5).astype(int) == lab.feed_cls.values[te]).mean())))
        allp.append(pd.DataFrame(dict(variant=name, run=run, t=lab.t.values[te], psi=y, psi_pred=p, feed_prob=fp)))
        print("  ", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in folds[-1].items()})
    f = pd.DataFrame(folds)
    metrics[name] = dict(run_rmse=float(np.sqrt(((f.pred - f.true) ** 2).mean())),
                         run_rmse_baseline=float(np.sqrt(((f.base - f.true) ** 2).mean())),
                         frame_mae=float((f.mae * f.n).sum() / f.n.sum()), frame_mae_baseline=float((f.mae_base * f.n).sum() / f.n.sum()),
                         feed_acc=float((f.feed_acc * f.n).sum() / f.n.sum()),
                         feed_runs_correct=int(sum(((r.feed_acc > 0.5)) for r in f.itertuples())), folds=folds)
    print(name, {k: round(v, 3) for k, v in metrics[name].items() if k != "folds"}, f"{(time.time()-t0)/60:.1f} min")
pd.concat(allp).to_csv("/kaggle/working/preds_v2.csv", index=False)
metrics["minutes"] = (time.time() - t0) / 60
json.dump(metrics, open("/kaggle/working/metrics_v2.json", "w"), indent=1)
print("done")
