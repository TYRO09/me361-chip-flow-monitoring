"""ME361: CNN that reads the chip ejection angle (and the feed) from a camera crop at the tool tip.

Runs on Kaggle (GPU). Leave-one-run-out cross-validation over the 6 runs, then a final model on
all runs. Outputs in /kaggle/working: preds.csv, metrics.json, model_all_runs.pt
"""
import glob, json, os, time, zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

torch.manual_seed(0); np.random.seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "")

# ---------- locate data (Kaggle may keep crops as a zip) ----------
labels_path = glob.glob("/kaggle/input/**/labels.csv", recursive=True)[0]
base = os.path.dirname(labels_path)
if not os.path.isdir(os.path.join(base, "crops")):
    z = glob.glob(os.path.join(base, "**", "crops.zip"), recursive=True)[0]
    zipfile.ZipFile(z).extractall("/kaggle/working/data")
    base = "/kaggle/working/data"
lab = pd.read_csv(labels_path)
lab = lab.dropna(subset=["psi"]).reset_index(drop=True)
lab["feed_cls"] = (lab.feed > 0.1).astype(int)
print(lab.groupby("run").agg(n=("psi", "size"), psi=("psi", "mean"), feed=("feed", "first")))

PSI_MU, PSI_SD = 50.0, 15.0
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
# No flips or rotations: they would change the angle the label describes.
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
    transforms.ColorJitter(0.3, 0.3, 0.2, 0.02),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class Crops(Dataset):
    def __init__(self, df, tf):
        self.df, self.tf = df.reset_index(drop=True), tf
        self.cache = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        if r.file not in self.cache:
            self.cache[r.file] = Image.open(os.path.join(base, r.file)).convert("RGB")
        x = self.tf(self.cache[r.file])
        return x, torch.tensor((r.psi - PSI_MU) / PSI_SD, dtype=torch.float32), torch.tensor(r.feed_cls, dtype=torch.float32)


def make_model():
    try:
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        print("using ImageNet weights")
    except Exception as e:  # no internet
        print("pretrained weights unavailable:", e); m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)          # [angle, feed logit]
    return m.to(DEV)


def train(df, epochs=8):
    m = make_model()
    dl = DataLoader(Crops(df, train_tf), batch_size=64, shuffle=True, num_workers=2, drop_last=True)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=epochs * len(dl))
    for ep in range(epochs):
        m.train(); tot = 0
        for x, y, c in dl:
            x, y, c = x.to(DEV), y.to(DEV), c.to(DEV)
            out = m(x)
            loss = nn.functional.smooth_l1_loss(out[:, 0], y) + 0.5 * nn.functional.binary_cross_entropy_with_logits(out[:, 1], c)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        print(f"  epoch {ep+1}/{epochs} loss {tot/len(dl):.4f}")
    return m


@torch.no_grad()
def predict(m, df):
    m.eval(); dl = DataLoader(Crops(df, test_tf), batch_size=128, num_workers=2)
    ps, fs = [], []
    for x, _, _ in dl:
        out = m(x.to(DEV)).cpu()
        ps.append(out[:, 0] * PSI_SD + PSI_MU); fs.append(torch.sigmoid(out[:, 1]))
    return torch.cat(ps).numpy(), torch.cat(fs).numpy()


t_start = time.time()
preds, folds = [], []
for run in sorted(lab.run.unique()):
    tr, te = lab[lab.run != run], lab[lab.run == run]
    print(f"fold: hold out {run} (train {len(tr)}, test {len(te)})")
    m = train(tr)
    p, fp = predict(m, te)
    base_pred = tr.psi.mean()
    d = te[["file", "run", "t", "psi", "psi_raw", "feed", "ap"]].copy()
    d["psi_pred"], d["feed_prob_0.15"], d["baseline_pred"] = p, fp, base_pred
    preds.append(d)
    folds.append(dict(run=run, n=len(te), psi_true_mean=float(te.psi.mean()), psi_pred_mean=float(p.mean()),
                      frame_mae=float(np.abs(p - te.psi).mean()), frame_mae_baseline=float(np.abs(base_pred - te.psi).mean()),
                      run_mean_error=float(p.mean() - te.psi.mean()), run_mean_error_baseline=float(base_pred - te.psi.mean()),
                      feed_acc=float(((fp > 0.5).astype(int) == te.feed_cls).mean())))
    print("  ", folds[-1])

pd.concat(preds).to_csv("/kaggle/working/preds.csv", index=False)
f = pd.DataFrame(folds)
summary = dict(
    folds=folds,
    run_level_rmse=float(np.sqrt((f.run_mean_error ** 2).mean())),
    run_level_rmse_baseline=float(np.sqrt((f.run_mean_error_baseline ** 2).mean())),
    frame_mae=float((f.frame_mae * f.n).sum() / f.n.sum()),
    frame_mae_baseline=float((f.frame_mae_baseline * f.n).sum() / f.n.sum()),
    feed_acc=float((f.feed_acc * f.n).sum() / f.n.sum()),
    device=DEV, minutes=(time.time() - t_start) / 60)
print(json.dumps({k: v for k, v in summary.items() if k != "folds"}, indent=1))

print("final model on all runs")
m = train(lab)
torch.save(m.state_dict(), "/kaggle/working/model_all_runs.pt")
summary["minutes_total"] = (time.time() - t_start) / 60
json.dump(summary, open("/kaggle/working/metrics.json", "w"), indent=1)
print("done")
