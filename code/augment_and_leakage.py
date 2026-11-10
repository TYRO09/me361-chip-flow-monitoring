"""Does cutting the six runs into many sub-clips ("a bigger data set") help the settings -> angle model?

Two questions, answered with the per-frame measurements in results/frames_all.csv:
  A. Sub-clip augmentation: cut each run's steady part into 1-s / 2-s windows (median angle per window)
     and train on windows instead of run means. Scored by leave-one-run-out (LORO) on run means.
  B. Leakage: the same models scored with a *random* split of frames / windows. Neighbouring frames of one
     run are almost copies, so a random split puts the test run in the training set and the score is fake.
Writes results/augment_leakage.csv.
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FA = {"R1": (.08, .9), "R2": (.15, .9), "R3": (.08, .3), "R4": (.15, .3), "R5": (.08, .6), "R6": (.15, .6)}


def table(win):
    fr = pd.read_csv(os.path.join(RES, "frames_all.csv"))
    fr = fr[(fr.valid == 1) & fr.steady].copy()
    fr["f"] = fr.run.map(lambda r: FA[r][0]); fr["ap"] = fr.run.map(lambda r: FA[r][1])
    fr["tau"] = fr.groupby("run").t.transform(lambda t: (t - t.min()) / (t.max() - t.min()))
    if win == 0:
        return fr.rename(columns={"psi_peak": "psi"})[["run", "f", "ap", "tau", "psi"]]
    fr["w"] = (fr.groupby("run").t.transform(lambda t: t - t.min()) // win).astype(int)
    return fr.groupby(["run", "w"]).agg(f=("f", "first"), ap=("ap", "first"), tau=("tau", "mean"),
                                        psi=("psi_peak", "median")).reset_index()


MODELS = {
    "linear f+ap": (lambda: LinearRegression(), ["f", "ap"]),
    "random forest f+ap": (lambda: RandomForestRegressor(200, min_samples_leaf=3, random_state=0), ["f", "ap"]),
    "boosting f+ap+time": (lambda: GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05,
                                                             random_state=0), ["f", "ap", "tau"]),
}


def run_rmse(d, pred):
    m = d.assign(p=pred).groupby("run")[["psi", "p"]].mean()
    return float(np.sqrt(((m.psi - m.p) ** 2).mean()))


def main():
    rows = []
    run_means = table(0).groupby("run").psi.mean()
    for win, label in ((None, "6 run means"), (2, "2-s windows"), (1, "1-s windows"), (0, "single frames")):
        if win is None:
            d = run_means.reset_index().assign(f=lambda x: x.run.map(lambda r: FA[r][0]),
                                               ap=lambda x: x.run.map(lambda r: FA[r][1]), tau=0.5)
        else:
            d = table(win)
        for name, (mk, cols) in MODELS.items():
            if win is None and "time" in name:
                continue
            p = np.zeros(len(d))
            for r in FA:                                      # LORO
                tr, te = d.run != r, d.run == r
                p[te.values] = mk().fit(d.loc[tr, cols], d.loc[tr, "psi"]).predict(d.loc[te, cols])
            loro = run_rmse(d, p)
            rnd = np.nan
            if win is not None:                               # random 5-fold over samples (leaky)
                q = np.zeros(len(d))
                for tr, te in KFold(5, shuffle=True, random_state=0).split(d):
                    q[te] = mk().fit(d.iloc[tr][cols], d.iloc[tr].psi).predict(d.iloc[te][cols])
                rnd = run_rmse(d, q)
            tm = d.groupby("run").psi.mean()                  # this table's own run-level target
            base = np.sqrt(np.mean([(tm[r] - tm.drop(r).mean()) ** 2 for r in FA]))
            # same target, but the model only sees the 6 run means: separates "more samples" from "cleaner target"
            agg = tm.reset_index().assign(f=lambda x: x.run.map(lambda r: FA[r][0]), ap=lambda x: x.run.map(lambda r: FA[r][1]))
            p6 = np.zeros(len(agg))
            if "time" not in name:
                for r in FA:
                    tr, te = agg.run != r, agg.run == r
                    p6[te.values] = mk().fit(agg.loc[tr, cols], agg.loc[tr, "psi"]).predict(agg.loc[te, cols])
                six = float(np.sqrt(((agg.psi - p6) ** 2).mean()))
            else:
                six = np.nan
            rows.append(dict(data=label, samples=len(d), model=name, loro_run_rmse=round(loro, 2),
                             loro_same_target_6_means=round(six, 2),
                             random_split_run_rmse=round(rnd, 2), baseline_same_target=round(float(base), 2)))
            print(rows[-1])
    pd.DataFrame(rows).to_csv(os.path.join(RES, "augment_leakage.csv"), index=False)


if __name__ == "__main__":
    main()
