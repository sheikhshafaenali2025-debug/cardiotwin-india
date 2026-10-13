"""Generate synthetic EHR-like profiles. No patient records are imported."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

def generate_ehr(n: int = 180, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(35, 81, n)
    sex = rng.choice(["F", "M"], n)
    bmi = np.clip(rng.normal(27 + 0.03 * (age - 55), 4.5, n), 17, 46)
    hypertension = rng.binomial(1, np.clip(.35 + .009 * (age - 35), .2, .8), n)
    diabetes = rng.binomial(1, np.clip(.08 + .004 * (age - 35), .05, .3), n)
    ckd = rng.binomial(1, .04 + .0009 * (age - 35), n)
    smoker = rng.binomial(1, .18, n)
    sbp = np.clip(rng.normal(122 + 16 * hypertension + 5 * diabetes, 12, n), 90, 200)
    dbp = np.clip(rng.normal(77 + 8 * hypertension, 8, n), 55, 125)
    ldl = np.clip(rng.normal(112 + 12 * diabetes + 8 * smoker, 25, n), 40, 240)
    hba1c = np.clip(rng.normal(5.5 + 1.2 * diabetes, .8, n), 4, 12)
    medication = np.where(hypertension, rng.choice(["ACEi", "ARB", "CCB", "thiazide"], n), "none")
    return pd.DataFrame({"patient_id":[f"P{i:04d}" for i in range(1,n+1)],"age":age,"sex":sex,"bmi":bmi.round(1),"hypertension":hypertension,"diabetes":diabetes,"ckd":ckd,"smoker":smoker,"sbp_history":sbp.round(0),"dbp_history":dbp.round(0),"ldl":ldl.round(1),"hba1c":hba1c.round(1),"medication":medication})

if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--n",type=int,default=180); parser.add_argument("--seed",type=int,default=7); parser.add_argument("--out",default="data/generated/ehr.csv"); args=parser.parse_args()
    path=Path(args.out); path.parent.mkdir(parents=True,exist_ok=True); generate_ehr(args.n,args.seed).to_csv(path,index=False); print(path)
