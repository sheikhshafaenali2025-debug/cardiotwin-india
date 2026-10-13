"""Simulate 14 days of hourly wearable-like readings with diurnal patterns."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

def generate_wearable(ehr: pd.DataFrame, days: int = 14, seed: int = 8) -> pd.DataFrame:
    rng=np.random.default_rng(seed); rows=[]
    for p in ehr.itertuples(index=False):
        risk=.35*p.hypertension+.22*p.diabetes+.12*p.ckd+.08*p.smoker+.012*max(p.age-45,0)+.012*max(p.bmi-25,0)
        risk=float(np.clip(risk,0,1.5)); base_hr=66+13*risk+rng.normal(0,4); base_hrv=57-21*risk+rng.normal(0,7)
        daily_event=False
        for h in range(days*24):
            t=pd.Timestamp("2026-01-01",tz="UTC")+pd.Timedelta(hours=h); hour=t.hour
            if h%24==0: daily_event=(h//24>=7 and rng.random()<(.06+.18*risk))
            sleep=hour>=22 or hour<7
            hr=base_hr + (8 if not sleep and 9<=hour<=19 else -7 if sleep else 0)+rng.normal(0,3)
            hrv=base_hrv+(9 if sleep else -5)+rng.normal(0,7)
            stage=("deep" if sleep and hour in (1,2,3) else "rem" if sleep and hour in (4,5) else "light" if sleep else "awake")
            steps=max(0,int(rng.normal(310 if not sleep and 8<=hour<=20 else 8,120)))
            # A simulated episode is sustained for a day so hourly aggregation
            # can detect it; this is a synthetic modeling device, not physiology.
            episode=daily_event
            if episode: hr+=rng.uniform(20,35); hrv*=rng.uniform(.35,.62)
            sbp=float(np.clip(p.sbp_history+rng.normal(0,7)+(18 if episode else 0),90,210))
            rows.append((p.patient_id,t,round(float(hr),1),round(float(max(5,hrv)),1),stage,int(steps),round(float(rng.uniform(0.78,.99) if sleep else 0),2),sbp))
    return pd.DataFrame(rows,columns=["patient_id","timestamp","heart_rate","hrv_ms","sleep_stage","steps","sleep_efficiency","systolic_bp"])

if __name__ == "__main__":
    from data.ehr_synthetic.generate import generate_ehr
    parser=argparse.ArgumentParser(); parser.add_argument("--n",type=int,default=180); parser.add_argument("--out",default="data/generated/wearable.csv"); args=parser.parse_args()
    path=Path(args.out); path.parent.mkdir(parents=True,exist_ok=True); generate_wearable(generate_ehr(args.n)).to_csv(path,index=False); print(path)
