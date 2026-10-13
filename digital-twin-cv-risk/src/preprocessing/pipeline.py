"""Hourly alignment, daily aggregation, weekly sequences and proxy labels."""
from __future__ import annotations
import numpy as np
import pandas as pd

SEQ_FEATURES=["resting_hr","nocturnal_hrv","sleep_hours","sleep_efficiency","steps","systolic_bp"]
EHR_FEATURES=["age","bmi","hypertension","diabetes","ckd","smoker","sbp_history","dbp_history","ldl","hba1c","sex_M","medication_count"]

def daily_features(wearable: pd.DataFrame) -> pd.DataFrame:
    w=wearable.copy(); w["timestamp"]=pd.to_datetime(w.timestamp,utc=True)
    hourly=[]
    for pid,g in w.groupby("patient_id"):
        g=g.set_index("timestamp").sort_index()
        h=g.resample("1h").agg({"heart_rate":"median","hrv_ms":"median","sleep_stage":lambda s:s.mode().iloc[0] if not s.mode().empty else "awake","steps":"sum","sleep_efficiency":"mean","systolic_bp":"mean"})
        h["patient_id"]=pid; hourly.append(h.reset_index())
    w=pd.concat(hourly,ignore_index=True); w["date"]=w.timestamp.dt.floor("D")
    w["is_night"]=w.timestamp.dt.hour.isin([0,1,2,3,4,5,6,22,23])
    d=w.groupby(["patient_id","date"]).apply(lambda g: pd.Series({"resting_hr":g.heart_rate.quantile(.1),"nocturnal_hrv":g.loc[g.is_night,"hrv_ms"].median(),"sleep_hours":g.sleep_stage.isin(["light","deep","rem"]).sum()/1.0,"sleep_efficiency":g.loc[g.is_night,"sleep_efficiency"].replace(0,np.nan).mean(),"steps":g.steps.sum(),"systolic_bp":g.systolic_bp.mean()}),include_groups=False).reset_index()
    d[SEQ_FEATURES]=d.groupby("patient_id")[SEQ_FEATURES].transform(lambda x:x.ffill().bfill())
    d["sleep_hours"]=d.sleep_hours.clip(0,14)
    return d

def make_windows(ehr: pd.DataFrame, wearable: pd.DataFrame):
    daily=daily_features(wearable); windows=[]
    for pid,g in daily.groupby("patient_id"):
        g=g.sort_values("date").reset_index(drop=True)
        if len(g)<14: continue
        for end in range(7,len(g)-6):
            hist=g.iloc[end-7:end].copy(); nxt=g.iloc[end:end+7]
            avg_hr=hist.resting_hr.mean(); avg_hrv=hist.nocturnal_hrv.mean()
            flags=[]
            for future in nxt.itertuples(index=False):
                day_flags=[future.resting_hr>=100 or future.resting_hr>=1.2*avg_hr,future.nocturnal_hrv<20 or future.nocturnal_hrv<.7*avg_hrv,future.sleep_hours<5,future.systolic_bp>=160]
                flags.append(sum(day_flags)>=2)
            label=int(any(flags))
            windows.append({"patient_id":pid,"end_date":hist.date.iloc[-1],"sequence":hist[SEQ_FEATURES].to_numpy(dtype=np.float32),"label":label,"weekly":hist[SEQ_FEATURES].mean().to_dict()})
    e=ehr.copy(); e["sex_M"]=(e.sex=="M").astype(int); e["medication_count"]=(e.medication!="none").astype(int)
    static=e.set_index("patient_id")[EHR_FEATURES]
    return windows,static

def sequence_matrix(windows):
    return np.stack([w["sequence"] for w in windows]).astype(np.float32)
