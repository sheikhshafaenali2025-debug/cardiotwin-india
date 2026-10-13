"""Score a patient's latest seven days and return transparent proxy drivers."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import torch
from src.models.hybrid import HybridRiskModel
from src.preprocessing.pipeline import SEQ_FEATURES, EHR_FEATURES, daily_features

def explain(sequence,static_row):
    x=pd.DataFrame(sequence,columns=SEQ_FEATURES); drivers=[]
    hr=x.resting_hr.mean(); hrv=x.nocturnal_hrv.mean(); sleep=x.sleep_hours.mean()
    if hr>90: drivers.append(f"Elevated resting HR ({hr:.0f} bpm weekly mean)")
    if hrv<30: drivers.append(f"Low nocturnal HRV ({hrv:.0f} ms weekly mean)")
    if sleep<6: drivers.append(f"Short sleep ({sleep:.1f} h nightly mean)")
    if x.steps.mean()<4000: drivers.append("Low activity (weekly mean below 4,000 steps/day)")
    if x.systolic_bp.mean()>=150: drivers.append("Elevated observed systolic BP in the synthetic stream")
    if static_row.get("hypertension",0): drivers.append("Synthetic EHR hypertension history")
    return drivers[:5] or ["No single heuristic driver crossed the display threshold"]

def predict(patient_id,artifact_dir=Path("artifacts"),ehr_path=None,wearable_path=None,threshold=.5):
    artifact_dir=Path(artifact_dir); ehr=pd.read_csv(ehr_path or artifact_dir/"ehr.csv"); wearable=pd.read_csv(wearable_path or artifact_dir/"wearable.csv")
    person=ehr[ehr.patient_id==patient_id]
    if person.empty: raise ValueError(f"Unknown patient: {patient_id}")
    daily=daily_features(wearable[wearable.patient_id==patient_id]).sort_values("date")
    if len(daily)<7: raise ValueError("At least seven daily observations are needed")
    seq=daily[SEQ_FEATURES].tail(7).to_numpy(dtype=np.float32)
    row=person.iloc[0].copy(); row["sex_M"]=int(row.sex=="M"); row["medication_count"]=int(row.medication!="none")
    static=row[EHR_FEATURES].to_numpy(dtype=np.float32)
    ck=torch.load(artifact_dir/"hybrid.pt",map_location="cpu",weights_only=False); static=ck["static_scaler"].transform(static.reshape(1,-1)); seq=ck["sequence_scaler"].transform(seq).reshape(1,7,-1)
    model=HybridRiskModel(ck["static_dim"],ck["sequence_features"]); model.load_state_dict(ck["state_dict"]); model.eval()
    with torch.no_grad(): score=float(torch.sigmoid(model(torch.tensor(static,dtype=torch.float32),torch.tensor(seq,dtype=torch.float32))).item())
    return {"patient_id":patient_id,"risk_score":round(score,4),"alert":score>=threshold,"threshold":threshold,"risk_tier":"high" if score>=.7 else "moderate" if score>=.4 else "lower","top_factors":explain(daily[SEQ_FEATURES].tail(7).to_numpy(),row.to_dict()),"warning":"Synthetic research output; not a diagnosis or medical advice."}

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--patient",default="P0001"); p.add_argument("--threshold",type=float,default=.5); a=p.parse_args(); print(json.dumps(predict(a.patient,threshold=a.threshold),indent=2))
