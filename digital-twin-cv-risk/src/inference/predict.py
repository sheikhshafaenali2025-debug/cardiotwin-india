"""Score a patient's latest seven days and return transparent proxy drivers."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import torch
from src.models.hybrid import HybridRiskModel
from src.preprocessing.pipeline import SEQ_FEATURES, SEQ_MODEL_FEATURES, EHR_FEATURES, daily_features

def explain(sequence,static_row,baseline_sequence=None):
    """Describe recent changes; these are context signals, not model attributions."""
    recent=pd.DataFrame(sequence,columns=SEQ_FEATURES); signals=[]
    if baseline_sequence is not None and len(baseline_sequence):
        prior=pd.DataFrame(baseline_sequence,columns=SEQ_FEATURES)
        changes=[
            ("resting_hr",5.0,"bpm","higher"),
            ("nocturnal_hrv",0.20,"fraction","lower"),
            ("sleep_hours",0.5,"hours","lower"),
            ("steps",0.20,"fraction","lower"),
            ("systolic_bp",5.0,"mmHg","higher"),
        ]
        for feature,threshold,unit,direction in changes:
            before=float(prior[feature].mean()); after=float(recent[feature].mean())
            delta=after-before
            if unit=="fraction":
                if before<=0: continue
                magnitude=abs(delta)/abs(before)
                crossed=magnitude>=threshold
                value=f"{magnitude:.0%}"
            else:
                magnitude=abs(delta); crossed=magnitude>=threshold
                value=f"{magnitude:.1f} {unit}"
            if crossed and ((direction=="higher" and delta>0) or (direction=="lower" and delta<0)):
                signals.append((magnitude,f"{feature.replace('_',' ').title()} {value} {direction} than the previous week"))
        signals=[message for _,message in sorted(signals,reverse=True)]
    if static_row.get("hypertension",0):
        signals.append("Synthetic EHR indicates a hypertension history")
    return signals[:5] or ["No large week-to-week change crossed the display thresholds"]

def predict(patient_id,artifact_dir=Path("artifacts"),ehr_path=None,wearable_path=None,threshold=.5,scenario=None):
    artifact_dir=Path(artifact_dir); ehr=pd.read_csv(ehr_path or artifact_dir/"ehr.csv"); wearable=pd.read_csv(wearable_path or artifact_dir/"wearable.csv")
    person=ehr[ehr.patient_id==patient_id]
    if person.empty: raise ValueError(f"Unknown patient: {patient_id}")
    daily=daily_features(wearable[wearable.patient_id==patient_id]).sort_values("date")
    if len(daily)<7: raise ValueError("At least seven daily observations are needed")
    ck=torch.load(artifact_dir/"hybrid.pt",map_location="cpu",weights_only=False)
    if ck.get("pipeline_version") != 2:
        raise RuntimeError("Model artifacts use an older preprocessing pipeline. Regenerate them with `python run_demo.py` before scoring.")
    sequence_columns=ck["sequence_columns"]
    model_window=daily[SEQ_MODEL_FEATURES].tail(7).copy()
    if scenario:
        for feature in ("sleep_hours","steps"):
            if feature in scenario and scenario[feature] is not None:
                model_window.loc[:,feature]=float(scenario[feature])
                model_window.loc[:,f"{feature}_missing"]=0.0
    seq=model_window[sequence_columns].to_numpy(dtype=np.float32)
    row=person.iloc[0].copy(); row["sex_M"]=int(row.sex=="M"); row["medication_count"]=int(row.medication!="none")
    static=row[EHR_FEATURES].to_numpy(dtype=np.float32); static=ck["static_scaler"].transform(static.reshape(1,-1)); seq=ck["sequence_scaler"].transform(seq).reshape(1,7,-1)
    model=HybridRiskModel(ck["static_dim"],ck["sequence_features"]); model.load_state_dict(ck["state_dict"]); model.eval()
    with torch.no_grad(): score=float(torch.sigmoid(model(torch.tensor(static,dtype=torch.float32),torch.tensor(seq,dtype=torch.float32))).item())
    baseline_window=daily[SEQ_FEATURES].iloc[-14:-7].to_numpy() if len(daily)>=14 else None
    return {"patient_id":patient_id,"risk_score":round(score,4),"alert":score>=threshold,"threshold":threshold,"risk_tier":"high" if score>=.7 else "moderate" if score>=.4 else "lower","top_factors":explain(daily[SEQ_FEATURES].tail(7).to_numpy(),row.to_dict(),baseline_window),"scenario_applied":bool(scenario),"warning":"Synthetic proxy score; not a calibrated event probability, diagnosis, or medical advice."}

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--patient",default="P0001"); p.add_argument("--threshold",type=float,default=.5); a=p.parse_args(); print(json.dumps(predict(a.patient,threshold=a.threshold),indent=2))
