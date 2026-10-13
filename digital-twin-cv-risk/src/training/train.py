"""Train weekly aggregated baselines and hybrid sequence model."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import joblib
import torch
from torch import nn
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
from data.ehr_synthetic.generate import generate_ehr
from data.wearable.generate import generate_wearable
from src.preprocessing.pipeline import make_windows, EHR_FEATURES, SEQ_FEATURES
from src.models.hybrid import HybridRiskModel

def metrics(y,p):
    y=np.asarray(y); p=np.asarray(p); out={"auroc":float(roc_auc_score(y,p)) if len(np.unique(y))>1 else float("nan"),"auprc":float(average_precision_score(y,p)) if len(np.unique(y))>1 else float("nan"),"brier":float(brier_score_loss(y,p))}
    if len(np.unique(y))>1:
        fpr,tpr,_=roc_curve(y,p); eligible=np.where(fpr<=.10)[0]; out["sensitivity_at_10pct_fpr"]=float(tpr[eligible[-1]]) if len(eligible) else 0.0
    else: out["sensitivity_at_10pct_fpr"]=float("nan")
    return out

def train(patients=180,epochs=12,seed=7,out=Path("artifacts")):
    out.mkdir(parents=True,exist_ok=True); ehr=generate_ehr(patients,seed); wearable=generate_wearable(ehr,14,seed+1); windows,static=make_windows(ehr,wearable)
    if not windows: raise RuntimeError("No 7-day windows generated")
    y=np.array([w["label"] for w in windows]); groups=np.array([w["patient_id"] for w in windows]); seq=np.stack([w["sequence"] for w in windows]); stat=static.loc[groups].to_numpy(dtype=np.float32)
    split=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed); outer_tr,te=next(split.split(stat,y,groups)); split2=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed+1); tr_rel,va_rel=next(split2.split(stat[outer_tr],y[outer_tr],groups[outer_tr])); tr=outer_tr[tr_rel]; va=outer_tr[va_rel]
    ss=StandardScaler().fit(stat[tr]); xs=ss.transform(stat); qs=StandardScaler().fit(seq[tr].reshape(-1,len(SEQ_FEATURES))); xq=qs.transform(seq.reshape(-1,len(SEQ_FEATURES))).reshape(seq.shape)
    agg=np.concatenate([xs,np.nan_to_num(xq.mean(axis=1))],axis=1)
    baseline=make_pipeline(StandardScaler(),LogisticRegression(max_iter=500,class_weight="balanced",random_state=seed)); baseline.fit(agg[tr],y[tr]); joblib.dump(baseline,out/"baseline.joblib")
    xgb=XGBClassifier(n_estimators=120,max_depth=3,learning_rate=.04,subsample=.85,colsample_bytree=.85,eval_metric="logloss",n_jobs=2,random_state=seed); xgb.fit(agg[tr],y[tr]); joblib.dump(xgb,out/"xgboost.joblib")
    device="cuda" if torch.cuda.is_available() else "cpu"; model=HybridRiskModel(stat.shape[1]).to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3); loss_fn=nn.BCEWithLogitsLoss()
    tensors=[torch.tensor(a,dtype=torch.float32,device=device) for a in (xs,xq)]; yt=torch.tensor(y,dtype=torch.float32,device=device)
    best=float("inf"); best_state=None
    for _ in range(epochs):
        model.train(); order=np.random.default_rng(seed+_).permutation(tr)
        for batch in np.array_split(order,max(1,int(np.ceil(len(order)/64)))):
            opt.zero_grad(); loss=loss_fn(model(tensors[0][batch],tensors[1][batch]),yt[batch]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): val_loss=float(loss_fn(model(tensors[0][va],tensors[1][va]),yt[va]).item())
        if val_loss<best: best=val_loss; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad(): prob=torch.sigmoid(model(tensors[0][te],tensors[1][te])).cpu().numpy()
    base_prob=baseline.predict_proba(agg[te])[:,1]; xgb_prob=xgb.predict_proba(agg[te])[:,1]
    torch.save({"state_dict":model.state_dict(),"static_dim":stat.shape[1],"sequence_features":len(SEQ_FEATURES),"static_scaler":ss,"sequence_scaler":qs,"ehr_features":EHR_FEATURES,"sequence_columns":SEQ_FEATURES},out/"hybrid.pt")
    report={"proxy_label_prevalence":float(y.mean()),"n_patients":int(ehr.patient_id.nunique()),"n_windows":len(windows),"logistic_regression":metrics(y[te],base_prob),"xgboost":metrics(y[te],xgb_prob),"hybrid":metrics(y[te],prob),"split":"patient-level holdout; metrics reflect synthetic heuristic only"}
    (out/"metrics.json").write_text(json.dumps(report,indent=2)); ehr.to_csv(out/"ehr.csv",index=False); wearable.to_csv(out/"wearable.csv",index=False)
    print(json.dumps(report,indent=2)); return report

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--patients",type=int,default=180); p.add_argument("--epochs",type=int,default=12); p.add_argument("--seed",type=int,default=7); p.add_argument("--out",default="artifacts"); a=p.parse_args(); train(a.patients,a.epochs,a.seed,Path(a.out))
