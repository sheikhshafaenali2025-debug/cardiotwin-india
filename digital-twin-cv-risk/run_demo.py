"""One-command synthetic end-to-end project demo."""
import argparse, subprocess, sys
from pathlib import Path
from data.ehr_synthetic.generate import generate_ehr
from data.wearable.generate import generate_wearable

def main():
    p=argparse.ArgumentParser(); p.add_argument("--patients",type=int,default=300); p.add_argument("--days",type=int,default=30); p.add_argument("--epochs",type=int,default=12); p.add_argument("--generate-only",action="store_true"); p.add_argument("--launch-dashboard",action="store_true"); a=p.parse_args()
    out=Path("data/generated"); out.mkdir(parents=True,exist_ok=True); ehr=generate_ehr(a.patients); wear=generate_wearable(ehr,a.days)
    ehr.to_csv(out/"ehr.csv",index=False); wear.to_csv(out/"wearable.csv",index=False)
    if a.generate_only: print(f"Generated {len(ehr)} virtual patients and {len(wear)} hourly observations under {out}"); return
    subprocess.run([sys.executable,"-m","src.training.train","--patients",str(a.patients),"--days",str(a.days),"--epochs",str(a.epochs)],check=True)
    print("Training complete. Example: python -m src.inference.predict --patient P0001")
    if a.launch_dashboard: subprocess.run([sys.executable,"-m","streamlit","run","src/dashboard/app.py"],check=True)
    else: print("Dashboard: streamlit run src/dashboard/app.py")

if __name__=="__main__": main()
