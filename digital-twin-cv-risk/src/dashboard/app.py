"""Streamlit dashboard for virtual patients in the generated cohort."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from src.inference.predict import predict
from src.preprocessing.pipeline import daily_features

ROOT=Path(__file__).resolve().parents[2]; ART=ROOT/"artifacts"
st.set_page_config(page_title="CardioTwin India | Research demo",layout="wide")
st.title("CardioTwin India")
st.caption("Synthetic research demonstration · Not for clinical use")
if not (ART/"ehr.csv").exists() or not (ART/"hybrid.pt").exists():
    st.warning("No trained demo artifacts found. From the project root run: python run_demo.py"); st.stop()
ehr=pd.read_csv(ART/"ehr.csv"); wearable=pd.read_csv(ART/"wearable.csv")
patient=st.sidebar.selectbox("Virtual patient",ehr.patient_id.tolist()); threshold=st.sidebar.slider("Alert threshold",0.1,0.9,0.5,0.05)
person=ehr[ehr.patient_id==patient].iloc[0]; result=predict(patient,ART,threshold=threshold)
daily=daily_features(wearable[wearable.patient_id==patient]).sort_values("date").tail(14)
daily=daily.reset_index(drop=True); daily["proxy_alert"]=False
for i in range(len(daily)):
    hist=daily.iloc[max(0,i-7):i]
    if hist.empty: continue
    row=daily.iloc[i]; flags=[row.resting_hr>=100 or row.resting_hr>=1.2*hist.resting_hr.mean(),row.nocturnal_hrv<20 or row.nocturnal_hrv<.7*hist.nocturnal_hrv.mean(),row.sleep_hours<5,row.systolic_bp>=160]
    daily.loc[i,"proxy_alert"]=sum(flags)>=2
left,right=st.columns([1,2]);
with left:
    st.subheader("Patient summary")
    st.write(f"Age {person.age} · Sex {person.sex} · BMI {person.bmi}")
    st.write(f"BP history {person.sbp_history:.0f}/{person.dbp_history:.0f} mmHg")
    st.write(f"Medication {person.medication} · HTN {bool(person.hypertension)} · Diabetes {bool(person.diabetes)}")
    st.metric("7-day proxy score",f"{result['risk_score']:.0%}",result["risk_tier"].title())
    st.progress(result["risk_score"])
    st.error("Alert threshold crossed") if result["alert"] else st.success("Below alert threshold")
    st.subheader("Top contributing factors")
    for factor in result["top_factors"]: st.write("• "+factor)
    st.caption(result["warning"])
with right:
    st.subheader("14-day wearable timeline")
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=daily.date,y=daily.resting_hr,name="Resting HR (bpm)",yaxis="y"))
    fig.add_trace(go.Scatter(x=daily.date,y=daily.nocturnal_hrv,name="Nocturnal HRV (ms)",yaxis="y2"))
    fig.add_trace(go.Bar(x=daily.date,y=daily.steps,name="Steps",opacity=.25,yaxis="y3"))
    marks=daily[daily.proxy_alert]
    if not marks.empty: fig.add_trace(go.Scatter(x=marks.date,y=marks.resting_hr,name="Synthetic alert day",mode="markers",marker=dict(color="#C94F45",size=10,symbol="x"),yaxis="y"))
    fig.update_layout(height=430,margin=dict(l=20,r=40,t=20,b=30),legend=dict(orientation="h"),yaxis=dict(title="HR"),yaxis2=dict(title="HRV",overlaying="y",side="right"),yaxis3=dict(title="Steps",anchor="free",overlaying="y",side="right",position=.94,showgrid=False))
    st.plotly_chart(fig,use_container_width=True)
    st.line_chart(daily.set_index("date")[["sleep_hours","sleep_efficiency"]],height=190)
    st.area_chart(pd.DataFrame({"Latest-window risk reference":[result["risk_score"]]*len(daily)},index=daily.date),height=100)
    st.caption("The flat risk band repeats the latest-window score for context; it is not a retrospective daily risk series. Red markers show synthetic rule days, not clinical events.")
st.subheader("What-if projection")
c1,c2=st.columns(2)
with c1: sleep_goal=st.slider("Sleep target (hours/night)",5.0,9.0,7.0,.5)
with c2: activity_goal=st.slider("Activity target (steps/day)",2000,12000,7000,500)
baseline=result["risk_score"]; projected=max(0,min(1,baseline-0.025*(sleep_goal-7)-0.015*((activity_goal-7000)/1000)))
st.write(f"Conceptual adjusted score: **{projected:.0%}** (current {baseline:.0%}). This illustrative arithmetic is not model output or a treatment recommendation.")
st.subheader("Alert history")
history=daily[daily.proxy_alert][["date"]].copy(); history["score"]=baseline; history["alert"]=True; history["outcome"]="Synthetic proxy rule fired; no clinical outcome available"
if history.empty: st.info("No historical synthetic rule alerts in this displayed window.")
else: st.dataframe(history,use_container_width=True,hide_index=True)
