"""Streamlit dashboard for a synthetic cardiovascular-risk research demo."""
from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ARTIFACTS = ROOT / "artifacts"

from src.inference.predict import predict
from src.preprocessing.pipeline import daily_features


st.set_page_config(
    page_title="CardioTwin India | Research demo",
    page_icon=":material/ecg_heart:",
    layout="wide",
)


@st.cache_data(ttl=300, show_spinner=False)
def load_inputs():
    """Read the generated cohort once per session window, not on each widget rerun."""
    return (
        pd.read_csv(ARTIFACTS / "ehr.csv"),
        pd.read_csv(ARTIFACTS / "wearable.csv"),
    )


@st.cache_data(ttl=300, show_spinner=False)
def load_prediction(patient_id: str, threshold: float, sleep_hours=None, steps=None):
    scenario = None
    if sleep_hours is not None or steps is not None:
        scenario = {"sleep_hours": sleep_hours, "steps": steps}
    return predict(patient_id, ARTIFACTS, threshold=threshold, scenario=scenario)


def add_rule_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Mark simulated days meeting the transparent two-signal rule."""
    frame = frame.copy().reset_index(drop=True)
    frame["proxy_alert"] = False
    frame["rule_factors"] = ""
    for i in range(7, len(frame)):
        history = frame.iloc[i - 7 : i]
        row = frame.iloc[i]
        if row.sensor_coverage < 0.75 or (history.sensor_coverage < 0.75).any():
            continue
        checks = {
            "resting HR elevated": row.resting_hr >= 100
            or row.resting_hr >= 1.2 * history.resting_hr.mean(),
            "nocturnal HRV low": row.nocturnal_hrv < 20
            or row.nocturnal_hrv < 0.7 * history.nocturnal_hrv.mean(),
            "sleep under 5 hours": row.sleep_hours < 5,
            "systolic BP at least 160": row.systolic_bp >= 160,
        }
        factors = [name for name, triggered in checks.items() if triggered]
        frame.loc[i, "proxy_alert"] = len(factors) >= 2
        frame.loc[i, "rule_factors"] = "; ".join(factors)
    return frame


def timeline_figure(frame: pd.DataFrame) -> go.Figure:
    titles = [
        "Resting heart rate (bpm)",
        "Nocturnal HRV (ms)",
        "Sleep duration (hours)",
        "Sleep efficiency (%)",
        "Steps per day",
    ]
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        subplot_titles=titles,
    )
    series = [
        ("resting_hr", 1, "#176B63", "lines"),
        ("nocturnal_hrv", 2, "#4677A8", "lines"),
        ("sleep_hours", 3, "#9B6A35", "lines"),
        ("sleep_efficiency_pct", 4, "#7961A8", "lines"),
        ("steps", 5, "#55905B", "lines+markers"),
    ]
    event_days = frame[frame["proxy_alert"]]
    for column, row, color, mode in series:
        fig.add_trace(
            go.Scatter(
                x=frame["date"],
                y=frame[column],
                mode=mode,
                name=titles[row - 1],
                line={"color": color, "width": 2},
                marker={"color": color, "size": 5},
                showlegend=False,
                hovertemplate="%{x|%b %d}: %{y:.1f}<extra></extra>",
            ),
            row=row,
            col=1,
        )
        if not event_days.empty:
            fig.add_trace(
                go.Scatter(
                    x=event_days["date"],
                    y=event_days[column],
                    mode="markers",
                    name="Synthetic rule day",
                    marker={"color": "#C94F45", "size": 9, "symbol": "x"},
                    showlegend=(row == 1),
                    hovertemplate="Synthetic rule day · %{x|%b %d}<extra></extra>",
                ),
                row=row,
                col=1,
            )
    fig.update_layout(
        height=790,
        margin={"l": 12, "r": 22, "t": 55, "b": 25},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.04, "x": 0},
    )
    fig.update_xaxes(showgrid=False)
    return fig


st.title("CardioTwin India")
st.caption(
    "Synthetic research demonstration. Scores and rule events are not clinical predictions."
)

required_files = [ARTIFACTS / name for name in ("ehr.csv", "wearable.csv", "hybrid.pt")]
missing = [path.name for path in required_files if not path.exists()]
if missing:
    st.warning(
        "Demo files are missing: "
        + ", ".join(missing)
        + ". From the project root, run `python run_demo.py`."
    )
    st.stop()

with st.sidebar:
    st.subheader("Demo controls")
    try:
        ehr, wearable = load_inputs()
    except Exception as exc:
        st.error(f"Could not read the generated cohort: {exc}")
        st.stop()
    patient_id = st.selectbox("Virtual patient", ehr.patient_id.tolist())
    alert_threshold = st.slider("Display alert cutoff", 0.1, 0.9, 0.5, 0.05)
    st.caption("Changing the cutoff changes the displayed alert only.")

person = ehr.loc[ehr.patient_id == patient_id].iloc[0]
try:
    result = load_prediction(patient_id, alert_threshold)
except Exception as exc:
    st.error(f"Could not score this virtual patient: {exc}")
    st.stop()

patient_stream = wearable.loc[wearable.patient_id == patient_id]
all_daily = daily_features(patient_stream).sort_values("date")
daily = all_daily.tail(14).copy()
daily = add_rule_flags(daily)
daily["sleep_efficiency_pct"] = daily.sleep_efficiency * 100
stream_timestamps = pd.to_datetime(patient_stream.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
coverage = float(daily.sensor_coverage.mean()) if not daily.empty else 0.0

summary, status = st.columns([2, 1], vertical_alignment="top")
with summary:
    st.subheader("Virtual patient summary")
    with st.container(border=True):
        st.write(f"**Age:** {person.age} · **Sex:** {person.sex} · **BMI:** {person.bmi}")
        st.write(
            f"**Synthetic baseline BP:** {person.sbp_history:.0f}/{person.dbp_history:.0f} mmHg"
        )
        st.write(
            f"**Medication:** {person.medication} · **Hypertension:** "
            f"{bool(person.hypertension)} · **Diabetes:** {bool(person.diabetes)}"
        )
        st.write(
            f"**Wearable coverage, latest 14 days:** {coverage:.0%} · "
            f"**Latest reading:** {stream_timestamps.max().strftime('%d %b %Y, %H:%M IST')}"
        )
with status:
    st.subheader("Next 7-day synthetic forecast")
    st.metric("Synthetic proxy score", f"{result['risk_score']:.2f}")
    st.caption(
        f"Display band: {result['risk_tier'].title()} · "
        f"Alert cutoff: {alert_threshold:.0%}. Unitless score, not a calibrated event probability."
    )
    if result["alert"]:
        st.warning("Synthetic score is above the display cutoff. Review the data context; this is not a diagnosis.")
    else:
        st.info("Score is below the display cutoff. This is not an all-clear or a clinical assessment.")

st.subheader("Recent signal context")
st.caption("Week-to-week changes and synthetic EHR context; these are not model explanations or clinical criteria.")
with st.container(border=True):
    for factor in result["top_factors"]:
        st.write(f"- {factor}")

st.subheader("14-day simulated wearable timeline")
st.plotly_chart(
    timeline_figure(daily),
    width="stretch",
    alt="Fourteen-day synthetic trends for heart rate, HRV, sleep, and steps; red marks denote simulated rule days.",
)
st.caption(
    "Dates belong to the fixed synthetic simulation. Red marks show days meeting the "
    "transparent two-signal rule, not model alerts or clinical events. The score above "
    "applies only to the latest seven-day input window."
)

st.subheader("Illustrative what-if controls")
st.caption(
    "This reruns the model with the latest seven days replaced by these wearable values. "
    "It is an input scenario only, not a treatment effect or clinical recommendation."
)
sleep_col, activity_col = st.columns(2)
with sleep_col:
    sleep_goal = st.slider("Sleep target (hours/night)", 5.0, 9.0, 7.0, 0.5)
with activity_col:
    activity_goal = st.slider("Activity target (steps/day)", 2000, 12000, 7000, 500)

scenario_result = load_prediction(patient_id, alert_threshold, sleep_goal, activity_goal)
scenario_change = scenario_result["risk_score"] - result["risk_score"]
scenario_col, comparison_col = st.columns(2)
with scenario_col:
    st.metric("Model score in this input scenario", f"{scenario_result['risk_score']:.2f}")
with comparison_col:
    st.metric("Difference from current input score", f"{scenario_change:+.2f}", delta_color="off")
st.caption(
    "The change reflects the model response to edited inputs in this synthetic dataset. "
    "It does not show that changing sleep or activity will change a real person's health outcome."
)

st.subheader("Synthetic rule-event history")
event_history = daily.loc[daily.proxy_alert, ["date", "rule_factors"]].rename(
    columns={"date": "Simulation date", "rule_factors": "Rule indicators"}
)
if event_history.empty:
    st.info("No days in this displayed window met the synthetic two-signal rule.")
else:
    st.dataframe(
        event_history,
        hide_index=True,
        width="stretch",
        alt="Simulated days when at least two rule indicators were present.",
    )
st.caption("No clinical outcomes are simulated or available in this project.")
