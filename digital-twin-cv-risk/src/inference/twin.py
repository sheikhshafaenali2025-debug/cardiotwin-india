"""Personal baseline drift and bounded synthetic future scenario simulation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.hybrid import HybridRiskModel
from src.preprocessing.pipeline import (
    EHR_FEATURES,
    SEQ_FEATURES,
    SEQ_MODEL_FEATURES,
    daily_features,
)


DRIFT_FEATURES = {
    "resting_hr": ("Resting heart rate", "bpm", "higher", 3.0),
    "nocturnal_hrv": ("Nocturnal HRV", "ms", "lower", 4.0),
    "sleep_hours": ("Sleep duration", "hours", "lower", 0.5),
    "sleep_efficiency": ("Sleep efficiency", "fraction", "lower", 0.05),
    "steps": ("Steps", "steps/day", "lower", 750.0),
    "systolic_bp": ("Systolic BP", "mmHg", "higher", 7.0),
}


def personalized_drift(daily: pd.DataFrame) -> dict:
    """Compare the latest week with the previous two weeks for one patient.

    Thresholds here are display heuristics for a synthetic demo, not medical
    cutoffs. Only days with adequate raw sensor coverage are used.
    """
    frame = daily.sort_values("date").reset_index(drop=True)
    if len(frame) < 21:
        return {"ready": False, "message": "At least 21 simulated days are needed to form a 14-day baseline and compare the latest week."}

    baseline = frame.iloc[-21:-7]
    recent = frame.iloc[-7:]
    baseline = baseline.loc[baseline.sensor_coverage >= 0.75]
    recent = recent.loc[recent.sensor_coverage >= 0.75]
    quality = min(
        float(baseline.sensor_coverage.mean()) if len(baseline) else 0.0,
        float(recent.sensor_coverage.mean()) if len(recent) else 0.0,
    )
    quality_label = "Good" if quality >= 0.9 else "Usable" if quality >= 0.75 else "Limited"
    if len(baseline) < 7 or len(recent) < 4:
        return {
            "ready": False,
            "message": "Not enough adequately covered days in the baseline or recent week to compare reliably.",
            "quality": quality,
            "quality_label": quality_label,
            "baseline_days": len(baseline),
            "recent_days": len(recent),
        }

    rows = []
    drift_signals = []
    for feature, (label, unit, direction, floor) in DRIFT_FEATURES.items():
        base_valid = baseline.loc[baseline[f"{feature}_missing"] < 0.5, feature].astype(float)
        recent_valid = recent.loc[recent[f"{feature}_missing"] < 0.5, feature].astype(float)
        if len(base_valid) < 5 or len(recent_valid) < 3:
            rows.append({
                "Signal": label,
                "Personal baseline": np.nan,
                "Recent week": np.nan,
                "Change": np.nan,
                "Unit": unit,
                "Days outside baseline": 0,
                "Valid days (baseline / recent)": f"{len(base_valid)} / {len(recent_valid)}",
                "Drift": "Insufficient signal data",
            })
            continue
        base_values = base_valid
        recent_values = recent_valid
        center = float(base_values.median())
        mad = float((base_values - center).abs().median())
        robust_scale = max(1.4826 * mad, floor / 2.0)
        recent_center = float(recent_values.median())
        delta = recent_center - center
        signed_z = delta / robust_scale
        adverse = signed_z if direction == "higher" else -signed_z
        daily_adverse = (recent_values - center) if direction == "higher" else (center - recent_values)
        sustained_days = int((daily_adverse >= max(floor, 2.0 * robust_scale)).sum())
        notable = adverse >= 2.0 and sustained_days >= 3
        row = {
            "Signal": label,
            "Personal baseline": center,
            "Recent week": recent_center,
            "Change": delta,
            "Unit": unit,
            "Days outside baseline": sustained_days,
            "Valid days (baseline / recent)": f"{len(base_values)} / {len(recent_values)}",
            "Drift": "Sustained synthetic deviation" if notable else "No sustained deviation",
        }
        rows.append(row)
        if notable:
            drift_signals.append(
                f"{label}: {delta:+.1f} {unit} from personal baseline on {sustained_days} of 7 days"
            )

    return {
        "ready": True,
        "quality": quality,
        "quality_label": quality_label,
        "baseline_days": len(baseline),
        "recent_days": len(recent),
        "signals": drift_signals,
        "table": pd.DataFrame(rows),
    }


def _load_patient_model(patient_id: str, artifact_dir: Path):
    ehr = pd.read_csv(artifact_dir / "ehr.csv")
    wearable = pd.read_csv(artifact_dir / "wearable.csv")
    person = ehr.loc[ehr.patient_id == patient_id]
    if person.empty:
        raise ValueError(f"Unknown patient: {patient_id}")
    daily = daily_features(wearable.loc[wearable.patient_id == patient_id]).sort_values("date")
    if len(daily) < 7:
        raise ValueError("At least seven daily observations are needed")

    checkpoint = torch.load(artifact_dir / "hybrid.pt", map_location="cpu", weights_only=False)
    if checkpoint.get("pipeline_version") != 2:
        raise RuntimeError("Model artifacts use an older preprocessing pipeline. Regenerate them with `python run_demo.py` before scoring.")
    row = person.iloc[0].copy()
    row["sex_M"] = int(row.sex == "M")
    row["medication_count"] = int(row.medication != "none")
    static = row[EHR_FEATURES].to_numpy(dtype=np.float32).reshape(1, -1)
    static = checkpoint["static_scaler"].transform(static)
    model = HybridRiskModel(checkpoint["static_dim"], checkpoint["sequence_features"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return daily, checkpoint, model, static


def simulate_futures(
    patient_id: str,
    artifact_dir: Path = Path("artifacts"),
    horizon: int = 14,
    simulations: int = 30,
    seed: int = 17,
) -> pd.DataFrame:
    """Bootstrap recent synthetic days and score three hypothetical branches.

    Scores are the model's response to sampled synthetic input sequences. They
    are not causal treatment effects or probabilities of real clinical events.
    Shared bootstrap draws make branch comparisons less sensitive to random
    variation between simulations.
    """
    daily, checkpoint, model, static = _load_patient_model(patient_id, Path(artifact_dir))
    if horizon < 1 or simulations < 1:
        raise ValueError("horizon and simulations must be positive")

    source = daily[SEQ_MODEL_FEATURES].tail(7).to_numpy(dtype=np.float32)
    if len(source) < 7:
        raise ValueError("At least seven days are needed to simulate futures")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(source), size=(simulations, horizon))
    branches = (
        ("Continue current pattern", None),
        ("More regular sleep", "sleep_hours"),
        ("Gradual activity increase", "steps"),
    )
    sequence_columns = checkpoint["sequence_columns"]
    sequence_scaler = checkpoint["sequence_scaler"]
    output = []

    for branch_name, intervention in branches:
        rolling = np.repeat(source[None, :, :], simulations, axis=0)
        scores_by_day = []
        for day in range(horizon):
            sampled = source[draws[:, day]].copy()
            progress = (day + 1) / horizon
            if intervention == "sleep_hours":
                sampled[:, sequence_columns.index("sleep_hours")] = np.clip(
                    sampled[:, sequence_columns.index("sleep_hours")] + 0.5 * progress,
                    0,
                    10,
                )
                sampled[:, sequence_columns.index("sleep_hours_missing")] = 0.0
            elif intervention == "steps":
                step_idx = sequence_columns.index("steps")
                sampled[:, step_idx] = np.clip(sampled[:, step_idx] * (1.0 + 0.15 * progress), 0, 20000)
                sampled[:, sequence_columns.index("steps_missing")] = 0.0

            rolling = np.concatenate([rolling[:, 1:, :], sampled[:, None, :]], axis=1)
            scaled = sequence_scaler.transform(rolling.reshape(-1, len(sequence_columns))).reshape(rolling.shape)
            static_batch = np.repeat(static, simulations, axis=0)
            with torch.no_grad():
                logits = model(
                    torch.tensor(static_batch, dtype=torch.float32),
                    torch.tensor(scaled, dtype=torch.float32),
                )
                scores_by_day.append(torch.sigmoid(logits).cpu().numpy().reshape(-1))

        scores = np.stack(scores_by_day, axis=1)
        for day in range(horizon):
            output.append({
                "day": day + 1,
                "branch": branch_name,
                "median": float(np.quantile(scores[:, day], 0.50)),
                "lower": float(np.quantile(scores[:, day], 0.10)),
                "upper": float(np.quantile(scores[:, day], 0.90)),
            })

    dates = pd.date_range(daily["date"].iloc[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    result = pd.DataFrame(output)
    result["date"] = result["day"].map(dict(enumerate(dates, start=1)))
    return result
