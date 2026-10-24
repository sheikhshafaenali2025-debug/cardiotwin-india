# CardioTwin India

> **Synthetic research prototype only.** CardioTwin India is not a medical device, validated risk calculator, diagnosis, treatment recommendation, or substitute for professional medical advice. It uses synthetic data and rule-generated proxy labels; its scores must not be used for clinical decisions.

## Team details

- **Team size:** 1 (individual participant)
- **Team/member name:** Sheikh Shafaen Ali
- **College/incubator:** Vit Vellore
- **Contact:** sheikh.shafaenali2025@vitstudent.ac.in

## Project title

**CardioTwin India: A Wearable and EHR Research Prototype for 7-Day Cardiovascular Risk**

## Problem statement

Cardiovascular risk factors can change over time, while a single measurement may not describe a person's recent pattern. This project demonstrates a research workflow that combines a synthetic patient profile with longitudinal wearable summaries to explore changes in a simulated cardiovascular-risk proxy over a seven-day window.

The project does not predict real heart attacks, strokes, or other clinical events. The labels are generated from illustrative rules, not observed health outcomes. Results only measure how well a model recovers those synthetic rules.

## Healthcare use case

The prototype is designed as an educational demonstration of how a future, properly governed digital-twin system might organize longitudinal signals for review. A user can select a virtual patient, inspect synthetic EHR context and recent wearable trends, see a model score alongside its limitations, compare recent signals with that patient's simulated baseline, and explore hypothetical future input paths.

It is not intended to diagnose, treat, triage, or reassure real patients. No real patient data is included.

## Technical stack

- **Language:** Python
- **Data and features:** pandas, NumPy
- **Classical ML:** scikit-learn logistic regression and XGBoost
- **Sequence model:** PyTorch, using a 1D CNN followed by a GRU and a fusion head for static and wearable inputs
- **Dashboard and charts:** Streamlit and Plotly
- **Artifacts:** joblib model files and a PyTorch checkpoint

### AI/ML model and framework details

The pipeline generates synthetic EHR-like patient profiles and hourly wearable streams, aligns and aggregates the signals, and builds seven-day windows. Logistic regression and XGBoost provide baseline comparisons. The hybrid model combines static profile features with a seven-day wearable sequence. Training uses patient-grouped development, validation, and holdout splits so windows from a patient do not cross those partitions. Reported metrics include AUROC, AUPRC, sensitivity at 10% false-positive rate, and Brier score.

The dashboard also includes:

- **Personalized baseline and Twin Drift:** compares the latest seven adequately covered simulated days with the prior fourteen days, reports coverage, and highlights sustained deviations using robust within-person statistics. These are display heuristics, not clinical thresholds or model explanations.
- **Branching Futures Simulator:** compares a continuation path with hypothetical gradual sleep and activity changes. It bootstraps recent synthetic days and shows the model's response range. It is not a causal estimate of an intervention's effect.

## Demo video

**Video link:** in work

## Open-source license

This project is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for the full license text.

## Architecture diagram

- [Architecture diagram (PDF)](docs/architecture.pdf)
- [Editable Mermaid source](docs/architecture.mmd)

## Project presentation

- [Project presentation (PowerPoint)](docs/presentation.pptx)

The PDF/PPT files are stored in this repository so they can be opened by anyone with access to the repository. Keep the repository public for submission if the competition requires public access.

## Data, labels, and ethics

This repository uses synthetic data only. The generators create EHR-like profiles and wearable time series; no real patient data or protected health information is collected or distributed. The demo does not require an external data source.

The synthetic target is a rule-generated proxy label, not an observed cardiovascular event. A seven-day window is marked positive when at least two illustrative signals occur on a day in the following week: elevated resting heart rate, low nocturnal HRV, sleep under five hours, or systolic blood pressure at least 160 mmHg when available. These thresholds are demonstration rules, not validated medical criteria. Synthetic correlations can produce misleadingly strong scores, and model metrics do not establish clinical validity, safety, fairness, calibration, or generalizability.

The code is not a determination of DPDP Act or HIPAA compliance. Any future work with real data would require appropriate legal and ethics review, consent and governance, clinical collaboration, external validation, bias and safety evaluation, and regulatory advice.

## Run locally

Run these commands in the `digital-twin-cv-risk` project root (the folder containing `run_demo.py`). The first run installs dependencies and trains the demo models; this may take several minutes on CPU.

### Windows PowerShell

```powershell
python3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_demo.py --patients 300 --days 30 --epochs 12 --launch-dashboard
```

If PowerShell blocks environment activation, run the virtual-environment interpreter directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_demo.py --patients 300 --days 30 --epochs 12 --launch-dashboard
```

The dashboard opens at the local URL printed by Streamlit, usually `http://localhost:8501` (or another port if that one is busy). Generated CSVs, metrics, and model files are written to `artifacts/`. They are intentionally excluded from Git by `.gitignore`; run the demo command after cloning to generate them locally.

### Generate or train separately

```powershell
python run_demo.py --generate-only --patients 300 --days 30
python -m src.training.train --patients 300 --days 30 --epochs 12
python -m src.inference.predict --patient P0001
python -m streamlit run src/dashboard/app.py
```

Run these from the project root with its virtual environment active. The full `run_demo.py` command is the simplest setup because it generates and trains the matching artifacts before launching the dashboard.

## Repository map

```text
data/                 Synthetic EHR and wearable generators and sample CSVs
src/preprocessing/    Alignment, resampling, windows, and feature extraction
src/models/           Hybrid CNN-GRU model
src/training/         Grouped splits, training, and evaluation metrics
src/inference/        Patient scoring, Twin Drift, and future-path simulation
src/dashboard/        Streamlit dashboard
notebooks/            Analysis plan and reproducible analysis material
docs/                 Architecture PDF/source and presentation deck
run_demo.py           End-to-end generation, training, and launch entry point
```

## Evaluation note

The repository includes an evaluation snapshot from a previous 180-patient, 14-day baseline run with a 36-patient holdout and one seven-day input window per patient. Its numbers describe agreement with the synthetic proxy labels only; they are not clinical performance results. AUROC was near chance in that run, and the holdout was small. A new run writes its own `artifacts/metrics.json` report.

| Model | AUROC | AUPRC | Sensitivity at 10% FPR | Brier score |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression | 0.462 | 0.636 | 0.043 | 0.263 |
| XGBoost | 0.555 | 0.685 | 0.217 | 0.252 |
| CNN-GRU hybrid | 0.502 | 0.724 | 0.261 | 0.242 |

The proxy-label prevalence in that run was 57.8%. These values must not be presented as clinical prediction results.
