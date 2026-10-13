# Analysis plan

The training entry point writes a patient-level holdout metrics report to `artifacts/metrics.json`. This project keeps a short, reproducible analysis plan rather than shipping fabricated findings.

1. Inspect age, sex, hypertension, diabetes, BMI and generated wearable distributions.
2. Compare logistic regression with the CNN-GRU hybrid on identical patient holdout IDs.
3. Repeat with wearable-only, EHR-only, and fused inputs for an ablation.
4. Plot reliability diagrams and inspect calibration; any apparent calibration is against the synthetic rule label only.
5. Review errors by age, sex and comorbidity strata; small simulated cohorts are not fairness evidence.

Do not interpret results as clinical validation.
