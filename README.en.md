# AeroRivet-QC-RF ✈️

*[Türkçe](README.md) | English*

An end-to-end **Machine Learning-based Quality Control** project for aircraft structural assembly (rivet/hole) processes. A Random Forest classification model predicts, from manufacturing-line measurement data, whether a part **requires rework**, and presents the results through an interactive Streamlit dashboard.

## Project Summary

In aircraft fuselage assembly, rivet holes are subject to tight tolerances on parameters such as diameter, countersink depth, and squeeze force. Exceeding these tolerances can lead to defective connections that compromise structural integrity. This project:

1. Generates a synthetic manufacturing dataset based on **FAA-H-8083-31A** (Aircraft Structural Repair) and the **NAS1097** (100° Flush/Countersunk Head Rivet) rivet/countersink tolerance standard,
2. Trains a **Random Forest** classifier on this data,
3. Optimizes Recall using a **low decision threshold (0.25)**, prioritizing flight safety,
4. Presents the results in a **Streamlit dashboard** usable by field operators.

## Quality Logic: Why Recall Matters More Than Accuracy

In standard classification problems, Accuracy is a common success metric. In aviation quality control, however, the real risk is a **False Negative** (a missed defect): if the model labels a part that is actually out of tolerance and requires rework as "sound," that defective part can be installed on the aircraft.

- **False Negative (Missed Defect):** A defective part is passed as "sound" → **flight safety risk**.
- **False Positive (Unnecessary Rework):** A sound part is flagged as "defective" and inspected again → only **time/cost loss**.

Because of this asymmetry, the model runs with a **0.25 threshold** instead of the default 0.50: as soon as a part's rework probability exceeds 25%, an "attention" signal is raised. This trades some Precision for higher Recall — a direct application of the "when in doubt, inspect" safety philosophy in aviation.

After adding realistic measurement noise (see the section below), the 0.25 threshold alone no longer fully reaches the 99%+ Recall target (~94.7% Recall) — this is an **expected and realistic** outcome, since measurement uncertainty is never zero on a real production line. Reaching 99%+ Recall on the test data requires lowering the threshold to ~0.003, at which point Precision drops to ~48% (nearly every other alert is a false alarm). This is exactly **why the dashboard has a live threshold slider**: a quality engineer can adjust the Recall/Precision trade-off live, based on that day's risk tolerance.

## Realism Model: Three Noise Layers

In the first version, the `Rework_Gerekli` (rework-required) label was derived from the same 4 features fed to the model via a *noise-free, deterministic* rule — which let the Random Forest memorize the rule and score 100% (a situation never seen in the real world). To fix this, three realistic noise layers were added to the data generation script:

1. **Operator experience effect (heteroscedastic process variation):** Inexperienced operators (e.g., 1 year) have looser process control, so the **true physical values** of the assembly are generated from a wider distribution (~1.4x standard deviation) than for experienced operators.
2. **Measurement/sensor noise:** Instruments such as calipers, pressure sensors, and depth gauges have their own precision limits. The rework decision is made based on the **true (physical) value**, but the dataset (and therefore the model) only contains the **measured (noisy) values** — just as on a real production line. This makes perfect separation (100% accuracy) statistically impossible.
3. **Human inspection/recording error:** To simulate quality inspector error on borderline cases, ~1.5% of labels are randomly flipped.

Example results after these changes (reproducible with seed=42):

| Metric | Value (Threshold = 0.25) |
|---|---|
| Accuracy | 90.10% |
| Precision | 84.27% |
| Recall | 94.65% |
| F1-Score | 89.16% |
| False Negatives | 46 / 860 positive cases |
| Threshold needed for 99%+ Recall | ~0.003 (Precision drops to ~48.25% at that point) |

## Project Structure

```
AeroRivet-QC-RF/
├── 01_veri_uretimi.py       # Synthetic assembly data generation (10,000 rows)
├── 02_model_egitimi.py      # Random Forest training + threshold optimization
├── app.py                   # Streamlit quality control dashboard
├── requirements.txt         # Python dependencies
├── havacilik_montaj_verisi.csv  # Generated synthetic data (created when the script runs)
├── montaj_modeli.pkl        # Trained model (created when the script runs)
└── metrikler.pkl            # Test data, probabilities, and metrics (created when the script runs)
```

## Dataset Variables

> **Rivet type note:** Countersinking is only applied to flush/countersunk-head (100° Flush Head) rivets such as **NAS1097**. **MS20470** (Universal/Protruding Head) rivets are not countersunk, since the head sits proud of the surface — this project therefore represents **NAS1097-type flush rivet assembly**; MS20470 is out of scope.

The `Delik_Capi_mm`, `Baski_Kuvveti_PSI`, and `Havsa_Derinligi_mm` columns in the CSV are **measured** (sensor/caliper-noisy) values — the model only ever sees these. The rework decision, however, is made based on the **true (physical)** values generated internally, so for some borderline rows the measured value and the label may not perfectly agree (realistic measurement uncertainty).

| Variable | Description | Underlying Distribution (true value) |
|---|---|---|
| `Delik_Capi_mm` | Measured hole diameter (nominal 4.76 mm) | Normal(μ=4.76, σ=0.025, ±1.4x by operator experience) + measurement noise (σ=0.008) |
| `Baski_Kuvveti_PSI` | Measured rivet squeeze force | Uniform(2000, 4000) + measurement noise (σ=40) |
| `Havsa_Derinligi_mm` | Measured countersink depth | Normal(μ=1.20, σ=0.015, ±1.4x by operator experience) + measurement noise (σ=0.006) |
| `Operator_Tecrube_Yil` | Operator experience in years | Integer, 1–15 |
| `Rework_Gerekli` | Target variable (0=Sound, 1=Rework) | Rule-based (on true values) + ~1.5% human inspection error |

**Rework rule (applied to the true/physical values):** A part requires rework if the hole diameter is `< 4.74 mm` or `> 4.84 mm` **OR** the squeeze force is `> 3500 PSI` **OR** the countersink depth is `> 1.25 mm`. See the [Realism Model](#realism-model-three-noise-layers) section for details.

## Setup

```bash
pip install -r requirements.txt
```

## Running the Project

```bash
# 1. Generate synthetic data
python 01_veri_uretimi.py

# 2. Train the model, run threshold analysis, save .pkl files
python 02_model_egitimi.py

# 3. Launch the dashboard
streamlit run app.py
```

Once the dashboard opens in the browser:
- **Tab 1 — Live Field Quality Control:** The operator enters measurement values; the system returns an instant quality decision.
- **Tab 2 — Model Performance & Reliability Analysis:** Accuracy/Recall/Precision/F1 metric cards, a live threshold slider, and a Precision-Recall curve (with the 99% Recall target point marked) are displayed.

## Technology Stack

- **scikit-learn** — Random Forest classifier
- **pandas / numpy** — data generation and processing
- **streamlit** — interactive dashboard
- **matplotlib / seaborn** — Precision-Recall visualization
- **joblib** — model and metrics serialization
