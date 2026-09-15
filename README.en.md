# AeroRivet-QC-RF

*[Türkçe](README.md) | English*

An end-to-end **Machine Learning-based Quality Control** project for aircraft structural assembly (rivet/hole) processes. Separately trained Random Forest classification models for two rivet types (**MS20470** and **NAS1097**) predict, from manufacturing-line measurement data, whether a part **requires rework**, and present the results through an interactive Streamlit dashboard.

## Project Summary

In aircraft fuselage assembly, rivet holes are subject to tight tolerances on parameters such as diameter, countersink depth, and squeeze force. Exceeding these tolerances can lead to defective connections that compromise structural integrity. But not every rivet type goes through the same checks: **MS20470** (Universal/Protruding Head, dome-shaped) rivets are never countersunk, while for **NAS1097** (100° Flush/Countersunk Head) rivets, countersink depth is also a critical quality parameter. This project:

1. Generates two separate synthetic manufacturing datasets based on the **FAA-H-8083-31A** (Aircraft Structural Repair) standard, each applying the tolerance rules appropriate to its rivet type,
2. Trains a separate **Random Forest** classifier per rivet type, each with its own feature set,
3. Optimizes Recall using a **low decision threshold (0.25)**, prioritizing flight safety,
4. Presents the results in a **Streamlit dashboard** where the user first selects the rivet type, then enters the relevant parameters.

## Two Rivet Types: MS20470 and NAS1097

| | **MS20470** | **NAS1097** |
|---|---|---|
| Head type | Universal / Protruding Head (dome-shaped) | 100° Flush / Countersunk Head |
| Countersink | Not cut — head sits proud of the surface | Cut — head sits flush with the surface |
| Model input features | Hole Diameter, Squeeze Force, Operator Experience (3 features) | Hole Diameter, Squeeze Force, Countersink Depth, Operator Experience (4 features) |
| Model/data file suffix | `_MS20470` | `_NAS1097` |

Because these two rivet types have **genuinely different feature schemas**, two independent models are trained instead of one — assigning a fake/placeholder countersink value to MS20470 data would be misleading, so that approach was avoided. In the dashboard, the user first selects the rivet type, and the form fields adjust automatically to match.

## Quality Logic: Why Recall Matters More Than Accuracy

In standard classification problems, Accuracy is a common success metric. In aviation quality control, however, the real risk is a **False Negative** (a missed defect): if the model labels a part that is actually out of tolerance and requires rework as "sound," that defective part can be installed on the aircraft.

- **False Negative (Missed Defect):** A defective part is passed as "sound" → **flight safety risk**.
- **False Positive (Unnecessary Rework):** A sound part is flagged as "defective" and inspected again → only **time/cost loss**.

Because of this asymmetry, both models run with a **0.25 threshold** instead of the default 0.50: as soon as a part's rework probability exceeds 25%, an "attention" signal is raised. This trades some Precision for higher Recall — a direct application of the "when in doubt, inspect" safety philosophy in aviation.

After adding realistic measurement noise (see the section below), the 0.25 threshold alone no longer fully reaches the 99%+ Recall target (~94% Recall) — this is an **expected and realistic** outcome, since measurement uncertainty is never zero on a real production line. This is exactly **why the dashboard has a live threshold slider**: a quality engineer can adjust the Recall/Precision trade-off live, based on that day's risk tolerance.

## Realism Model: Three Noise Layers

In the first version, the `Rework_Gerekli` (rework-required) label was derived from the same features fed to the model via a *noise-free, deterministic* rule — which let the Random Forest memorize the rule and score 100% (a situation never seen in the real world). To fix this, three realistic noise layers were added to the data generation script (applied to both rivet types):

1. **Operator experience effect (heteroscedastic process variation):** Inexperienced operators (e.g., 1 year) have looser process control, so the **true physical values** of the assembly are generated from a wider distribution (~1.4x standard deviation) than for experienced operators.
2. **Measurement/sensor noise:** Instruments such as calipers, pressure sensors, and depth gauges have their own precision limits. The rework decision is made based on the **true (physical) value**, but the dataset (and therefore the model) only contains the **measured (noisy) values** — just as on a real production line. This makes perfect separation (100% accuracy) statistically impossible.
3. **Human inspection/recording error:** To simulate quality inspector error on borderline cases, ~1.5% of labels are randomly flipped.

Example results after these changes (reproducible with seed=42):

| Metric | MS20470 (Threshold = 0.25) | NAS1097 (Threshold = 0.25) |
|---|---|---|
| Accuracy | 90.35% | 90.40% |
| Precision | 84.97% | 84.50% |
| Recall | 94.03% | 94.97% |
| F1-Score | 89.27% | 89.43% |
| False Negatives | 51 / 854 positive cases | 43 / 855 positive cases |
| Threshold needed for 99%+ Recall | ~0.010 (Precision drops to ~58.55%) | ~0.023 (Precision drops to ~61.38%) |

## Project Structure

```
AeroRivet-QC-RF/
├── 01_veri_uretimi.py                      # Synthetic data generation for both rivet types (10,000 rows each)
├── 02_model_egitimi.py                     # Separate Random Forest training + threshold analysis per rivet type
├── app.py                                  # Streamlit dashboard with a rivet-type selector
├── requirements.txt                        # Python dependencies
├── havacilik_montaj_verisi_MS20470.csv     # MS20470 synthetic data (created when the script runs)
├── havacilik_montaj_verisi_NAS1097.csv     # NAS1097 synthetic data (created when the script runs)
├── montaj_modeli_MS20470.pkl               # MS20470 trained model (created when the script runs)
├── montaj_modeli_NAS1097.pkl               # NAS1097 trained model (created when the script runs)
├── metrikler_MS20470.pkl                   # MS20470 test data/metrics (created when the script runs)
└── metrikler_NAS1097.pkl                   # NAS1097 test data/metrics (created when the script runs)
```

## Dataset Variables

The measurement columns in the CSVs are **measured** (sensor/caliper-noisy) values — the model only ever sees these. The rework decision, however, is made based on the **true (physical)** values generated internally, so for some borderline rows the measured value and the label may not perfectly agree (realistic measurement uncertainty).

| Variable | Description | Underlying Distribution (true value) | MS20470 | NAS1097 |
|---|---|---|:---:|:---:|
| `Delik_Capi_mm` | Measured hole diameter (nominal 4.76 mm) | Normal(μ=4.76, σ=0.025, ±1.4x by operator experience) + measurement noise (σ=0.008) | Yes | Yes |
| `Baski_Kuvveti_PSI` | Measured rivet squeeze force | Uniform(2000, 4000) + measurement noise (σ=40) | Yes | Yes |
| `Havsa_Derinligi_mm` | Measured countersink depth | Normal(μ=1.20, σ=0.015, ±1.4x by operator experience) + measurement noise (σ=0.006) | No | Yes |
| `Operator_Tecrube_Yil` | Operator experience in years | Integer, 1–15 | Yes | Yes |
| `Rework_Gerekli` | Target variable (0=Sound, 1=Rework) | Rule-based (on true values) + ~1.5% human inspection error | Yes | Yes |

**Rework rule (applied to the true/physical values):**
- **MS20470:** A part requires rework if the hole diameter is `< 4.74 mm` or `> 4.84 mm` **OR** the squeeze force is `> 3500 PSI`.
- **NAS1097:** In addition to the two conditions above, a part also requires rework if the countersink depth is `> 1.25 mm`.

See the [Realism Model](#realism-model-three-noise-layers) section for details.

## Setup

```bash
pip install -r requirements.txt
```

## Running the Project

```bash
# 1. Generate synthetic data for both rivet types
python 01_veri_uretimi.py

# 2. Train both models, run threshold analysis, save .pkl files
python 02_model_egitimi.py

# 3. Launch the dashboard
streamlit run app.py
```

Once the dashboard opens in the browser:
- **Tab 1 — Live Field Quality Control:** The user first selects the rivet type (MS20470 / NAS1097); the form only shows the parameters relevant to that type (the Countersink Depth field is hidden for MS20470). The operator enters measurement values, and the system returns an instant quality decision.
- **Tab 2 — Model Performance & Reliability Analysis:** Based on the selected rivet type, Accuracy/Recall/Precision/F1 metric cards, a live threshold slider, and a Precision-Recall curve (with the 99% Recall target point marked) are displayed.

## Technology Stack

- **scikit-learn** — Random Forest classifier
- **pandas / numpy** — data generation and processing
- **streamlit** — interactive dashboard
- **matplotlib / seaborn** — Precision-Recall visualization
- **joblib** — model and metrics serialization
